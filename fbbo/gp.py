# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import copy
import torch
import botorch
import gpytorch
import gpytorch.constraints
from botorch.models.gpytorch import GPyTorchModel


class GP_FixedNoise(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(
        self, train_x, train_y, ls_prior, os_prior, noise_size=1e-4, ARD=False
    ):

        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            torch.full_like(train_y, noise_size)
        )
        super(GP_FixedNoise, self).__init__(train_x, train_y, likelihood)

        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=ls_prior,
            nu=5 / 2,
            ard_num_dims=train_x.shape[-1] if ARD else None,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, outputscale_prior=os_prior,
        )
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_NoisyOutput(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(
        self, train_x, train_y, ls_prior, os_prior, noise_prior, ARD=False
    ):

        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.constraints.GreaterThan(
                1e-4, transform=None, initial_value=noise_prior_mode,
            ),
        )
        super(GP_NoisyOutput, self).__init__(train_x, train_y, likelihood)

        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=ls_prior,
            nu=5 / 2,
            ard_num_dims=train_x.shape[-1] if ARD else None,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, outputscale_prior=os_prior,
        )
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_restarts(mll, n_restarts, verbose=False):
    def model_loss(mll):
        mll.train()
        output = mll.model(*mll.model.train_inputs)
        loss = -mll(output, mll.model.train_targets)
        return loss.sum().item()

    noise_prior = "noise_prior" in mll.model.likelihood.noise_covar._priors

    # start by assuming the current parameters are the best
    best_params = copy.deepcopy(mll.state_dict())
    best_loss = model_loss(mll)

    for i in range(n_restarts):
        params_sampled = []
        # sample new hyperparameters from the kernel priors
        # only if they require gradient info, i.e. are optimisable
        if mll.model.covar_module.base_kernel.lengthscale.requires_grad:
            mll.model.covar_module.base_kernel.sample_from_prior(
                "lengthscale_prior"
            )
            params_sampled.append("lengthscale")

        if mll.model.covar_module.outputscale.requires_grad:
            mll.model.covar_module.sample_from_prior("outputscale_prior")
            params_sampled.append("outputscale")

        #  if we have one, sample from noise prior
        if noise_prior and mll.model.likelihood.noise.requires_grad:
            mll.model.likelihood.noise_covar.sample_from_prior("noise_prior")
            params_sampled.append("noise")

        # try and fit the model using bfgs, starting at the sampled params
        botorch.fit_gpytorch_model(mll, method="L-BFGS-B")

        # calculate the loss
        curr_loss = model_loss(mll)
        if verbose:
            print(
                f"  LML: {i:d}: Loss: {curr_loss:0.4f} Sampled: {params_sampled}"
            )

        # if we've ended up with better hyperparams, save them to use
        if curr_loss < best_loss:
            best_params = copy.deepcopy(mll.state_dict())
            best_loss = curr_loss

    # load the best found parameters into the model
    mll.load_state_dict(best_params)

    if verbose:
        ls = mll.model.covar_module.base_kernel.lengthscale.detach().numpy()
        ops = mll.model.covar_module.outputscale.item()
        print("Best found hyperparameters:")
        print(f"\tLengthscale(s): {ls.ravel()}")
        print(f"\tOutputscale: {ops}")
        if noise_prior:
            ns = mll.model.likelihood.noise.item()
            print(f"\tNoise: {ns}")


class ExactMarginalLogLikelihood(
    gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood
):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian process with a
    Gaussian likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(
            likelihood, gpytorch.likelihoods._GaussianLikelihoodBase
        ):
            raise RuntimeError(
                "Likelihood must be Gaussian for exact inference"
            )
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for _, module, prior, closure, _ in self.named_priors():
            val = prior.log_prob(closure(module))

            if val.ndim == 3:
                val = val.sum((1, 2))
            elif val.ndim == 2:
                val = val.sum(1)

            res.add_(val.squeeze())

        return res

    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(
            function_dist, gpytorch.distributions.MultivariateNormal
        ):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)

    def pyro_factor(self, output, target, *params):
        import pyro

        mll = self(output, target, *params)
        pyro.factor("gp_mll", mll)
        return mll
