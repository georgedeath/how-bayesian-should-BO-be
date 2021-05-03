# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Utility functions for the the mcmc/advi sampling
"""

import torch
import pymc3 as pm
import numpy as np
from typing import Any, Dict

# make pymc3 stop spamming
import logging

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


def emcee_samples_into_pyro_samples(
    trace: np.ndarray,
    prior_info: Dict[int, Dict[str, Any]],
    samples_required: int,
) -> Dict:
    # reshape the samples from (chains, samples, dim) to (chains * samples, dim)
    ndim = sum(v["dim"] for v in prior_info.values())
    trace = np.reshape(trace, (-1, ndim))

    # ensure that we only give back the required number of samples
    if trace.shape[0] != samples_required:
        trace = trace[-samples_required:, :]

    # put samples into a dictionary in a form that can be loaded by gpytorch
    sample_dict = {}
    sample_idx = 0

    for i in prior_info:
        name = prior_info[i]["name"]
        dim = prior_info[i]["dim"]
        dtype = prior_info[i]["dtype"]

        # slice out the correct indices data for the corresponding hyperparameter
        samples = trace[:, sample_idx : sample_idx + dim]

        # convert from numpy to torch and cast to the correct dtype
        samples = torch.from_numpy(samples).squeeze()
        samples = samples.to(dtype=dtype)

        # reshape the lengthscale to (#samples, 1, #lengthscales) for gpytorch
        if "lengthscale" in name:
            samples = samples.view(-1, 1, dim)

        sample_dict[name] = samples

        sample_idx += dim

    return sample_dict


def pm_samples_into_pyro_samples(trace, prior_info, thin=1):
    # put samples into a dictionary in a form that can be loaded by gpytorch
    sample_dict = {}

    for i in prior_info:
        name = prior_info[i]["name"]
        dtype = prior_info[i]["dtype"]
        dim = prior_info[i]["dim"]

        # convert to the advi name - we just remove 'raw_' here
        advi_name = name.split(".")[-1].split("_")[0]
        samples = trace[advi_name, ::thin]

        # convert from numpy to torch and cast to the correct dtype
        samples = torch.from_numpy(samples).squeeze()
        samples = samples.to(dtype=dtype)

        # reshape the lengthscale to (#samples, 1, #lengthscales) for gpytorch
        if "lengthscale" in name:
            samples = samples.view(-1, 1, dim)

        # noise needs to be (#samples, 1) for whatever reason
        # note that outputscale is fine being (#samples, )
        if "noise" in name:
            samples = samples.view(-1, 1)

        sample_dict[name] = samples

    return sample_dict


def create_pm3_gp(mll):
    # extract the stuff we need from the mll
    X = mll.model.train_inputs[0].numpy()
    y = mll.model.train_targets.numpy()
    d = X.shape[1]

    # determine if noisy
    noisy = hasattr(mll.model.likelihood.noise_covar, "noise_prior")

    if noisy:
        gp_noise_prior = mll.model.likelihood.noise_covar.noise_prior
        noise_size = None
    else:
        gp_noise_prior = None
        noise_size = mll.likelihood.noise_covar.noise[0].item()

    gp_ls_prior = mll.model.covar_module.base_kernel.lengthscale_prior
    gp_os_prior = mll.model.covar_module.outputscale_prior

    # check if we're using ARD
    n_lengthscales = mll.model.covar_module.base_kernel.raw_lengthscale.numel()
    ard_dim = n_lengthscales if n_lengthscales > 1 else 1

    # create the Gaussian process
    with pm.Model() as model:

        # extract Gamma prior parameters from model
        ls_alpha = gp_ls_prior.concentration.item()
        ls_beta = gp_ls_prior.rate.item()
        os_alpha = gp_os_prior.concentration.item()
        os_beta = gp_os_prior.rate.item()

        # priors
        ls_prior = pm.Gamma(
            name="lengthscale", alpha=ls_alpha, beta=ls_beta, shape=ard_dim
        )
        os_prior = pm.Gamma(name="outputscale", alpha=os_alpha, beta=os_beta)

        if noisy:
            noise_alpha = gp_noise_prior.concentration.item()
            noise_beta = gp_noise_prior.rate.item()
            noise_prior = pm.Gamma(
                name="noise", alpha=noise_alpha, beta=noise_beta
            )
        else:
            noise_prior = None

        # Specify the covariance function.
        cov = pm.gp.cov.Matern52(d, ls=ls_prior)
        cov *= pm.gp.cov.Constant(os_prior)

        # set up the gp
        gp = pm.gp.Marginal(cov_func=cov)

        # get the marginal likelihood
        _ = gp.marginal_likelihood(
            "y", X=X, y=y, noise=noise_prior if noisy else noise_size
        )

    return model


def get_prior_info_dict(mll):
    prior_info = {}

    for i, (name, module, prior, closure, inv_closure) in enumerate(
        mll.model.named_priors()
    ):
        data = closure(module)

        prior_info[i] = {
            "name": name,
            "prior": prior,
            "getter": lambda: closure(module),
            "setter": inv_closure,
            "dim": data.numel(),
            "dtype": data.dtype,
            "shape": data.size(),
        }

    return prior_info
