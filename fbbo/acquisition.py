# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import typing
import botorch
import gpytorch
import numpy as np
from botorch.models.model import Model
import torch.distributions


class PosteriorMeanMinimize(botorch.acquisition.PosteriorMean):
    def __init__(self, model, objective=None, maximize=False):
        super().__init__(model=model, objective=objective)

    def forward(self, X):
        # negate the posterior mean because we maximise this acq function
        return -super().forward(X)


def averager(X: torch.Tensor, take_average: bool = True) -> torch.Tensor:
    if take_average:
        X = X.mean(dim=0).view(-1)
    return X


# here we're assuming that model is actually a batch of >1 models
class BatchAverageExpectedImprovement(botorch.acquisition.ExpectedImprovement):
    def __init__(
        self,
        model: Model,
        best_f: typing.Union[float, torch.Tensor],
        objective: typing.Optional[
            botorch.acquisition.ScalarizedObjective
        ] = None,
        maximize: bool = True,
        return_average: bool = True,
    ) -> None:
        super().__init__(model, best_f, objective, maximize)
        self.return_average = return_average

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # optimize_acqf will give b1 x 1 x d tensors,
        # so reshape to b1 x d
        if X.ndim == 3:
            X = X.squeeze(1)

        # calculate EI for each model
        self.best_f = self.best_f.to(X)

        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        # at this point ei will be (nmodels, b1, 1)
        ei = averager(ei, self.return_average)
        return ei


class BatchAverageUCB(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: float,
        maximize: bool = True,
        return_average: bool = True,
    ) -> None:
        super().__init__(model=model)

        self.sqrtbeta = torch.sqrt(torch.tensor(float(beta)))
        self.maximize = maximize
        self.return_average = return_average

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # optimize_acqf will give b1 x 1 x d tensors,
        # so reshape to b1 x d
        if X.ndim == 3:
            X = X.squeeze(1)

        # calculate EI for each model
        self.sqrtbeta = self.sqrtbeta.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()

        if self.maximize:
            ucb = mean + self.sqrtbeta * sigma
        else:
            ucb = -(mean - self.sqrtbeta * sigma)

        # at this point ucb will be (nmodels, b1, 1)
        ucb = averager(ucb, self.return_average)

        return ucb


class BatchAveragePosteriorMean(
    botorch.acquisition.AnalyticAcquisitionFunction
):
    def __init__(
        self, model: Model, maximize: bool = True, return_average: bool = True,
    ):
        super().__init__(model=model)
        self.maximize = maximize
        self.return_average = return_average

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # optimize_acqf will give b1 x 1 x d tensors,
        # so reshape to b1 x d
        if X.ndim == 3:
            X = X.squeeze(1)

        posterior = self._get_posterior(X=X)

        # mean should be (nmodels, b1, 1) shape
        mean = posterior.mean

        if not self.maximize:
            mean = -mean

        mean = averager(mean, self.return_average)

        return mean


class AcqOptimiserBase:
    def __init__(self, mll, problem_bounds):
        self.model = mll.model
        self.problem_bounds = problem_bounds

        # check if we're in batch mode, in which case the training data will
        # have shape (b, n, d) instead of (n, d)
        self.batchmode = self.model.train_inputs[0].ndim == 3
        self.acq_params = {"maximize": False}
        self.acq_func = self._get_acq_func()

    def _get_acq_func(self):
        raise NotImplementedError

    def optimise(self, num_restarts=10, raw_samples=1000, batch_limit=500):
        # num_restarts: number of best samples to select
        # raw_samples: number of samples to generate to select from
        # batch_limit: batch size chunks of the raw samples evaluation
        #              i.e. we perform bl lots of raw_samples/bl evaluations

        # optimise acquisition function
        with gpytorch.settings.cholesky_jitter(1e-3):
            train_xnew, acq_f = botorch.optim.optimize_acqf(
                acq_function=self.acq_func,
                bounds=self.problem_bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": batch_limit},
            )

        return train_xnew


class AcqOptEI(AcqOptimiserBase):
    def __init__(self, mll, problem_bounds):
        AcqOptimiserBase.__init__(self, mll, problem_bounds)

    def _get_acq_func(self):
        self.acq_params["best_f"] = self.model.train_targets.min()

        if self.batchmode:
            acq_func = BatchAverageExpectedImprovement
        else:
            acq_func = botorch.acquisition.ExpectedImprovement

        return acq_func(self.model, **self.acq_params)


class AcqOptUCB(AcqOptimiserBase):
    def __init__(self, mll, problem_bounds):
        AcqOptimiserBase.__init__(self, mll, problem_bounds)

    def _get_acq_func(self):
        t = self.model.train_targets.numel()
        delta = 0.01
        D = self.model.train_inputs[0].shape[1]

        self.acq_params["beta"] = 2 * np.log(
            D * t ** 2 * np.pi ** 2 / (6 * delta)
        )

        if self.batchmode:
            acq_func = BatchAverageUCB
        else:
            acq_func = botorch.acquisition.UpperConfidenceBound

        return acq_func(self.model, **self.acq_params)


class AcqOptMean(AcqOptimiserBase):
    def __init__(self, mll, problem_bounds):
        AcqOptimiserBase.__init__(self, mll, problem_bounds)

    def _get_acq_func(self):
        if self.batchmode:
            acq_func = BatchAveragePosteriorMean
        else:
            acq_func = PosteriorMeanMinimize

        return acq_func(self.model, **self.acq_params)
