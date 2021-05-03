# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .sampling import (
    create_pm3_gp,
    pm_samples_into_pyro_samples,
    get_prior_info_dict,
)
import numpy as np
import pymc3 as pm
from itertools import cycle
from . import gp


def perform_mcmc(
    mll: gp.ExactMarginalLogLikelihood,
    samples_required: int = 128,
    discard: int = 200,
    thin: int = 20,
    chains: int = 4,
    progress: bool = False,
    max_failures: int = 4,
):
    n_failures = 0

    init_selector = cycle(
        ["advi_map", "advi+adapt_diag", "jitter+adapt_diag", "map"]
    )

    # sometimes the mcmc will fail due to rubbish chain ending up in a location
    # with zero gradient; when that happens, restart up to max_failures times
    while n_failures < max_failures:
        model = create_pm3_gp(mll)
        draws = np.ceil(samples_required / chains).astype("int") * thin
        init_method = next(init_selector)

        if progress:
            print("Initialisation method:", init_method)

        try:
            with model:
                trace = pm.sample(
                    draws=draws,
                    tune=discard,
                    chains=chains,
                    progressbar=progress,
                    compute_convergence_checks=True,
                    cores=chains,
                    target_accept=0.95,
                    init=init_method,
                    return_inferencedata=False,  # so we get returned the trace
                    # max advi steps
                    n_init=20000,
                )

            # convert to gpytorch dictionary for sample loading
            prior_info = get_prior_info_dict(mll)
            sample_dict = pm_samples_into_pyro_samples(
                trace, prior_info, thin=thin
            )

            return sample_dict

        except Exception:
            import traceback

            traceback.print_exc()
            n_failures += 1

    if n_failures == max_failures:
        errmsg = f"Failed to complete MCMC {max_failures} times."
        raise ValueError(errmsg)
