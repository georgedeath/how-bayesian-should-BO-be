# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pymc3.variational

from .sampling import (
    create_pm3_gp,
    pm_samples_into_pyro_samples,
    get_prior_info_dict,
)

# make pymc3 stop spamming
import logging

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


def get_vi_func(method_name):
    if method_name == "advi":
        return pymc3.variational.ADVI()
    elif method_name == "fullrank_advi":
        return pymc3.variational.FullRankADVI()
    else:
        raise ValueError(f"Invalid method name: {method_name:s}")


def perform_advi(
    mll, n_steps=20000, samples_required=1000, method="advi", progress=False,
):
    # methods:
    # "advi" for mean-field advi
    # "fullrank_advi" for full rank advi

    success = False
    stepsize = 500
    vi = None

    while (n_steps > stepsize) and not success:
        with create_pm3_gp(mll):
            vi = get_vi_func(method)
            vi.fit(
                n=stepsize, progressbar=progress, score=progress,
            )

            success = True

            starting_steps = stepsize

            for starting_steps in range(
                stepsize, n_steps + stepsize, stepsize
            ):
                try:
                    vi.refine(n=stepsize, progressbar=progress)
                except (ValueError, FloatingPointError) as e:
                    print(e)
                    print("VI error, trying to continue with less steps")
                    success = False
                    raise

            n_steps = starting_steps - stepsize

    if not success:
        print("Unable to do any VI, randomly drawing hyperparams from prior")
        with create_pm3_gp(mll):
            vi = get_vi_func(method)

    # draw samples
    trace = vi.approx.sample(samples_required)

    # convert to gpytorch dictionary for sample loading
    prior_info = get_prior_info_dict(mll)
    sample_dict = pm_samples_into_pyro_samples(trace, prior_info)

    return sample_dict
