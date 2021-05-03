# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import torch
import gpytorch
import numpy as np
from . import (
    acquisition,
    transforms,
    gp,
    advi,
    util,
    mcmc_pm3,
)


def perform_BO_step(
    Xtr,
    Ytr,
    problem_bounds,
    method,
    acq_name,
    transform_name="Transform_Standardize",
    use_ard=False,
    mcmc_settings={},
    verbose=False,
    noisy=False,
):

    # scale Y
    output_transform = getattr(transforms, transform_name)
    T_out = output_transform(Ytr)
    train_y = T_out.scale_mean(Ytr)
    train_x = Xtr

    # get an untrained gpytorch model
    ls_prior = gpytorch.priors.GammaPrior(3, 6)
    os_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
    noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)

    if noisy:
        # use a noise prior
        model = gp.GP_NoisyOutput(
            train_x, train_y, ls_prior, os_prior, noise_prior, ARD=use_ard
        )
    else:
        # fixed noise size
        noise_size = 1e-4
        model = gp.GP_FixedNoise(
            train_x, train_y, ls_prior, os_prior, noise_size, ARD=use_ard
        )
    mll = gp.ExactMarginalLogLikelihood(model.likelihood, model)

    # either fit the model (if we're doing MAP) or perform the MCMC/VI
    if method == "map":
        gp.train_model_restarts(mll, n_restarts=10, verbose=verbose)

    else:
        if "vi" in method:
            if method == "vimf":
                advi_method = "advi"

            elif method == "vifr":
                advi_method = "fullrank_advi"

            sample_dict = advi.perform_advi(
                mll,
                n_steps=40000,
                samples_required=mcmc_settings["samples_required"],
                method=advi_method,  # type: ignore
                progress=verbose,
            )

        elif method == "mcmc_pm3":
            chains = 1

            sample_dict = mcmc_pm3.perform_mcmc(
                mll,
                mcmc_settings["samples_required"],
                mcmc_settings["discard"],
                mcmc_settings["thin"],
                chains,
                progress=verbose,
            )

        # load the mcmc/VI samples into the model
        model.pyro_load_from_samples(sample_dict)  # type: ignore

    # optimise the acquisition function
    acq_classes = {
        "ei": acquisition.AcqOptEI,
        "ucb": acquisition.AcqOptUCB,
    }

    acq_optimiser_class = acq_classes[acq_name]
    acq_optimiser = acq_optimiser_class(mll, problem_bounds)

    # put the model into evaluation mode for optimisation
    mll.eval()
    train_xnew = acq_optimiser.optimise(batch_limit=500)

    # make sure we give back a location further than 1e-4 away from
    # all training locations to (hopefully) avoid numerical issues.
    train_xnew = util.give_safer_location(
        train_x,
        train_xnew,
        lb=problem_bounds[0],
        ub=problem_bounds[1],
        tol=1e-4,
    )

    return train_xnew


def bo(
    problem_name,
    problem_params,
    run_no,
    budget,
    method,
    acq_name,
    use_ard,
    data_path,
    save_path,
    save_every=10,
    verbose=False,
    mcmc_settings={"discard": 2048, "thin": 50, "samples_required": 256},
    noise_file=None,
):
    noisy = noise_file is not None

    # check if we're resuming a saved run
    if os.path.exists(save_path):
        load_path = save_path
        print("Resuming run:", save_path)
    else:
        load_path = data_path
        print("Starting run:", data_path)

    # load the training data - we're assuming here it is in [0, 1]^d
    data = torch.load(load_path)
    Xtr = data["Xtr"]
    Ytr = data["Ytr"]

    # if it has additional arguments add them to the dictionary passed to f
    if "problem_params" in data:
        problem_params.update(data["problem_params"])
    problem_params["run_no"] = run_no

    print(f"Training data shape: {Xtr.shape}")

    # load the problem instance
    f = util.test_func_getter(problem_name, problem_params)

    # if noisy, load the noise level for the problem and wrap it
    if noisy:
        with np.load(noise_file, allow_pickle=True) as data:
            pnd = data["pnd"].item()

        D = pnd[problem_name][Xtr.shape[1]]

        try:
            stdev = D[run_no]["stdev"]
        except KeyError:
            stdev = D["stdev"]

        f = util.NoisyProblem(f, stdev)

    # wrap the problem for torch and so that it resides in [0, 1]^d
    f = util.TorchProblem(util.UniformProblem(f))
    problem_bounds = torch.stack((f.lb, f.ub))

    assert (Xtr.dtype == Ytr.dtype) and (Xtr.dtype == problem_bounds.dtype)

    while Xtr.shape[0] < budget:

        train_xnew = perform_BO_step(
            Xtr,
            Ytr,
            problem_bounds,
            method,
            acq_name,
            transform_name="Transform_Standardize",
            use_ard=use_ard,
            mcmc_settings=mcmc_settings,
            verbose=verbose,
            noisy=noisy,
        )

        # evaluate the function
        train_ynew = f(train_xnew)

        # store the result
        Xtr = torch.cat((Xtr, train_xnew))
        Ytr = torch.cat((Ytr, train_ynew))

        # print some info about the run
        s = f"Iteration {Xtr.shape[0]:> 3d}"
        s += f"\n\tFitness: {train_ynew.item():g}"
        if hasattr(f, "yopt"):
            try:
                yopt = f.yopt.item()
            # ValueError: only one element tensors can be converted to Python scalars
            except ValueError:
                yopt = f.yopt[0].item()

            s += f"\n\tRegret: {float(Ytr.min().item()) - float(yopt):g}\n"
        print(s)

        # save the run
        if (Xtr.shape[0] % save_every == 0) or (Xtr.shape[0] == budget):
            save_dict = {
                "Xtr": Xtr,
                "Ytr": Ytr,
                "problem_params": problem_params,
            }
            torch.save(obj=save_dict, f=save_path)

    print("Finished run")
