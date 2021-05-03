# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from . import test_problems

import os
import torch
import numpy as np
from typing import Dict


class UniformProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.real_lb = problem.lb
        self.real_ub = problem.ub

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        if problem.xopt is not None:
            self.xopt = (problem.xopt - problem.lb) / (problem.ub - problem.lb)
        else:
            self.xopt = problem.xopt

        self.yopt = problem.yopt

        self.real_cf = problem.cf
        self.set_cf()

    def __call__(self, x: np.ndarray):
        x = np.atleast_2d(x)

        # map x back to original space
        x = x * (self.real_ub - self.real_lb) + self.real_lb

        return self.problem(x)

    def set_cf(self):
        if self.real_cf is None:
            self.cf = None
            return

        def cf_wrapper(x: np.ndarray):
            x = np.atleast_2d(x)

            # map x back to original space
            x = x * (self.real_ub - self.real_lb) + self.real_lb

            return self.real_cf(x)

        self.cf = cf_wrapper


class NoisyProblem:
    def __init__(self, problem, std: float):
        self.problem = problem
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.xopt = problem.xopt
        self.yopt = problem.yopt
        self.cf = problem.cf

        self.std = std

    def __call__(self, x: np.ndarray):
        # call the problem
        fx = self.problem(x)

        # add noise
        fx += np.random.normal(loc=0, scale=self.std, size=fx.shape)

        return fx


class TorchProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.lb = torch.from_numpy(problem.lb)
        self.ub = torch.from_numpy(problem.ub)

        if problem.xopt is not None:
            self.xopt = torch.from_numpy(problem.xopt)
        else:
            self.xopt = problem.xopt

        self.yopt = torch.from_numpy(problem.yopt)

        if self.problem.cf is not None:

            def cf(x):
                if not isinstance(x, np.ndarray):
                    x = x.numpy()
                return self.problem.cf(x)

            self.cf = cf
        else:
            self.cf = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.problem(x.numpy())
        torchfx = torch.from_numpy(fx)

        # cast to same datatype as x
        torchfx = torchfx.to(x.dtype)
        return torchfx


def generate_save_filename(
    method_name: str,  # map, vimf, vifr, mcmc
    acq_name: str,  # ei, ucb, egreedy
    budget: int,  # 200
    problem_name,  # function name
    run_no: int,
    problem_params={},
    use_ard=False,
    repeat_no=None,
    results_dir_prefix="results",
    noise_level: float = 0.0,
) -> str:
    noisy = noise_level > 0.0

    if not noisy:
        results_dir = results_dir_prefix
    else:
        results_dir = f"{results_dir_prefix:s}_noise_{noise_level:g}"

    # append dim if different from default
    if "d" in problem_params:
        problem_name = f'{problem_name:s}{problem_params["d"]:d}'

    fname_components = [
        "fbbo",
        "_noisy" if noisy else "",
        f"_{budget:d}",
        "_ARD" if use_ard else "",
        f"_{method_name:s}",
        f"_{acq_name:s}",
        f"_{problem_name:s}",
        f"_run={run_no:03d}",
        f"-{repeat_no:d}" if repeat_no is not None else "",
        ".pt",
    ]

    fname = "".join(fname_components)

    return os.path.join(results_dir, fname)


def generate_data_filename(
    problem_name: str,
    run_no: int,
    problem_params: Dict = {},
    repeat_no: int = None,
    data_dir: str = "data",
    noise_level: float = 0.0,
) -> str:
    noisy = noise_level > 0.0

    if not noisy:
        data_dir = data_dir
    else:
        data_dir = f"{data_dir:s}_noise_{noise_level:g}"

    # append dim if different from default
    if "d" in problem_params:
        problem_name = f'{problem_name:s}{problem_params["d"]:d}'

    fname_components = [
        f"{problem_name:s}",
        "_noisy" if noisy else "",
        f"_{run_no:03d}",
        f"-{repeat_no:d}" if repeat_no is not None else "",
        ".pt",
    ]

    fname = "".join(fname_components)

    return os.path.join(data_dir, fname)


def give_safer_location(
    Xtr: torch.Tensor,
    xnew: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tol: float = 1e-7,
    MAX_ATTEMPTS: int = 10000,
) -> torch.Tensor:
    xn = xnew.squeeze()

    # check the distance from xnew to all of Xtr
    D = torch.norm(Xtr - xn, dim=1)

    # get the locations that are within the tolerance
    mask = torch.nonzero(D < tol, as_tuple=False).reshape(-1)

    # if the mask is empty, xnew is fine
    maskn = mask.numel()
    if maskn == 0:
        return xnew

    # else we need to sample a new location that is
    # more than tol away from the close samples
    Xtr = Xtr[mask, :]
    xdim = Xtr.shape[1]

    N = 10000

    for _ in range(MAX_ATTEMPTS):
        # randomly choose one of the points that are too close
        idx = torch.randint(low=0, high=maskn, size=(1,))

        # draw N samples around that location with a stdev of tol
        newX = Xtr[idx, :] + torch.randn((N, xdim)) * tol

        # find those that are > tol from all locations
        newX = newX.unsqueeze(1)  # shape (N, 1, d)
        sXtr = Xtr.unsqueeze(0)  # shape (1, maskn, d)

        D = torch.norm(newX - sXtr, dim=2)
        valid_mask = torch.all(D > tol, dim=1)

        # also check that they are in bounds
        valid_mask &= torch.all(newX >= lb, dim=2).squeeze()
        valid_mask &= torch.all(newX <= ub, dim=2).squeeze()

        nvalid = valid_mask.sum().item()

        if nvalid == 0:
            continue

        # indices of valid locations in newX
        valid_inds = torch.arange(N)[valid_mask]

        # randomly choose an index
        ridx = int(torch.randint(int(nvalid), size=(1,)).item())
        selected_idx = valid_inds[ridx]

        xn = newX[selected_idx]

        # sanity check
        D = torch.norm(Xtr - xn, dim=1)
        mask = torch.nonzero(D < tol, as_tuple=False).reshape(-1)
        assert mask.numel() == 0

        # reshape the new location to match the original one's
        xn = xn.reshape(xnew.shape)

        return xn

    # if we've got to this point, we've failed MAX_ATTEMPTS times..
    err = "Unable to generate a location further away from the training data"
    err += f"\nthan {tol:g} after {MAX_ATTEMPTS:d} attempts."
    raise ValueError(err)


def generate_noise_filename(noise_level: float) -> str:
    noise_file = f"problem_noise_dict_{noise_level:g}.npz"

    if not os.path.exists(noise_file):
        raise ValueError(
            f"Noise file ({noise_file:s}) corresponding to"
            f" noise level ({noise_level:g}) does not exist."
        )
    return noise_file


def test_func_getter(name: str, problem_params: Dict = {}):
    if "run_no" in problem_params:
        del problem_params["run_no"]

    f = getattr(test_problems, name)(**problem_params)

    return f
