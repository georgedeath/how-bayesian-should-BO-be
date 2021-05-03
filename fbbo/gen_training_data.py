# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import torch
import numpy as np

from pyDOE2 import lhs
from . import util


def generate_training_data_LHS(
    problem_name,
    n_exp_start=1,
    n_exp_end=51,
    n_samples=None,
    n_repeats=None,
    additional_arguments={},
    noise_level: float = 0.0,
):
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)
    N = len(exp_nos)
    noisy = noise_level > 0.0

    if noisy:
        noise_file = util.generate_noise_filename(noise_level)
        with np.load(noise_file, allow_pickle=True) as data:
            pnd = data["pnd"].item()
    else:
        pnd = None

    # check that there are the same number of arguments as there are
    # experimental training data to construct
    for _, v in additional_arguments.items():
        assert len(v) == N, (
            "There should be as many elements for each "
            "optional arguments as there are experimental runs"
        )

    for i, run_no in enumerate(exp_nos):
        # get the optional arguments for this problem instance (if they exist)
        problem_params = {k: v[i] for (k, v) in additional_arguments.items()}
        problem_params["run_no"] = run_no

        # instantiate the function, uniform wrap it and torch wrap it
        f_original = util.test_func_getter(problem_name, problem_params)

        if noisy:
            D = pnd[problem_name][f_original.dim]
            if run_no in D:
                D = D[run_no]

            stdev = D["stdev"]

            f_original = util.NoisyProblem(f_original, stdev)

        f_uniform = util.UniformProblem(f_original)
        f = util.TorchProblem(f_uniform)

        # default data type
        dtype = f.yopt.dtype

        # if n_samples isn't specified, generate 2 * D samples
        n_samples = 2 * f.dim if (n_samples is None) else n_samples

        # if we've got no repeats, then just set up a loop with one item,
        # which corresponds to setting repeat_no=None for one run.
        reprange = [None] if (n_repeats is None) else range(1, n_repeats + 1)

        for repeat_no in reprange:

            save_path = util.generate_data_filename(
                problem_name,
                run_no,
                problem_params,
                repeat_no=repeat_no,
                noise_level=noise_level,
            )

            if os.path.exists(save_path):
                print(f"File exists, skipping: {save_path:s}")
                continue

            # storage
            D = {
                "problem_params": problem_params,
                "Xtr": torch.as_tensor(
                    lhs(f.dim, n_samples, criterion="maximin"), dtype=dtype,
                ),
            }

            # LHS
            try:
                D["Ytr"] = f(D["Xtr"])
            except ValueError:
                D["Ytr"] = torch.cat([f(x) for x in D["Xtr"]])

            # save the training data
            torch.save(obj=D, f=save_path)
            print("Saved: {:s}".format(save_path))


def generate_synthetic_training_data(noise_level: float = 0.0):
    synth_problems = [
        # --- initial problems ---
        "Branin",
        "Eggholder",
        "GoldsteinPrice",
        "SixHumpCamel",
        "Hartmann3",
        "Ackley:5",
        "Hartmann6",
        "Michalewicz:10",
        "Rosenbrock:10",
        "StyblinskiTang:10",
        # --- further problems ---
        "Michalewicz:5",
        "StyblinskiTang:5",
        "Rosenbrock:7",
        "StyblinskiTang:7",
        "Ackley:10",
    ]

    for problem_name in synth_problems:
        additional_arguments = {}

        if ":" in problem_name:
            problem_name, dim = problem_name.split(":")
            dim = int(dim)
            additional_arguments["d"] = [dim] * 51

        generate_training_data_LHS(
            problem_name,
            additional_arguments=additional_arguments,
            n_repeats=None,
            noise_level=noise_level,
        )


if __name__ == "__main__":
    import sys

    noise_level = float(sys.argv[1])

    generate_synthetic_training_data(noise_level=noise_level)
