# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch

torch.set_num_threads(1)

if __name__ == "__main__":
    import fbbo
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--problem",
        type=str,
        choices=[
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
            "Michalewicz:5",
            "StyblinskiTang:5",
            "Rosenbrock:7",
            "StyblinskiTang:7",
            "Ackley:10",
        ],
        required=True,
    )

    ap.add_argument(
        "--run_no",
        type=int,
        required=True,
        help="Run number (1 to 51 inclusive)",
    )

    ap.add_argument(
        "--method",
        type=str,
        choices=["map", "vimf", "vifr", "mcmc_pm3"],
        required=True,
        help="Inference method",
    )

    ap.add_argument(
        "--acq_name", type=str, choices=["ei", "ucb"], required=True
    )

    ap.add_argument("--budget", type=int, default=200)

    ap.add_argument(
        "--noise",
        type=float,
        default=0.0,
        choices=[0, 0.05, 0.1, 0.2],
        help="Function noise standard deviation",
    )

    ap.add_argument(
        "--ard", action="store_true", default=False, help="Use ARD"
    )

    ap.add_argument(
        "--verbose", action="store_true", default=False,
    )

    args = ap.parse_args()
    problem_name = args.problem
    problem_params = {"run_no": args.run_no}

    noise_file = None

    if args.noise > 0:
        noise_file = fbbo.util.generate_noise_filename(args.noise)

    if ":" in problem_name:
        problem_name, dim = problem_name.split(":")
        problem_params["d"] = int(dim)

    save_path = fbbo.util.generate_save_filename(
        args.method,
        args.acq_name,
        args.budget,
        problem_name,
        args.run_no,
        problem_params,
        args.ard,
        noise_level=args.noise,
    )

    data_path = fbbo.util.generate_data_filename(
        problem_name,
        args.run_no,
        problem_params,
        data_dir="data",
        noise_level=args.noise,
    )

    if args.verbose:
        print("Arguments:", args)

        if args.ard:
            print("Using ARD")
        print(f"Data path: {data_path:s}")
        print(f"Save path: {save_path:s}\n")  # extra new line here

    for _ in range(10):
        try:
            fbbo.bo.bo(
                problem_name=problem_name,
                problem_params=problem_params,
                run_no=args.run_no,
                budget=args.budget,
                method=args.method,
                acq_name=args.acq_name,
                use_ard=args.ard,
                data_path=data_path,
                save_path=save_path,
                save_every=10,
                verbose=args.verbose,
                noise_file=noise_file,
            )
            break

        except:  # noqa: E722
            raise
