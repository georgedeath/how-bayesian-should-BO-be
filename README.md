# How Bayesian Should Bayesian Optimisation Be?

This repository contains the Python3 code for the experiments presented in:
> George De Ath, Richard M. Everson, and Jonathan E. Fieldsend.  2021.  How Bayesian Should Bayesian Optimisation Be? In Genetic and Evolutionary Computation Conference Companion (GECCO 21 Companion), July 10--14, 2021, Lille, France. ACM, New York, NY, USA, 10 pages.</br>
> **Paper**: <https://doi.org/10.1145/3449726.3463164> (to appear)</br>
> **Preprint**: <https://arxiv.org/abs/TBA>

## Citation

If you use any part of this code in your work, please cite:

```bibtex
@inproceedings{death:howbayesian:2021,
    title={How Bayesian Should Bayesian Optimisation Be},
    author = {George {De Ath} and Richard M. Everson and Jonathan E. Fieldsend},
    year = {2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    doi = {10.1145/3449726.3463164},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
}
```

## Reproduction of experiments

The python file `run_exp.py` provides a convenient way to reproduce an individual experimental evaluation carried out the paper. It has the following syntax:

```script
> python run_exp.py --help
usage: run_exp.py [-h] --problem {Branin, Eggholder, GoldsteinPrice, SixHumpCamel,
                                  Hartmann3, Ackley:5, Hartmann6, Michalewicz:10,
                                  Rosenbrock:10, StyblinskiTang:10, Michalewicz:5,
                                  StyblinskiTang:5, Rosenbrock:7, StyblinskiTang:7,
                                  Ackley:10}
                  --run_no RUN_NO
                  --method {map, vimf, vifr, mcmc_pm3}
                  --acq_name {ei,ucb}
                  [--budget BUDGET]
                  [--noise {0, 0.05, 0.1, 0.2}]
                  [--ard]
                  [--verbose]
```

In order to run the all experiments carried in this paper, e.g.  on a single 8-core machine, will take roughly 80 years, and, therefore, the use of high-performance computing resources is highly recommended.

## Reproduction of figures and tables in the paper

- [FBBO - Results plots.ipynb](FBBO%20-%20Results%20plots.ipynb) contains the code to load and process the optimisation results (stored in the directories prefixed with `results`), as well as the code to produce all results figures and tables used in the paper and supplementary material.

- [FBBO - marginal likelihood plot.ipynb](FBBO%20-%20marginal%20likelihood%20plot.ipynb) contains the code to reproduce the marginal likelihood optimisation plot, Figure 1. in the paper.

- [FBBO - Noisy function setup.ipynb](FBBO%20-%20Noisy%20function%20setup.ipynb) contains the code to estimate the function value range for the synthetic test functions and save the results as a `problem_noise_dict` object for the optimisation runs.
