# How Bayesian Should Bayesian Optimisation Be?

This repository contains the Python3 code for the experiments presented in:
> George De Ath, Richard M. Everson, and Jonathan E. Fieldsend.  2021.  How Bayesian Should Bayesian Optimisation Be? In Genetic and Evolutionary Computation Conference Companion (GECCO 21 Companion), July 10--14, 2021, Lille, France. ACM, New York, NY, USA, 10 pages.</br>
> **Paper**: <https://doi.org/10.1145/3449726.3463164> (to appear)</br>
> **Preprint**: <https://arxiv.org/abs/TBA>

The repository also contains all training data used for the initialisation of the optimisation runs carried out, the optimisation results of each of the runs, and jupyter notebooks to generate the results, figures and tables in the paper.

We note that the code is generally tightly coupled and, therefore, not directly usable for additional test problems or methods. However, the code does contain useful methods for performing [VI](fbbo/advi.py) and [MCMC](fbbo/mcmc_pm3.py) that we believe others will find useful. If you have any questions, please raise a [github issue](https://github.com/georgedeath/how-bayesian-should-BO-be/issues) and we will try to help where possible.

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

## Core packages

- [PyMC3](https://github.com/pymc-devs/pymc3), [PyTorch](https://github.com/pytorch/pytorch), [BoTorch](https://github.com/pytorch/botorch), [gpytorch](https://github.com/cornellius-gp/gpytorch), [pyDOE2](https://pypi.org/project/pyDOE2/), [tqdm](https://github.com/tqdm/tqdm), [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), and [matplotlib](https://github.com/matplotlib/matplotlib).

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
                  --acq_name {ei, ucb}
                  [--budget BUDGET]
                  [--noise {0, 0.05, 0.1, 0.2}]
                  [--ard]
                  [--verbose]
```

In order to run the all experiments carried in this paper, e.g. on a single 8-core machine, will take approximately 80 years, and, therefore, the use of high-performance computing resources is highly recommended.

## Reproduction of figures and tables in the paper

- [FBBO - Results plots.ipynb](FBBO%20-%20Results%20plots.ipynb) contains the code to load and process the optimisation results (stored in the directories prefixed with `results`), as well as the code to produce all results figures and tables used in the paper and supplementary material.

- [FBBO - marginal likelihood plot.ipynb](FBBO%20-%20marginal%20likelihood%20plot.ipynb) contains the code to reproduce the marginal likelihood optimisation plot, Figure 1. in the paper.

- [FBBO - Noisy function setup.ipynb](FBBO%20-%20Noisy%20function%20setup.ipynb) contains the code to estimate the function value range for the synthetic test functions and save the results as a `problem_noise_dict` object for the optimisation runs.

## Training data

The initial training locations for each of the 51 sets of [Latin hypercube](https://www.jstor.org/stable/1268522) samples for the various noise levels are located in the `data` and `data_X.X` directories in this repository, where `X.X` corresponds to the noise level. The files have the following naming structure: `ProblemName_number`, e.g. the first set of training locations for the Branin problem is stored in `Branin_001.npz`. Each of these files is a compressed numpy file created with [torch.save](https://pytorch.org/docs/stable/torch.html#torch.save). It has two [torch.tensor](https://pytorch.org/docs/stable/torch.html#torch.tensor) arrays (`Xtr` and `Ytr`) containing the 2*D initial locations and their corresponding fitness values. Note that for problems that have a non-default dimensionality (e.g. Ackley with d=5), then the data files have the dimensionality appended, e.g. `Ackley5_001.pt`; see the suite of [available test problems](fbbo/test_problems/synthetic_problems.py). To load and inspect the training data, use the following instructions:

```python
> python
>>> import torch
>>> data = torch.load('data/Ackley5_001.pt')
>>> Xtr = data['Xtr']  # Training data locations
>>> Ytr = data['Ytr']  # Corresponding function values
>>> Xtr.shape, Ytr.shape
(torch.Size([10, 5]), torch.Size([10]))
```

## Optimisation results

The results of all optimisation runs can be found in the `results` and `results_X.X` directories, corresponding to the noise-free and noisy problems. The filenames have the following structure for the isotropic kernel: `fbbo_200_METHOD_ACQ_ProblemName_run=XXX.pt` and `fbbo_200_ARD_METHOD_ACQ_ProblemName_run=XXX.pt` for the ARD kernel. Here, the methods are one of `[map, mcmc_pm3, vimf, vifr]`, corresponding to the *maximum a posteriori* estimate, MCMC using [PyMC3](https://docs.pymc.io), and mean-field and full-rank variational inference respectively. The acquisition function can be either `ei` or `ucb`, and run numbers range from `001` to `051` inclusive. Note that the save optimisation runs include the initial LHS samples. To load and inspect the training data, use the following instructions:

```python
> python
>>> import torch
>>> data = data = torch.load('results/fbbo_200_ARD_map_ei_Ackley5_run=001.pt')
>>> Xtr = data['Xtr']  # Evaluated locations
>>> Ytr = data['Ytr']  # Corresponding function values
>>> Xtr.shape, Ytr.shape
(torch.Size([200, 5]), torch.Size([200]))
```
