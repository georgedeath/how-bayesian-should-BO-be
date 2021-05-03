# Copyright (c) 2021 George De Ath
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import tqdm.auto
import torch
import warnings
import numpy as np
from typing import Any, List, Dict
from . import util

from scipy.stats import median_abs_deviation, wilcoxon
from statsmodels.stats.multitest import multipletests

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.lines
import matplotlib.patheffects
import matplotlib.ticker
from matplotlib import pyplot as plt


def load_results(
    problems: Dict[str, Dict[str, Any]],
    methods: Dict[str, Dict[str, str]],
    acq_funcs: Dict[str, str],
    n_runs: int,
    results_dir_prefix: str = "results",
    budget: int = 200,
    use_ard: List[bool] = [False, True],
    noise_level: float = 0.0,
) -> Dict:
    D = {}

    total = (
        len(use_ard) * len(problems) * len(methods) * len(acq_funcs) * n_runs
    )

    with tqdm.auto.tqdm(total=total, leave=True) as pbar:
        for pn in problems:
            function_name = problems[pn]["fname"]
            function_params = problems[pn]["params"]

            if pn not in D:
                D[pn] = {}

            for ard in use_ard:
                D[pn][ard] = {}

                for method_name in methods:
                    D[pn][ard][method_name] = {}

                    for acq_func in acq_funcs:
                        res = np.zeros((n_runs, budget))

                        for i, run_no in enumerate(range(1, n_runs + 1)):
                            function_params["run_no"] = run_no

                            f = util.test_func_getter(
                                function_name, function_params
                            )

                            fn = util.generate_save_filename(
                                method_name=method_name,
                                acq_name=acq_func,
                                budget=budget,
                                problem_name=function_name,
                                run_no=run_no,
                                problem_params=function_params,
                                use_ard=ard,
                                repeat_no=None,
                                results_dir_prefix=results_dir_prefix,
                                noise_level=noise_level,
                            )

                            try:
                                data = torch.load(fn)

                                # if noisy, we have to reevaluate using the
                                # non-noisy function to get meaningful
                                # conv curves
                                if noise_level > 0.0:
                                    # get the evaluated locations
                                    Xtr = data["Xtr"].numpy()

                                    # get the function
                                    f = util.UniformProblem(f)

                                    # try to batch do it
                                    try:
                                        Ytr = f(Xtr).ravel()
                                    # else we need sequential eval
                                    except ValueError:
                                        Ytr = np.array([f(x) for x in Xtr])

                                else:
                                    Ytr = data["Ytr"].numpy().ravel()

                                n = Ytr.size

                                res[i, :n] = Ytr - f.yopt.ravel()[0]

                                if n != budget:
                                    print("Not full:", fn, Ytr.shape)

                            except FileNotFoundError:
                                print("Missing", os.path.basename(fn))
                                pass
                            except Exception:
                                print(fn)
                                # pass

                            pbar.update()

                        res = np.minimum.accumulate(res, axis=1)
                        res.flat[res.flat == 0] = 1e-9
                        if np.any(res < 0):
                            print("res < 0", pn)

                        D[pn][ard][method_name][acq_func] = res

    return D


def load_distances(
    problems: Dict[str, Dict[str, Any]],
    methods: Dict[str, Dict[str, str]],
    acq_funcs: Dict[str, str],
    n_runs: int,
    results_dir_prefix: str = "results",
    budget: int = 200,
    use_ard: List[bool] = [False, True],
    noise_level: float = 0.0,
    normalise_distances: bool = False,
) -> Dict:

    D = {}

    total = (
        len(use_ard) * len(problems) * len(methods) * len(acq_funcs) * n_runs
    )

    with tqdm.auto.tqdm(total=total, leave=True) as pbar:
        for pn in problems:
            function_name = problems[pn]["fname"]
            function_params = problems[pn]["params"]

            f = util.test_func_getter(function_name, function_params)
            dim = f.dim

            maxdist = np.sqrt(dim)
            start = 2 * dim - 1

            if pn not in D:
                D[pn] = {}

            for ard in use_ard:
                D[pn][ard] = {}

                for method_name in methods:
                    D[pn][ard][method_name] = {}

                    for acq_func in acq_funcs:
                        res = np.zeros((n_runs, budget - start - 1))

                        for i, run_no in enumerate(range(1, n_runs + 1)):
                            function_params["run_no"] = run_no

                            fn = util.generate_save_filename(
                                method_name=method_name,
                                acq_name=acq_func,
                                budget=budget,
                                problem_name=function_name,
                                run_no=run_no,
                                problem_params=function_params,
                                use_ard=ard,
                                repeat_no=None,
                                results_dir_prefix=results_dir_prefix,
                                noise_level=noise_level,
                            )

                            data = torch.load(fn)
                            Xtr = data["Xtr"].numpy()
                            Xtr = Xtr[start:]

                            # Euclidean distance between consecutive locations
                            dist = np.linalg.norm(np.diff(Xtr, axis=0), axis=1)
                            if normalise_distances:
                                dist /= maxdist

                            res[i, :] = dist
                            pbar.update()

                        D[pn][ard][method_name][acq_func] = res
    return D


def create_table_data(
    results: Dict,
    problems: Dict[str, Dict[str, Any]],
    ard: bool,
    acq_funcs: Dict[str, str],
    methods: Dict[str, Dict[str, str]],
    n_runs: int,
    time: int = -1,
) -> Dict:
    method_names = np.array(list(methods.keys()))
    n_methods = len(method_names)

    # D[pn][ard][method_name][acq_func]

    # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
    table_data = {}

    for acq in acq_funcs:
        table_data[acq] = {}

        for pn in problems:
            best_seen_values = np.zeros((n_methods, n_runs))

            for i, method_name in enumerate(method_names):
                # best seen evaluate at the end of the optimisation run
                best_seen_values[i, :] = results[pn][ard][method_name][acq][
                    :, time
                ]

            medians = np.median(best_seen_values, axis=1)
            MADS = median_abs_deviation(
                best_seen_values, scale="normal", axis=1
            )

            # best method -> lowest median value
            best_method_idx = np.argmin(medians)

            # mask of methods equivlent to the best
            stats_equal_to_best_mask = np.zeros(n_methods, dtype="bool")
            stats_equal_to_best_mask[best_method_idx] = True

            # perform wilcoxon signed rank test between best and all other methods
            p_values = []
            for i, method_name in enumerate(method_names):
                if i == best_method_idx:
                    continue
                # a ValueError will be thrown if the runs are all identical,
                # therefore we can assign a p-value of 0 as they are identical
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, p_value = wilcoxon(
                            best_seen_values[best_method_idx, :],
                            best_seen_values[i, :],
                        )
                    p_values.append(p_value)

                except ValueError:
                    p_values.append(0)

            # calculate the Holm-Bonferroni correction
            reject_hyp, pvals_corrected, _, _ = multipletests(
                p_values, alpha=0.05, method="holm"
            )

            for reject, method_name in zip(
                reject_hyp,
                [
                    m
                    for m in method_names
                    if m != method_names[best_method_idx]
                ],
            ):
                # if we can't reject the hypothesis that a technique is
                # statistically equivalent to the best method
                if not reject:
                    idx = np.where(np.array(method_names) == method_name)[0][0]
                    stats_equal_to_best_mask[idx] = True

            # store the data
            table_data[acq][pn] = {
                "medians": medians,
                "MADS": MADS,
                "stats_equal_to_best_mask": stats_equal_to_best_mask,
            }

    return table_data


def create_table(
    table_data: Dict,
    acq_name: str,
    methods: Dict[str, Dict[str, str]],
    problem_name_rows: List[List[str]],
    problem_name_paper_rows: List[List[str]],
    problem_dim_rows: List[List[int]],
    caption: str = "",
    tablename: str = "",
    tab_size: int = 4,
) -> List[str]:

    tab = " " * tab_size

    # storage for returning the list of strings
    table_text = []

    # table heading stuff
    table_text += [
        r"\begin{table}[H]",
        r"\setlength{\tabcolsep}{2pt}",
        r"\sisetup{table-format=1.2e-1, table-number-alignment=center}",
        r"\resizebox{1\textwidth}{!}{",
        r"\begin{tabular}{l Sz Sz Sz Sz Sz}",
    ]

    for probs, probs_paper, probs_dim in zip(
        problem_name_rows, problem_name_paper_rows, problem_dim_rows
    ):
        table_text += [r"\toprule", tab + r"\bfseries Method"]

        # column titles: Problem name (dim)
        for prob, dim in zip(probs_paper, probs_dim):
            table_text += [
                tab
                + r"& \multicolumn{2}{c}{\bfseries "
                + f"{prob:s} ({dim:d})"
                + "}"
            ]

        # stick a newline on the last element
        table_text[-1] += r" \\"

        # column titles: Median MAD
        for prob in probs:
            table_text += [
                tab + r"& \multicolumn{1}{c}{Median} & \multicolumn{1}{c}{MAD}"
            ]
        table_text[-1] += r" \\ \midrule"

        # results printing
        for i, method_name in enumerate(methods):
            text = f"{tab:s}{methods[method_name]['papername']:s} & "

            for prob in probs:
                med = "{:4.2e}".format(
                    table_data[acq_name][prob]["medians"][i]
                )
                mad = "{:4.2e}".format(table_data[acq_name][prob]["MADS"][i])

                best_methods = table_data[acq_name][prob][
                    "stats_equal_to_best_mask"
                ]
                best_idx = np.argmin(table_data[acq_name][prob]["medians"])

                if i == best_idx:
                    med = r"\best " + med
                    mad = r"\best " + mad

                elif best_methods[i]:
                    med = r"\statsimilar " + med
                    mad = r"\statsimilar " + mad

                text = f"{text:s} {med:s} & {mad:s} & "

            text = text[:-2] + r"\\"
            table_text.append(text)

        table_text.append(r"\bottomrule")

    # footer
    table_text += [
        r"\end{tabular}",
        r"}%",  # to close the resizebox
        r"\caption{" + f"{caption:s}" + "}",
        r"\label{tbl:" + f"{tablename:s}" + "}",
        r"\end{table}",
    ]

    return table_text


def summary_barchart(
    ax: matplotlib.axes.Axes,
    table_data: Dict,
    acq_name: str,
    ard: bool,
    problem_sets: Dict[str, Dict[str, str]],
    problems: Dict[str, Dict[str, Any]],
    methods: Dict[str, Dict[str, str]],
    draw_legend: bool = False,
):

    n_problems = len(problems)
    n_methods = len(methods)
    offset_increment = 1.5

    xv = np.linspace(0, 1 - 1 / n_methods, n_methods)
    widths = 0.9 * (1 / n_methods)
    labels = []

    offset = 0

    for problem_set in problem_sets:
        td = table_data[problem_set][ard][acq_name]
        best_counts = np.zeros(n_methods)

        for pn in problems:
            best_counts += td[pn]["stats_equal_to_best_mask"]

        ax.bar(
            x=xv + offset,
            height=best_counts,
            width=widths,
            color=[d["color"] for d in methods.values()],
        )
        labels.append(f"{problem_sets[problem_set]['papername']}")

        offset += offset_increment

    # make the fake legend
    lh = []
    for method_name in methods:
        lh.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                ls="-",
                c=methods[method_name]["color"],
                label=methods[method_name]["papername"],
                ms=10,
                alpha=1,
            )
        )
    if draw_legend:
        ax.legend(handles=lh, loc="upper center", ncol=n_methods, fontsize=7)

    # set up the limits
    ax.set_ylim([0, n_problems + 4])

    # set up the bar labels
    x = np.arange(0.375, 4 * 1.5, 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # y labels
    yticks = [0, 5, 10, 15]
    ax.set_yticks(yticks)


def compare_mcmc_map_plot(
    table_data: Dict,
    acq_funcs: Dict[str, str],
    use_ard: List[bool],
    problem_sets: Dict[str, Dict[str, str]],
    problems: Dict[str, Dict[str, Any]],
    save_fname: str = "",
):
    cmap = matplotlib.colors.ListedColormap(["white", "tab:blue", "tab:red"])

    k = 0

    fig, ax = plt.subplots(1, 4, figsize=(6, 4), sharex=True, sharey=True)

    for acq_name in acq_funcs:
        for ard in use_ard:

            res = np.zeros((15, 4))

            for j, problem_set in enumerate(problem_sets):
                td = table_data[problem_set][ard][acq_name]

                for i, pn in enumerate(problems):
                    # MAP = idx 0, MCMC = idx 1
                    mask = td[pn]["stats_equal_to_best_mask"]
                    if np.all(mask[[0, 1]]):
                        continue

                    elif mask[0]:
                        res[i, j] = 1

                    else:
                        res[i, j] = 2

            ttle = acq_funcs[acq_name] + "\n"
            ttle += "Isotropic" if not ard else "ARD"

            ttle = f"{acq_funcs[acq_name]:s} ("
            ttle += "Isotropic)" if not ard else "ARD)"

            ax[k].set_title(ttle, fontsize=10)
            ax[k].imshow(
                res,
                cmap=cmap,
                vmin=0,
                vmax=2,
                aspect="auto",
                interpolation="none",
            )
            if k > 0:
                ax[k].tick_params(left=False)

            k += 1

    # set the problem names
    ylabels = []
    for pn in problems:
        f = util.test_func_getter(
            problems[pn]["fname"], problems[pn]["params"]
        )
        dim = f.dim
        name = f"{problems[pn]['fname']:s} (${dim:d}$)"
        ylabels.append(name)

    yr = np.arange(len(ylabels))
    ax[0].set_yticks(yr)
    ax[0].set_yticklabels(ylabels, fontsize=10)

    # ensure the tick labels (i.e. the problem names) are all
    # the same distance from the ticks
    for tick in ax[0].get_yaxis().get_major_ticks():
        tick.set_pad(2.0)

    # set the scenarios and gridlines
    xlabels = [f"{problem_sets[ps]['papername']}" for ps in problem_sets]
    xr = np.arange(len(xlabels))
    for k in range(k):
        ax[k].set_xticks(xr)
        ax[k].set_xticklabels(xlabels, rotation=90)

        # sort out gridlines
        ax[k].set_xticks(xr - 0.5, minor=True)
        ax[k].set_yticks(yr - 0.5, minor=True)
        ax[k].grid(which="minor")

        for ticks in [
            ax[k].xaxis.get_minor_ticks(),
            ax[k].yaxis.get_minor_ticks(),
        ]:
            for tick in ticks:
                tick.tick1line.set_visible(False)

    # make the fake legend
    lh = []
    for i, label in enumerate(["Equal", "MAP", "MCMC"]):
        lh.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                ls="-",
                c=cmap(i),
                label=label,
                alpha=1,
                lw=5,
                path_effects=[
                    matplotlib.patheffects.Stroke(linewidth=6, foreground="k"),
                    matplotlib.patheffects.Normal(),
                ],
            )
        )

    plt.legend(
        handles=lh,
        bbox_to_anchor=(0.13, 0.13),
        bbox_transform=fig.transFigure,
        fontsize=10,
    )
    plt.subplots_adjust(wspace=0.05)

    if save_fname != "":
        plt.savefig(save_fname)

    plt.show()


def results_plot_maker(
    ax,
    yvals,
    xvals,
    xlabel,
    ylabel,
    title,
    colors,
    LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    TICK_FONTSIZE,
    use_fill_between=True,
    fix_ticklabels=False,
    method_names=None,
    line_width=0.5,
):
    # here we assume we're plotting to a matplotlib axis object
    # and yvals is a LIST of arrays of size (n_runs, iterations),
    # where each can be different sized
    # and if xvals is given then len(xvals) == len(yvals)

    # set the labelling
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if method_names is None:
        method_names = [None] * len(colors)

    Y_valid = 0
    for Y in yvals:
        if not np.all(Y == Y[0, 0]):
            Y_valid = np.percentile(Y, [50], axis=0)[0][0]
            break

    for color, x, Y, mn in zip(colors, xvals, yvals, method_names):
        if np.all(Y == Y[0, 0]):
            ax.plot(Y_valid, Y_valid, "-", color=color, label=mn)
            continue
        bot, mid, top = [
            _.ravel() for _ in np.percentile(Y, [25, 50, 75], axis=0)
        ]

        if use_fill_between:
            ax.fill_between(x, bot.flat, top.flat, color=color, alpha=0.15)

        ax.plot(x, mid, color=color, label=mn, lw=line_width)

        ax.plot(x, bot, "--", color=color, alpha=0.15, lw=line_width)
        ax.plot(x, top, "--", color=color, alpha=0.15, lw=line_width)

    # set the xlim
    min_x = np.min([np.min(x) for x in xvals])
    max_x = np.max([np.max(x) for x in xvals])
    ax.set_xlim([0, max_x])

    ax.axvline(min_x, linestyle="dashed", color="gray", linewidth=1, alpha=0.5)

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)

    # set the alignment for outer ticklabels
    ax.set_xticks([0, 50, 100, 150, 200])
    if fix_ticklabels:
        ticklabels = ax.get_xticklabels()
        if len(ticklabels) > 0:
            ticklabels[0].set_ha("left")
            ticklabels[-1].set_ha("right")

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter("{x:>4.1f}")
    )


def create_gridplot(
    results,
    problem_set: str,
    ard: bool,
    acq_name: str,
    methods: Dict[str, Dict[str, str]],
    problems: Dict[str, Dict[str, Any]],
    problem_name_order: List[str],
    budget: int = 200,
    LABEL_FONTSIZE: int = 7,
    TITLE_FONTSIZE: int = 8,
    TICK_FONTSIZE: int = 5,
    savename: str = "",
    showfig: bool = False,
    line_width: int = 1,
):
    fig, axes = plt.subplots(3, 5, figsize=(8, 4), sharex=True)

    for i, (ax, pn) in enumerate(zip(axes.flat, problem_name_order)):
        f = util.test_func_getter(
            problems[pn]["fname"], problems[pn]["params"]
        )
        dim = f.dim

        start = 2 * dim - 1
        end = budget

        x = np.arange(start + 1, end + 1)

        xvals = []
        yvals = []

        for method_name in methods:
            Y = results[problem_set][pn][ard][method_name][acq_name][
                :, start:end
            ]
            Y = np.log(Y)

            yvals.append(Y)
            xvals.append(x)

        title = f"{problems[pn]['fname']:s} ({dim:d})"

        ylabel = r"$\log(R_t)$" if i % 5 == 0 else ""
        xlabel = "Function evaluations" if i >= 10 else ""

        results_plot_maker(
            ax,
            yvals,
            xvals,
            xlabel,
            ylabel,
            title,
            [d["color"] for d in methods.values()],
            LABEL_FONTSIZE=LABEL_FONTSIZE,
            TITLE_FONTSIZE=TITLE_FONTSIZE,
            TICK_FONTSIZE=TICK_FONTSIZE,
            use_fill_between=True,
            fix_ticklabels=True,
            method_names=[methods[m]["papername"] for m in methods],
            line_width=line_width,
        )

        ax.get_xaxis().set_label_coords(0.5, -0.2)

        # ensure labels are all in the same place!
        ax.get_yaxis().set_label_coords(-0.24, 0.5)

        # move the title down a bit
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE, y=0.95)

    plt.subplots_adjust(wspace=0.26, hspace=0.25)
    if savename != "":
        plt.savefig(savename, bbox_inches="tight", transparent=True)

    if showfig:
        plt.show()
    else:
        plt.close()


def create_distance_gridplot(
    results,
    problem_set: str,
    ard: bool,
    acq_name: str,
    methods: Dict[str, Dict[str, str]],
    problems: Dict[str, Dict[str, Any]],
    problem_name_order: List[str],
    budget: int = 200,
    LABEL_FONTSIZE: int = 7,
    TITLE_FONTSIZE: int = 8,
    TICK_FONTSIZE: int = 5,
    savename: str = "",
    showfig: bool = False,
    line_width: int = 1,
    normalise_distances: bool = False,
):
    if normalise_distances:
        ylab = "Normalised distance"
    else:
        ylab = "Euclidean distance"

    fig, axes = plt.subplots(3, 5, figsize=(8, 4), sharex=True)

    for i, (ax, pn) in enumerate(zip(axes.flat, problem_name_order)):
        f = util.test_func_getter(
            problems[pn]["fname"], problems[pn]["params"]
        )
        dim = f.dim

        start = 2 * dim - 1
        end = budget

        x = np.arange(start + 2, end + 1)

        xvals = []
        yvals = []

        for method_name in methods:
            Y = results[problem_set][pn][ard][method_name][acq_name][:, :]

            yvals.append(Y)
            xvals.append(x)

        title = f"{problems[pn]['fname']:s} ({dim:d})"

        ylabel = ylab if i % 5 == 0 else ""
        xlabel = "Function evaluations" if i >= 10 else ""

        results_plot_maker(
            ax,
            yvals,
            xvals,
            xlabel,
            ylabel,
            title,
            [d["color"] for d in methods.values()],
            LABEL_FONTSIZE=LABEL_FONTSIZE,
            TITLE_FONTSIZE=TITLE_FONTSIZE,
            TICK_FONTSIZE=TICK_FONTSIZE,
            use_fill_between=True,
            fix_ticklabels=True,
            method_names=[methods[m]["papername"] for m in methods],
            line_width=line_width,
        )

        ax.get_xaxis().set_label_coords(0.5, -0.2)

        # ensure labels are all in the same place!
        ax.get_yaxis().set_label_coords(-0.24, 0.5)

        # move the title down a bit
        ax.set_title(ax.get_title(), fontsize=TITLE_FONTSIZE, y=0.95)

        if normalise_distances:
            ax.set_ylim([0, 1])

    plt.subplots_adjust(wspace=0.26, hspace=0.25)
    if savename != "":
        plt.savefig(savename, bbox_inches="tight", transparent=True)

    if showfig:
        plt.show()
    else:
        plt.close()
