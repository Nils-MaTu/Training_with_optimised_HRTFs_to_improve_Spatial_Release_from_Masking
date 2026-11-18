#!/usr/bin/env python3
"""Plot the reported post-on-baseline ANCOVA without a coupled change score."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
DATA = ROOT / "results" / "baseline_dependence_values.csv"
BLUE, ORANGE = "#147CE5", "#E77900"
COLORS = {"individual": BLUE, "optimised": ORANGE}


def fit_ancova(frame):
    frame = frame.copy()
    frame["hrtf"] = pd.Categorical(frame["hrtf"], ["individual", "optimised"])
    return smf.ols("post ~ baseline_c * C(hrtf)", data=frame).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["participant"], "use_correction": True},
    )


def prediction(fit, frame, hrtf, grid):
    centre = frame["baseline"].mean()
    names = list(fit.params.index)
    rows = []
    for baseline in grid:
        values = {
            "Intercept": 1,
            "C(hrtf)[T.optimised]": float(hrtf == "optimised"),
            "baseline_c": baseline - centre,
            "baseline_c:C(hrtf)[T.optimised]":
                (baseline - centre) * float(hrtf == "optimised"),
        }
        rows.append([values.get(name, 0) for name in names])
    design = np.asarray(rows)
    mean = design @ fit.params.to_numpy()
    covariance = fit.cov_params().to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
    critical = stats.t.ppf(.975, 17)
    return mean, mean - critical * se, mean + critical * se


def render():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(DATA)
    outcomes = ("Front-back accuracy", "Separated-speech accuracy")
    titles = ("Front-back discrimination", "Speech Intelligibility")
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.3), layout="constrained")
    for ax, outcome, title in zip(axes, outcomes, titles):
        frame = data[data.outcome == outcome].copy()
        fit = fit_ancova(frame)
        lower = np.floor(min(frame.baseline.min(), frame.post.min()) / 5) * 5 - 5
        upper = min(102, np.ceil(max(frame.baseline.max(), frame.post.max()) / 5) * 5 + 5)
        ax.plot([lower, upper], [lower, upper], ls="--", color=".48", lw=.8,
                label="No change")
        for hrtf in ("individual", "optimised"):
            subset = frame[frame.hrtf == hrtf]
            ax.scatter(subset.baseline, subset.post, s=24, color=COLORS[hrtf],
                       alpha=.82, edgecolor="white", linewidth=.4, zorder=3)
            grid = np.linspace(subset.baseline.min(), subset.baseline.max(), 160)
            mean, lo, hi = prediction(fit, frame, hrtf, grid)
            ax.fill_between(grid, lo, hi, color=COLORS[hrtf], alpha=.14, linewidth=0)
            ax.plot(grid, mean, color=COLORS[hrtf], lw=1.3)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.set_xlabel("Pre-training accuracy (%)")
        ax.grid(color=".91", lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Post-training accuracy (%)")
    axes[1].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", lw=0, markersize=3.5,
               label="Individual"),
        Line2D([], [], color=ORANGE, marker="o", lw=0, markersize=3.5,
               label="Optimised"),
    ], loc="lower right", frameon=False, fontsize=6.4, handlelength=.8,
       handletextpad=.3, borderaxespad=.35)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"figure4_baseline_ancova.{suffix}", dpi=240,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
