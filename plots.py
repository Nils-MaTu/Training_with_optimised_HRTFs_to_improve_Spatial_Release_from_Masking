#!/usr/bin/env python3
"""Reproduce the five figures included in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARTICIPANTS = DATA / "participants"
BLUE, ORANGE, RED, INK = "#147CE5", "#FF8A00", "#FF0000", "#202124"
GREY, GRID = "#A8A8AD", "#E5E5EA"
HRTF_COLORS = {"individual": BLUE, "optimised": ORANGE}
SESSION_COLORS = {"before": INK, "after": RED}
HRTFS = ("individual", "optimised")
SESSIONS = ("before", "after")
CONFIGURATIONS = ("T30 M30", "T30 M150", "T150 M30")
SEPARATED = ("T30 M150", "T150 M30")
MARKERS = {"T30 M150": "s", "T150 M30": "^"}
RNG = np.random.default_rng(20260611)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#C7C7CC", "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def style_axis(axis: plt.Axes, grid: bool = True) -> None:
    if grid:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def load_participant_data(filename: str) -> pd.DataFrame:
    paths = sorted(PARTICIPANTS.glob(f"sub-*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No participant files named {filename!r} in {PARTICIPANTS}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_trials() -> tuple[pd.DataFrame, pd.DataFrame]:
    fb = load_participant_data("front_back_trials.csv")
    srm = load_participant_data("srm_trials.csv")
    srm["configuration"] = (
        "T" + srm["target_azimuth_deg"].astype(int).astype(str)
        + " M" + srm["masker_azimuth_deg"].astype(int).astype(str)
    )
    return fb, srm


def means(trials: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    return (
        trials.groupby(["participant", *factors], observed=True)["correct"]
        .mean().mul(100).rename("accuracy_pct").reset_index()
    )


def add_box(axis: plt.Axes, values: np.ndarray, position: float, color: str,
            hatch: str = "", width: float = 0.22) -> None:
    box = axis.boxplot(
        [values], positions=[position], widths=width, patch_artist=True,
        showfliers=False, manage_ticks=False,
        medianprops={"color": INK, "linewidth": 1.3},
        whiskerprops={"color": "#8E8E93", "linewidth": 0.9},
        capprops={"color": "#8E8E93", "linewidth": 0.9},
        boxprops={"edgecolor": color, "linewidth": 0.9},
    )
    box["boxes"][0].set_facecolor(color)
    if hatch:
        box["boxes"][0].set_hatch(hatch)
        box["boxes"][0].set_edgecolor("white")
    jitter = RNG.normal(0, 0.012, len(values))
    axis.scatter(
        position + jitter, values, s=13, color=color, edgecolor="white",
        linewidth=0.35, alpha=0.88, zorder=4,
    )


def figure_hrtf(output: Path) -> None:
    archive = np.load(DATA / "hrtf_spectra_summary.npz")
    spectra = pd.DataFrame({name: archive[name] for name in archive.files})
    gain = load_participant_data("better_ear_snr_gain.csv")
    frequency = spectra["frequency_hz"].to_numpy()
    mask = (frequency >= 80) & (frequency <= 20000)
    fig = plt.figure(figsize=(3.54, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.06)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    for axis, azimuth in zip(axes, (30, 150)):
        # Preserve the ear-role convention used in the submitted figure.
        ear_map = {"ipsi": "left", "contra": "right"}
        for role, linestyle in (("ipsi", "-"), ("contra", "--")):
            ear = ear_map[role]
            mean = spectra[f"az{azimuth}_{ear}_mean_difference_db"].to_numpy()
            sd = spectra[f"az{azimuth}_{ear}_sd_difference_db"].to_numpy()
            axis.fill_between(frequency[mask], (mean - sd)[mask], (mean + sd)[mask],
                              color=INK, alpha=0.09, linewidth=0)
            axis.plot(frequency[mask], mean[mask], color=INK, linestyle=linestyle,
                      linewidth=1.2, label="Ipsilateral" if role == "ipsi" else "Contralateral")
        axis.axhline(0, color=GREY, linestyle=":", linewidth=0.8)
        axis.set_xscale("log"); axis.set_xlim(80, 20000); axis.set_ylim(-52, 42)
        axis.set_xticks([100, 500, 1000, 5000, 10000, 20000])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.text(0.98, 0.95, f"Azimuth {azimuth}°", transform=axis.transAxes,
                  ha="right", va="top", fontsize=9)
        axis.legend(frameon=False, loc="upper left")
        style_axis(axis)
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.supylabel("Magnitude difference (dB)\nOptimised - individual", x=0.01, y=0.70)

    axis = fig.add_subplot(grid[2])
    pair_colors = {"T30 M150": INK, "T150 M30": RED}
    for pair in SEPARATED:
        summary = (
            gain[gain["pair"] == pair].groupby("band_fc_hz")["better_ear_snr_gain_db"]
            .agg(["mean", "std"]).reset_index()
        )
        axis.errorbar(summary["band_fc_hz"], summary["mean"], yerr=summary["std"],
                      color=pair_colors[pair], marker="o", markersize=3.4,
                      markeredgecolor="white", markeredgewidth=0.35,
                      linewidth=1, elinewidth=0.7, capsize=1.2, label=pair)
    axis.axhline(0, color=GREY, linewidth=0.8)
    axis.set_xscale("log"); axis.set_xlim(20, 20000); axis.set_ylim(-25, 25)
    axis.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000])
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.tick_params(axis="x", labelrotation=30)
    axis.set_xlabel("ERB-band centre frequency (Hz)")
    axis.set_ylabel("Better-ear SNR gain (dB)\nOptimised - individual")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    save(fig, output, "hrtf_spectra_better_ear_combined")


def figure_front_back(fb: pd.DataFrame, output: Path) -> None:
    cell = means(fb, ["session", "hrtf", "location_deg"])
    fig, axes = plt.subplots(1, 2, figsize=(3.54, 2.55), sharey=True)
    positions = {
        ("before", "individual"): 0, ("after", "individual"): 0.5,
        ("before", "optimised"): 1.15, ("after", "optimised"): 1.65,
    }
    for axis, location in zip(axes, (30, 150)):
        part = cell[cell["location_deg"] == location]
        for hrtf in HRTFS:
            wide = part[part["hrtf"] == hrtf].pivot(
                index="participant", columns="session", values="accuracy_pct"
            )
            for row in wide.itertuples():
                axis.plot([positions[("before", hrtf)], positions[("after", hrtf)]],
                          [row.before, row.after], color=GREY, alpha=0.25, linewidth=0.5)
        for session in SESSIONS:
            for hrtf in HRTFS:
                values = part[(part["session"] == session) & (part["hrtf"] == hrtf)]["accuracy_pct"].to_numpy()
                add_box(axis, values, positions[(session, hrtf)], HRTF_COLORS[hrtf],
                        "//" if session == "after" else "")
        axis.axhline(50, color=GREY, linestyle=":", linewidth=0.8)
        axis.set_xticks(list(positions.values()), ["Before", "After", "Before", "After"], fontsize=7)
        axis.set_xlim(-0.3, 1.95); axis.set_ylim(0, 105)
        axis.set_title(f"Azimuth {location}°", fontsize=8.5)
        style_axis(axis)
    axes[0].set_ylabel("Correct (%)")
    fig.legend(handles=[Patch(color=BLUE, label="Individual"), Patch(color=ORANGE, label="Optimised")],
               ncol=2, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, output, "fb_accuracy")


def figure_srm(srm: pd.DataFrame, output: Path) -> None:
    cell = means(srm, ["session", "hrtf", "configuration"])
    fig, axis = plt.subplots(figsize=(7.48, 3.45))
    base = np.arange(3)
    offsets = {
        ("before", "individual"): -0.27, ("after", "individual"): -0.09,
        ("before", "optimised"): 0.09, ("after", "optimised"): 0.27,
    }
    for index, configuration in enumerate(CONFIGURATIONS):
        part = cell[cell["configuration"] == configuration]
        for hrtf in HRTFS:
            wide = part[part["hrtf"] == hrtf].pivot(index="participant", columns="session", values="accuracy_pct")
            for row in wide.itertuples():
                axis.plot([base[index] + offsets[("before", hrtf)], base[index] + offsets[("after", hrtf)]],
                          [row.before, row.after], color=GREY, alpha=0.25, linewidth=0.55)
        for session in SESSIONS:
            for hrtf in HRTFS:
                values = part[(part["session"] == session) & (part["hrtf"] == hrtf)]["accuracy_pct"].to_numpy()
                add_box(axis, values, base[index] + offsets[(session, hrtf)], HRTF_COLORS[hrtf],
                        "//" if session == "after" else "", width=0.15)
    axis.set_xticks(base, [value.replace(" ", "\n") for value in CONFIGURATIONS])
    axis.set_ylim(0, 105); axis.set_ylabel("Correct (%)")
    axis.set_xlabel("Target-masker configuration")
    handles = [Patch(color=BLUE, label="Individual HRTF"), Patch(color=ORANGE, label="Optimised HRTF"),
               Patch(facecolor=INK, label="Before"), Patch(facecolor=INK, hatch="//", edgecolor="white", label="After")]
    axis.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    style_axis(axis); fig.tight_layout()
    save(fig, output, "srm_accuracy")


def srm_prediction_points(srm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = (
        srm.groupby(["participant", "session", "hrtf", "configuration"], observed=True)
        .agg(accuracy_pct=("correct", lambda x: 100 * x.mean()),
             predicted_srm_db=("predicted_srm_db", "mean"))
        .reset_index()
    )
    colocated = cell[cell["configuration"] == "T30 M30"][
        ["participant", "session", "hrtf", "accuracy_pct", "predicted_srm_db"]
    ].rename(columns={"accuracy_pct": "colocated_accuracy_pct",
                      "predicted_srm_db": "predicted_colocated_srm_db"})
    points = cell.merge(colocated, on=["participant", "session", "hrtf"], validate="many_to_one")
    points["behavioral_srm_pp"] = points["accuracy_pct"] - points["colocated_accuracy_pct"]
    points["predicted_delta_db"] = points["predicted_srm_db"] - points["predicted_colocated_srm_db"]
    points = points[points["configuration"].isin(SEPARATED)].copy()
    wide = points.pivot(
        index=["participant", "session", "configuration"], columns="hrtf",
        values=["behavioral_srm_pp", "predicted_delta_db", "predicted_srm_db"],
    )
    benefit = pd.DataFrame({
        "behavioral_gain_pp": wide[("behavioral_srm_pp", "optimised")] - wide[("behavioral_srm_pp", "individual")],
        "predicted_gain_db": wide[("predicted_srm_db", "optimised")] - wide[("predicted_srm_db", "individual")],
    }).reset_index()
    return points, benefit


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_normal = stats.shapiro(x).pvalue >= 0.05
    y_normal = stats.shapiro(y).pvalue >= 0.05
    result = stats.pearsonr(x, y) if x_normal and y_normal else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def p_text(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def scatter_panel(axis: plt.Axes, data: pd.DataFrame, x: str, y: str,
                  title: str = "", gain: bool = False) -> None:
    annotations = []
    for session in SESSIONS:
        subset = data[data["session"] == session]
        for configuration in SEPARATED:
            points = subset[subset["configuration"] == configuration]
            axis.scatter(points[x], points[y], marker=MARKERS[configuration], s=17,
                         color=SESSION_COLORS[session], alpha=0.88, linewidth=0)
        xv, yv = subset[x].to_numpy(), subset[y].to_numpy()
        r, p = correlation(xv, yv)
        annotations.append(f"{session.title()}: $r$={r:.2f}, {p_text(p)}")
        if len(np.unique(xv)) > 1:
            slope, intercept = np.polyfit(xv, yv, 1)
            line = np.linspace(xv.min(), xv.max(), 80)
            axis.plot(line, intercept + slope * line, color=SESSION_COLORS[session],
                      linestyle="--" if session == "before" else "-", linewidth=1.1)
    axis.axhline(0, color=GREY, linestyle=":", linewidth=0.7)
    axis.axvline(0, color=GREY, linestyle=":", linewidth=0.7)
    if title:
        axis.set_title(title, loc="left", fontsize=9)
    axis.text(0.03, 0.96, "\n".join(annotations), transform=axis.transAxes,
              ha="left", va="top", fontsize=7)
    style_axis(axis)


def figure_lavandier(srm: pd.DataFrame, output: Path) -> None:
    points, benefit = srm_prediction_points(srm)
    fig, axes = plt.subplots(3, 1, figsize=(3.54, 7.9))
    for axis, hrtf, title in zip(axes[:2], HRTFS, ("Individual", "Optimised")):
        scatter_panel(axis, points[points["hrtf"] == hrtf], "predicted_delta_db",
                      "behavioral_srm_pp", title)
        axis.set_ylabel("Behavioural SRM (%)")
        axis.set_xlabel("Predicted SRM (dB)")
    scatter_panel(axes[2], benefit, "predicted_gain_db", "behavioral_gain_pp", gain=True)
    axes[2].set_ylabel("Behavioural SRM Gain (%)")
    axes[2].set_xlabel("Predicted SRM Gain (dB)")
    handles = [
        Line2D([], [], marker="o", linestyle="None", color=INK, label="Before"),
        Line2D([], [], marker="o", linestyle="None", color=RED, label="After"),
        Line2D([], [], marker="s", linestyle="None", color=INK, label="T30M150"),
        Line2D([], [], marker="^", linestyle="None", color=INK, label="T150M30"),
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False,
               handletextpad=0.35, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.1)
    save(fig, output, "lavandier_esrm_prediction_stacked")


def regression_with_ci(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    line = np.linspace(x.min(), x.max(), 100)
    prediction = fit.get_prediction(sm.add_constant(line)).summary_frame(alpha=0.05)
    axis.fill_between(line, prediction["mean_ci_lower"].to_numpy(),
                      prediction["mean_ci_upper"].to_numpy(), color=color, alpha=0.16, linewidth=0)
    axis.plot(line, prediction["mean"].to_numpy(), color=color, linewidth=1.1)


def figure_baseline(fb: pd.DataFrame, srm: pd.DataFrame, output: Path) -> None:
    fb_cell = means(fb, ["session", "hrtf"])
    srm_cell = means(srm, ["session", "hrtf", "configuration"])
    srm_cell = (
        srm_cell[srm_cell["configuration"].isin(SEPARATED)]
        .groupby(["participant", "session", "hrtf"], observed=True)["accuracy_pct"]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.54, 2.55))
    for axis, cell, title in zip(axes, (fb_cell, srm_cell), ("Front–back discrimination", "SRM")):
        annotations = []
        for hrtf in HRTFS:
            wide = cell[cell["hrtf"] == hrtf].pivot(index="participant", columns="session", values="accuracy_pct")
            x = wide["before"].to_numpy(); y = (wide["after"] - wide["before"]).to_numpy()
            r, p = correlation(x, y)
            short = "Indiv." if hrtf == "individual" else "Opt."
            annotations.append(f"{short} $r$={r:.2f}, {p_text(p)}")
            axis.scatter(x, y, color=HRTF_COLORS[hrtf], s=20, edgecolor="white", linewidth=0.4, alpha=0.9)
            regression_with_ci(axis, x, y, HRTF_COLORS[hrtf])
        axis.axhline(0, color=GREY, linestyle=":", linewidth=0.8)
        axis.text(0.03, 0.96, "\n".join(annotations), transform=axis.transAxes,
                  ha="left", va="top", fontsize=7)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlabel("Before accuracy (%)")
        style_axis(axis)
    axes[0].set_ylabel("Improvement (%)\nAfter - before")
    axes[1].legend(handles=[Line2D([], [], marker="o", linestyle="None", color=BLUE, label="Indiv."),
                            Line2D([], [], marker="o", linestyle="None", color=ORANGE, label="Opt.")],
                   frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    save(fig, output, "baseline_improvement_vs_before")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    fb, srm = load_trials()
    figure_hrtf(output)
    figure_front_back(fb, output)
    figure_srm(srm, output)
    figure_lavandier(srm, output)
    figure_baseline(fb, srm, output)
    print(f"Wrote five manuscript figures to {output}")


if __name__ == "__main__":
    main()
