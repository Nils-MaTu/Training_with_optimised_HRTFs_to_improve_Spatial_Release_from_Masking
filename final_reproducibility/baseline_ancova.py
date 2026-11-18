#!/usr/bin/env python3
"""Baseline-dependent change analyses avoiding baseline/change coupling."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "original"))
import analysis as a  # noqa: E402


def make_wide_tables():
    fb, srm = a.load_data()
    fb_cell = a.participant_means(fb, ["session", "hrtf"])
    fb_wide = fb_cell.pivot(
        index="participant", columns=["session", "hrtf"], values="accuracy_pct"
    )

    srm_cell, _ = a.srm_participant_cells(srm)
    speech = (
        srm_cell[srm_cell["configuration"].isin(a.SEPARATED)]
        .groupby(["participant", "session", "hrtf"], observed=True)["accuracy_pct"]
        .mean()
        .reset_index()
    )
    speech_wide = speech.pivot(
        index="participant", columns=["session", "hrtf"], values="accuracy_pct"
    )
    return fb_wide, speech_wide


def long_from_wide(wide, outcome):
    rows = []
    for participant in wide.index:
        for hrtf in a.HRTFS:
            baseline = float(wide.loc[participant, ("before", hrtf)])
            post = float(wide.loc[participant, ("after", hrtf)])
            rows.append(
                {
                    "participant": participant,
                    "outcome": outcome,
                    "hrtf": hrtf,
                    "baseline": baseline,
                    "post": post,
                    "change": post - baseline,
                    "mean_level": (post + baseline) / 2,
                }
            )
    frame = pd.DataFrame(rows)
    frame["baseline_c"] = frame["baseline"] - frame["baseline"].mean()
    return frame


def contrast(fit, vector, null, label, outcome):
    vector = np.asarray(vector, dtype=float)
    estimate = float(vector @ fit.params.to_numpy())
    se = float(np.sqrt(vector @ fit.cov_params().to_numpy() @ vector))
    df = 17
    statistic = (estimate - null) / se
    p = 2 * stats.t.sf(abs(statistic), df)
    critical = stats.t.ppf(0.975, df)
    return {
        "outcome": outcome,
        "contrast": label,
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "null": null,
        "t17": statistic,
        "p": p,
    }


def ancova(frame, outcome):
    frame = frame.copy()
    frame["hrtf"] = pd.Categorical(frame["hrtf"], ["individual", "optimised"])
    fit = smf.ols("post ~ baseline_c * C(hrtf)", data=frame).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["participant"], "use_correction": True},
    )
    names = list(fit.params.index)

    def v(**weights):
        return [weights.get(name, 0.0) for name in names]

    baseline_name = "baseline_c"
    interaction_name = "baseline_c:C(hrtf)[T.optimised]"
    individual = v(**{baseline_name: 1})
    optimised = v(**{baseline_name: 1, interaction_name: 1})
    interaction = v(**{interaction_name: 1})
    rows = [
        contrast(fit, individual, 1, "Individual: post-on-baseline slope versus 1", outcome),
        contrast(fit, optimised, 1, "Optimised: post-on-baseline slope versus 1", outcome),
        contrast(fit, interaction, 0, "Optimised minus individual slope", outcome),
    ]
    return fit, pd.DataFrame(rows)


def oldham(frame, outcome):
    rows = []
    for hrtf in a.HRTFS:
        subset = frame[frame["hrtf"] == hrtf]
        pearson = stats.pearsonr(subset["mean_level"], subset["change"])
        pearson_ci = pearson.confidence_interval(0.95)
        spearman = stats.spearmanr(subset["mean_level"], subset["change"])
        rows.append(
            {
                "outcome": outcome,
                "hrtf": hrtf,
                "n": len(subset),
                "pre_sd": subset["baseline"].std(ddof=1),
                "post_sd": subset["post"].std(ddof=1),
                "pearson_r": pearson.statistic,
                "pearson_ci_low": pearson_ci.low,
                "pearson_ci_high": pearson_ci.high,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )
    return pd.DataFrame(rows)


def plot_results(frames, fits):
    colors = {"individual": "#2166AC", "optimised": "#B2182B"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.5), constrained_layout=True)
    for column, outcome in enumerate(["Front-back accuracy", "Separated-speech accuracy"]):
        frame = frames[outcome]
        fit = fits[outcome]
        ax = axes[0, column]
        lower = min(frame["baseline"].min(), frame["post"].min()) - 3
        upper = max(frame["baseline"].max(), frame["post"].max()) + 3
        ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
        for hrtf in a.HRTFS:
            subset = frame[frame["hrtf"] == hrtf]
            ax.scatter(
                subset["baseline"], subset["post"], label=hrtf.capitalize(),
                color=colors[hrtf], alpha=0.75, edgecolor="white", linewidth=0.5,
            )
            grid = np.linspace(subset["baseline"].min(), subset["baseline"].max(), 100)
            prediction = pd.DataFrame(
                {
                    "baseline": grid,
                    "baseline_c": grid - frame["baseline"].mean(),
                    "hrtf": hrtf,
                }
            )
            ax.plot(grid, fit.predict(prediction), color=colors[hrtf], linewidth=2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_title(outcome, weight="bold")
        ax.set_xlabel("Baseline accuracy (%)")
        ax.set_ylabel("Post-training accuracy (%)")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(frameon=False)

        ax = axes[1, column]
        for hrtf in a.HRTFS:
            subset = frame[frame["hrtf"] == hrtf]
            ax.scatter(
                subset["mean_level"], subset["change"], label=hrtf.capitalize(),
                color=colors[hrtf], alpha=0.75, edgecolor="white", linewidth=0.5,
            )
            slope, intercept = np.polyfit(subset["mean_level"], subset["change"], 1)
            grid = np.linspace(subset["mean_level"].min(), subset["mean_level"].max(), 100)
            ax.plot(grid, intercept + slope * grid, color=colors[hrtf], linewidth=2)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xlabel("Mean of baseline and post-training accuracy (%)")
        ax.set_ylabel("Post minus baseline accuracy (pp)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Baseline-dependent change: ANCOVA (top) and Oldham sensitivity analysis (bottom)",
        weight="bold",
    )
    path = OUTPUT / "baseline_dependence_preview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fb, speech = make_wide_tables()
    frames = {
        "Front-back accuracy": long_from_wide(fb, "Front-back accuracy"),
        "Separated-speech accuracy": long_from_wide(speech, "Separated-speech accuracy"),
    }
    fits = {}
    ancova_tables = []
    oldham_tables = []
    for outcome, frame in frames.items():
        fit, table = ancova(frame, outcome)
        fits[outcome] = fit
        ancova_tables.append(table)
        oldham_tables.append(oldham(frame, outcome))
    ancova_table = pd.concat(ancova_tables, ignore_index=True)
    oldham_table = pd.concat(oldham_tables, ignore_index=True)
    raw = pd.concat(frames.values(), ignore_index=True)
    ancova_table.to_csv(OUTPUT / "baseline_dependence_ancova.csv", index=False)
    oldham_table.to_csv(OUTPUT / "baseline_dependence_oldham.csv", index=False)
    raw.to_csv(OUTPUT / "baseline_dependence_values.csv", index=False)
    plot_path = plot_results(frames, fits)
    print("ANCOVA")
    print(ancova_table.to_string(index=False))
    print("\nOLDHAM")
    print(oldham_table.to_string(index=False))
    print(f"\n{plot_path}")


if __name__ == "__main__":
    main()
