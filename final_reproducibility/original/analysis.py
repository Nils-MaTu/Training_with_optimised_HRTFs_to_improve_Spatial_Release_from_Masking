#!/usr/bin/env python3
"""Reproduce the statistical analyses reported in the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARTICIPANTS = DATA / "participants"
ALPHA = 0.05
SESSIONS = ("before", "after")
HRTFS = ("individual", "optimised")
SEPARATED = ("T30 M150", "T150 M30")


def load_participant_data(filename: str) -> pd.DataFrame:
    paths = sorted(PARTICIPANTS.glob(f"sub-*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No participant files named {filename!r} in {PARTICIPANTS}")
    tables = [pd.read_csv(path) for path in paths]
    for path, table in zip(paths, tables):
        expected = path.parent.name
        if set(table["participant"]) != {expected}:
            raise ValueError(f"Participant ID in {path} does not match its directory")
    return pd.concat(tables, ignore_index=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fb = load_participant_data("front_back_trials.csv")
    srm = load_participant_data("srm_trials.csv")
    fb_required = {"participant", "session", "hrtf", "location_deg", "correct"}
    srm_required = {
        "participant", "session", "hrtf", "target_azimuth_deg",
        "masker_azimuth_deg", "correct", "predicted_srm_db",
    }
    for name, table, required in (("FB", fb, fb_required), ("SRM", srm, srm_required)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {sorted(missing)}")
        if table["participant"].nunique() != 18:
            raise ValueError(f"{name} data must contain 18 participants")
        if set(table["session"]) != set(SESSIONS) or set(table["hrtf"]) != set(HRTFS):
            raise ValueError(f"Unexpected {name} session or HRTF labels")
        if not table["correct"].isin([0, 1]).all():
            raise ValueError(f"{name} correct column must be binary")
    srm["configuration"] = (
        "T" + srm["target_azimuth_deg"].astype(int).astype(str)
        + " M" + srm["masker_azimuth_deg"].astype(int).astype(str)
    )
    if set(srm["configuration"]) != {"T30 M30", *SEPARATED}:
        raise ValueError("Unexpected target-masker configuration")
    return fb, srm


def participant_means(
    trials: pd.DataFrame, factors: list[str], value: str = "correct"
) -> pd.DataFrame:
    return (
        trials.groupby(["participant", *factors], observed=True)[value]
        .mean().mul(100).rename("accuracy_pct").reset_index()
    )


def rm_anova(data: pd.DataFrame, within: list[str], outcome: str) -> pd.DataFrame:
    table = AnovaRM(
        data, depvar=outcome, subject="participant", within=within
    ).fit().anova_table
    rows = []
    for effect, row in table.iterrows():
        f_value = float(row["F Value"])
        df_num = float(row["Num DF"])
        df_den = float(row["Den DF"])
        rows.append({
            "effect": effect,
            "F": f_value,
            "df_num": df_num,
            "df_den": df_den,
            "p": float(row["Pr > F"]),
            "partial_eta_squared": f_value * df_num / (f_value * df_num + df_den),
        })
    return pd.DataFrame(rows)


def shapiro(values: np.ndarray) -> tuple[float, float, bool]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan, False
    if np.allclose(values, values[0]):
        return np.nan, 1.0, True
    result = stats.shapiro(values)
    return float(result.statistic), float(result.pvalue), bool(result.pvalue >= ALPHA)


def paired_test(first: pd.Series, second: pd.Series, label: str) -> dict[str, object]:
    pair = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    difference = pair["first"].to_numpy() - pair["second"].to_numpy()
    w, p_shapiro, normal = shapiro(difference)
    if normal:
        result = stats.ttest_rel(pair["first"], pair["second"])
        test = "paired t-test"
        sd = difference.std(ddof=1)
        effect_name = "Cohen dz"
        effect = difference.mean() / sd if sd else 0.0
    elif np.allclose(difference, 0):
        result = type("Result", (), {"statistic": 0.0, "pvalue": 1.0})()
        test = "Wilcoxon signed-rank"
        effect_name, effect = "rank-biserial correlation", 0.0
    else:
        result = stats.wilcoxon(pair["first"], pair["second"])
        test = "Wilcoxon signed-rank"
        nonzero = difference[difference != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        positive = ranks[nonzero > 0].sum()
        negative = ranks[nonzero < 0].sum()
        effect_name = "rank-biserial correlation"
        effect = (positive - negative) / (positive + negative)
    return {
        "contrast": label,
        "n": len(difference),
        "test": test,
        "statistic": float(result.statistic),
        "p_raw": float(result.pvalue),
        "mean_difference_pp": float(difference.mean()),
        "median_difference_pp": float(np.median(difference)),
        "effect_size_name": effect_name,
        "effect_size": float(effect),
        "shapiro_W": w,
        "shapiro_p": p_shapiro,
    }


def wilcoxon_test(first: pd.Series, second: pd.Series, label: str) -> dict[str, object]:
    """Pre-specified Wilcoxon test used for the optimised configuration ordering."""
    pair = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    difference = pair["first"].to_numpy() - pair["second"].to_numpy()
    result = stats.wilcoxon(pair["first"], pair["second"])
    nonzero = difference[difference != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    w, p_shapiro, _ = shapiro(difference)
    return {
        "contrast": label, "n": len(difference), "test": "Wilcoxon signed-rank",
        "statistic": float(result.statistic), "p_raw": float(result.pvalue),
        "mean_difference_pp": float(difference.mean()),
        "median_difference_pp": float(np.median(difference)),
        "effect_size_name": "rank-biserial correlation",
        "effect_size": float((positive - negative) / (positive + negative)),
        "shapiro_W": w, "shapiro_p": p_shapiro,
    }


def front_back_analysis(fb: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    cell = participant_means(fb, ["session", "hrtf", "location_deg"])
    anova = rm_anova(cell, ["session", "hrtf", "location_deg"], "accuracy_pct")
    averaged = (
        cell.groupby(["participant", "session", "hrtf"], observed=True)
        ["accuracy_pct"].mean().reset_index()
    )
    wide = averaged.pivot(index="participant", columns=["session", "hrtf"], values="accuracy_pct")
    contrasts = pd.DataFrame([
        paired_test(wide[("before", "optimised")], wide[("before", "individual")],
                    "H2: optimised - individual before"),
        paired_test(wide[("after", "optimised")], wide[("after", "individual")],
                    "optimised - individual after"),
        paired_test(wide[("after", "individual")], wide[("before", "individual")],
                    "individual after - before"),
    ])
    improvement_individual = wide[("after", "individual")] - wide[("before", "individual")]
    improvement_optimised = wide[("after", "optimised")] - wide[("before", "optimised")]
    h3 = pd.DataFrame([
        paired_test(improvement_optimised, improvement_individual,
                    "H3: optimised improvement - individual improvement")
    ])
    h3.insert(1, "mean_improvement_individual_pp", improvement_individual.mean())
    h3.insert(2, "mean_improvement_optimised_pp", improvement_optimised.mean())
    descriptives = (
        cell.groupby(["session", "hrtf", "location_deg"], observed=True)["accuracy_pct"]
        .agg(["mean", "sem"]).reset_index()
    )
    cell.to_csv(output / "front_back_participant_means.csv", index=False)
    descriptives.to_csv(output / "front_back_descriptives.csv", index=False)
    anova.to_csv(output / "front_back_anova.csv", index=False)
    contrasts.to_csv(output / "front_back_contrasts.csv", index=False)
    h3.to_csv(output / "front_back_h3_contrast.csv", index=False)
    return {"anova": anova, "contrasts": contrasts, "h3": h3, "cell": cell}


def srm_participant_cells(srm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = participant_means(srm, ["session", "hrtf", "configuration"])
    colocated = (
        cell[cell["configuration"] == "T30 M30"]
        .drop(columns="configuration")
        .rename(columns={"accuracy_pct": "colocated_accuracy_pct"})
    )
    esrm = cell.merge(colocated, on=["participant", "session", "hrtf"], validate="many_to_one")
    esrm["srm_pp"] = esrm["accuracy_pct"] - esrm["colocated_accuracy_pct"]
    return cell, esrm


def srm_analysis(srm: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    cell, esrm = srm_participant_cells(srm)
    separated = esrm[esrm["configuration"].isin(SEPARATED)].copy()
    anova_input = separated[["participant", "session", "hrtf", "configuration", "srm_pp"]]
    anova = rm_anova(anova_input, ["session", "hrtf", "configuration"], "srm_pp")

    wide = separated.pivot(
        index="participant", columns=["session", "hrtf", "configuration"], values="srm_pp"
    )
    h4_rows = []
    for configuration in SEPARATED:
        before = wide[("before", "optimised", configuration)] - wide[("before", "individual", configuration)]
        after = wide[("after", "optimised", configuration)] - wide[("after", "individual", configuration)]
        row = paired_test(after, before, f"H4: optimisation-benefit change, {configuration}")
        row["configuration"] = configuration
        row["benefit_before_pp"] = before.mean()
        row["benefit_after_pp"] = after.mean()
        h4_rows.append(row)
    h4 = pd.DataFrame(h4_rows)
    reject, adjusted, _, _ = multipletests(h4["p_raw"], alpha=ALPHA, method="holm")
    h4["p_holm"] = adjusted
    h4["reject_holm"] = reject

    descriptives = (
        cell.groupby(["session", "hrtf", "configuration"], observed=True)["accuracy_pct"]
        .agg(["mean", "sem"]).reset_index()
    )
    predicted_descriptives = (
        srm.groupby(["hrtf", "configuration"], observed=True)["predicted_srm_db"]
        .agg(["mean", "std"]).reset_index()
    )

    cell.to_csv(output / "srm_accuracy_participant_means.csv", index=False)
    descriptives.to_csv(output / "srm_accuracy_descriptives.csv", index=False)
    predicted_descriptives.to_csv(output / "predicted_srm_descriptives.csv", index=False)
    separated.to_csv(output / "srm_participant_means.csv", index=False)
    anova.to_csv(output / "srm_anova.csv", index=False)
    h4.to_csv(output / "srm_h4_contrasts.csv", index=False)
    return {"anova": anova, "h4": h4, "cell": cell, "esrm": esrm}


def predicted_srm_analysis(srm: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    trials = srm.copy()
    trials["post"] = (trials["session"] == "after").astype(int)
    trials["separated"] = (trials["configuration"] != "T30 M30").astype(int)
    trials["predicted_srm_z"] = (
        trials["predicted_srm_db"] - trials["predicted_srm_db"].mean()
    ) / trials["predicted_srm_db"].std(ddof=1)
    individual = trials[trials["hrtf"] == "individual"].copy()
    fit = smf.gee(
        "correct ~ predicted_srm_z * post + separated * post",
        groups=individual["participant"], data=individual,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    ci = fit.conf_int()
    coefficients = pd.DataFrame({
        "term": fit.params.index,
        "log_odds": fit.params.values,
        "robust_se": fit.bse.values,
        "z": fit.tvalues.values,
        "p": fit.pvalues.values,
        "odds_ratio": np.exp(fit.params.values),
        "odds_ratio_ci_low": np.exp(ci.iloc[:, 0].values),
        "odds_ratio_ci_high": np.exp(ci.iloc[:, 1].values),
    })

    optimised = participant_means(
        trials[(trials["hrtf"] == "optimised") & trials["configuration"].isin(SEPARATED)],
        ["session", "configuration"],
    ).pivot(index="participant", columns=["session", "configuration"], values="accuracy_pct")
    ordering_rows = []
    for session in SESSIONS:
        first = optimised[(session, "T150 M30")]
        second = optimised[(session, "T30 M150")]
        row = wilcoxon_test(first, second, f"T150 M30 - T30 M150, {session}")
        row["n_higher"] = int((first > second).sum())
        ordering_rows.append(row)
    ordering = pd.DataFrame(ordering_rows)
    coefficients.to_csv(output / "predicted_srm_gee_coefficients.csv", index=False)
    ordering.to_csv(output / "optimised_configuration_contrasts.csv", index=False)
    return {"coefficients": coefficients, "ordering": ordering}


def correlation_result(x: pd.Series, y: pd.Series, label: str) -> dict[str, object]:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    x_values, y_values = pair["x"].to_numpy(), pair["y"].to_numpy()
    x_w, x_p, x_normal = shapiro(x_values)
    y_w, y_p, y_normal = shapiro(y_values)
    if x_normal and y_normal:
        result, method = stats.pearsonr(x_values, y_values), "Pearson"
    else:
        result, method = stats.spearmanr(x_values, y_values), "Spearman"
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return {
        "outcome": label, "n": len(pair), "method": method,
        "correlation": float(result.statistic), "p": float(result.pvalue),
        "slope": float(slope), "intercept": float(intercept),
        "x_shapiro_W": x_w, "x_shapiro_p": x_p,
        "y_shapiro_W": y_w, "y_shapiro_p": y_p,
    }


def baseline_analysis(
    fb: pd.DataFrame, srm_cells: pd.DataFrame, output: Path
) -> pd.DataFrame:
    fb_cell = participant_means(fb, ["session", "hrtf"])
    fb_wide = fb_cell.pivot(index="participant", columns=["session", "hrtf"], values="accuracy_pct")
    srm_sep = srm_cells[srm_cells["configuration"].isin(SEPARATED)]
    srm_mean = (
        srm_sep.groupby(["participant", "session", "hrtf"], observed=True)
        ["accuracy_pct"].mean().reset_index()
    )
    srm_wide = srm_mean.pivot(index="participant", columns=["session", "hrtf"], values="accuracy_pct")
    rows = []
    for hrtf in HRTFS:
        rows.append(correlation_result(
            fb_wide[("before", hrtf)],
            fb_wide[("after", hrtf)] - fb_wide[("before", hrtf)],
            f"front-back, {hrtf}",
        ))
    for hrtf in HRTFS:
        rows.append(correlation_result(
            srm_wide[("before", hrtf)],
            srm_wide[("after", hrtf)] - srm_wide[("before", hrtf)],
            f"SRM, {hrtf}",
        ))
    table = pd.DataFrame(rows)
    table.to_csv(output / "baseline_improvement_correlations.csv", index=False)
    return table


def manuscript_summary(results: dict[str, object]) -> str:
    fb = results["fb"]
    srm = results["srm"]
    predicted = results["predicted"]
    baseline = results["baseline"]
    fb_anova = fb["anova"].set_index("effect")
    srm_anova = srm["anova"].set_index("effect")
    gee = predicted["coefficients"].set_index("term")
    order = predicted["ordering"].set_index("contrast")
    corr = baseline.set_index("outcome")
    lines = [
        "Manuscript result check (N=18)",
        "",
        f"FB HRTF x location: F(1,17)={fb_anova.loc['hrtf:location_deg','F']:.2f}, p={fb_anova.loc['hrtf:location_deg','p']:.4g}, partial eta^2={fb_anova.loc['hrtf:location_deg','partial_eta_squared']:.2f}",
        f"FB session x HRTF: F(1,17)={fb_anova.loc['session:hrtf','F']:.2f}, p={fb_anova.loc['session:hrtf','p']:.4g}, partial eta^2={fb_anova.loc['session:hrtf','partial_eta_squared']:.2f}",
        f"H2 contrast: mean difference={fb['contrasts'].iloc[0]['mean_difference_pp']:.1f} pp, t={fb['contrasts'].iloc[0]['statistic']:.2f}, p={fb['contrasts'].iloc[0]['p_raw']:.4g}",
        f"H3 contrast: mean difference={fb['h3'].iloc[0]['mean_difference_pp']:.1f} pp, t={fb['h3'].iloc[0]['statistic']:.2f}, p={fb['h3'].iloc[0]['p_raw']:.4g}",
        f"Post-training optimised-individual: mean difference={fb['contrasts'].iloc[1]['mean_difference_pp']:.1f} pp, t={fb['contrasts'].iloc[1]['statistic']:.2f}, p={fb['contrasts'].iloc[1]['p_raw']:.4g}",
        "",
        f"SRM HRTF: F(1,17)={srm_anova.loc['hrtf','F']:.2f}, p={srm_anova.loc['hrtf','p']:.4g}, partial eta^2={srm_anova.loc['hrtf','partial_eta_squared']:.2f}",
        f"SRM configuration: F(1,17)={srm_anova.loc['configuration','F']:.2f}, p={srm_anova.loc['configuration','p']:.4g}, partial eta^2={srm_anova.loc['configuration','partial_eta_squared']:.2f}",
        f"SRM session: F(1,17)={srm_anova.loc['session','F']:.2f}, p={srm_anova.loc['session','p']:.4g}, partial eta^2={srm_anova.loc['session','partial_eta_squared']:.2f}",
        f"SRM HRTF x configuration: F(1,17)={srm_anova.loc['hrtf:configuration','F']:.2f}, p={srm_anova.loc['hrtf:configuration','p']:.4g}, partial eta^2={srm_anova.loc['hrtf:configuration','partial_eta_squared']:.2f}",
        f"SRM 3-way interaction: F(1,17)={srm_anova.loc['session:hrtf:configuration','F']:.2f}, p={srm_anova.loc['session:hrtf:configuration','p']:.4g}, partial eta^2={srm_anova.loc['session:hrtf:configuration','partial_eta_squared']:.2f}",
        f"H4 T30M150: change={srm['h4'].iloc[0]['mean_difference_pp']:.1f} pp, t={srm['h4'].iloc[0]['statistic']:.2f}, Holm p={srm['h4'].iloc[0]['p_holm']:.4g}",
        f"H4 T150M30: change={srm['h4'].iloc[1]['mean_difference_pp']:.1f} pp, t={srm['h4'].iloc[1]['statistic']:.2f}, Holm p={srm['h4'].iloc[1]['p_holm']:.4g}",
        "",
        f"GEE predicted SRM: OR={gee.loc['predicted_srm_z','odds_ratio']:.2f}, 95% CI [{gee.loc['predicted_srm_z','odds_ratio_ci_low']:.2f}, {gee.loc['predicted_srm_z','odds_ratio_ci_high']:.2f}], p={gee.loc['predicted_srm_z','p']:.4g}",
        f"GEE separated: OR={gee.loc['separated','odds_ratio']:.2f}, 95% CI [{gee.loc['separated','odds_ratio_ci_low']:.2f}, {gee.loc['separated','odds_ratio_ci_high']:.2f}], p={gee.loc['separated','p']:.4g}",
        "",
        f"Optimised T150M30-T30M150 before: {order.loc['T150 M30 - T30 M150, before','mean_difference_pp']:.1f} pp, Wilcoxon p={order.loc['T150 M30 - T30 M150, before','p_raw']:.4g}",
        f"Optimised T150M30-T30M150 after: {order.loc['T150 M30 - T30 M150, after','mean_difference_pp']:.1f} pp, Wilcoxon p={order.loc['T150 M30 - T30 M150, after','p_raw']:.4g}",
        "",
    ]
    for label in corr.index:
        symbol = "rho" if corr.loc[label, "method"] == "Spearman" else "r"
        lines.append(f"Baseline/improvement {label}: {symbol}={corr.loc[label,'correlation']:.2f}, p={corr.loc[label,'p']:.4g}, slope={corr.loc[label,'slope']:.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "statistics")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fb, srm = load_data()
    fb_results = front_back_analysis(fb, output)
    srm_results = srm_analysis(srm, output)
    predicted_results = predicted_srm_analysis(srm, output)
    baseline = baseline_analysis(fb, srm_results["cell"], output)
    summary = manuscript_summary({
        "fb": fb_results, "srm": srm_results,
        "predicted": predicted_results, "baseline": baseline,
    })
    (output / "manuscript_results.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Detailed tables: {output}")


if __name__ == "__main__":
    main()
