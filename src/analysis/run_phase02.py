"""Phase 02 pipeline: exploratory analysis, temporal feasibility and leakage checks.

Usage:
    python -m src.analysis.run_phase02            # both datasets
    python -m src.analysis.run_phase02 --only uci

Outputs (no model is trained):
    research/inspection/phase02_oulad.json, phase02_uci.json   (versioned, aggregates only)
    research/inspection/*.csv                                    (small aggregate tables)
    research/figures/*.png                                       (versioned)
    data/interim/oulad_*.csv.gz                                  (NOT versioned)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import eda, leakage, plots
from src.data.inspect_dataset import profile_dataframe
from src.data.loaders import OULAD_KEY, PROJECT_ROOT, load_oulad, load_student_vle, load_uci
from src.data.quality import oulad_integrity, uci_consistency
from src.feature_engineering import temporal

INSPECTION = PROJECT_ROOT / "research" / "inspection"
FIGURES = PROJECT_ROOT / "research" / "figures"
INTERIM = PROJECT_ROOT / "data" / "interim"

WEEKS = (2, 4, 6)
CUTOFFS = {f"S{w}": temporal.cutoff_day_for_week(w) for w in WEEKS}

UCI_NOMINAL = ("Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
               "Mother's qualification", "Father's qualification", "Mother's occupation",
               "Father's occupation")

AS_OF_INDICATORS = {
    "total_clicks": "Cliques acumulados",
    "active_days": "Dias ativos",
    "distinct_sites": "Materiais distintos",
    "clicks_pre_start": "Cliques antes do início",
    "clicks_last_7_days": "Cliques nos últimos 7 dias",
    "days_since_last_activity": "Dias desde o último acesso",
    "n_submitted": "Avaliações entregues",
    "mean_score": "Nota média (entregues)",
    "n_missing_submissions": "Avaliações vencidas não entregues",
}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, pd.DataFrame):
        o = o.copy()
        if isinstance(o.columns, pd.MultiIndex):
            o.columns = [" | ".join(map(str, c)) for c in o.columns]
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    raise TypeError(type(o))


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _round(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return df.round(n)


# --------------------------------------------------------------------------- OULAD
def analyse_oulad() -> dict:
    t = load_oulad()
    sv = load_student_vle()
    base = temporal.student_base(t)
    r: dict = {}

    # 3.1 Structure -------------------------------------------------------------
    r["structure"] = {name: profile_dataframe(df) for name, df in t.items()}
    r["structure"]["studentVle"] = {
        "n_rows": int(len(sv)), "n_columns": int(sv.shape[1]),
        "dtypes": {c: str(d) for c, d in sv.dtypes.items()},
        "total_missing_cells": int(sv.isna().sum().sum()),
        "n_full_row_duplicates": int(sv.duplicated().sum()),
        "date_min": int(sv["date"].min()), "date_max": int(sv["date"].max()),
        "sum_click_quantiles": sv["sum_click"].quantile([0.5, 0.9, 0.99, 1.0]).to_dict(),
    }
    r["integrity"] = oulad_integrity(t, sv)
    per_student = t["studentInfo"].groupby("id_student").size()
    r["students_in_multiple_presentations"] = {
        "n_students": int((per_student > 1).sum()),
        "n_unique_students": int(per_student.size),
        "max_presentations_per_student": int(per_student.max()),
    }

    # 3.2 Academic ---------------------------------------------------------------
    r["outcome_distribution"] = eda.distribution(base["final_result"])
    r["outcome_by_module"] = _round(pd.crosstab(base["code_module"], base["final_result"], normalize="index"))
    r["outcome_by_presentation"] = _round(pd.crosstab(base["code_presentation"], base["final_result"], normalize="index"))
    r["binary_label_options"] = {
        "A_withdrawn_vs_pass_distinction (Fail excluded)": {
            "n": int(base["final_result"].isin(["Withdrawn", "Pass", "Distinction"]).sum()),
            "positive_rate": round(float((base["final_result"] == "Withdrawn").sum()
                                         / base["final_result"].isin(["Withdrawn", "Pass", "Distinction"]).sum()), 4),
        },
        "B_withdrawn_vs_all_others": {
            "n": int(len(base)), "positive_rate": round(float((base["final_result"] == "Withdrawn").mean()), 4),
        },
        "C_withdrawn_or_fail_vs_pass_distinction": {
            "n": int(len(base)),
            "positive_rate": round(float(base["final_result"].isin(["Withdrawn", "Fail"]).mean()), 4),
        },
    }
    unreg = base.loc[base["final_result"] == "Withdrawn", "date_unregistration"]
    bins = [-np.inf, -1, 13, 27, 41, np.inf]
    labels = ["antes do dia 0", "semanas 1-2", "semanas 3-4", "semanas 5-6", "após semana 6"]
    timing = pd.cut(unreg, bins=bins, labels=labels).value_counts(dropna=False).reindex(labels + [np.nan])
    r["withdrawal_timing"] = {
        "quantiles": unreg.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "counts_by_window": {str(k): int(v) for k, v in timing.items()},
    }

    withdrawn = base["final_result"] == "Withdrawn"
    overall = float(withdrawn.mean())
    base_cat = base.assign(
        studied_credits_band=pd.cut(base["studied_credits"], [0, 60, 90, 120, 700],
                                    labels=["≤60", "61-90", "91-120", ">120"]),
        num_of_prev_attempts_cap=base["num_of_prev_attempts"].clip(upper=3).astype(str).replace("3", "3+"),
        registration_band=pd.cut(base["date_registration"], [-400, -90, -30, 0, 400],
                                 labels=["≤-90", "-89..-30", "-29..0", ">0 (após início)"]),
    )
    rate_cols = ["highest_education", "imd_band", "age_band", "num_of_prev_attempts_cap",
                 "studied_credits_band", "registration_band", "disability", "gender", "code_module"]
    rates = {c: eda.rate_by_category(base_cat, c, withdrawn) for c in rate_cols}
    r["withdrawal_rate_by_attribute"] = rates
    r["cramers_v_with_final_result"] = {
        c: round(eda.cramers_v(base_cat[c].astype(str), base["final_result"]), 4)
        for c in rate_cols + ["region"]
    }

    # Scores of non-exam assessments (whole presentation: descriptive only)
    sa = t["studentAssessment"].merge(t["assessments"], on="id_assessment", how="left")
    sa = sa.merge(base[[*OULAD_KEY, "final_result"]], on=OULAD_KEY, how="left")
    non_exam = sa.loc[sa["assessment_type"] != "Exam"]
    r["non_exam_score_by_outcome"] = _round(non_exam.groupby("final_result")["score"].describe())
    first_ass = (t["assessments"].loc[t["assessments"]["assessment_type"] != "Exam"]
                 .sort_values("date").groupby(["code_module", "code_presentation"]).head(1))
    first_sub = base[[*OULAD_KEY, "final_result", "date_unregistration"]].merge(
        first_ass[["code_module", "code_presentation", "id_assessment", "date"]], on=["code_module", "code_presentation"])
    first_sub = first_sub.loc[first_sub["date_unregistration"].isna() | (first_sub["date_unregistration"] > first_sub["date"])]
    first_sub = first_sub.merge(t["studentAssessment"], on=["id_assessment", "id_student"], how="left")
    r["first_assessment"] = {
        "due_day": first_ass["date"].describe().to_dict(),
        "share_submitted_by_outcome (students still registered at due date)":
            first_sub.assign(s=first_sub["score"].notna()).groupby("final_result")["s"].mean().round(4).to_dict(),
        "median_score_by_outcome": first_sub.groupby("final_result")["score"].median().to_dict(),
    }
    plots.score_distribution(first_sub, "score", "final_result", plots.OULAD_COLORS,
                             "OULAD — Nota na 1ª avaliação não-final, por resultado final",
                             "score (0–100)", FIGURES / "oulad_first_assessment_score.png")

    # Silent disengagement among Fail (label-quality diagnostic, NOT a feature)
    last = eda.last_activity_profile(sv, base)
    r["last_activity_ratio_by_outcome"] = _round(last.groupby("final_result")["last_day_ratio"].describe())
    r["share_last_activity_before_half"] = (
        last.assign(h=last["last_day_ratio"] < 0.5).groupby("final_result")["h"].mean().round(4).to_dict())
    exam_ids = t["assessments"].loc[t["assessments"]["assessment_type"] == "Exam", "id_assessment"]
    sat_exam = t["studentAssessment"].loc[t["studentAssessment"]["id_assessment"].isin(exam_ids), "id_student"]
    modules_with_exam_records = sa.loc[sa["assessment_type"] == "Exam", "code_module"].unique().tolist()
    fail_in_exam_modules = base.loc[(base["final_result"] == "Fail") & base["code_module"].isin(modules_with_exam_records)]
    r["fail_without_exam_submission"] = {
        "modules_with_exam_records": sorted(modules_with_exam_records),
        "share": round(float((~fail_in_exam_modules["id_student"].isin(sat_exam)).mean()), 4),
        "n_fail_in_those_modules": int(len(fail_in_exam_modules)),
    }

    # 3.3 Engagement ---------------------------------------------------------------
    weekly = temporal.weekly_clicks(sv)
    INTERIM.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(INTERIM / "oulad_weekly_clicks.csv.gz", index=False)
    wbo = eda.mean_weekly_clicks_by_outcome(weekly, base, range(-3, 40))
    wbo.to_csv(INSPECTION / "oulad_weekly_clicks_by_outcome.csv", index=False)
    r["weekly_mean_clicks_weeks_1_6"] = (
        wbo.loc[wbo["week"].between(1, 6)].pivot(index="week", columns="final_result", values="mean").round(1))
    rel = eda.clicks_relative_to_withdrawal(weekly, base, weeks_before=8)
    r["clicks_relative_to_withdrawal"] = _round(rel, 3)

    sv_w4 = sv.loc[sv["date"] <= CUTOFFS["S4"]].merge(t["vle"][["id_site", "activity_type"]], on="id_site")
    sv_w4 = sv_w4.merge(base[[*OULAD_KEY, "final_result"]].astype({"code_module": "category", "code_presentation": "category"}),
                        on=OULAD_KEY, how="left")
    act = sv_w4.groupby(["activity_type", "final_result"], observed=True)["sum_click"].sum().unstack()
    act_share = (act / act.sum()).fillna(0)
    top = act_share.mean(axis=1).sort_values(ascending=False).head(10).index
    r["activity_type_click_share_until_week4"] = _round(act_share.loc[top][list(plots.OULAD_COLORS)], 4)
    del sv_w4

    # 3.4 Temporal -------------------------------------------------------------------
    snaps, aucs, per_cutoff = [], [], {}
    for label, cut in CUTOFFS.items():
        s = temporal.snapshot(t, sv, cut)
        s.to_csv(INTERIM / f"oulad_snapshot_{label.lower()}.csv.gz", index=False)
        s["week_label"] = f"Semana {label[1:]}"
        snaps.append(s[[*OULAD_KEY, "final_result", "week_label", "total_clicks"]])
        bin_a = s.loc[s["final_result"].isin(["Withdrawn", "Pass", "Distinction"])]
        y = (bin_a["final_result"] == "Withdrawn").astype(int)
        for col, name in AS_OF_INDICATORS.items():
            auc = eda.univariate_auc(bin_a[col], y)
            aucs.append({"week_label": f"Semana {label[1:]}", "indicator": name, "variable": col,
                         "auc": round(auc, 4), "separation": round(eda.separation(auc), 4),
                         "coverage": round(float(bin_a[col].notna().mean()), 4)})
        per_cutoff[label] = {
            "cutoff_day": cut,
            "population": int(len(s)),
            "excluded_already_unregistered": int(len(base) - len(s)),
            "label_counts": s["final_result"].value_counts().to_dict(),
            "binary_A_positive_rate": round(float(y.mean()), 4),
            "medians_by_outcome": _round(s.groupby("final_result")[list(AS_OF_INDICATORS)].median(), 2),
            "share_zero_clicks_by_outcome": s.assign(z=s["total_clicks"] == 0).groupby("final_result")["z"].mean().round(4).to_dict(),
            "share_with_submission": round(float((s["n_submitted"] > 0).mean()), 4),
            "share_presentations_with_due_assessment": round(float((s["n_due"] > 0).mean()), 4),
            "remaining_withdrawals_after_cutoff_share_of_all_withdrawn":
                round(float((s["final_result"] == "Withdrawn").sum() / (base["final_result"] == "Withdrawn").sum()), 4),
        }
    r["temporal_cutoffs"] = per_cutoff
    auc_df = pd.DataFrame(aucs)
    auc_df.to_csv(INSPECTION / "oulad_univariate_separation_by_cutoff.csv", index=False)
    r["univariate_separation_by_cutoff"] = auc_df

    # Leakage reference: whole-presentation aggregates (PROHIBITED) on the week-4 population.
    s4 = pd.read_csv(INTERIM / "oulad_snapshot_s4.csv.gz")
    s4 = s4.loc[s4["final_result"].isin(["Withdrawn", "Pass", "Distinction"])]
    full =temporal.vle_as_of(sv, int(sv["date"].max()))[[*OULAD_KEY, "total_clicks", "last_active_day"]]
    full_sub = t["studentAssessment"].merge(t["assessments"][["id_assessment", "code_module", "code_presentation"]],
                                            on="id_assessment").groupby(OULAD_KEY).size().rename("n_submitted_full").reset_index()
    s4 = s4.merge(full.rename(columns={"total_clicks": "clicks_full", "last_active_day": "last_day_full"}),
                  on=OULAD_KEY, how="left").merge(full_sub, on=OULAD_KEY, how="left")
    s4[["clicks_full", "n_submitted_full"]] = s4[["clicks_full", "n_submitted_full"]].fillna(0)
    y4 = (s4["final_result"] == "Withdrawn").astype(int)  # after the merge: same index as s4
    r["leakage_reference_week4_population"] = {
        name: {"auc": round(a, 4), "separation": round(eda.separation(a), 4)}
        for name, a in {
            "total clicks over whole presentation": eda.univariate_auc(s4["clicks_full"], y4),
            "last active day over whole presentation": eda.univariate_auc(s4["last_day_full"], y4),
            "assessments submitted over whole presentation": eda.univariate_auc(s4["n_submitted_full"], y4),
        }.items()
    }

    # Availability over weeks 1..8
    avail = []
    for w in range(1, 9):
        cut = temporal.cutoff_day_for_week(w)
        pop = temporal.population_at_cutoff(base, cut)
        act_keys = sv.loc[sv["date"] <= cut, OULAD_KEY].drop_duplicates()
        act_keys = act_keys.astype({"code_module": str, "code_presentation": str}).assign(active=True)
        sub_keys = temporal.assessments_as_of(t["studentAssessment"], t["assessments"], cut)[OULAD_KEY].assign(sub=True)
        due = temporal.assessments_due_by(t["assessments"], cut)
        p = pop.merge(act_keys, on=OULAD_KEY, how="left").merge(sub_keys, on=OULAD_KEY, how="left")
        p = p.merge(due, on=["code_module", "code_presentation"], how="left")
        avail.append({
            "week": w, "cutoff_day": cut, "population": int(len(p)),
            "share_with_vle_activity": round(float(p["active"].notna().mean()), 4),
            "share_with_submission": round(float(p["sub"].notna().mean()), 4),
            "share_presentations_with_due_assessment": round(float((p["n_due"].fillna(0) > 0).mean()), 4),
        })
    avail_df = pd.DataFrame(avail)
    r["information_availability_by_week"] = avail_df

    # Figures --------------------------------------------------------------------------
    plots.outcome_bars(base["final_result"].value_counts(), plots.OULAD_COLORS,
                       "OULAD — Resultado final (32.593 matrículas estudante×módulo)",
                       FIGURES / "oulad_outcome_distribution.png")
    plots.outcome_share_by_group(base, "code_module", "final_result", plots.OULAD_COLORS,
                                 "OULAD — Resultado final por módulo", FIGURES / "oulad_outcome_by_module.png")
    plots.withdrawal_timing(unreg, CUTOFFS, FIGURES / "oulad_withdrawal_timing.png")
    plots.weekly_engagement(wbo.loc[wbo["week"] <= 38], CUTOFFS, FIGURES / "oulad_weekly_clicks_by_outcome.png")
    plots.disengagement_before_withdrawal(rel, FIGURES / "oulad_disengagement_before_withdrawal.png")
    plots.engagement_at_cutoffs(pd.concat(snaps), FIGURES / "oulad_clicks_at_cutoffs.png")
    ref = r["leakage_reference_week4_population"]
    plots.auc_by_cutoff(auc_df, FIGURES / "oulad_separation_by_cutoff.png",
                        reference={"Cliques no módulo inteiro": ref["total clicks over whole presentation"]["separation"]})
    plots.rate_panels({c: rates[c] for c in rate_cols if c != "code_module"}, overall,
                      "OULAD — Taxa de Withdrawn por característica de ingresso",
                      FIGURES / "oulad_withdrawal_rate_by_attribute.png")
    plots.activity_type_shares(r["activity_type_click_share_until_week4"], FIGURES / "oulad_activity_type_until_week4.png")
    plots.availability_over_weeks(avail_df, FIGURES / "oulad_information_availability.png")
    return r


# --------------------------------------------------------------------------- UCI
def analyse_uci() -> dict:
    df = load_uci()
    r: dict = {"structure": profile_dataframe(df, target="Target"), "consistency": uci_consistency(df)}
    r["target_distribution"] = eda.distribution(df["Target"])

    key_vars = ["Admission grade", "Previous qualification (grade)", "Age at enrollment",
                "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)"]
    r["medians_by_class"] = _round(df.groupby("Target")[key_vars].median(), 2)
    binary_vars = ["Debtor", "Tuition fees up to date", "Scholarship holder", "Displaced", "Gender",
                   "International", "Educational special needs", "Daytime/evening attendance"]
    r["binary_share_by_class"] = _round(df.groupby("Target")[binary_vars].mean(), 3)
    for sem in ("1st", "2nd"):
        p = f"Curricular units {sem} sem "
        enr = df[p + "(enrolled)"]
        df[f"_approval_ratio_{sem}"] = (df[p + "(approved)"] / enr.where(enr > 0))
    r["approval_ratio_by_class"] = _round(df.groupby("Target")[["_approval_ratio_1st", "_approval_ratio_2nd"]].describe(), 3)
    df = df.drop(columns=["_approval_ratio_1st", "_approval_ratio_2nd"])

    # Label options
    r["binary_label_options"] = {
        "A_dropout_vs_graduate (Enrolled excluded)": {"n": int(df["Target"].isin(["Dropout", "Graduate"]).sum()),
                                                      "positive_rate": round(1421 / (1421 + 2209), 4)},
        "B_dropout_vs_graduate_plus_enrolled": {"n": int(len(df)), "positive_rate": round(float((df["Target"] == "Dropout").mean()), 4)},
        "C_dropout_plus_enrolled_vs_graduate (Enrolled as 'not completed on time')": {
            "n": int(len(df)), "positive_rate": round(float(df["Target"].isin(["Dropout", "Enrolled"]).mean()), 4)},
    }

    # Associations: each feature vs Dropout×Graduate (A) and Dropout×rest (B)
    cat = leakage.catalog_frame(leakage.UCI_CATALOG).set_index("variable")
    a_mask = df["Target"].isin(["Dropout", "Graduate"])
    ya = (df.loc[a_mask, "Target"] == "Dropout").astype(int)
    yb = (df["Target"] == "Dropout").astype(int)
    rows = []
    for col in df.columns.drop("Target"):
        if col in UCI_NOMINAL:  # code order is arbitrary: rank AUC is meaningless, use Cramér's V
            auc_a = auc_b = float("nan")
        else:
            auc_a = eda.univariate_auc(df.loc[a_mask, col], ya)
            auc_b = eda.univariate_auc(df[col], yb)
        rows.append({"variable": col, "auc_A": round(auc_a, 4), "separation": round(eda.separation(auc_a), 4),
                     "separation_B": round(eda.separation(auc_b), 4),
                     "cramers_v_3class": round(eda.cramers_v(df[col], df["Target"]), 4) if df[col].nunique() <= 50 else None,
                     "class": cat.loc[col, "class_sem1_weeks_2_6"], "class_sem2": cat.loc[col, "class_sem2_weeks_2_6"]})
    sep = pd.DataFrame(rows).sort_values(["separation", "cramers_v_3class"], ascending=False)
    sep.to_csv(INSPECTION / "uci_univariate_separation.csv", index=False)
    r["univariate_separation"] = sep
    r["max_separation_by_scenario"] = {
        "sem1_weeks_2_6_permitted_only": float(sep.loc[sep["class"] == "PERMITTED", "separation"].max()),
        "sem1_weeks_2_6_permitted_or_conditional": float(sep.loc[sep["class"] != "PROHIBITED", "separation"].max()),
        "sem2_weeks_2_6_permitted_only": float(sep.loc[sep["class_sem2"] == "PERMITTED", "separation"].max()),
        "prohibited_any": float(sep.loc[sep["class_sem2"] == "PROHIBITED", "separation"].max()),
    }
    r["outcome_revealing_rules"] = leakage.uci_outcome_revealing_rules(df)
    macro = df["Unemployment rate"].astype(str) + "|" + df["Inflation rate"].astype(str) + "|" + df["GDP"].astype(str)
    r["cohort_proxy_class_share"] = _round(pd.crosstab(macro, df["Target"], normalize="index"))
    r["course_class_share"] = _round(pd.crosstab(df["Course"], df["Target"], normalize="index"))

    # Figures
    plots.outcome_bars(df["Target"].value_counts(), plots.UCI_COLORS,
                       "UCI 697 — Distribuição de Target (4.424 estudantes)", FIGURES / "uci_target_distribution.png")
    plots.uci_semester_performance(df, FIGURES / "uci_semester_approval_by_class.png")
    plots.score_distribution(df, "Admission grade", "Target", plots.UCI_COLORS,
                             "UCI 697 — Nota de admissão por classe (disponível no ingresso)",
                             "Admission grade (0–200)", FIGURES / "uci_admission_grade_by_class.png")
    plots.uci_availability_separation(sep, FIGURES / "uci_separation_by_availability.png")
    plots.uci_cohort_proxy(df, FIGURES / "uci_cohort_proxy.png")
    numeric = df.drop(columns=["Target"])
    corr = numeric.corr(method="spearman")
    plots.correlation_heatmap(corr, "UCI 697 — Correlação de Spearman entre variáveis",
                              FIGURES / "uci_spearman_correlation.png")
    strong = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack())
    strong = strong.loc[strong.abs() >= 0.7].round(3).sort_values(ascending=False)
    r["strong_correlations_abs_ge_0_7"] = {f"{a} ~ {b}": v for (a, b), v in strong.items()}
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 02 exploratory analysis")
    parser.add_argument("--only", choices=["oulad", "uci"], default=None)
    args = parser.parse_args(argv)
    INSPECTION.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "uci"):
        _write_json(analyse_uci(), INSPECTION / "phase02_uci.json")
        print("UCI done")
    if args.only in (None, "oulad"):
        _write_json(analyse_oulad(), INSPECTION / "phase02_oulad.json")
        print("OULAD done")
    _write_json({"oulad": leakage.OULAD_CATALOG, "uci": leakage.UCI_CATALOG}, INSPECTION / "leakage_catalog.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
