"""Phase 05 analyses (no API, no Brazilian data):

1. Split integrity on the real datasets (records vs students, disjointness).
2. Q17: retrain every OULAD experiment with predictor set phase05-v2 (no adm_imd_band)
   into reports/modeling_v2/ — Phase 04 results in reports/modeling/ are left untouched.
3. v1 x v2 comparison (metrics, calibration, paired bootstrap, subgroup results).
4. Q18: disability deep dive (calibration gap with CI, module standardisation, power, representativeness).
5. Q19: threshold-methodology illustration on the CALIBRATION subset (never the test).

    python -m src.modeling.run_phase05              # ~15 min (retraining)
    python -m src.modeling.run_phase05 --skip-train # reuse reports/modeling_v2
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.data.loaders import PROJECT_ROOT
from src.modeling import evaluation, fairness, figures, splitting, thresholds
from src.modeling.run_phase04 import summarise
from src.modeling.train import _default, load_verified_oulad, proba, run_oulad_experiment

V1 = PROJECT_ROOT / "reports" / "modeling"
V2 = PROJECT_ROOT / "reports" / "modeling_v2"
OUT = PROJECT_ROOT / "reports" / "phase05"
TARGETS = ("target_withdrawn", "target_adverse_outcome")
MODELS = ("logistic_regression", "gradient_boosting")


def _dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def split_integrity() -> dict:
    out = {}
    for week in (2, 4, 6):
        df, meta, _ = load_verified_oulad(week)
        sp = splitting.oulad_split(df, meta["target_columns"])
        parts = {"train": sp.train, "validation": sp.validation, "calibration": sp.calibration, "test_main": sp.test}
        ids = {k: set(v["student_id"]) for k, v in parts.items()}
        pool = df[df["presentation_year"] == 2013]
        per = pool.groupby("student_id").size()
        lab = pd.concat([v[["student_id"]].assign(part=k) for k, v in parts.items() if k != "test_main"])
        out[f"week_{week:02d}"] = {
            "rows_2013": int(len(pool)), "students_2013": int(pool["student_id"].nunique()),
            "students_with_more_than_one_record_2013": int((per > 1).sum()),
            "sum_rows_train_val_cal": int(sum(len(parts[k]) for k in ("train", "validation", "calibration"))),
            "sum_students_train_val_cal": int(sum(len(ids[k]) for k in ("train", "validation", "calibration"))),
            "students_in_more_than_one_part": int((lab.groupby("student_id")["part"].nunique() > 1).sum()),
            "shared_students": {f"{a}&{b}": len(ids[a] & ids[b]) for i, a in enumerate(ids) for b in list(ids)[i + 1:]},
            "rows_by_part": {k: int(len(v)) for k, v in parts.items()},
            "students_by_part": {k: len(v) for k, v in ids.items()},
        }
    return out


def retrain_v2() -> dict:
    res = {}
    for target in TARGETS:
        for week in (2, 4, 6):
            full = target == "target_withdrawn" and week == 4
            print(f"[v2] week {week} {target}")
            res[(week, target)] = run_oulad_experiment(
                week, target, extras=full, interpret=target == "target_withdrawn",
                out_dir=V2 / "oulad" / f"week_{week:02d}" / target, model_version="phase05-v2")
    return res


def load_v2() -> dict:
    res = {}
    for target in TARGETS:
        for week in (2, 4, 6):
            d = V2 / "oulad" / f"week_{week:02d}" / target
            r = json.loads((d / "results.json").read_text(encoding="utf-8"))
            r["_predictions"] = pd.read_parquet(d / "test_predictions.parquet")
            res[(week, target)] = r
    return res


def compare_versions(v2: dict) -> dict:
    rows, boots = [], {}
    for (week, target), r2 in v2.items():
        d1 = V1 / "oulad" / f"week_{week:02d}" / target
        r1 = json.loads((d1 / "results.json").read_text(encoding="utf-8"))
        p1 = pd.read_parquet(d1 / "test_predictions.parquet")
        p2 = r2["_predictions"]
        assert p1["student_id"].tolist() == p2["student_id"].tolist(), "test sets must be identical"
        for m in MODELS:
            for ver, r in (("v1", r1), ("v2", r2)):
                t = r["test_main"][m]
                rows.append({"week": week, "target": target, "model": m, "version": ver,
                             "n_predictors": r["experiment"]["n_predictors"], "pr_auc": t["pr_auc"],
                             "roc_auc": t["roc_auc"], "brier": t["brier"], "ece": t["ece"],
                             "recall": t["at_threshold"]["recall"], "precision": t["at_threshold"]["precision"],
                             "threshold": t["at_threshold"]["threshold"],
                             "recall_top10": t["top_10pct"]["recall"], "precision_top10": t["top_10pct"]["precision"]})
            boots[f"week_{week:02d}/{target}/{m}"] = evaluation.paired_bootstrap(
                p1["y"], p1[f"p_{m}"], p2[f"p_{m}"], p1["student_id"])
    return {"table": pd.DataFrame(rows), "bootstrap_v2_minus_v1": boots}


def subgroup_versions(v2: dict) -> dict:
    r1 = json.loads((V1 / "oulad" / "week_04" / "target_withdrawn" / "results.json").read_text(encoding="utf-8"))
    r2 = v2[(4, "target_withdrawn")]
    return {m: {"v1": r1["fairness"][m], "v2": r2["fairness"][m]} for m in MODELS}


def disability_deep_dive(v2: dict) -> dict:
    out = {}
    for week in (2, 4, 6):
        df, meta, _ = load_verified_oulad(week)
        sp = splitting.oulad_split(df, meta["target_columns"])
        pred = v2[(week, "target_withdrawn")]["_predictions"]
        test = sp.test.reset_index(drop=True)
        assert test["student_id"].tolist() == pred["student_id"].tolist()
        y = pred["y"].to_numpy()
        thr = v2[(week, "target_withdrawn")]["calibration_subset"]["thresholds"]
        w = {"calibration_gap": {}, "fairness": {}}
        for m in MODELS:
            p = pred[f"p_{m}"].to_numpy()
            w["calibration_gap"][m] = fairness.calibration_gap(test, y, p, "disability")
            w["fairness"][m] = fairness.group_report(test, y, p, thr[m], "disability", n_boot=1000)
            if week == 4:
                w.setdefault("module_standardised", {})[m] = fairness.module_standardised_gap(test, y, p, "disability")
                w.setdefault("calibration_curve", {})[m] = fairness.calibration_curve_by_group(test, y, p, "disability")
                w.setdefault("gender_calibration_gap", {})[m] = fairness.calibration_gap(test, y, p, "gender")
                w.setdefault("imd_calibration_gap", {})[m] = fairness.calibration_gap(test, y, p, "imd_group")
        n_y = int((test["adm_disability"] == 1).sum())
        w["power"] = {"n_disability": n_y, "n_other": int(len(test) - n_y),
                      "positives_disability": int(y[test["adm_disability"] == 1].sum()),
                      "min_detectable_recall_difference": fairness.minimum_detectable_difference(
                          int(y[test["adm_disability"] == 1].sum()), int(y[test["adm_disability"] == 0].sum()), 0.5)}
        w["representativeness"] = fairness.representativeness(
            {"train": sp.train, "validation": sp.validation, "calibration": sp.calibration, "test_main": sp.test,
             "test_recurrent": sp.test_recurrent})
        w["by_module"] = (test.assign(y=y).groupby(["code_module", "adm_disability"])["y"]
                          .agg(["size", "mean"]).round(4).reset_index().to_dict("records"))
        out[f"week_{week:02d}"] = w
    return out


def threshold_methodology(v2: dict) -> dict:
    """Illustrative consequences on the 2013 CALIBRATION subset of the v2 week-4 LR."""
    df, meta, _ = load_verified_oulad(4)
    sp = splitting.oulad_split(df, meta["target_columns"])
    bundle = joblib.load(V2 / "oulad" / "week_04" / "target_withdrawn" / "models.joblib")
    cols = bundle["predictors"]
    out = {}
    for m in MODELS:
        p = bundle["calibrators"][m].transform(proba(bundle["models"][m], sp.calibration[cols]))
        y = sp.calibration["target_withdrawn"].to_numpy()
        cap = {f"{int(c * 100)}%": thresholds.consequences(y, p, thresholds.capacity_threshold(p, c))
               for c in (0.05, 0.10, 0.20, 0.30)}
        cost = {f"FN custa {r}x FP": thresholds.consequences(y, p, thresholds.cost_optimal_threshold(r, 1))
                for r in (2, 3, 5, 10)}
        nb = [{"threshold": float(t), "net_benefit": thresholds.net_benefit(y, p, t),
               "treat_all": thresholds.net_benefit(y, np.ones_like(p), t)} for t in np.arange(0.05, 0.55, 0.05)]
        out[m] = {"by_capacity": cap, "by_cost_ratio": cost, "net_benefit": nb,
                  "data": "2013 calibration subset (never the 2014 test)"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args(argv)
    _dump(split_integrity(), OUT / "split_integrity.json")
    v2 = retrain_v2() if not a.skip_train else load_v2()
    summary = summarise(v2)
    (V2 / "comparisons").mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(V2 / "comparisons" / "oulad_test_summary_v2.csv", index=False)
    comp = compare_versions(v2)
    comp["table"].round(4).to_csv(OUT / "v1_vs_v2_metrics.csv", index=False)
    _dump({"bootstrap_v2_minus_v1": comp["bootstrap_v2_minus_v1"], "subgroups_week04": subgroup_versions(v2)},
          OUT / "v1_vs_v2_comparison.json")
    dd = disability_deep_dive(v2)
    _dump(dd, OUT / "disability_audit.json")
    _dump(threshold_methodology(v2), OUT / "threshold_methodology.json")
    figs = OUT / "figures"
    figures.model_comparison(summary[summary["target"] == "target_withdrawn"], figs / "v2_pr_auc_by_week.png",
                             "OULAD v2 (sem IMD) — PR-AUC por corte; pontilhado = prevalência")
    r4 = v2[(4, "target_withdrawn")]
    for m in MODELS:
        figures.fairness_plot(r4["fairness"][m], f"OULAD semana 4 — v2 sem IMD — {figures.LABELS[m]}",
                              figs / f"v2_week04_fairness_{m}.png")
    figures.group_calibration(dd["week_04"]["calibration_curve"]["logistic_regression"],
                              "Semana 4 — calibração por deficiência (RL v2, quintis)", figs / "v2_disability_calibration.png")
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
