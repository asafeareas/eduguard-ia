"""Run every Phase 04 experiment and build the comparison reports.

    python -m src.modeling.run_phase04                 # everything (~10-15 min)
    python -m src.modeling.run_phase04 --only uci

Each OULAD cutoff x target is an independent experiment (own split, fit, selection,
calibration and test). Week 4 + primary target receives the full analysis.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.modeling import evaluation, figures
from src.modeling.train import REPORTS, run_oulad_experiment
from src.modeling.uci_experiment import run_uci_experiment

PRIMARY, SECONDARY = "target_withdrawn", "target_adverse_outcome"
FIG = REPORTS / "figures"
MODEL_ORDER = ["baseline_prevalence", "baseline_module_rate", "baseline_inactivity_rule",
               "logistic_regression", "gradient_boosting"]


def summarise(results: dict) -> pd.DataFrame:
    rows = []
    for (week, target), r in results.items():
        pred = r["_predictions"]
        for m in MODEL_ORDER:
            t = r["test_main"][m]
            ci = evaluation.bootstrap_ci(pred["y"], pred[f"p_{m}"], pred["student_id"])
            rows.append({
                "week": week, "target": target, "model": m, "prevalence": t["prevalence"],
                "pr_auc": t["pr_auc"], "ci_low": ci[0], "ci_high": ci[1], "roc_auc": t["roc_auc"],
                "brier": t["brier"], "log_loss": t["log_loss"], "ece": t["ece"],
                "threshold": t["at_threshold"]["threshold"], "precision": t["at_threshold"]["precision"],
                "recall": t["at_threshold"]["recall"], "f1": t["at_threshold"]["f1"],
                "recall_top10": t["top_10pct"]["recall"], "precision_top10": t["top_10pct"]["precision"],
                "recall_top20": t["top_20pct"]["recall"], "precision_top20": t["top_20pct"]["precision"],
                "val_pr_auc": r["validation_metrics"][m]["pr_auc"],
            })
    return pd.DataFrame(rows)


def run_oulad() -> pd.DataFrame:
    results = {}
    for target in (PRIMARY, SECONDARY):
        for week in (2, 4, 6):
            full = target == PRIMARY and week == 4
            print(f"[run] OULAD week {week} {target}{' (full analysis)' if full else ''}")
            results[(week, target)] = run_oulad_experiment(week, target, extras=full,
                                                           interpret=target == PRIMARY)
    summary = summarise(results)
    comp = REPORTS / "comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(comp / "oulad_test_summary.csv", index=False)

    # figures
    for target, label in ((PRIMARY, "cancelamento de módulo"), (SECONDARY, "resultado adverso")):
        figures.model_comparison(summary[summary["target"] == target],
                                 FIG / f"oulad_{target}_pr_auc_by_week.png",
                                 f"OULAD — PR-AUC por corte ({label}); pontilhado = prevalência")
    r4 = results[(4, PRIMARY)]
    figures.pr_curves(r4["_predictions"], "OULAD semana 4 — curvas precision-recall (teste 2014)",
                      FIG / "oulad_week04_pr_curves.png")
    figures.calibration_plot(r4["calibration_tables_test"], r4["_predictions"],
                             "OULAD semana 4 — calibração no teste 2014", FIG / "oulad_week04_calibration.png")
    for m in ("logistic_regression", "gradient_boosting"):
        figures.fairness_plot(r4["fairness"][m], f"OULAD semana 4 — {figures.LABELS[m]}",
                              FIG / f"oulad_week04_fairness_{m}.png")
    gi = [{"week": f"S{w}", "group": g, "drop": v["pr_auc_drop_mean"]}
          for w in (2, 4, 6)
          for g, v in results[(w, PRIMARY)]["interpretability"]["gradient_boosting"]["per_group"].items()]
    figures.group_importance(pd.DataFrame(gi), FIG / "oulad_group_importance_by_week.png")
    figures.top_features(r4["interpretability"]["logistic_regression"]["coefficients"], "coef",
                         "OULAD semana 4 — coeficientes da regressão logística (padronizados)",
                         FIG / "oulad_week04_lr_coefficients.png")
    figures.top_features(r4["interpretability"]["gradient_boosting"]["per_feature"], "importance_mean",
                         "OULAD semana 4 — permutation importance do gradient boosting (validação)",
                         FIG / "oulad_week04_hgb_permutation.png")

    # compact cross-experiment json (versioned)
    compact = {f"week_{w:02d}/{t}": {
        "split_audit": r["split_audit"], "selected": {m: r["selection"][m].get("selected") for m in
                                                      ("logistic_regression", "gradient_boosting")},
        "inactivity_n": r["selection"]["baseline_inactivity_rule"]["selected_n"],
        "calibrators": r["calibration_subset"]["calibrators"],
        "bootstrap": r["test_comparisons_bootstrap"]} for (w, t), r in results.items()}
    (comp / "oulad_experiments_overview.json").write_text(json.dumps(compact, indent=2, default=str), encoding="utf-8")
    return summary


def run_uci() -> dict:
    print("[run] UCI admission")
    r = run_uci_experiment()
    rows = [{"model": m, **{k: r["test"][m][k] for k in ("pr_auc", "roc_auc", "brier", "ece", "prevalence")},
             "recall": r["test"][m]["at_threshold"]["recall"], "precision": r["test"][m]["at_threshold"]["precision"],
             "cv_pr_auc": r["cv_out_of_fold_metrics"][m]["pr_auc"]} for m in r["test"]]
    comp = REPORTS / "comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(comp / "uci_test_summary.csv", index=False)
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["oulad", "uci"])
    a = p.parse_args(argv)
    if a.only in (None, "oulad"):
        print(run_oulad().round(3).to_string())
    if a.only in (None, "uci"):
        run_uci()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
