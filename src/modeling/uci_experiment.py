"""UCI 697 admission-time experiment — independent from OULAD (never combined).

Validation design (no reliable time information exists in UCI):
- Stratified hold-out test = 20% of the supervised rows (fixed seed), untouched until the end.
- Remaining 80%: stratified 5-fold CV for model/hyperparameter selection; the out-of-fold
  predictions of the selected configuration are used to select the calibrator and threshold.
  The final model is refitted on the whole 80%.
- Robustness (NOT temporal): leave-one-cohort-out over `cohort_proxy` inside the 80%. The order
  of cohorts is unknown, so this is a group-shift check, not a chronological validation.
Predictors: the 18 approved admission features MINUS Gender and Educational special needs
(kept for the fairness audit, consistent with decision Q16).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from src.feature_engineering.build_datasets import content_hash, load_dataset
from src.modeling import baselines, calibration, evaluation, preprocessing
from src.modeling.train import (HGB_GRID, LR_GRID, PROCESSED, REPORTS, SEED, _default, git_commit,
                                library_versions, make_hgb, make_lr, proba)

TARGET = "target_dropout"


def load_verified_uci():
    base = PROCESSED / "uci" / "admission"
    df, meta = load_dataset(base / "uci_admission.parquet", base / "uci_admission.metadata.json")
    manifest = json.loads((base / "uci_admission.manifest.json").read_text(encoding="utf-8"))
    if content_hash(df) != manifest["output"]["content_sha256"]:
        raise RuntimeError("uci_admission: content hash differs from manifest")
    return df, meta, {"dataset": "uci_admission", "content_sha256": content_hash(df)}


def _oof(builder, params, X, y, folds):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = proba(builder(**params).fit(X.iloc[tr], y[tr]), X.iloc[te])
    return oof


def run_uci_experiment(df: pd.DataFrame | None = None, meta: dict | None = None, out_dir=None,
                       n_boot: int = 1000) -> dict:
    provenance = None
    if df is None:
        df, meta, provenance = load_verified_uci()
    out_dir = out_dir or REPORTS / "uci" / "admission"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = preprocessing.predictor_columns(meta)
    y_all = df[TARGET].to_numpy()
    dev, test = train_test_split(df, test_size=0.2, stratify=y_all, random_state=SEED)
    dev, test = dev.sort_values("row_id"), test.sort_values("row_id")
    Xd, yd, Xt, yt = dev[cols], dev[TARGET].to_numpy(), test[cols], test[TARGET].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd))

    builders = {"logistic_regression": (lambda **p: make_lr(meta, cols, **p), LR_GRID),
                "gradient_boosting": (lambda **p: make_hgb(meta, cols, **p), HGB_GRID)}
    selection, oof_best, final = {}, {}, {}
    for name, (builder, grid) in builders.items():
        scores = [{"params": p, "cv_pr_auc": float(average_precision_score(yd, _oof(builder, p, Xd, yd, folds)))}
                  for p in grid]
        best = max(scores, key=lambda s: s["cv_pr_auc"])
        oof_best[name] = _oof(builder, best["params"], Xd, yd, folds)
        final[name] = builder(**best["params"]).fit(Xd, yd)
        selection[name] = {"selected": best, "grid": scores}

    # baselines (fitted on dev; course rate = analogue of the module-rate baseline)
    base_models = {"baseline_prevalence": baselines.PrevalenceBaseline(),
                   "baseline_course_rate": baselines.ModuleRateBaseline(column="Course")}
    for name, m in base_models.items():
        oof_best[name] = np.zeros(len(yd))
        for tr, te in folds:
            oof_best[name][te] = proba(type(m)(**({"column": "Course"} if "course" in name else {})).fit(
                Xd.iloc[tr], yd[tr]), Xd.iloc[te])
        final[name] = m.fit(Xd, yd)

    calibrators, thresholds, preds, test_metrics = {}, {}, {}, {}
    for name in final:
        if name in builders:
            cal, info = calibration.select_calibrator(oof_best[name], yd, np.arange(len(yd)))
            selection[name]["calibration"] = info
        else:
            cal = calibration.Calibrator("none")
        calibrators[name] = cal
        thresholds[name] = evaluation.best_f1_threshold(yd, cal.transform(oof_best[name]))
        preds[name] = cal.transform(proba(final[name], Xt))
        test_metrics[name] = evaluation.full_report(yt, preds[name], thresholds[name])

    cv_metrics = {n: evaluation.probability_metrics(yd, p) for n, p in oof_best.items()}
    groups = test["row_id"].to_numpy()
    comparisons = {
        "gradient_boosting_minus_logistic_regression": evaluation.paired_bootstrap(
            yt, preds["logistic_regression"], preds["gradient_boosting"], groups, n_boot=n_boot),
        "logistic_regression_minus_baseline_course_rate": evaluation.paired_bootstrap(
            yt, preds["baseline_course_rate"], preds["logistic_regression"], groups, n_boot=n_boot),
    }

    # leave-one-cohort-out (group shift, not time)
    loco = []
    best_lr = selection["logistic_regression"]["selected"]["params"]
    for tr, te in GroupKFold(n_splits=dev["cohort_proxy"].nunique()).split(Xd, yd, dev["cohort_proxy"]):
        m = make_lr(meta, cols, **best_lr).fit(Xd.iloc[tr], yd[tr])
        p = proba(m, Xd.iloc[te])
        loco.append({"cohort": str(dev["cohort_proxy"].iloc[te[0]]), "n": int(len(te)),
                     "prevalence": float(yd[te].mean()), "pr_auc": float(average_precision_score(yd[te], p))})

    from src.modeling import fairness
    audit_df = test.assign(adm_gender=test["Gender"].map({1: "M", 0: "F"}), student_id=test["row_id"])
    fair = {}
    for n in ("logistic_regression", "gradient_boosting"):
        fair[n] = {
            "gender": fairness.group_report(audit_df, yt, preds[n], thresholds[n], "gender"),
            "educational_special_needs": fairness.group_report(
                audit_df.assign(esn=test["Educational special needs"].map({1: "Y", 0: "N"})), yt, preds[n],
                thresholds[n], "esn"),
            "age_at_enrollment": fairness.group_report(
                audit_df.assign(age=pd.cut(test["Age at enrollment"], [0, 20, 25, 100],
                                           labels=["≤20", "21-25", ">25"]).astype(str)), yt, preds[n], thresholds[n], "age"),
        }

    result = {
        "experiment": {"dataset": "UCI 697", "task": "dropout risk at admission (Dropout vs Graduate)",
                       "target": TARGET, "moment": "admission", "seed": SEED, "predictors": cols,
                       "n_predictors": len(cols),
                       "excluded_sensitive": sorted(preprocessing.EXCLUDED_SENSITIVE & set(meta["feature_columns"]))},
        "provenance": {**(provenance or {}), "code": git_commit(), "libraries": library_versions()},
        "split": {"dev_rows": int(len(dev)), "test_rows": int(len(test)), "dev_rate": float(yd.mean()),
                  "test_rate": float(yt.mean()), "cv": "StratifiedKFold(5, shuffle, seed=42)",
                  "note": "No reliable time information: no chronological validation is simulated."},
        "selection": selection,
        "cv_out_of_fold_metrics": cv_metrics,
        "thresholds": thresholds,
        "calibrators": {n: c.method for n, c in calibrators.items()},
        "test": test_metrics,
        "test_comparisons_bootstrap": comparisons,
        "leave_one_cohort_out_lr": loco,
        "fairness": fair,
    }
    joblib.dump({"models": final, "calibrators": calibrators, "thresholds": thresholds, "predictors": cols},
                out_dir / "models.joblib")
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_default),
                                          encoding="utf-8")
    return result
