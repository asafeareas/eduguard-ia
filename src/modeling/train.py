"""OULAD experiment runner (one independent experiment per cutoff x target).

Protocol (docs/MODELING_METHODOLOGY.md):
    1. Load the Phase 03 dataset and verify its content hash against the manifest.
    2. Split: 2013 -> train/validation/calibration by student; 2014 minus recurrent students -> test.
    3. Baselines + logistic regression grid + gradient-boosting grid, all fitted on TRAIN.
    4. Model/hyperparameter/N selection on VALIDATION (PR-AUC; natural class distribution).
    5. Calibrator + alert threshold + risk bands on CALIBRATION.
    6. One final evaluation on TEST (never used before). Paired student bootstrap for comparisons.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.data.loaders import PROJECT_ROOT
from src.feature_engineering.build_datasets import content_hash, load_dataset
from src.modeling import baselines, calibration, evaluation, preprocessing, splitting

SEED = splitting.SEED
REPORTS = PROJECT_ROOT / "reports" / "modeling"
PROCESSED = PROJECT_ROOT / "data" / "processed"

LR_GRID = [{"penalty": pen, "C": c} for pen in ("l1", "l2") for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
HGB_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl, "max_iter": it}
            for lr, leaves, msl, it in itertools.product((0.05, 0.1), (15, 31), (50, 200), (150, 300))]


# ----------------------------------------------------------------------------- builders
def make_lr(meta, cols, penalty="l2", C=1.0, class_weight=None):
    return Pipeline([
        ("prep", preprocessing.build_preprocessor(meta, cols, scale=True)),
        ("clf", LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=2000,
                                   class_weight=class_weight, random_state=SEED)),
    ])


def make_hgb(meta, cols, **params):
    return Pipeline([
        ("prep", preprocessing.build_preprocessor(meta, cols, scale=False)),
        ("clf", HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False,
                                               random_state=SEED, **params)),
    ])


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- provenance
def library_versions() -> dict:
    import pyarrow
    import scipy
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "pyarrow": pyarrow.__version__}


def git_commit() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "src_uncommitted_changes": bool(run("status", "--porcelain", "--", "src"))}


def load_verified_oulad(week: int) -> tuple[pd.DataFrame, dict, dict]:
    name = f"oulad_week_{week:02d}"
    df, meta = load_dataset(PROCESSED / "oulad" / f"week_{week:02d}" / f"{name}.parquet",
                            PROCESSED / "oulad" / "metadata" / f"{name}.metadata.json")
    manifest = json.loads((PROCESSED / "oulad" / "metadata" / f"{name}.manifest.json").read_text(encoding="utf-8"))
    observed = content_hash(df)
    if observed != manifest["output"]["content_sha256"]:
        raise RuntimeError(f"{name}: content hash differs from manifest — rebuild the dataset first")
    return df, meta, {"dataset": name, "content_sha256": observed, "manifest_commit": manifest["code"]["git_commit"]}


# ----------------------------------------------------------------------------- selection
def select_from_grid(builder, grid, X_tr, y_tr, X_val, y_val):
    scores = []
    for params in grid:
        m = builder(**params).fit(X_tr, y_tr)
        scores.append({"params": params, "val_pr_auc": float(average_precision_score(y_val, proba(m, X_val)))})
    best = max(scores, key=lambda s: s["val_pr_auc"])
    return builder(**best["params"]).fit(X_tr, y_tr), best, scores


def run_oulad_experiment(week: int, target: str, extras: bool = False, interpret: bool = False,
                         out_dir: Path | None = None, df: pd.DataFrame | None = None, meta: dict | None = None,
                         provenance: dict | None = None, n_boot: int = 1000) -> dict:
    """Run one experiment. `df/meta` can be injected (tests); otherwise loaded and hash-verified."""
    if df is None:
        df, meta, provenance = load_verified_oulad(week)
    out_dir = out_dir or REPORTS / "oulad" / f"week_{week:02d}" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff_day = 7 * week - 1

    sp = splitting.oulad_split(df, meta["target_columns"])
    cols = preprocessing.predictor_columns(meta)
    X = {k: getattr(sp, k)[cols] for k in ("train", "validation", "calibration", "test", "test_recurrent")}
    y = {k: getattr(sp, k)[target].to_numpy() for k in X}
    groups_test = sp.test["student_id"].to_numpy()

    # ---- fit + select on validation ----------------------------------------------------
    models, selection = {}, {}
    models["baseline_prevalence"] = baselines.PrevalenceBaseline().fit(X["train"], y["train"])
    models["baseline_module_rate"] = baselines.ModuleRateBaseline().fit(X["train"], y["train"])
    rule, n_scores = baselines.InactivityRuleBaseline.select_n(
        X["train"], y["train"], X["validation"], y["validation"], range(1, cutoff_day + 3))
    models["baseline_inactivity_rule"] = rule
    selection["baseline_inactivity_rule"] = {"validation_f1_by_n": n_scores, "selected_n": rule.n_days}

    lr, lr_best, lr_scores = select_from_grid(lambda **p: make_lr(meta, cols, **p), LR_GRID,
                                              X["train"], y["train"], X["validation"], y["validation"])
    hgb, hgb_best, hgb_scores = select_from_grid(lambda **p: make_hgb(meta, cols, **p), HGB_GRID,
                                                 X["train"], y["train"], X["validation"], y["validation"])
    models["logistic_regression"], models["gradient_boosting"] = lr, hgb
    selection["logistic_regression"] = {"selected": lr_best, "grid": lr_scores}
    selection["gradient_boosting"] = {"selected": hgb_best, "grid": hgb_scores}

    validation = {name: evaluation.probability_metrics(y["validation"], proba(m, X["validation"]))
                  for name, m in models.items()}

    # ---- calibration + thresholds on the calibration subset -----------------------------
    calibrators, thresholds, raw_cal = {}, {}, {}
    cal_groups = sp.calibration["student_id"].to_numpy()
    for name, m in models.items():
        p_cal = proba(m, X["calibration"])
        raw_cal[name] = p_cal
        if name in ("logistic_regression", "gradient_boosting"):
            cal, info = calibration.select_calibrator(p_cal, y["calibration"], cal_groups)
            selection[name]["calibration"] = info
        else:
            cal = calibration.Calibrator("none")
        calibrators[name] = cal
        thresholds[name] = evaluation.best_f1_threshold(y["calibration"], cal.transform(p_cal))

    def predict(name, part):
        return calibrators[name].transform(proba(models[name], X[part]))

    # ---- final test evaluation (single pass) -------------------------------------------
    test, test_raw, preds = {}, {}, {}
    for name in models:
        preds[name] = predict(name, "test")
        test[name] = evaluation.full_report(y["test"], preds[name], thresholds[name])
        if name in ("logistic_regression", "gradient_boosting"):
            test_raw[name] = evaluation.probability_metrics(y["test"], proba(models[name], X["test"]))

    best_baseline = max((n for n in models if n.startswith("baseline_")), key=lambda n: validation[n]["pr_auc"])
    comparisons = {
        "gradient_boosting_minus_logistic_regression": evaluation.paired_bootstrap(
            y["test"], preds["logistic_regression"], preds["gradient_boosting"], groups_test, n_boot=n_boot),
        f"logistic_regression_minus_{best_baseline}": evaluation.paired_bootstrap(
            y["test"], preds[best_baseline], preds["logistic_regression"], groups_test, n_boot=n_boot),
    }

    result = {
        "experiment": {"dataset": "OULAD", "task": "module withdrawal prediction" if target == "target_withdrawn"
                       else "adverse academic outcome", "target": target, "week": week, "cutoff_day": cutoff_day,
                       "seed": SEED, "predictors": cols, "n_predictors": len(cols),
                       "excluded_sensitive": sorted(preprocessing.EXCLUDED_SENSITIVE & set(meta["feature_columns"]))},
        "provenance": {**(provenance or {}), "code": git_commit(), "libraries": library_versions()},
        "split_audit": sp.audit,
        "selection": selection,
        "validation_metrics": validation,
        "calibration_subset": {"thresholds": thresholds,
                               "calibrators": {n: c.method for n, c in calibrators.items()}},
        "test_main": test,
        "test_main_uncalibrated": test_raw,
        "test_comparisons_bootstrap": comparisons,
        "best_baseline_on_validation": best_baseline,
    }

    if extras or interpret:
        from src.modeling import fairness, interpretability
    if extras:
        # risk bands from calibration data only
        p_cal_lr = calibrators["logistic_regression"].transform(raw_cal["logistic_regression"])
        p_cal_hgb = calibrators["gradient_boosting"].transform(raw_cal["gradient_boosting"])
        prev = float(y["train"].mean())
        bands = {n: calibration.propose_risk_bands(pc, y["calibration"], prev, thresholds[n])
                 for n, pc in (("logistic_regression", p_cal_lr), ("gradient_boosting", p_cal_hgb))}
        for n, b in bands.items():
            assigned = calibration.assign_bands(preds[n], b)
            b["test_main"] = {k: {"share": float((assigned == k).mean()), "n": int((assigned == k).sum()),
                                  "observed_rate": float(y["test"][assigned == k].mean()) if (assigned == k).any() else None}
                              for k in ("low", "moderate", "high")}
        result["risk_bands"] = bands

        # recurrent students: separate, secondary scenario
        result["test_recurrent_secondary"] = {
            n: evaluation.full_report(y["test_recurrent"], predict(n, "test_recurrent"), thresholds[n])
            for n in ("logistic_regression", "gradient_boosting", best_baseline)}

        # class_weight='balanced' variant of the selected LR (comparison only, not selection)
        lr_bal = make_lr(meta, cols, class_weight="balanced", **lr_best["params"]).fit(X["train"], y["train"])
        bal_cal, bal_info = calibration.select_calibrator(proba(lr_bal, X["calibration"]), y["calibration"], cal_groups)
        thr_bal = evaluation.best_f1_threshold(y["calibration"], bal_cal.transform(proba(lr_bal, X["calibration"])))
        result["balanced_class_weight_variant"] = {
            "validation": evaluation.probability_metrics(y["validation"], proba(lr_bal, X["validation"])),
            "test_uncalibrated": evaluation.full_report(y["test"], proba(lr_bal, X["test"]), 0.5),
            "test_calibrated": evaluation.full_report(y["test"], bal_cal.transform(proba(lr_bal, X["test"])), thr_bal),
            "calibrator": bal_info,
        }

        # with vs without the socio-economic deprivation band
        cols_wo = [c for c in cols if c != "adm_imd_band"]
        wo = {}
        for n, builder, params in (("logistic_regression", make_lr, lr_best["params"]),
                                   ("gradient_boosting", make_hgb, hgb_best["params"])):
            m = builder(meta, cols_wo, **params).fit(sp.train[cols_wo], y["train"])
            c, _ = calibration.select_calibrator(proba(m, sp.calibration[cols_wo]), y["calibration"], cal_groups)
            p_wo = c.transform(proba(m, sp.test[cols_wo]))
            wo[n] = {"validation": evaluation.probability_metrics(y["validation"], proba(m, sp.validation[cols_wo])),
                     "test": evaluation.probability_metrics(y["test"], p_wo),
                     "bootstrap_with_minus_without": evaluation.paired_bootstrap(
                         y["test"], p_wo, preds[n], groups_test, n_boot=n_boot),
                     "fairness_imd": fairness.group_report(sp.test, y["test"], p_wo, thresholds[n], "imd_group")}
        result["without_imd_band"] = wo

        # fairness audit (sensitive attributes used ONLY to segment results)
        result["fairness"] = {n: fairness.full_audit(sp.test, y["test"], preds[n], thresholds[n])
                              for n in ("logistic_regression", "gradient_boosting")}
        result["calibration_tables_test"] = {n: evaluation.calibration_table(y["test"], preds[n]).to_dict("records")
                                             for n in ("logistic_regression", "gradient_boosting")}

    if interpret:
        result["interpretability"] = {
            "logistic_regression": interpretability.lr_report(lr, sp.train, y["train"], cols, meta, lr_best["params"]),
            "gradient_boosting": interpretability.permutation_report(hgb, X["validation"], y["validation"], cols, meta),
            "logistic_regression_permutation": interpretability.permutation_report(lr, X["validation"], y["validation"],
                                                                                    cols, meta),
        }

    # artefacts (models/predictions are not versioned)
    joblib.dump({"models": models, "calibrators": calibrators, "thresholds": thresholds, "predictors": cols},
                out_dir / "models.joblib")
    pred_df = sp.test[["student_id", "presentation_id"]].assign(y=y["test"], **{f"p_{n}": p for n, p in preds.items()})
    pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_default),
                                          encoding="utf-8")
    result["_predictions"] = pred_df
    return result


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)
