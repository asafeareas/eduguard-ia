"""Evaluation metrics. Accuracy is deliberately not a headline metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

EPS = 1e-6


def top_fraction(y: np.ndarray, p: np.ndarray, fraction: float) -> dict:
    """Recall/precision among the `fraction` highest-risk records (ties broken by order, stable)."""
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(fraction * len(y)))
    idx = np.argsort(-p, kind="mergesort")[:k]
    caught = int(y[idx].sum())
    total = int(y.sum())
    return {"fraction": fraction, "group_size": k, "positives_in_group": caught, "positives_total": total,
            "recall": caught / total if total else float("nan"), "precision": caught / k if k else float("nan")}


def threshold_metrics(y, p, threshold: float) -> dict:
    y, pred = np.asarray(y), (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "flagged_share": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins of predicted probability vs observed frequency."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, len(df)), labels=False)
    t = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    t["gap"] = t["mean_predicted"] - t["observed_rate"]
    return t.reset_index()


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def probability_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "mean_predicted": float(p.mean()),
        "calibration_in_the_large": float(p.mean() - y.mean()),
    }


def full_report(y, p, threshold: float) -> dict:
    return {
        **probability_metrics(y, p),
        "at_threshold": threshold_metrics(y, p, threshold),
        "top_10pct": top_fraction(y, p, 0.10),
        "top_20pct": top_fraction(y, p, 0.20),
    }


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1 (used ONLY on calibration/validation data)."""
    y, p = np.asarray(y), np.asarray(p)
    cands = np.unique(np.round(p, 4))
    if len(cands) > 400:
        cands = np.quantile(p, np.linspace(0.01, 0.99, 400))
    f1 = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


def paired_bootstrap(y, p_a, p_b, groups, metric=average_precision_score, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """Paired bootstrap (resampling STUDENTS) of metric(b) - metric(a) on the same records."""
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).to_list()
    diffs, a_vals, b_vals = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a, b = metric(y[idx], p_a[idx]), metric(y[idx], p_b[idx])
        a_vals.append(a), b_vals.append(b), diffs.append(b - a)
    diffs = np.array(diffs)
    return {
        "observed_a": float(metric(y, p_a)), "observed_b": float(metric(y, p_b)),
        "diff_mean": float(diffs.mean()), "diff_ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "share_b_better": float((diffs > 0).mean()),
        "a_ci95": [float(np.quantile(a_vals, 0.025)), float(np.quantile(a_vals, 0.975))],
        "b_ci95": [float(np.quantile(b_vals, 0.025)), float(np.quantile(b_vals, 0.975))],
        "n_boot": int(len(diffs)),
    }


def bootstrap_ci(y, p, groups, metric=average_precision_score, n_boot: int = 1000, seed: int = 42) -> list[float]:
    """95% CI of a single model's metric, resampling students."""
    r = paired_bootstrap(y, p, p, groups, metric=metric, n_boot=n_boot, seed=seed)
    return r["a_ci95"]
