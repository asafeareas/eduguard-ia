"""Fairness audit: sensitive attributes are used ONLY to segment results on the test set.

They are never predictors and never influence selection, thresholds or calibration.
Groups with fewer than MIN_N records are reported but flagged as insufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

MIN_N = 100
MIN_POSITIVES = 20

IMD_LABELS = {0: "0-20% (mais privação)", 1: "20-40%", 2: "40-60%", 3: "60-80%", 4: "80-100% (menos privação)"}
AGE_LABELS = {0: "0-35", 1: "35-55", 2: "55+"}


def group_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "gender":
        return df["adm_gender"].astype(str)
    if name == "disability":
        return df["adm_disability"].map({1: "Y", 0: "N"})
    if name == "imd_group":
        return (df["adm_imd_band"] // 2).map(IMD_LABELS).fillna("ausente")
    if name == "age_group":
        return df["adm_age_band"].map(AGE_LABELS)
    if name in df.columns:
        return df[name].astype(str)
    raise KeyError(name)


def _rates(y, p, thr):
    pred = (p >= thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "false_negative_rate": fn / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "flagged_share": float(pred.mean()),
    }


def group_report(df: pd.DataFrame, y, p, threshold: float, group: str, n_boot: int = 300, seed: int = 42) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    g = group_series(df, group).to_numpy()
    students = df["student_id"].to_numpy() if "student_id" in df.columns else np.arange(len(df))
    rng = np.random.default_rng(seed)
    rows = []
    for level in pd.unique(g):
        m = g == level
        yy, pp, ss = y[m], p[m], students[m]
        row = {"group": group, "level": str(level), "n_records": int(m.sum()), "n_students": int(len(np.unique(ss))),
               "positives": int(yy.sum()), "observed_rate": float(yy.mean()), "mean_predicted": float(pp.mean()),
               "calibration_in_the_large": float(pp.mean() - yy.mean()), "brier": float(brier_score_loss(yy, pp)),
               "pr_auc": float(average_precision_score(yy, pp)) if 0 < yy.sum() < len(yy) else np.nan,
               **_rates(yy, pp, threshold)}
        row["sufficient_sample"] = bool(m.sum() >= MIN_N and yy.sum() >= MIN_POSITIVES)
        if row["sufficient_sample"]:
            uniq, inv = np.unique(ss, return_inverse=True)
            idx_by = pd.Series(np.arange(len(yy))).groupby(inv).apply(np.array).to_list()
            boots = []
            for _ in range(n_boot):
                idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(uniq), len(uniq))])
                boots.append(_rates(yy[idx], pp[idx], threshold))
            for k in ("recall", "false_positive_rate", "precision"):
                vals = np.array([b[k] for b in boots], dtype=float)
                row[f"{k}_ci95"] = [float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))]
        rows.append(row)
    return sorted(rows, key=lambda r: r["level"])


def full_audit(df: pd.DataFrame, y, p, threshold: float) -> dict:
    return {g: group_report(df, y, p, threshold, g) for g in ("gender", "disability", "imd_group", "age_group")}
