"""Probability calibration and preliminary risk bands.

Everything here is fitted on the CALIBRATION subset (2013 students not used for
training or model selection). The 2014 test set is never seen.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """method: 'none' | 'sigmoid' (Platt on the logit) | 'isotonic'."""

    def __init__(self, method: str = "none"):
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(method)
        self.method = method

    def fit(self, p, y):
        y = np.asarray(y)
        if self.method == "sigmoid":
            with warnings.catch_warnings():  # scipy>=1.15 emits a harmless 'iprint' OptimizeWarning via lbfgs
                warnings.simplefilter("ignore")
                self.model_ = LogisticRegression(C=1e6).fit(_logit(p), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p), y)
        return self

    def transform(self, p):
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(p))[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model_.predict(np.asarray(p)), 0, 1)
        return np.asarray(p, dtype=float)


def select_calibrator(p, y, groups, n_splits: int = 5) -> tuple[Calibrator, dict]:
    """Choose none/sigmoid/isotonic by out-of-fold Brier score WITHIN the calibration subset
    (folds by student), then refit the winner on the whole calibration subset."""
    p, y, groups = map(np.asarray, (p, y, groups))
    scores = {}
    for method in ("none", "sigmoid", "isotonic"):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=n_splits).split(p, y, groups):
            oof[te] = Calibrator(method).fit(p[tr], y[tr]).transform(p[te])
        scores[method] = float(brier_score_loss(y, oof))
    # prefer the simpler method unless the gain is > 0.5% of the uncalibrated Brier
    best = min(scores, key=scores.get)
    if best != "none" and scores["none"] - scores[best] < 0.005 * scores["none"]:
        best = "none"
    return Calibrator(best).fit(p, y), {"oof_brier": scores, "selected": best}


def propose_risk_bands(p_cal, y_cal, train_prevalence: float, f1_threshold: float) -> dict:
    """Preliminary bands on CALIBRATED probabilities, from training/calibration data only.

    low       : p <  train_prevalence           (below the average risk of the training population)
    moderate  : train_prevalence <= p < high    (above average)
    high      : p >= max(f1_threshold, 2 * train_prevalence)
                f1_threshold is the F1-optimal alert threshold on the calibration subset; the
                2x-prevalence floor keeps "high" meaning at least double the base rate.
    """
    high = float(max(f1_threshold, 2 * train_prevalence))
    low = float(train_prevalence)
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    band = np.where(p_cal >= high, "high", np.where(p_cal >= low, "moderate", "low"))
    summary = {b: {"share": float((band == b).mean()), "observed_rate": float(y_cal[band == b].mean())
                   if (band == b).any() else None} for b in ("low", "moderate", "high")}
    return {"low_upper": low, "high_lower": high, "rule": propose_risk_bands.__doc__.strip(),
            "calibration_subset": summary}


def assign_bands(p, bands: dict) -> np.ndarray:
    p = np.asarray(p)
    return np.where(p >= bands["high_lower"], "high", np.where(p >= bands["low_upper"], "moderate", "low"))
