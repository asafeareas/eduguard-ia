"""Methodology support for FUTURE threshold definition with the pedagogical team (Q19).

Nothing here defines an operational threshold. The functions turn explicit, externally
supplied assumptions (team capacity, relative cost of a missed vs a false alert, intervention
effectiveness) into a candidate threshold and its expected consequences, computed on
CALIBRATION data. Probabilities are never turned into definitive student labels.
"""

from __future__ import annotations

import numpy as np


def capacity_threshold(p, capacity_fraction: float) -> float:
    """Smallest probability inside the top `capacity_fraction` of students (team can follow k%)."""
    p = np.sort(np.asarray(p))[::-1]
    k = max(1, int(np.ceil(capacity_fraction * len(p))))
    return float(p[k - 1])


def cost_optimal_threshold(cost_fn: float, cost_fp: float) -> float:
    """For CALIBRATED probabilities, alert when p * cost_fn > (1 - p) * cost_fp."""
    return float(cost_fp / (cost_fp + cost_fn))


def consequences(y, p, threshold: float, effectiveness: float = 0.0) -> dict:
    """What a threshold implies on labelled (calibration) data; `effectiveness` = share of true
    alerts whose withdrawal an intervention would prevent (an external assumption, default 0)."""
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    n = len(y)
    return {"threshold": float(threshold), "flagged": int(flag.sum()), "flagged_share": float(flag.mean()),
            "true_alerts": tp, "false_alerts": fp, "missed": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "alerts_per_true_case": (tp + fp) / tp if tp else float("inf"),
            "expected_prevented": tp * effectiveness, "n": n}


def net_benefit(y, p, threshold: float) -> float:
    """Decision-curve net benefit: TP/n - FP/n * t/(1-t)."""
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= threshold
    n = len(y)
    return float(((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * threshold / (1 - threshold)) / n)
