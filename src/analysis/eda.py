"""Descriptive statistics helpers for the exploratory analysis.

All functions are descriptive (counts, rates, rank statistics). None of them
fits a model. Associations reported here are NOT causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import OULAD_KEY


def distribution(series: pd.Series) -> pd.DataFrame:
    """Counts and proportions of a categorical series (missing kept)."""
    counts = series.value_counts(dropna=False)
    return pd.DataFrame({"count": counts, "proportion": (counts / counts.sum()).round(4)})


def rate_by_category(df: pd.DataFrame, col: str, event: pd.Series, min_count: int = 30) -> pd.DataFrame:
    """Event rate (e.g. withdrawal) per category of `col`, with counts.

    Categories with fewer than `min_count` rows are reported but flagged.
    """
    tmp = pd.DataFrame({col: df[col].astype("object").fillna("<missing>"), "event": event.astype(int)})
    out = tmp.groupby(col)["event"].agg(n="size", events="sum")
    out["rate"] = (out["events"] / out["n"]).round(4)
    out["small_sample"] = out["n"] < min_count
    return out.sort_values("rate", ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-uncorrected Cramér's V between two categorical series."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
    chi2 = ((observed - expected) ** 2 / expected).sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (observed.sum() * k)))


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    """Rank-based AUC (Mann-Whitney U / n1 n0) of `x` for binary `y`.

    0.5 = no separation; values far from 0.5 in either direction mean strong
    separation. This is a descriptive statistic, not a model. Rows with
    missing x are dropped.
    """
    if not x.index.equals(y.index):
        raise ValueError("x and y must share the same index (misaligned rows would corrupt the AUC)")
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(float), y[mask].astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = x.rank(method="average")
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def separation(auc: float) -> float:
    """Direction-free strength of separation: |AUC - 0.5| * 2 in [0, 1]."""
    return float(abs(auc - 0.5) * 2) if not np.isnan(auc) else float("nan")


def group_medians(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return df.groupby(by)[cols].median()


def mean_weekly_clicks_by_outcome(
    weekly: pd.DataFrame, base: pd.DataFrame, weeks: range
) -> pd.DataFrame:
    """Mean clicks per week for each outcome, counting zero for inactive students.

    Each student only contributes to weeks in which they were still registered
    (week start <= date_unregistration), so withdrawn students do not drag the
    mean down after leaving.
    """
    grid = base[[*OULAD_KEY, "final_result", "date_unregistration"]].merge(
        pd.DataFrame({"week": list(weeks)}), how="cross"
    )
    still = grid["date_unregistration"].isna() | (7 * (grid["week"] - 1) <= grid["date_unregistration"])
    grid = grid.loc[still]
    grid = grid.merge(weekly, on=[*OULAD_KEY, "week"], how="left").fillna({"sum_click": 0})
    out = grid.groupby(["final_result", "week"])["sum_click"].agg(["mean", "median", "size"]).reset_index()
    return out


def clicks_relative_to_withdrawal(
    weekly: pd.DataFrame, base: pd.DataFrame, weeks_before: int = 8
) -> pd.DataFrame:
    """Mean weekly clicks in the weeks preceding withdrawal (week 0 = week of unregistration).

    Only students who withdrew at least `weeks_before` weeks after the start are
    used, so every student contributes to every relative week.
    """
    wd = base.loc[
        (base["final_result"] == "Withdrawn") & (base["date_unregistration"] >= 7 * weeks_before),
        [*OULAD_KEY, "date_unregistration"],
    ].copy()
    wd["withdraw_week"] = np.floor_divide(wd["date_unregistration"], 7) + 1
    grid = wd.merge(pd.DataFrame({"rel_week": range(-weeks_before, 1)}), how="cross")
    grid["week"] = grid["withdraw_week"] + grid["rel_week"]
    grid = grid.merge(weekly, on=[*OULAD_KEY, "week"], how="left").fillna({"sum_click": 0})
    out = grid.groupby("rel_week")["sum_click"].agg(["mean", "median", "size"]).reset_index()
    out["share_inactive"] = grid.assign(z=grid["sum_click"] == 0).groupby("rel_week")["z"].mean().to_numpy()
    return out


def last_activity_profile(student_vle: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Last VLE activity day relative to presentation length, per outcome.

    Used to characterise "silent" disengagement among students labelled Fail
    (no formal unregistration). Uses the whole presentation on purpose: this is
    a LABEL-QUALITY diagnostic, never a feature.
    """
    last = student_vle.groupby(OULAD_KEY, observed=True)["date"].max().rename("last_day").reset_index()
    last["code_module"] = last["code_module"].astype(str)
    last["code_presentation"] = last["code_presentation"].astype(str)
    df = base.merge(last, on=OULAD_KEY, how="left")
    df["last_day_ratio"] = df["last_day"] / df["module_presentation_length"]
    return df[[*OULAD_KEY, "final_result", "last_day", "last_day_ratio", "module_presentation_length"]]
