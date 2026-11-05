"""As-of (point-in-time) aggregation for OULAD.

Core rule, enforced by every function here:

    An aggregate computed for cutoff day `t` only uses events whose own
    timestamp is <= t. The population at `t` only contains students who were
    registered and had not unregistered by `t`.

Week convention (documented in docs/TEMPORAL_FEASIBILITY.md):
    day 0 is the first day of the presentation; week k covers days 7(k-1)..7k-1.
    "Prediction at the end of week k" therefore uses cutoff_day = 7k - 1.

These functions are exploratory aggregates for Phase 02. They are written to be
reusable, but the definitive feature set is NOT selected here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import OULAD_KEY, PRESENTATION_KEY

# Withdrawn = event of interest. "Fail" is kept separate (decision pending).
OUTCOME_ORDER = ["Withdrawn", "Fail", "Pass", "Distinction"]


def cutoff_day_for_week(week: int) -> int:
    """Last day (inclusive) of week `week` counted from day 0."""
    if week < 1:
        raise ValueError("week must be >= 1")
    return 7 * week - 1


def week_index(day: pd.Series | np.ndarray) -> pd.Series:
    """Map a day relative to start onto week number (day 0..6 -> week 1).

    Days before the start (negative) map to week <= 0.
    """
    return np.floor_divide(day, 7) + 1


def student_base(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per student-presentation: demographics + registration + outcome + length."""
    base = tables["studentInfo"].merge(tables["studentRegistration"], on=OULAD_KEY, how="left")
    return base.merge(tables["courses"], on=PRESENTATION_KEY, how="left")


def population_at_cutoff(base: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Students eligible for a prediction at `cutoff_day`.

    Included: registered by the cutoff (or registration date unknown) and not
    unregistered on or before the cutoff. Students who already left cannot be
    "predicted" -- their outcome is already observed.
    """
    registered = base["date_registration"].isna() | (base["date_registration"] <= cutoff_day)
    still_in = base["date_unregistration"].isna() | (base["date_unregistration"] > cutoff_day)
    return base.loc[registered & still_in].copy()


def vle_as_of(student_vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Per-student VLE aggregates using only interactions with date <= cutoff_day.

    Returns columns: total_clicks, active_days, distinct_sites, clicks_pre_start,
    first_active_day, last_active_day, days_since_last_activity,
    clicks_last_7_days, clicks_prev_7_days.
    """
    sv = student_vle.loc[student_vle["date"] <= cutoff_day]
    g = sv.groupby(OULAD_KEY, observed=True)
    out = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "distinct_sites": g["id_site"].nunique(),
        "first_active_day": g["date"].min(),
        "last_active_day": g["date"].max(),
    })
    pre = sv.loc[sv["date"] < 0].groupby(OULAD_KEY, observed=True)["sum_click"].sum()
    last7 = sv.loc[sv["date"] > cutoff_day - 7].groupby(OULAD_KEY, observed=True)["sum_click"].sum()
    prev7 = sv.loc[(sv["date"] > cutoff_day - 14) & (sv["date"] <= cutoff_day - 7)].groupby(
        OULAD_KEY, observed=True)["sum_click"].sum()
    out["clicks_pre_start"] = pre
    out["clicks_last_7_days"] = last7
    out["clicks_prev_7_days"] = prev7
    out[["clicks_pre_start", "clicks_last_7_days", "clicks_prev_7_days"]] = out[
        ["clicks_pre_start", "clicks_last_7_days", "clicks_prev_7_days"]].fillna(0)
    out["days_since_last_activity"] = cutoff_day - out["last_active_day"]
    out = out.reset_index()
    out["code_module"] = out["code_module"].astype(str)
    out["code_presentation"] = out["code_presentation"].astype(str)
    return out


def assessments_as_of(
    student_assessment: pd.DataFrame, assessments: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """Per-student assessment aggregates known at `cutoff_day`.

    - Submissions count only if date_submitted <= cutoff_day.
    - Banked results (is_banked == 1, carried over from a previous presentation)
      are excluded: their date semantics are not documented (see leakage audit).
    - Scores are assumed visible on the submission day (optimistic: the real
      release date is not in OULAD).
    Returns n_submitted, mean_score, n_late (submitted after due date).
    """
    sa = student_assessment.merge(
        assessments[["id_assessment", *PRESENTATION_KEY, "assessment_type", "date"]],
        on="id_assessment", how="left",
    )
    sa = sa.loc[(sa["date_submitted"] <= cutoff_day) & (sa["is_banked"] == 0)]
    sa = sa.assign(late=(sa["date_submitted"] > sa["date"]).astype(int))
    g = sa.groupby(OULAD_KEY)
    return pd.DataFrame({
        "n_submitted": g["id_assessment"].count(),
        "mean_score": g["score"].mean(),
        "n_late": g["late"].sum(),
    }).reset_index()


def assessments_due_by(assessments: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Number of non-exam assessments whose due date is <= cutoff, per presentation."""
    a = assessments.loc[(assessments["assessment_type"] != "Exam") & (assessments["date"] <= cutoff_day)]
    return a.groupby(PRESENTATION_KEY).size().rename("n_due").reset_index()


def snapshot(
    tables: dict[str, pd.DataFrame], student_vle: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """Population at cutoff joined with as-of VLE and assessment aggregates."""
    pop = population_at_cutoff(student_base(tables), cutoff_day)
    vle = vle_as_of(student_vle, cutoff_day)
    ass = assessments_as_of(tables["studentAssessment"], tables["assessments"], cutoff_day)
    due = assessments_due_by(tables["assessments"], cutoff_day)
    snap = pop.merge(vle, on=OULAD_KEY, how="left").merge(ass, on=OULAD_KEY, how="left")
    snap = snap.merge(due, on=PRESENTATION_KEY, how="left")
    zero_fill = ["total_clicks", "active_days", "distinct_sites", "clicks_pre_start",
                 "clicks_last_7_days", "clicks_prev_7_days", "n_submitted", "n_late", "n_due"]
    snap[zero_fill] = snap[zero_fill].fillna(0)
    snap["n_missing_submissions"] = (snap["n_due"] - snap["n_submitted"]).clip(lower=0)
    # Students with no VLE activity: recency measured from the presentation start
    # (or registration if later) -- documented as an assumption.
    no_act = snap["last_active_day"].isna()
    ref = snap.loc[no_act, "date_registration"].fillna(0).clip(lower=0)
    snap.loc[no_act, "days_since_last_activity"] = cutoff_day - ref
    snap["cutoff_day"] = cutoff_day
    return snap


def weekly_clicks(student_vle: pd.DataFrame, max_week: int | None = None) -> pd.DataFrame:
    """Long table: clicks per student-presentation per week (week<=0 = before start)."""
    wk = week_index(student_vle["date"].astype("int32"))
    df = student_vle.assign(week=wk)
    if max_week is not None:
        df = df.loc[df["week"] <= max_week]
    out = df.groupby([*OULAD_KEY, "week"], observed=True)["sum_click"].sum().reset_index()
    out["code_module"] = out["code_module"].astype(str)
    out["code_presentation"] = out["code_presentation"].astype(str)
    return out
