"""OULAD modelling datasets per cutoff (week 2 / 4 / 6).

Reuses the Phase 02 temporal rules (src/feature_engineering/temporal.py) and the
feature catalog (src/feature_engineering/catalog.py). Guarantees:

1. Only events dated <= cutoff_day are read (VLE clicks, submissions).
2. Population = registered by the cutoff and not unregistered by the cutoff,
   minus rows whose label/date are inconsistent (cannot be placed in time).
3. Outcome columns (final_result, date_unregistration) only feed targets,
   population and audit columns — never features.
4. Scores are not used (quarantined: no release date in OULAD).
5. Output is deterministic: fixed row order and column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import OULAD_KEY, PRESENTATION_KEY
from src.feature_engineering import catalog
from src.feature_engineering.temporal import cutoff_day_for_week, population_at_cutoff, student_base

EDUCATION_ORDER = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
                   "HE Qualification": 3, "Post Graduate Qualification": 4}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}


# ----------------------------------------------------------------------------- population
def eligible_population(tables: dict[str, pd.DataFrame], cutoff_day: int) -> tuple[pd.DataFrame, dict]:
    """Eligible students at `cutoff_day` plus an exclusion report."""
    base = student_base(tables)
    withdrawn = base["final_result"] == "Withdrawn"
    has_unreg = base["date_unregistration"].notna()
    inconsistent = (withdrawn & ~has_unreg) | (~withdrawn & has_unreg)
    clean = base.loc[~inconsistent]
    pop = population_at_cutoff(clean, cutoff_day)
    not_registered = clean["date_registration"].notna() & (clean["date_registration"] > cutoff_day)
    report = {
        "base_rows": int(len(base)),
        "excluded_inconsistent_label": int(inconsistent.sum()),
        "excluded_withdrawn_without_date": int((withdrawn & ~has_unreg).sum()),
        "excluded_not_withdrawn_with_date": int((~withdrawn & has_unreg).sum()),
        "excluded_not_yet_registered": int(not_registered.sum()),
        "excluded_unregistered_by_cutoff": int(len(clean) - len(pop) - not_registered.sum()),
        "eligible": int(len(pop)),
    }
    return pop.reset_index(drop=True), report


# ----------------------------------------------------------------------------- engagement
def prepare_vle(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """Attach activity_type to each interaction (static course metadata)."""
    types = vle[["id_site", "activity_type"]].astype({"id_site": "int32"})
    out = student_vle.merge(types, on="id_site", how="left")
    out["activity_type"] = out["activity_type"].astype("category")
    return out


def _keys(pop: pd.DataFrame) -> pd.DataFrame:
    k = pop[OULAD_KEY].copy()
    k["code_module"] = k["code_module"].astype(str)
    k["code_presentation"] = k["code_presentation"].astype(str)
    return k


def _str_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose module/presentation key columns are plain strings."""
    df = df.reset_index() if not set(OULAD_KEY).issubset(df.columns) else df.copy()
    df["code_module"] = df["code_module"].astype(str)
    df["code_presentation"] = df["code_presentation"].astype(str)
    return df


def engagement_features(sv: pd.DataFrame, pop: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Engagement features for `pop` using only interactions with date <= cutoff_day."""
    k = (cutoff_day + 1) // 7
    keys = _keys(pop)
    s = sv.loc[sv["date"] <= cutoff_day]
    # Merge on categoricals aligned with studentVle (avoids materialising 10M strings).
    keys_cat = keys.copy()
    for c in ("code_module", "code_presentation"):
        keys_cat[c] = pd.Categorical(keys_cat[c], categories=s[c].cat.categories)
    s = s.merge(keys_cat, on=OULAD_KEY, how="inner")

    daily = s.groupby([*OULAD_KEY, "date"], observed=True)["sum_click"].sum().reset_index()
    daily = _str_keys(daily)
    g = daily.groupby(OULAD_KEY)
    out = keys.copy()

    def put(name, series, fill=0):
        nonlocal out
        out = out.merge(_str_keys(series.rename(name).reset_index()), on=OULAD_KEY, how="left")
        if fill is not None:
            out[name] = out[name].fillna(fill)

    put("eng_total_clicks", g["sum_click"].sum())
    put("eng_active_days", g["date"].nunique())
    put("_first", g["date"].min(), fill=None)
    put("_last", g["date"].max(), fill=None)
    pre = daily.loc[daily["date"] < 0].groupby(OULAD_KEY)
    put("eng_clicks_pre_start", pre["sum_click"].sum())
    put("eng_active_days_pre_start", pre["date"].nunique())
    in_pres = daily.loc[daily["date"] >= 0]
    put("_clicks_in_presentation", in_pres.groupby(OULAD_KEY)["sum_click"].sum())
    recent = daily.loc[daily["date"] >= cutoff_day - 13].groupby(OULAD_KEY)
    put("eng_clicks_last_14_days", recent["sum_click"].sum())
    put("eng_active_days_last_14_days", recent["date"].nunique())
    put("eng_distinct_sites", s.groupby(OULAD_KEY, observed=True)["id_site"].nunique())
    put("eng_distinct_activity_types", s.groupby(OULAD_KEY, observed=True)["activity_type"].nunique())

    by_type = _str_keys(s.groupby([*OULAD_KEY, "activity_type"], observed=True)["sum_click"].sum().reset_index())
    by_type["p"] = by_type["sum_click"] / by_type.groupby(OULAD_KEY)["sum_click"].transform("sum")
    by_type["h"] = -by_type["p"] * np.log2(by_type["p"])
    put("eng_activity_type_entropy", by_type.groupby(OULAD_KEY)["h"].sum())

    # Weekly matrix (weeks 1..k, zeros for inactive weeks)
    weekly = in_pres.assign(week=in_pres["date"] // 7 + 1)
    weekly = weekly.groupby([*OULAD_KEY, "week"])["sum_click"].sum().unstack("week")
    weekly = weekly.reindex(columns=range(1, k + 1)).fillna(0)
    weekly.columns = [f"eng_clicks_week_{w:02d}" for w in range(1, k + 1)]
    out = out.merge(weekly.reset_index(), on=OULAD_KEY, how="left")
    wcols = list(weekly.columns)
    out[wcols] = out[wcols].fillna(0)
    W = out[wcols].to_numpy(dtype=float)
    x = np.arange(1, k + 1, dtype=float)
    xc = x - x.mean()
    mean_w = W.mean(axis=1)
    std_w = W.std(axis=1, ddof=0)

    out["eng_any_activity"] = (out["eng_total_clicks"] > 0).astype(int)
    out["eng_clicks_per_active_day"] = np.where(out["eng_active_days"] > 0,
                                                out["eng_total_clicks"] / out["eng_active_days"].clip(lower=1), 0.0)
    out["eng_clicks_per_elapsed_day"] = out["_clicks_in_presentation"] / (cutoff_day + 1)
    out["eng_active_weeks"] = (W > 0).sum(axis=1)
    out["eng_inactive_weeks"] = k - out["eng_active_weeks"]
    out["eng_active_weeks_ratio"] = out["eng_active_weeks"] / k
    out["eng_days_since_last_activity"] = np.where(out["_last"].notna(), cutoff_day - out["_last"], cutoff_day + 1)
    out["eng_first_activity_day"] = out["_first"].fillna(cutoff_day + 1)
    out["eng_weekly_clicks_std"] = std_w
    out["eng_weekly_clicks_cv"] = np.where(mean_w > 0, std_w / np.where(mean_w > 0, mean_w, 1), 0.0)
    out["eng_weekly_trend_slope"] = (W - mean_w[:, None]) @ xc / (xc ** 2).sum()
    out["eng_last_week_change"] = W[:, k - 1] - W[:, k - 2]
    out["eng_clicks_pct_in_presentation"] = out.groupby(PRESENTATION_KEY)["eng_total_clicks"].rank(
        pct=True, method="average")
    out["max_vle_day_used"] = out["_last"]
    return out.drop(columns=["_first", "_last", "_clicks_in_presentation"])


# ----------------------------------------------------------------------------- assessment
def assessment_features(tables: dict[str, pd.DataFrame], pop: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Participation/submission features known at cutoff_day. Scores are NOT used.

    Banked results (is_banked=1; all recorded on day -1, i.e. carried over before
    the start) make an assessment "not applicable" for that student: they are
    removed from the student's due denominators instead of counting as missing.
    """
    keys = _keys(pop)
    a = tables["assessments"]
    a = a.loc[(a["assessment_type"] != "Exam") & a["date"].notna(),
              ["id_assessment", *PRESENTATION_KEY, "date", "weight"]]
    first = a.sort_values(["date", "id_assessment"]).groupby(PRESENTATION_KEY).head(1)
    due = a.loc[a["date"] <= cutoff_day]

    sa = tables["studentAssessment"].merge(a, on="id_assessment", how="inner")
    sa = sa.merge(keys, on=OULAD_KEY, how="inner")
    banked = sa.loc[sa["is_banked"] == 1, ["id_assessment", "id_student"]].assign(banked=True)
    subs = sa.loc[(sa["is_banked"] == 0) & (sa["date_submitted"] <= cutoff_day)].copy()
    subs["is_due"] = subs["date"] <= cutoff_day
    subs["on_time"] = subs["date_submitted"] <= subs["date"]

    # student x due-assessment grid, minus banked
    grid = keys.merge(due, on=PRESENTATION_KEY, how="inner")
    grid = grid.merge(banked, on=["id_assessment", "id_student"], how="left")
    grid = grid.loc[grid["banked"].isna()].drop(columns="banked")
    grid = grid.merge(subs[["id_assessment", "id_student"]].assign(sub=1), on=["id_assessment", "id_student"],
                      how="left").fillna({"sub": 0})

    out = keys.copy()

    def put(name, series, fill=0):
        nonlocal out
        out = out.merge(series.rename(name).reset_index(), on=OULAD_KEY, how="left")
        if fill is not None:
            out[name] = out[name].fillna(fill)

    gg = grid.groupby(OULAD_KEY)
    put("asm_n_due", gg.size())
    put("asm_n_due_submitted", gg["sub"].sum())
    put("asm_missed_weight", grid.loc[grid["sub"] == 0].groupby(OULAD_KEY)["weight"].sum())
    gs = subs.groupby(OULAD_KEY)
    put("asm_n_submitted", gs.size())
    put("asm_n_on_time", subs.loc[subs["on_time"]].groupby(OULAD_KEY).size())
    put("asm_n_late", subs.loc[~subs["on_time"]].groupby(OULAD_KEY).size())
    put("asm_mean_days_before_deadline", (subs["date"] - subs["date_submitted"]).groupby(
        [subs[c] for c in OULAD_KEY]).mean(), fill=None)
    out["asm_n_missing"] = out["asm_n_due"] - out["asm_n_due_submitted"]
    out["asm_submission_rate"] = np.where(out["asm_n_due"] > 0,
                                          out["asm_n_due_submitted"] / out["asm_n_due"].clip(lower=1), np.nan)

    f = keys.merge(first[[*PRESENTATION_KEY, "id_assessment", "date"]], on=PRESENTATION_KEY, how="left")
    f = f.merge(subs[["id_assessment", "id_student"]].assign(s=1), on=["id_assessment", "id_student"], how="left")
    f = f.merge(banked, on=["id_assessment", "id_student"], how="left")
    out["asm_first_due_passed"] = (f["date"] <= cutoff_day).astype(int).to_numpy()
    out["asm_first_submitted"] = (f["s"].notna() | f["banked"].notna()).astype(int).to_numpy()
    sub_last = subs.groupby(OULAD_KEY)["date_submitted"].max()
    out = out.merge(sub_last.rename("max_submission_day_used").reset_index(), on=OULAD_KEY, how="left")
    return out


# ----------------------------------------------------------------------------- static
def static_features(pop: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Context + admission features, all known at registration/start."""
    out = _keys(pop)
    p = pop.reset_index(drop=True)
    a = tables["assessments"]
    planned = a.loc[a["assessment_type"] != "Exam"].groupby(PRESENTATION_KEY).size().rename(
        "ctx_n_planned_assessments").reset_index()
    out["ctx_code_module"] = p["code_module"].astype(str)
    out["ctx_presentation_start"] = p["code_presentation"].astype(str).str[-1]
    out["ctx_module_length"] = p["module_presentation_length"].astype(int)
    out = out.merge(planned, on=PRESENTATION_KEY, how="left").fillna({"ctx_n_planned_assessments": 0})
    out["adm_gender"] = p["gender"].astype(str)
    out["adm_region"] = p["region"].astype(str)
    out["adm_highest_education"] = p["highest_education"].map(EDUCATION_ORDER)
    out["adm_imd_band"] = p["imd_band"].str.extract(r"^(\d+)")[0].astype(float) // 10
    out["adm_age_band"] = p["age_band"].map(AGE_ORDER)
    out["adm_disability"] = (p["disability"] == "Y").astype(int)
    out["adm_num_prev_attempts"] = p["num_of_prev_attempts"].astype(int)
    out["adm_registration_lead_days"] = -p["date_registration"]
    return out


# ----------------------------------------------------------------------------- dataset
def build_week(tables: dict[str, pd.DataFrame], sv_typed: pd.DataFrame, week: int) -> tuple[pd.DataFrame, dict]:
    """Full modelling table for one cutoff: ids + features + targets + audit columns."""
    cutoff_day = cutoff_day_for_week(week)
    pop, report = eligible_population(tables, cutoff_day)
    df = static_features(pop, tables)
    df = df.merge(engagement_features(sv_typed, pop, cutoff_day), on=OULAD_KEY, how="left")
    df = df.merge(assessment_features(tables, pop, cutoff_day), on=OULAD_KEY, how="left")
    p = _keys(pop).assign(
        final_result=pop["final_result"].to_numpy(),
        date_registration=pop["date_registration"].to_numpy(),
        date_unregistration=pop["date_unregistration"].to_numpy(),
    )
    df = df.merge(p, on=OULAD_KEY, how="left")

    df["student_id"] = df["id_student"].astype("int64")
    df["presentation_id"] = df["code_module"] + "_" + df["code_presentation"]
    df["target_withdrawn"] = (df["final_result"] == "Withdrawn").astype(int)
    df["target_adverse_outcome"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    df["cutoff_week"] = week
    df["cutoff_day"] = cutoff_day
    df["presentation_year"] = df["code_presentation"].str[:4].astype(int)

    features = catalog.feature_names_for_week(week)
    cols = [*catalog.OULAD_ID_COLUMNS, *features, *catalog.OULAD_TARGET_COLUMNS, *catalog.OULAD_AUDIT_COLUMNS]
    df = df[cols].sort_values(["presentation_id", "student_id"]).reset_index(drop=True)
    return canonical_dtypes(df, week), report


_DTYPE_BY_FTYPE = {"count": "int64", "binary": "int64", "ordinal": "float64", "numeric": "float64",
                   "nominal": "object"}


def canonical_dtypes(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Fixed dtypes per feature type, so output never depends on input dtypes.

    ordinal is float64 because it may hold missing values (imputation happens later,
    fitted on the training split only).
    """
    df = df.copy()
    for f in catalog.oulad_features_for_week(week):
        dtype = _DTYPE_BY_FTYPE[f.ftype]
        df[f.name] = df[f.name].astype(str) if dtype == "object" else df[f.name].astype(dtype)
    for c in ("target_withdrawn", "target_adverse_outcome", "cutoff_week", "cutoff_day", "presentation_year",
              "student_id"):
        df[c] = df[c].astype("int64")
    for c in ("date_registration", "date_unregistration", "max_vle_day_used", "max_submission_day_used"):
        df[c] = df[c].astype("float64")
    return df
