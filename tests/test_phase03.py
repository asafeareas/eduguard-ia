"""Phase 03 tests: feature calculations, cutoffs, targets, eligibility, temporal
invariance, absence of leakage, reproducibility and processed-dataset consistency.

Synthetic tests run offline; real-data tests are skipped when the data or the
processed datasets are not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import loaders
from src.data.download import sha256_of
from src.feature_engineering import build_datasets as bd
from src.feature_engineering import catalog, oulad_features as of, uci_admission

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
K = {"code_module": "AAA", "code_presentation": "2013J"}
HAS_OULAD = (loaders.OULAD_DIR / "studentVle.csv").exists()
HAS_UCI = loaders.UCI_CSV.exists()
HAS_PROCESSED = (PROCESSED / "oulad" / "week_04" / "oulad_week_04.parquet").exists()


# =============================================================================== fixtures
def make_tables():
    info = pd.DataFrame([
        {**K, "id_student": 1, "final_result": "Withdrawn", "gender": "M", "region": "R1",
         "highest_education": "A Level or Equivalent", "imd_band": "10-20", "age_band": "0-35",
         "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N"},      # left day -3
        {**K, "id_student": 2, "final_result": "Withdrawn", "gender": "F", "region": "R2",
         "highest_education": "No Formal quals", "imd_band": "90-100%", "age_band": "35-55",
         "num_of_prev_attempts": 1, "studied_credits": 120, "disability": "Y"},     # leaves day 20
        {**K, "id_student": 3, "final_result": "Pass", "gender": "F", "region": "R1",
         "highest_education": "HE Qualification", "imd_band": np.nan, "age_band": "55<=",
         "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N"},
        {**K, "id_student": 4, "final_result": "Fail", "gender": "M", "region": "R3",
         "highest_education": "Lower Than A Level", "imd_band": "0-10%", "age_band": "0-35",
         "num_of_prev_attempts": 2, "studied_credits": 90, "disability": "N"},      # registers day 30
        {**K, "id_student": 5, "final_result": "Distinction", "gender": "F", "region": "R2",
         "highest_education": "Post Graduate Qualification", "imd_band": "50-60%", "age_band": "0-35",
         "num_of_prev_attempts": 0, "studied_credits": 30, "disability": "N"},
        {**K, "id_student": 6, "final_result": "Withdrawn", "gender": "M", "region": "R1",
         "highest_education": "A Level or Equivalent", "imd_band": "20-30%", "age_band": "0-35",
         "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N"},      # NO unreg date
        {**K, "id_student": 7, "final_result": "Pass", "gender": "M", "region": "R1",
         "highest_education": "A Level or Equivalent", "imd_band": "20-30%", "age_band": "0-35",
         "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N"},      # Pass WITH unreg date
    ])
    reg = pd.DataFrame([
        {**K, "id_student": 1, "date_registration": -40, "date_unregistration": -3},
        {**K, "id_student": 2, "date_registration": -40, "date_unregistration": 20},
        {**K, "id_student": 3, "date_registration": -10, "date_unregistration": np.nan},
        {**K, "id_student": 4, "date_registration": 30, "date_unregistration": np.nan},
        {**K, "id_student": 5, "date_registration": np.nan, "date_unregistration": np.nan},
        {**K, "id_student": 6, "date_registration": -5, "date_unregistration": np.nan},
        {**K, "id_student": 7, "date_registration": -5, "date_unregistration": 100},
    ])
    courses = pd.DataFrame([{**K, "module_presentation_length": 260}])
    assessments = pd.DataFrame([
        {**K, "id_assessment": 10, "assessment_type": "TMA", "date": 19.0, "weight": 10.0},
        {**K, "id_assessment": 11, "assessment_type": "CMA", "date": 26.0, "weight": 5.0},
        {**K, "id_assessment": 12, "assessment_type": "TMA", "date": 40.0, "weight": 20.0},
        {**K, "id_assessment": 13, "assessment_type": "Exam", "date": np.nan, "weight": 100.0},
    ])
    sa = pd.DataFrame([
        {"id_assessment": 10, "id_student": 3, "date_submitted": 18, "is_banked": 0, "score": 80.0},
        {"id_assessment": 11, "id_student": 3, "date_submitted": 27, "is_banked": 0, "score": 60.0},  # late
        {"id_assessment": 12, "id_student": 3, "date_submitted": 39, "is_banked": 0, "score": 70.0},
        {"id_assessment": 10, "id_student": 2, "date_submitted": 15, "is_banked": 0, "score": 50.0},
        {"id_assessment": 10, "id_student": 5, "date_submitted": -1, "is_banked": 1, "score": 99.0},  # banked
    ])
    vle = pd.DataFrame([
        {"id_site": 100, **K, "activity_type": "resource", "week_from": np.nan, "week_to": np.nan},
        {"id_site": 101, **K, "activity_type": "forumng", "week_from": np.nan, "week_to": np.nan},
        {"id_site": 102, **K, "activity_type": "quiz", "week_from": np.nan, "week_to": np.nan},
    ])
    return {"studentInfo": info, "studentRegistration": reg, "courses": courses,
            "assessments": assessments, "studentAssessment": sa, "vle": vle}


def make_vle(rows, tables):
    sv = pd.DataFrame([{**K, "id_student": s, "id_site": site, "date": d, "sum_click": c}
                       for s, site, d, c in rows])
    sv = sv.astype({"code_module": "category", "code_presentation": "category", "id_student": "int32",
                    "id_site": "int32", "date": "int16", "sum_click": "int32"})
    return of.prepare_vle(sv, tables["vle"])


BASE_CLICKS = [
    (3, 100, -2, 4), (3, 101, 0, 2), (3, 100, 3, 6), (3, 102, 8, 1), (3, 100, 13, 3),  # <= 13
    (3, 100, 20, 10), (3, 101, 27, 5),                                                 # weeks 3-4
    (2, 100, 1, 7), (2, 101, 10, 1), (2, 100, 30, 99),
    (5, 102, 12, 2),
]


@pytest.fixture
def tables():
    return make_tables()


@pytest.fixture
def sv(tables):
    return make_vle(BASE_CLICKS, tables)


def features_of(df, week):
    return df[["student_id", *catalog.feature_names_for_week(week)]].set_index("student_id")


# =============================================================================== catalog
def test_catalog_is_consistent():
    names = [f.name for f in catalog.OULAD_FEATURES]
    assert len(names) == len(set(names))
    reserved = set(catalog.OULAD_ID_COLUMNS) | set(catalog.OULAD_TARGET_COLUMNS) | set(catalog.OULAD_AUDIT_COLUMNS)
    for f in catalog.OULAD_FEATURES:
        assert f.name not in reserved
        assert set(f.cutoffs) <= set(catalog.WEEKS)
        assert f.ftype in {"numeric", "count", "binary", "ordinal", "nominal"}
        if f.status == "quarantined":
            assert f.cutoffs == ()
        else:
            assert f.status == "included" and f.cutoffs


def test_week_feature_sets_respect_availability():
    s2, s4, s6 = (set(catalog.feature_names_for_week(w)) for w in (2, 4, 6))
    assert not any(n.startswith("asm_") for n in s2), "week 2 is engagement + static only"
    assert {"asm_n_due", "asm_n_missing", "asm_first_submitted"} <= s4
    for w, s in ((2, s2), (4, s4), (6, s6)):
        weekly = {n for n in s if n.startswith("eng_clicks_week_")}
        assert weekly == {f"eng_clicks_week_{i:02d}" for i in range(1, w + 1)}
    quarantined = {f.name for f in catalog.OULAD_FEATURES if f.status == "quarantined"}
    assert {"asm_mean_score", "studied_credits", "adm_registration_missing"} <= quarantined
    assert not quarantined & (s2 | s4 | s6)
    assert not any("score" in n for n in s2 | s4 | s6)


# =============================================================================== eligibility & targets
def test_eligibility_rules(tables):
    pop, rep = of.eligible_population(tables, 13)
    assert set(pop["id_student"]) == {2, 3, 5}           # 1 left, 4 not registered, 6/7 inconsistent
    assert rep["excluded_withdrawn_without_date"] == 1 and rep["excluded_not_withdrawn_with_date"] == 1
    assert rep["excluded_not_yet_registered"] == 1 and rep["excluded_unregistered_by_cutoff"] == 1
    assert rep["eligible"] + rep["excluded_inconsistent_label"] + rep["excluded_not_yet_registered"] \
        + rep["excluded_unregistered_by_cutoff"] == rep["base_rows"]
    assert set(of.eligible_population(tables, 20)[0]["id_student"]) == {3, 5}   # leaves ON day 20
    assert set(of.eligible_population(tables, 41)[0]["id_student"]) == {3, 4, 5}


def test_target_definitions(tables, sv):
    df, _ = of.build_week(tables, sv, 6)
    t = df.set_index("student_id")[["final_result", "target_withdrawn", "target_adverse_outcome"]]
    assert t.loc[4].tolist() == ["Fail", 0, 1]
    assert t.loc[3].tolist() == ["Pass", 0, 0]
    assert t.loc[5].tolist() == ["Distinction", 0, 0]
    df2, _ = of.build_week(tables, sv, 2)
    assert df2.set_index("student_id").loc[2, ["target_withdrawn", "target_adverse_outcome"]].tolist() == [1, 1]
    for d in (df, df2):
        assert set(d["target_withdrawn"]) <= {0, 1} and d["target_withdrawn"].notna().all()


# =============================================================================== feature values
def test_engagement_feature_values(tables, sv):
    f = features_of(of.build_week(tables, sv, 2)[0], 2).loc[3]
    assert f["eng_total_clicks"] == 16                  # 4 + 2 + 6 + 1 + 3
    assert f["eng_clicks_pre_start"] == 4 and f["eng_active_days_pre_start"] == 1
    assert f["eng_active_days"] == 5
    assert f["eng_clicks_per_active_day"] == pytest.approx(16 / 5)
    assert f["eng_clicks_per_elapsed_day"] == pytest.approx(12 / 14)
    assert f["eng_clicks_week_01"] == 8 and f["eng_clicks_week_02"] == 4
    assert f["eng_active_weeks"] == 2 and f["eng_inactive_weeks"] == 0 and f["eng_active_weeks_ratio"] == 1
    assert f["eng_weekly_trend_slope"] == pytest.approx(-4)
    assert f["eng_weekly_clicks_std"] == pytest.approx(2) and f["eng_weekly_clicks_cv"] == pytest.approx(2 / 6)
    assert f["eng_days_since_last_activity"] == 0 and f["eng_first_activity_day"] == -2
    assert f["eng_distinct_sites"] == 3 and f["eng_distinct_activity_types"] == 3
    p = np.array([13, 2, 1]) / 16                       # resource 4+6+3, forumng 2, quiz 1
    assert f["eng_activity_type_entropy"] == pytest.approx(-(p * np.log2(p)).sum())


def test_never_active_student_gets_deterministic_values(tables, sv):
    f = features_of(of.build_week(tables, sv, 4)[0], 4)
    # student 5 only clicked on day 12; at week 4 recency = 27 - 12
    assert f.loc[5, "eng_days_since_last_activity"] == 15
    tables2 = make_tables()
    sv2 = make_vle([r for r in BASE_CLICKS if r[0] != 5], tables2)
    g = features_of(of.build_week(tables2, sv2, 4)[0], 4).loc[5]
    assert g["eng_any_activity"] == 0 and g["eng_total_clicks"] == 0
    assert g["eng_days_since_last_activity"] == 28 and g["eng_first_activity_day"] == 28


def test_recent_window_features(tables, sv):
    f = features_of(of.build_week(tables, sv, 4)[0], 4).loc[3]
    assert f["eng_clicks_last_14_days"] == 15           # days 14..27: 10 + 5
    assert f["eng_active_days_last_14_days"] == 2
    assert f["eng_last_week_change"] == 5 - 10          # week 4 (days 21-27) minus week 3 (14-20)


def test_assessment_feature_values(tables, sv):
    f = features_of(of.build_week(tables, sv, 4)[0], 4)
    s3 = f.loc[3]
    assert s3["asm_n_due"] == 2                         # 10 (day 19), 11 (day 26)
    assert s3["asm_n_submitted"] == 2                   # day-39 submission is after the cutoff
    assert s3["asm_n_due_submitted"] == 2 and s3["asm_n_missing"] == 0
    assert s3["asm_n_on_time"] == 1 and s3["asm_n_late"] == 1
    assert s3["asm_mean_days_before_deadline"] == pytest.approx(((19 - 18) + (26 - 27)) / 2)
    assert s3["asm_first_due_passed"] == 1 and s3["asm_first_submitted"] == 1
    s5 = f.loc[5]                                       # banked first TMA -> not applicable
    assert s5["asm_n_due"] == 1 and s5["asm_n_missing"] == 1 and s5["asm_missed_weight"] == 5
    assert s5["asm_first_submitted"] == 1 and s5["asm_n_submitted"] == 0
    assert np.isnan(s5["asm_mean_days_before_deadline"])


def test_submission_rate_undefined_without_due(tables, sv):
    tables["assessments"].loc[tables["assessments"]["id_assessment"].isin([10, 11]), "date"] = 35.0
    f = features_of(of.build_week(tables, sv, 4)[0], 4)
    assert (f["asm_n_due"] == 0).all() and f["asm_submission_rate"].isna().all()


def test_static_feature_mapping(tables, sv):
    f = features_of(of.build_week(tables, sv, 2)[0], 2)
    assert f.loc[2, "adm_imd_band"] == 9 and np.isnan(f.loc[3, "adm_imd_band"])
    assert f.loc[2, "adm_highest_education"] == 0 and f.loc[5, "adm_highest_education"] == 4
    assert f.loc[3, "adm_age_band"] == 2 and f.loc[2, "adm_disability"] == 1
    assert f.loc[2, "adm_registration_lead_days"] == 40
    assert f.loc[3, "ctx_presentation_start"] == "J" and f.loc[3, "ctx_n_planned_assessments"] == 3


# =============================================================================== leakage & invariance
@pytest.mark.parametrize("week", catalog.WEEKS)
def test_temporal_invariance_to_post_cutoff_events(tables, sv, week):
    cutoff = week * 7 - 1
    before = features_of(of.build_week(tables, sv, week)[0], week)
    t2 = make_tables()
    future_clicks = [(s, site, cutoff + d, 500) for s in (2, 3, 5) for site, d in ((100, 1), (102, 30))]
    sv2 = make_vle(BASE_CLICKS + future_clicks, t2)
    extra = pd.DataFrame([
        {"id_assessment": 12, "id_student": 5, "date_submitted": cutoff + 1, "is_banked": 0, "score": 10.0},
        {"id_assessment": 11, "id_student": 2, "date_submitted": cutoff + 3, "is_banked": 0, "score": 10.0},
    ])
    t2["studentAssessment"] = pd.concat([t2["studentAssessment"], extra], ignore_index=True)
    # A still-eligible student's (future) unregistration date moves, but stays after the cutoff.
    t2["studentRegistration"].loc[t2["studentRegistration"]["id_student"] == 3, "date_unregistration"] = cutoff + 50
    t2["studentInfo"].loc[t2["studentInfo"]["id_student"] == 3, "final_result"] = "Withdrawn"
    after = features_of(of.build_week(t2, sv2, week)[0], week)
    pd.testing.assert_frame_equal(before, after)


def test_features_do_not_depend_on_outcome(tables, sv):
    before = features_of(of.build_week(tables, sv, 4)[0], 4)
    t2 = make_tables()
    t2["studentInfo"].loc[t2["studentInfo"]["id_student"] == 5, "final_result"] = "Fail"
    t2["studentInfo"].loc[t2["studentInfo"]["id_student"] == 3, "final_result"] = "Distinction"
    after = features_of(of.build_week(t2, make_vle(BASE_CLICKS, t2), 4)[0], 4)
    pd.testing.assert_frame_equal(before, after)


def test_audit_bounds(tables, sv):
    for week in catalog.WEEKS:
        df, _ = of.build_week(tables, sv, week)
        cut = week * 7 - 1
        assert (df["max_vle_day_used"].dropna() <= cut).all()
        assert (df["max_submission_day_used"].dropna() <= cut).all()
        assert (df["date_unregistration"].isna() | (df["date_unregistration"] > cut)).all()


def test_split_roles_excludes_ids_targets_and_audit(tables, sv):
    df, _ = of.build_week(tables, sv, 4)
    meta = bd.oulad_metadata(4)
    X, y = bd.split_roles(df, meta, "target_withdrawn")
    forbidden = set(meta["id_columns"]) | set(meta["target_columns"]) | set(meta["audit_columns"])
    assert not forbidden & set(X.columns)
    assert list(X.columns) == catalog.feature_names_for_week(4)
    with pytest.raises(KeyError):
        bd.split_roles(df, meta, "final_result")


# =============================================================================== reproducibility
def test_build_is_deterministic_and_order_independent(tables, sv):
    a, _ = of.build_week(tables, sv, 4)
    b, _ = of.build_week(make_tables(), make_vle(BASE_CLICKS, make_tables()), 4)
    pd.testing.assert_frame_equal(a, b)
    assert bd.content_hash(a) == bd.content_hash(b)
    t3 = make_tables()
    t3 = {k: v.sample(frac=1, random_state=1).reset_index(drop=True) for k, v in t3.items()}
    c, _ = of.build_week(t3, make_vle(list(reversed(BASE_CLICKS)), t3), 4)
    pd.testing.assert_frame_equal(a, c)


# =============================================================================== UCI
@pytest.mark.skipif(not HAS_UCI, reason="UCI not downloaded")
def test_uci_admission_dataset():
    sup, enr, rep = uci_admission.build_uci_admission(loaders.load_uci())
    assert len(sup) == 3630 and len(enr) == 794
    assert set(sup["original_target"]) == {"Dropout", "Graduate"}
    assert sup["target_dropout"].sum() == 1421
    assert enr["target_dropout"].isna().all()
    feats = set(catalog.UCI_ADMISSION_FEATURES)
    assert not feats & (set(catalog.UCI_PROHIBITED) | set(catalog.UCI_QUARANTINED))
    assert not any("Curricular units" in c for c in feats)
    assert "Scholarship holder" not in feats and "Tuition fees up to date" not in feats
    assert sup["row_id"].is_unique


def test_uci_guard_rejects_forbidden_features(monkeypatch):
    monkeypatch.setattr(catalog, "UCI_ADMISSION_FEATURES", [*catalog.UCI_ADMISSION_FEATURES, "Debtor"])
    with pytest.raises(ValueError):
        uci_admission.build_uci_admission(pd.DataFrame())


# =============================================================================== real data
@pytest.mark.skipif(not HAS_OULAD, reason="OULAD not downloaded")
def test_real_data_invariance_on_one_presentation():
    t = loaders.load_oulad()
    key = {"code_module": "AAA", "code_presentation": "2013J"}
    sub = {n: (d.loc[(d["code_module"] == "AAA") & (d["code_presentation"] == "2013J")]
               if "code_module" in d.columns else d) for n, d in t.items()}
    ids = sub["assessments"]["id_assessment"]
    sub["studentAssessment"] = t["studentAssessment"].loc[t["studentAssessment"]["id_assessment"].isin(ids)]
    sv = loaders.load_student_vle()
    sv = sv.loc[(sv["code_module"] == "AAA") & (sv["code_presentation"] == "2013J")]
    sv_typed = of.prepare_vle(sv, sub["vle"])
    before = features_of(of.build_week(sub, sv_typed, 4)[0], 4)
    noise = sv.loc[sv["date"] <= 27].sample(2000, random_state=0).copy()
    noise["date"] = (noise["date"].clip(lower=0) + 28).astype("int16")
    noise["sum_click"] = 777
    after = features_of(of.build_week(sub, of.prepare_vle(pd.concat([sv, noise]), sub["vle"]), 4)[0], 4)
    pd.testing.assert_frame_equal(before, after)
    assert len(before) > 300 and key


@pytest.mark.skipif(not HAS_PROCESSED, reason="processed datasets not built")
@pytest.mark.parametrize("week", catalog.WEEKS)
def test_processed_dataset_matches_manifest_and_rules(week):
    name = f"oulad_week_{week:02d}"
    path = PROCESSED / "oulad" / f"week_{week:02d}" / f"{name}.parquet"
    manifest = json.loads((PROCESSED / "oulad" / "metadata" / f"{name}.manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((PROCESSED / "oulad" / "metadata" / f"{name}.metadata.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(path)
    assert sha256_of(path) == manifest["output"]["file_sha256"]
    assert bd.content_hash(df) == manifest["output"]["content_sha256"]
    assert manifest["n_rows"] == len(df) and manifest["n_features"] == len(meta["feature_columns"])
    assert meta["feature_columns"] == catalog.feature_names_for_week(week)
    cut = week * 7 - 1
    assert (df["cutoff_day"] == cut).all()
    assert (df["max_vle_day_used"].dropna() <= cut).all()
    assert (df["max_submission_day_used"].dropna() <= cut).all()
    assert (df["date_unregistration"].isna() | (df["date_unregistration"] > cut)).all()
    assert (df["date_registration"].isna() | (df["date_registration"] <= cut)).all()
    assert not df.duplicated(["student_id", "presentation_id"]).any()
    assert set(df["target_withdrawn"]) == {0, 1} and set(df["target_adverse_outcome"]) == {0, 1}
    assert ((df["target_withdrawn"] == 1) <= (df["target_adverse_outcome"] == 1)).all()
    assert (df.loc[df["target_withdrawn"] == 1, "final_result"] == "Withdrawn").all()


@pytest.mark.skipif(not (HAS_PROCESSED and HAS_OULAD), reason="processed datasets not built")
def test_processed_week2_is_reproducible_from_raw():
    t = loaders.load_oulad()
    sv = of.prepare_vle(loaders.load_student_vle(), t["vle"])
    rebuilt, _ = of.build_week(t, sv, 2)
    stored = pd.read_parquet(PROCESSED / "oulad" / "week_02" / "oulad_week_02.parquet")
    pd.testing.assert_frame_equal(rebuilt, stored)


@pytest.mark.skipif(not (PROCESSED / "uci" / "admission" / "uci_admission.parquet").exists(),
                    reason="UCI processed dataset not built")
def test_processed_uci_matches_manifest():
    base = PROCESSED / "uci" / "admission"
    manifest = json.loads((base / "uci_admission.manifest.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(base / "uci_admission.parquet")
    assert sha256_of(base / "uci_admission.parquet") == manifest["output"]["file_sha256"]
    assert len(df) == manifest["n_rows"] == 3630
    assert list(df.columns) == [*catalog.UCI_ID_COLUMNS, *catalog.UCI_ADMISSION_FEATURES,
                                *catalog.UCI_TARGET_COLUMNS, *catalog.UCI_AUDIT_COLUMNS]
