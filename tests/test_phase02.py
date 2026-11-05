"""Phase 02 tests: temporal cut-offs, aggregation correctness, leakage catalog,
data integrity on the real files (skipped when the data is not downloaded) and
raw-data immutability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import eda, leakage
from src.data import download, loaders, quality
from src.feature_engineering import temporal

ROOT = Path(__file__).resolve().parents[1]
K = {"code_module": "AAA", "code_presentation": "2013J"}
HAS_OULAD = (loaders.OULAD_DIR / "studentVle.csv").exists()
HAS_UCI = loaders.UCI_CSV.exists()


# --------------------------------------------------------------------------- fixtures
def _vle(rows):
    df = pd.DataFrame([{**K, "id_student": s, "id_site": site, "date": d, "sum_click": c}
                       for s, site, d, c in rows])
    return df.astype({"code_module": "category", "code_presentation": "category", "date": "int16"})


@pytest.fixture
def tables():
    info = pd.DataFrame([
        {**K, "id_student": 1, "final_result": "Withdrawn"},   # left before start
        {**K, "id_student": 2, "final_result": "Withdrawn"},   # leaves on day 20
        {**K, "id_student": 3, "final_result": "Pass"},
        {**K, "id_student": 4, "final_result": "Pass"},        # registers late (day 30)
        {**K, "id_student": 5, "final_result": "Fail"},        # registration date unknown
    ])
    reg = pd.DataFrame([
        {**K, "id_student": 1, "date_registration": -40, "date_unregistration": -3},
        {**K, "id_student": 2, "date_registration": -40, "date_unregistration": 20},
        {**K, "id_student": 3, "date_registration": -10, "date_unregistration": np.nan},
        {**K, "id_student": 4, "date_registration": 30, "date_unregistration": np.nan},
        {**K, "id_student": 5, "date_registration": np.nan, "date_unregistration": np.nan},
    ])
    courses = pd.DataFrame([{**K, "module_presentation_length": 260}])
    assessments = pd.DataFrame([
        {**K, "id_assessment": 10, "assessment_type": "TMA", "date": 19, "weight": 10},
        {**K, "id_assessment": 11, "assessment_type": "TMA", "date": 40, "weight": 10},
        {**K, "id_assessment": 12, "assessment_type": "Exam", "date": 250, "weight": 100},
    ])
    sa = pd.DataFrame([
        {"id_assessment": 10, "id_student": 3, "date_submitted": 18, "is_banked": 0, "score": 80},
        {"id_assessment": 11, "id_student": 3, "date_submitted": 39, "is_banked": 0, "score": 60},
        {"id_assessment": 10, "id_student": 2, "date_submitted": 22, "is_banked": 0, "score": 50},  # late
        {"id_assessment": 10, "id_student": 5, "date_submitted": 5, "is_banked": 1, "score": 99},   # banked
    ])
    return {"studentInfo": info, "studentRegistration": reg, "courses": courses,
            "assessments": assessments, "studentAssessment": sa}


# --------------------------------------------------------------------------- temporal
def test_cutoff_day_for_week():
    assert [temporal.cutoff_day_for_week(w) for w in (1, 2, 4, 6)] == [6, 13, 27, 41]
    with pytest.raises(ValueError):
        temporal.cutoff_day_for_week(0)


def test_week_index_boundaries():
    days = pd.Series([-8, -1, 0, 6, 7, 13, 14])
    assert temporal.week_index(days).tolist() == [-1, 0, 1, 1, 2, 2, 3]


def test_population_at_cutoff_rules(tables):
    base = temporal.student_base(tables)
    ids = lambda c: set(temporal.population_at_cutoff(base, c)["id_student"])  # noqa: E731
    assert ids(13) == {2, 3, 5}        # 1 already left, 4 not registered yet
    assert ids(20) == {3, 5}           # 2 leaves ON day 20 -> excluded at cutoff 20
    assert ids(41) == {3, 4, 5}


def test_vle_as_of_ignores_future_and_counts_correctly():
    sv = _vle([(1, 100, -2, 5), (1, 101, 0, 3), (1, 100, 10, 4), (1, 102, 13, 2),  # <= 13
               (1, 100, 14, 1000), (1, 100, 200, 1000)])                          # future
    out = temporal.vle_as_of(sv, 13).set_index("id_student").loc[1]
    assert out["total_clicks"] == 14
    assert out["active_days"] == 4
    assert out["distinct_sites"] == 3
    assert out["clicks_pre_start"] == 5
    assert out["last_active_day"] == 13 and out["days_since_last_activity"] == 0
    assert out["clicks_last_7_days"] == 6   # days 7..13
    assert out["clicks_prev_7_days"] == 3   # days 0..6


def test_vle_as_of_is_invariant_to_events_after_cutoff():
    past = [(1, 100, 1, 3), (2, 100, 5, 7), (2, 101, 13, 1)]
    a = temporal.vle_as_of(_vle(past), 13)
    b = temporal.vle_as_of(_vle(past + [(1, 100, 14, 999), (2, 100, 100, 999), (3, 100, 50, 1)]), 13)
    pd.testing.assert_frame_equal(a, b)


def test_assessments_as_of_respects_cutoff_and_banked(tables):
    out = temporal.assessments_as_of(tables["studentAssessment"], tables["assessments"], 27)
    out = out.set_index("id_student")
    assert set(out.index) == {2, 3}                       # 5 is banked -> excluded
    assert out.loc[3, "n_submitted"] == 1 and out.loc[3, "mean_score"] == 80   # day-39 one excluded
    assert out.loc[2, "n_late"] == 1


def test_snapshot_uses_nothing_after_cutoff(tables):
    sv = _vle([(2, 100, 3, 10), (3, 100, 12, 4), (3, 100, 30, 500), (5, 100, 40, 9)])
    s = temporal.snapshot(tables, sv, 13).set_index("id_student")
    assert set(s.index) == {2, 3, 5}
    assert s.loc[3, "total_clicks"] == 4
    assert s.loc[5, "total_clicks"] == 0                  # its only click is on day 40
    assert (s["n_submitted"] == 0).all()                  # first submission is on day 18
    assert (s["n_due"] == 0).all()
    s27 = temporal.snapshot(tables, sv, 27).set_index("id_student")
    assert s27.loc[3, "n_missing_submissions"] == 0 and s27.loc[5, "n_missing_submissions"] == 1


def test_weekly_clicks_aggregation():
    sv = _vle([(1, 100, -1, 2), (1, 101, 0, 3), (1, 100, 6, 4), (1, 100, 7, 5)])
    w = temporal.weekly_clicks(sv).set_index("week")["sum_click"].to_dict()
    assert w == {0: 2, 1: 7, 2: 5}


# --------------------------------------------------------------------------- eda helpers
def test_univariate_auc_extremes_and_alignment():
    y = pd.Series([0, 0, 1, 1])
    assert eda.univariate_auc(pd.Series([1, 2, 3, 4]), y) == 1.0
    assert eda.univariate_auc(pd.Series([4, 3, 2, 1]), y) == 0.0
    assert eda.univariate_auc(pd.Series([1, 1, 1, 1]), y) == 0.5
    assert eda.separation(0.0) == 1.0
    with pytest.raises(ValueError):
        eda.univariate_auc(pd.Series([1, 2, 3, 4], index=[9, 8, 7, 6]), y)


def test_cramers_v_and_rates():
    x = pd.Series(list("aabb"))
    assert eda.cramers_v(x, pd.Series(list("ccdd"))) == pytest.approx(1.0)
    r = eda.rate_by_category(pd.DataFrame({"g": list("aabb")}), "g", pd.Series([1, 0, 1, 1]))
    assert r.loc["b", "rate"] == 1.0 and r.loc["a", "rate"] == 0.5


def test_mean_weekly_clicks_excludes_students_after_withdrawal(tables):
    base = temporal.student_base(tables)
    weekly = pd.DataFrame([{**K, "id_student": 2, "week": 1, "sum_click": 10},
                           {**K, "id_student": 3, "week": 5, "sum_click": 8}])
    out = eda.mean_weekly_clicks_by_outcome(weekly, base, range(1, 6)).set_index(["final_result", "week"])
    # student 2 leaves on day 20 (week 3): contributes to weeks 1-3 only, student 1 to none (left day -3)
    assert out.loc[("Withdrawn", 1), "size"] == 1 and out.loc[("Withdrawn", 1), "mean"] == 10
    assert ("Withdrawn", 4) not in out.index
    assert out.loc[("Pass", 5), "mean"] == 4  # (8 + 0) / 2 students


# --------------------------------------------------------------------------- quality
def test_orphan_count_and_key_uniqueness():
    parent = pd.DataFrame({"k": [1, 2]})
    assert quality.orphan_count(pd.DataFrame({"k": [1, 3, 3]}), parent, ["k"]) == 2
    assert not quality.key_is_unique(pd.DataFrame({"k": [1, 1]}), ["k"])["unique"]


# --------------------------------------------------------------------------- leakage catalog
def test_uci_catalog_covers_every_column_exactly_once():
    names = [e["variable"] for e in leakage.UCI_CATALOG]
    assert len(names) == len(set(names)) == 37
    valid = {leakage.PERMITTED, leakage.CONDITIONAL, leakage.PROHIBITED}
    for e in leakage.UCI_CATALOG + leakage.OULAD_CATALOG:
        assert e.get("class", e.get("class_sem1_weeks_2_6")) in valid
        assert e["justification"]
    if HAS_UCI:
        assert set(names) == set(loaders.load_uci().columns)


def test_end_of_semester_variables_prohibited_for_first_semester():
    cat = {e["variable"]: e for e in leakage.UCI_CATALOG}
    for v in ("approved", "grade", "evaluations", "without evaluations"):
        assert cat[f"Curricular units 1st sem ({v})"]["class_sem1_weeks_2_6"] == leakage.PROHIBITED
        assert cat[f"Curricular units 2nd sem ({v})"]["class_sem2_weeks_2_6"] == leakage.PROHIBITED
    assert cat["Target"]["class_sem1_weeks_2_6"] == leakage.PROHIBITED


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not HAS_OULAD, reason="OULAD not downloaded")
def test_oulad_real_integrity():
    t = loaders.load_oulad()
    rep = quality.oulad_integrity(t)
    assert all(v == 0 for v in rep["orphans"].values()), rep["orphans"]
    for name in ("courses", "assessments", "vle", "studentInfo", "studentRegistration", "studentAssessment"):
        assert rep["granularity"][name]["unique"], name
    assert len(t["studentInfo"]) == 32593


@pytest.mark.skipif(not HAS_OULAD, reason="OULAD not downloaded")
def test_oulad_real_snapshot_never_uses_future():
    t = loaders.load_oulad()
    sv = loaders.load_student_vle()
    for week in (2, 4, 6):
        cut = temporal.cutoff_day_for_week(week)
        s = temporal.snapshot(t, sv, cut)
        assert (s["last_active_day"].dropna() <= cut).all()
        assert (s["date_unregistration"].isna() | (s["date_unregistration"] > cut)).all()
        assert (s["date_registration"].isna() | (s["date_registration"] <= cut)).all()


@pytest.mark.skipif(not HAS_UCI, reason="UCI not downloaded")
def test_uci_real_consistency():
    rep = quality.uci_consistency(loaders.load_uci())
    assert rep["duplicate_rows"] == 0 and rep["missing_cells"] == 0
    for sem in ("1st", "2nd"):
        assert rep[sem]["approved_gt_enrolled"] == 0


@pytest.mark.parametrize("source_id", ["uci_dropout_697", "oulad_349"])
def test_raw_files_unchanged_since_download(source_id):
    manifest = ROOT / "data" / "raw" / source_id / f"{source_id}.source.json"
    if not manifest.exists():
        pytest.skip(f"{source_id} not downloaded")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    for f in m["files"]:
        path = manifest.parent / f["path"]
        assert path.stat().st_size == f["size_bytes"], f["path"]
        assert download.sha256_of(path) == f["sha256"], f["path"]
