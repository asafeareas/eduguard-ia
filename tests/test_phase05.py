"""Phase 05 tests: Q17 predictor exclusion, student-split integrity (real data), compatibility
gate, external evaluation without refit, PSI, data contract, threshold helpers and the new
fairness diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import catalog
from src.modeling import fairness, preprocessing, splitting, thresholds
from src.modeling.train import run_oulad_experiment
from src.validation import data_contract, external, feature_mapping
from tests.test_phase04 import META, synthetic

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
V2 = ROOT / "reports" / "modeling_v2"
HAS_DATA = (PROCESSED / "oulad" / "week_04" / "oulad_week_04.parquet").exists()


# =============================================================================== Q17
def test_v2_predictor_set_excludes_imd_gender_disability():
    for week in catalog.WEEKS:
        meta = {"feature_columns": catalog.feature_names_for_week(week)}
        v1 = preprocessing.predictor_columns(meta, drop=preprocessing.MODEL_VERSIONS["phase04-v1"])
        v2 = preprocessing.predictor_columns(meta, drop=preprocessing.MODEL_VERSIONS["phase05-v2"])
        assert set(v1) - set(v2) == {"adm_imd_band"}
        assert not {"adm_imd_band", "adm_gender", "adm_disability"} & set(v2)
        assert len(v2) == len(catalog.feature_names_for_week(week)) - 3
    assert preprocessing.CURRENT_MODEL_VERSION == "phase05-v2"


def test_v2_experiment_never_uses_imd_but_audits_it(tmp_path):
    r = run_oulad_experiment(4, "target_withdrawn", extras=True, interpret=True, out_dir=tmp_path,
                             df=synthetic(), meta=META, n_boot=20, model_version="phase05-v2")
    assert "adm_imd_band" not in r["experiment"]["predictors"]
    assert r["experiment"]["audit_only_features"] == ["adm_disability", "adm_gender", "adm_imd_band"]
    coefs = {c["feature"] for c in r["interpretability"]["logistic_regression"]["coefficients"]}
    assert not any("imd" in c for c in coefs)
    assert {x["level"] for x in r["fairness"]["logistic_regression"]["imd_group"]}  # still audited
    assert r["without_imd_band"] == {}  # comparison only meaningful when IMD is a predictor


# =============================================================================== split integrity (real data)
@pytest.mark.skipif(not HAS_DATA, reason="processed datasets not built")
@pytest.mark.parametrize("week", catalog.WEEKS)
def test_real_split_is_by_student_and_counts_reconcile(week):
    df = pd.read_parquet(PROCESSED / "oulad" / f"week_{week:02d}" / f"oulad_week_{week:02d}.parquet")
    sp = splitting.oulad_split(df, ["target_withdrawn"])
    parts = [sp.train, sp.validation, sp.calibration]
    pool = df[df["presentation_year"] == 2013]
    assert sum(len(p) for p in parts) == len(pool)                                   # records
    assert sum(p["student_id"].nunique() for p in parts) == pool["student_id"].nunique()  # students
    lab = pd.concat([p[["student_id"]].assign(k=i) for i, p in enumerate(parts)])
    assert (lab.groupby("student_id")["k"].nunique() == 1).all()                      # never split
    assert not set(sp.test["student_id"]) & set(pool["student_id"])


# =============================================================================== compatibility / mapping
def test_every_model_feature_has_a_mapping():
    names = {f.name for f in catalog.OULAD_FEATURES if f.status == "included"}
    mapped = set(feature_mapping.by_feature())
    assert names <= mapped, names - mapped
    for m in feature_mapping.MAPPINGS:
        assert m.status in {"direct", "transform", "no_equivalent", "excluded"}


def test_direct_transfer_impossible_for_current_model():
    meta = {"feature_columns": catalog.feature_names_for_week(4)}
    preds = preprocessing.predictor_columns(meta, drop=preprocessing.MODEL_VERSIONS["phase05-v2"])
    everything = set(data_contract.REQUIRED) | set(data_contract.OPTIONAL)
    rep = external.compatibility_report(preds, everything)
    assert rep["direct_transfer_possible"] is False
    assert rep["status_counts"].get("direct", 0) <= 1
    ufrn = {c for cols in feature_mapping.UFRN_OPEN_DATA_FIELDS.values() for c in cols}
    assert external.compatibility_report(preds, ufrn)["direct_transfer_possible"] is False


def test_compatibility_allows_only_fully_direct_models():
    rep = external.compatibility_report(["adm_age_band"], {"birth_year"})
    assert rep["direct_transfer_possible"] is True
    assert external.compatibility_report(["adm_age_band"], set())["direct_transfer_possible"] is False
    assert external.compatibility_report(["adm_age_band", "eng_total_clicks"],
                                         {"birth_year", "ava_log_timestamp", "ava_log_student"})["direct_transfer_possible"] is False


# =============================================================================== external evaluation
def test_external_evaluation_applies_model_without_refit(tmp_path):
    df = synthetic()
    r = run_oulad_experiment(4, "target_withdrawn", out_dir=tmp_path, df=df, meta=META, n_boot=10,
                             model_version="phase05-v2")
    import joblib
    bundle = joblib.load(tmp_path / "models.joblib")
    ext = synthetic(n_students=300, seed=9).assign(course=lambda d: d["code_module"])
    with pytest.raises(ValueError):
        external.evaluate_external(bundle, "logistic_regression", ext, "target_withdrawn")
    before = bundle["models"]["logistic_regression"].named_steps["clf"].coef_.copy()
    res = external.evaluate_external(bundle, "logistic_regression", ext, "target_withdrawn",
                                     group_cols=["course"], reference=df, feature_types=META["feature_types"],
                                     require_compatibility=False)
    assert np.array_equal(before, bundle["models"]["logistic_regression"].named_steps["clf"].coef_)
    assert res["threshold_from_development"] == r["calibration_subset"]["thresholds"]["logistic_regression"]
    assert {g["level"] for g in res["by_course"]} == set(ext["course"])
    assert all("psi" in s or "unseen_category_share" in s for s in res["distribution_shift"])


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert external.population_stability_index(a, rng.normal(0, 1, 5000)) < 0.02
    assert external.population_stability_index(a, rng.normal(1.5, 1, 5000)) > 0.25


# =============================================================================== data contract
def test_data_contract_validation():
    good = pd.DataFrame({k: ["x"] for k in data_contract.REQUIRED}).assign(
        term="2019.1", final_status="CANCELADO", final_status_date="2019-04-01")
    assert data_contract.validate(good)["valid"]
    bad = good.drop(columns=["term"]).assign(cpf="000", final_status="ABANDONO", final_status_date=None)
    rep = data_contract.validate(bad)
    types = {p["type"] for p in rep["problems"]}
    assert {"missing_required", "forbidden_personal_data", "invalid_final_status"} <= types
    undated = good.assign(final_status_date=None)
    assert "cancellation_without_date" in {p["type"] for p in data_contract.validate(undated)["problems"]}


# =============================================================================== thresholds & fairness helpers
def test_threshold_helpers():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.7).astype(int)
    t = thresholds.capacity_threshold(p, 0.10)
    assert (p >= t).sum() == 10
    assert thresholds.cost_optimal_threshold(cost_fn=4, cost_fp=1) == pytest.approx(0.2)
    c = thresholds.consequences(y, p, 0.5, effectiveness=0.5)
    assert c["true_alerts"] + c["missed"] == y.sum() and c["expected_prevented"] == c["true_alerts"] * 0.5
    assert thresholds.net_benefit(y, p, 0.69) > 0


def test_calibration_gap_and_standardisation():
    df = pd.DataFrame({"student_id": range(400), "adm_disability": [1] * 100 + [0] * 300,
                       "code_module": (["A", "B"] * 200)})
    y = np.r_[np.ones(40), np.zeros(60), np.ones(60), np.zeros(240)]
    p = np.r_[np.full(100, 0.2), np.full(300, 0.2)]
    gaps = {g["level"]: g for g in fairness.calibration_gap(df, y, p, "disability", n_boot=200)}
    assert gaps["Y"]["gap"] == pytest.approx(0.2 - 0.4) and gaps["Y"]["gap_ci95"][1] < 0
    assert gaps["N"]["gap"] == pytest.approx(0.0)
    std = fairness.module_standardised_gap(df, y, p, "disability")
    assert std["Y"]["crude_gap"] == pytest.approx(-0.2)
    assert fairness.minimum_detectable_difference(300, 2000, 0.5) > fairness.minimum_detectable_difference(3000, 20000, 0.5)


# =============================================================================== real v2 results
def _v2(week, target="target_withdrawn"):
    p = V2 / "oulad" / f"week_{week:02d}" / target / "results.json"
    if not p.exists():
        pytest.skip("v2 reports not generated")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("week", catalog.WEEKS)
@pytest.mark.parametrize("target", ["target_withdrawn", "target_adverse_outcome"])
def test_real_v2_results_use_v2_predictors(week, target):
    r = _v2(week, target)
    expected = preprocessing.predictor_columns({"feature_columns": catalog.feature_names_for_week(week)},
                                               drop=preprocessing.MODEL_VERSIONS["phase05-v2"])
    assert r["experiment"]["model_version"] == "phase05-v2"
    assert r["experiment"]["predictors"] == expected
    manifest = json.loads((PROCESSED / "oulad" / "metadata" / f"oulad_week_{week:02d}.manifest.json").read_text(encoding="utf-8"))
    assert r["provenance"]["content_sha256"] == manifest["output"]["content_sha256"]
