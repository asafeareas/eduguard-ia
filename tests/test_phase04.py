"""Phase 04 tests: splits, predictor matrix, leakage-safe preprocessing, metrics,
baselines, calibration without test data, determinism, cutoff separation,
OULAD/UCI independence and consistency between manifests and results.

Synthetic experiments write to tmp_path; real-result tests are skipped when the
reports have not been generated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.feature_engineering import catalog
from src.modeling import baselines, calibration, evaluation, preprocessing, splitting
from src.modeling.train import make_hgb, make_lr, run_oulad_experiment

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "modeling"
PROCESSED = ROOT / "data" / "processed"


# =============================================================================== synthetic data
META = {
    "dataset": "synthetic",
    "feature_columns": ["eng_total_clicks", "eng_days_since_last_activity", "asm_submission_rate",
                        "adm_imd_band", "adm_registration_lead_days", "ctx_code_module", "adm_age_band",
                        "adm_gender", "adm_disability"],
    "feature_types": {"eng_total_clicks": "count", "eng_days_since_last_activity": "numeric",
                      "asm_submission_rate": "numeric", "adm_imd_band": "ordinal",
                      "adm_registration_lead_days": "numeric", "ctx_code_module": "nominal",
                      "adm_age_band": "ordinal", "adm_gender": "nominal", "adm_disability": "binary"},
    "feature_groups": {"eng_total_clicks": "engagement", "eng_days_since_last_activity": "engagement",
                       "asm_submission_rate": "assessment", "adm_imd_band": "admission",
                       "adm_registration_lead_days": "admission", "ctx_code_module": "context",
                       "adm_age_band": "admission", "adm_gender": "admission", "adm_disability": "admission"},
    "target_columns": ["target_withdrawn", "target_adverse_outcome"],
    "id_columns": ["student_id", "presentation_id", "code_module", "code_presentation"],
    "audit_columns": ["presentation_year", "final_result"],
}


def synthetic(n_students: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_students):
        years = [2013] if s < 400 else [2014] if s < 800 else [2013, 2014]  # last 100 recurrent
        for year in years:
            module = rng.choice(["AAA", "BBB", "CCC"])
            clicks = rng.poisson(80)
            inactive = rng.integers(0, 28)
            risk = 1 / (1 + np.exp(-(-1.0 - 0.02 * (clicks - 80) + 0.08 * (inactive - 14))))
            wd = int(rng.random() < risk)
            fail = int(not wd and rng.random() < 0.3)
            rows.append({
                "student_id": s, "presentation_id": f"{module}_{year}J", "code_module": module,
                "code_presentation": f"{year}J", "presentation_year": year,
                "eng_total_clicks": clicks, "eng_days_since_last_activity": float(inactive),
                "asm_submission_rate": np.nan if rng.random() < 0.2 else float(rng.random()),
                "adm_imd_band": np.nan if rng.random() < 0.05 else float(rng.integers(0, 10)),
                "adm_registration_lead_days": np.nan if rng.random() < 0.02 else float(rng.integers(-10, 200)),
                "ctx_code_module": module, "adm_age_band": float(rng.integers(0, 3)),
                "adm_gender": rng.choice(["M", "F"]), "adm_disability": int(rng.random() < 0.1),
                "target_withdrawn": wd, "target_adverse_outcome": int(wd or fail),
                "final_result": "Withdrawn" if wd else ("Fail" if fail else "Pass"),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df():
    return synthetic()


@pytest.fixture(scope="module")
def run_a(df, tmp_path_factory):
    return run_oulad_experiment(4, "target_withdrawn", extras=True, interpret=True,
                                out_dir=tmp_path_factory.mktemp("a"), df=df, meta=META, n_boot=50)


# =============================================================================== splitting
def test_temporal_split_and_recurrent_exclusion(df):
    sp = splitting.oulad_split(df, META["target_columns"])
    for part in (sp.train, sp.validation, sp.calibration):
        assert (part["presentation_year"] == 2013).all()
    assert (sp.test["presentation_year"] == 2014).all()
    s2013 = set(df.loc[df["presentation_year"] == 2013, "student_id"])
    assert not set(sp.test["student_id"]) & s2013, "recurrent students must not be in the main test"
    assert set(sp.test_recurrent["student_id"]) <= s2013
    a = sp.audit
    assert a["students_in_both_years"] == 100
    assert a["rows_2014"] == len(sp.test) + a["rows_excluded_from_main_test"]
    assert a["rows_2013"] == len(sp.train) + len(sp.validation) + len(sp.calibration)


def test_internal_split_is_by_student(df):
    sp = splitting.oulad_split(df, META["target_columns"])
    ids = [set(p["student_id"]) for p in (sp.train, sp.validation, sp.calibration, sp.test)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not ids[i] & ids[j]
    with pytest.raises(AssertionError):
        splitting.check_disjoint({"train": sp.train, "validation": sp.train.head(5)})


# =============================================================================== predictor matrix
def test_predictors_exclude_ids_targets_audit_and_sensitive():
    cols = preprocessing.predictor_columns(META)
    assert "adm_gender" not in cols and "adm_disability" not in cols
    forbidden = set(META["id_columns"]) | set(META["target_columns"]) | set(META["audit_columns"])
    assert not forbidden & set(cols)
    for week in catalog.WEEKS:
        meta = {"feature_columns": catalog.feature_names_for_week(week)}
        real = preprocessing.predictor_columns(meta)
        assert "adm_gender" not in real and "adm_disability" not in real
        assert "adm_imd_band" in real  # decision: deprivation band stays in the main model


def test_no_missing_indicator_for_quarantined_registration_missingness(df):
    cols = preprocessing.predictor_columns(META)
    prep = preprocessing.build_preprocessor(META, cols, scale=True).fit(df[cols])
    names = list(prep.get_feature_names_out())
    assert any("missingindicator_asm_submission_rate" in n for n in names)
    assert not any("missingindicator_adm_registration_lead_days" in n for n in names)


def test_imputation_fitted_on_train_only(df):
    cols = preprocessing.predictor_columns(META)
    train, test = df.iloc[:500], df.iloc[500:].copy()
    pipe = make_lr(META, cols).fit(train[cols], train["target_withdrawn"])
    imp = pipe.named_steps["prep"].named_transformers_["num_ind"].named_steps["impute"]
    expected = train[["asm_submission_rate", "adm_imd_band"]].median()
    # column order inside the transformer follows the predictor list
    order = preprocessing.column_groups(META, cols)["numeric_with_indicator"]
    assert np.allclose(imp.statistics_, expected[order].to_numpy())
    before = pipe.predict_proba(test[cols])[:, 1]
    test["asm_submission_rate"] = 999.0  # transforming the test must not refit anything
    pipe.predict_proba(test[cols])
    assert np.allclose(imp.statistics_, expected[order].to_numpy())
    assert len(before) == len(test)


def test_unknown_category_does_not_break(df):
    cols = preprocessing.predictor_columns(META)
    pipe = make_hgb(META, cols, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, max_iter=20)
    pipe.fit(df[cols], df["target_withdrawn"])
    new = df[cols].head(5).assign(ctx_code_module="ZZZ")
    p = pipe.predict_proba(new)[:, 1]
    assert np.isfinite(p).all()


# =============================================================================== metrics
def test_metrics_are_computed_correctly():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    t = evaluation.top_fraction(y, p, 0.2)
    assert t == {"fraction": 0.2, "group_size": 2, "positives_in_group": 1, "positives_total": 3,
                 "recall": 1 / 3, "precision": 0.5}
    m = evaluation.threshold_metrics(y, p, 0.65)
    assert m["confusion_matrix"] == {"tn": 6, "fp": 1, "fn": 1, "tp": 2}
    assert m["recall"] == pytest.approx(2 / 3) and m["false_positive_rate"] == pytest.approx(1 / 7)
    pm = evaluation.probability_metrics(y, p)
    assert pm["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert pm["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_bootstrap_ci_brackets_observed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo, hi = evaluation.bootstrap_ci(y, p, np.arange(400), n_boot=200)
    assert lo <= average_precision_score(y, p) <= hi


# =============================================================================== baselines
def test_module_rate_uses_training_only_with_fallback():
    X = pd.DataFrame({"ctx_code_module": ["A"] * 40 + ["B"] * 5})
    y = np.array([1] * 10 + [0] * 30 + [1] * 5)
    b = baselines.ModuleRateBaseline(min_count=30).fit(X, y)
    p = b.predict_proba(pd.DataFrame({"ctx_code_module": ["A", "B", "Z"]}))[:, 1]
    assert p[0] == pytest.approx(0.25)
    assert p[1] == pytest.approx(15 / 45) and p[2] == pytest.approx(15 / 45)  # fallback to global


def test_inactivity_rule_selects_n_on_validation():
    Xtr = pd.DataFrame({"eng_days_since_last_activity": [0, 1, 2, 10, 12, 15] * 10})
    ytr = np.array([0, 0, 0, 1, 1, 1] * 10)
    Xv = pd.DataFrame({"eng_days_since_last_activity": [0, 3, 5, 9, 11, 20]})
    yv = np.array([0, 0, 0, 1, 1, 1])
    rule, scores = baselines.InactivityRuleBaseline.select_n(Xtr, ytr, Xv, yv, range(1, 15))
    assert rule.n_days in {6, 7, 8, 9} and scores[rule.n_days] == 1.0


# =============================================================================== calibration
def test_calibrator_methods():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(int)  # miscalibrated on purpose
    for method in ("sigmoid", "isotonic"):
        c = calibration.Calibrator(method).fit(p, y)
        q = c.transform(np.sort(p))
        assert np.all(np.diff(q) >= -1e-12)
    cal, info = calibration.select_calibrator(p, y, np.arange(2000))
    assert info["selected"] in {"sigmoid", "isotonic"}
    assert info["oof_brier"][info["selected"]] < info["oof_brier"]["none"]


def test_test_labels_never_influence_learning(df, tmp_path):
    """Shuffling the 2014 TEST labels must not change selection, calibrators, thresholds or bands."""
    a = run_oulad_experiment(4, "target_withdrawn", extras=True, out_dir=tmp_path / "a", df=df, meta=META, n_boot=20)
    shuffled = df.copy()
    m = shuffled["presentation_year"] == 2014
    rng = np.random.default_rng(7)
    shuffled.loc[m, "target_withdrawn"] = rng.permutation(shuffled.loc[m, "target_withdrawn"].to_numpy())
    b = run_oulad_experiment(4, "target_withdrawn", extras=True, out_dir=tmp_path / "b", df=shuffled, meta=META,
                             n_boot=20)
    assert a["selection"]["logistic_regression"]["selected"] == b["selection"]["logistic_regression"]["selected"]
    assert a["selection"]["gradient_boosting"]["selected"] == b["selection"]["gradient_boosting"]["selected"]
    assert a["calibration_subset"] == b["calibration_subset"]
    for n in ("logistic_regression", "gradient_boosting"):
        assert a["risk_bands"][n]["low_upper"] == b["risk_bands"][n]["low_upper"]
        assert a["risk_bands"][n]["high_lower"] == b["risk_bands"][n]["high_lower"]
    pd.testing.assert_series_equal(a["_predictions"]["p_gradient_boosting"], b["_predictions"]["p_gradient_boosting"])


# =============================================================================== determinism & outputs
def test_experiment_is_deterministic(df, run_a, tmp_path):
    b = run_oulad_experiment(4, "target_withdrawn", extras=True, interpret=True, out_dir=tmp_path, df=df,
                             meta=META, n_boot=50)
    for m in run_a["test_main"]:
        assert run_a["test_main"][m]["pr_auc"] == b["test_main"][m]["pr_auc"]
        assert run_a["test_main"][m]["brier"] == b["test_main"][m]["brier"]
    pd.testing.assert_frame_equal(run_a["_predictions"], b["_predictions"])


def test_experiment_outputs_respect_q16(run_a):
    assert "adm_gender" not in run_a["experiment"]["predictors"]
    assert "adm_disability" not in run_a["experiment"]["predictors"]
    coefs = {c["feature"] for c in run_a["interpretability"]["logistic_regression"]["coefficients"]}
    assert not any("gender" in c or "disability" in c for c in coefs)
    assert set(run_a["fairness"]["gradient_boosting"]) == {"gender", "disability", "imd_group", "age_group"}
    assert "test_recurrent_secondary" in run_a and "without_imd_band" in run_a


# =============================================================================== real reports
def _results(week, target="target_withdrawn"):
    path = REPORTS / "oulad" / f"week_{week:02d}" / target / "results.json"
    if not path.exists():
        pytest.skip("Phase 04 reports not generated")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("week", catalog.WEEKS)
@pytest.mark.parametrize("target", ["target_withdrawn", "target_adverse_outcome"])
def test_real_results_consistent_with_manifest_and_cutoff(week, target):
    r = _results(week, target)
    manifest = json.loads((PROCESSED / "oulad" / "metadata" / f"oulad_week_{week:02d}.manifest.json")
                          .read_text(encoding="utf-8"))
    assert r["provenance"]["content_sha256"] == manifest["output"]["content_sha256"]
    assert r["experiment"]["cutoff_day"] == 7 * week - 1 and r["experiment"]["target"] == target
    expected = [c for c in catalog.feature_names_for_week(week) if c not in preprocessing.EXCLUDED_SENSITIVE]
    assert r["experiment"]["predictors"] == expected
    if week == 2:
        assert not any(c.startswith("asm_") for c in r["experiment"]["predictors"])
    parts = r["split_audit"]["parts"]
    assert parts["test_main"]["rows"] + r["split_audit"]["rows_excluded_from_main_test"] == r["split_audit"]["rows_2014"]


def test_real_uci_results_independent_from_oulad():
    path = REPORTS / "uci" / "admission" / "results.json"
    if not path.exists():
        pytest.skip("UCI report not generated")
    r = json.loads(path.read_text(encoding="utf-8"))
    oulad_names = {f.name for f in catalog.OULAD_FEATURES}
    assert not set(r["experiment"]["predictors"]) & oulad_names
    assert set(r["experiment"]["predictors"]) <= set(catalog.UCI_ADMISSION_FEATURES)
    assert "Gender" not in r["experiment"]["predictors"]
    assert r["provenance"]["dataset"] == "uci_admission"
    manifest = json.loads((PROCESSED / "uci" / "admission" / "uci_admission.manifest.json").read_text(encoding="utf-8"))
    assert r["provenance"]["content_sha256"] == manifest["output"]["content_sha256"]


def test_integer_nominal_codes_are_one_hot_encoded():
    meta = {"feature_types": {"Course": "nominal", "Admission grade": "numeric"}}
    X = pd.DataFrame({"Course": [33, 171, 9500, 33], "Admission grade": [120.0, 130.0, np.nan, 140.0]})
    prep = preprocessing.build_preprocessor(meta, ["Course", "Admission grade"], scale=False).fit(X)
    names = list(prep.get_feature_names_out())
    assert {"cat__Course_33", "cat__Course_171", "cat__Course_9500"} <= set(names)
    out = prep.transform(pd.DataFrame({"Course": [12345], "Admission grade": [125.0]}))
    assert np.isfinite(out).all()
