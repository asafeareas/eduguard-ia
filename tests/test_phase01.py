"""Phase 01 checks: project structure, source registry and data utilities.

These tests run offline (no network access).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import download
from src.data.inspect_dataset import profile_dataframe
from src.data.registry import SOURCES, get_source

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "data/raw", "data/processed", "data/external", "data/interim",
    "notebooks", "research", "tests", "docs",
    "src/data", "src/preprocessing", "src/feature_engineering",
    "src/models", "src/evaluation", "src/prediction", "src/analysis",
]
EXPECTED_DOCS = [
    "README.md", "requirements.txt", ".gitignore",
    "docs/DATASET_RESEARCH.md", "docs/DATA_DICTIONARY.md",
    "docs/PROJECT_ARCHITECTURE.md", "docs/TECHNICAL_DECISIONS.md",
    "docs/DATA_SOURCES.md",
    # Phase 02
    "docs/EXPLORATORY_DATA_ANALYSIS.md", "docs/TEMPORAL_FEASIBILITY.md",
    "docs/DATA_LEAKAGE_AUDIT.md", "docs/DATASET_COMPARISON.md", "docs/FEATURE_CANDIDATES.md",
    # Phase 03
    "docs/FEATURE_ENGINEERING.md", "docs/TARGET_DEFINITION.md", "docs/MODELING_DATASETS.md",
    "docs/FEATURE_VALIDATION.md",
    # Phase 04
    "docs/MODELING_METHODOLOGY.md", "docs/MODEL_EVALUATION.md", "docs/FAIRNESS_AUDIT.md",
    "docs/RISK_CLASSIFICATION.md",
    # Phase 05
    "docs/MODEL_V2_EVALUATION.md", "docs/THRESHOLD_METHODOLOGY.md", "docs/BRAZILIAN_FEATURE_MAPPING.md",
    "docs/BRAZILIAN_VALIDATION_PROTOCOL.md", "docs/UFRN_DATA_AVAILABILITY.md", "docs/GOVERNANCE_AND_PRIVACY.md",
]


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_expected_directories_exist(rel):
    assert (ROOT / rel).is_dir(), rel


@pytest.mark.parametrize("rel", EXPECTED_DOCS)
def test_expected_files_exist(rel):
    assert (ROOT / rel).is_file(), rel


def test_gitignore_blocks_data_and_secrets():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/**", ".env", "kaggle.json", ".venv/", "*.zip"):
        assert pattern in text


def test_registry_entries_are_consistent():
    assert SOURCES
    for key, src in SOURCES.items():
        assert key == src.source_id
        assert src.compatibility in {"high", "partial", "low"}
        assert src.license
        assert src.landing_url.startswith("https://")


def test_get_source_unknown_raises():
    with pytest.raises(KeyError):
        get_source("does_not_exist")


def test_non_auto_source_requires_approval():
    with pytest.raises(PermissionError):
        download.download_source("oulad_349", approved=False)


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        download.safe_extract(archive, tmp_path / "out")


def test_describe_file_reports_size_and_checksum(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n", encoding="utf-8")
    meta = download.describe_file(f, tmp_path)
    assert meta["size_bytes"] == f.stat().st_size
    assert meta["format"] == "csv"
    assert len(meta["sha256"]) == 64


def test_oulad_cutoff_excludes_students_already_unregistered():
    from src.data.inspect_oulad import temporal_analysis

    key = {"code_module": "AAA", "code_presentation": "2013J"}
    info = pd.DataFrame([
        {**key, "id_student": 1, "final_result": "Withdrawn"},  # left on day -5
        {**key, "id_student": 2, "final_result": "Withdrawn"},  # left on day 20
        {**key, "id_student": 3, "final_result": "Pass"},
    ])
    reg = pd.DataFrame([
        {**key, "id_student": 1, "date_registration": -30, "date_unregistration": -5},
        {**key, "id_student": 2, "date_registration": -30, "date_unregistration": 20},
        {**key, "id_student": 3, "date_registration": -30, "date_unregistration": None},
    ])
    assessments = pd.DataFrame([{**key, "id_assessment": 1, "assessment_type": "TMA", "date": 19, "weight": 10}])
    vle = pd.DataFrame([{**key, "id_student": s, "first_click_day": 0,
                         "clicks_until_day_14": 5, "clicks_until_day_28": 9,
                         "clicks_until_day_42": 12} for s in (2, 3)])
    out = temporal_analysis(
        {"studentInfo": info, "studentRegistration": reg, "assessments": assessments}, vle
    )
    assert out["per_cutoff"]["day_14"]["population"] == 2  # student 1 excluded
    assert out["per_cutoff"]["day_28"]["population"] == 1  # student 2 also excluded
    assert out["unregistered_before_day_0"] == 1


def test_profile_dataframe_counts():
    df = pd.DataFrame({"a": [1, 1, None], "y": ["Dropout", "Dropout", "Graduate"]})
    prof = profile_dataframe(df, target="y")
    assert prof["n_rows"] == 3
    assert prof["total_missing_cells"] == 1
    assert prof["target"]["counts"] == {"Dropout": 2, "Graduate": 1}
