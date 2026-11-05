"""Documentation must stay in sync with the feature catalog."""

from pathlib import Path

from src.feature_engineering import catalog

DOC = Path(__file__).resolve().parents[1] / "docs" / "FEATURE_ENGINEERING.md"


def test_every_catalog_feature_is_documented():
    text = DOC.read_text(encoding="utf-8")
    missing = [f.name for f in catalog.OULAD_FEATURES if f"`{f.name}`" not in text]
    missing += [c for c in catalog.UCI_ADMISSION_FEATURES if f"`{c}`" not in text]
    assert not missing, missing
