"""Read-only loaders for the downloaded datasets.

Centralises the quirks found in Phase 01 so every analysis reads the raw files
the same way:
- OULAD encodes missing values as "?".
- UCI 697 is ';'-separated, UTF-8 with BOM, and one header has a trailing TAB.

Nothing here writes to data/raw/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OULAD_DIR = PROJECT_ROOT / "data" / "raw" / "oulad_349" / "extracted"
UCI_CSV = PROJECT_ROOT / "data" / "raw" / "uci_dropout_697" / "extracted" / "data.csv"

# Student-in-presentation key used across OULAD tables.
OULAD_KEY = ["code_module", "code_presentation", "id_student"]
PRESENTATION_KEY = ["code_module", "code_presentation"]

OULAD_SMALL_TABLES = (
    "courses", "assessments", "vle", "studentInfo",
    "studentRegistration", "studentAssessment",
)

_VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}


def load_oulad_table(name: str, raw_dir: Path = OULAD_DIR) -> pd.DataFrame:
    """Load one small OULAD table with '?' treated as missing."""
    return pd.read_csv(raw_dir / f"{name}.csv", na_values="?")


def load_oulad(raw_dir: Path = OULAD_DIR) -> dict[str, pd.DataFrame]:
    """Load every OULAD table except studentVle (use load_student_vle)."""
    return {name: load_oulad_table(name, raw_dir) for name in OULAD_SMALL_TABLES}


def load_student_vle(raw_dir: Path = OULAD_DIR, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load studentVle.csv (~10.6M rows) with compact dtypes (~200 MB in memory)."""
    cols = usecols or list(_VLE_DTYPES)
    dtypes = {c: t for c, t in _VLE_DTYPES.items() if c in cols}
    return pd.read_csv(raw_dir / "studentVle.csv", usecols=cols, dtype=dtypes, na_values="?")


def load_uci(path: Path = UCI_CSV) -> pd.DataFrame:
    """Load UCI 697 with BOM removed and header whitespace stripped (in memory only)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df
