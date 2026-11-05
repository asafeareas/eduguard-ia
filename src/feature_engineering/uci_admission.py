"""UCI 697 admission-time dataset (independent experiment; never mixed with OULAD).

Target: Dropout = 1, Graduate = 0. Enrolled rows are excluded from the
supervised table and saved apart (unlabelled) for later sensitivity analyses.
Features: only variables documented as known at enrollment (catalog.UCI_ADMISSION_FEATURES).
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import catalog


def build_uci_admission(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (supervised, enrolled_unlabelled, report)."""
    forbidden = set(catalog.UCI_PROHIBITED) | set(catalog.UCI_QUARANTINED)
    leaked = forbidden & set(catalog.UCI_ADMISSION_FEATURES)
    if leaked:
        raise ValueError(f"Forbidden variables in admission feature list: {sorted(leaked)}")
    missing = set(catalog.UCI_ADMISSION_FEATURES) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found: {sorted(missing)}")

    out = pd.DataFrame({"row_id": range(len(df))})  # position in the original file (no personal ID exists)
    for col in catalog.UCI_ADMISSION_FEATURES:
        out[col] = df[col].to_numpy()
    out["target_dropout"] = df["Target"].map({"Dropout": 1, "Graduate": 0}).to_numpy()  # Enrolled -> NaN
    out["original_target"] = df["Target"].to_numpy()
    out["cohort_proxy"] = (df["Unemployment rate"].astype(str) + "|" + df["Inflation rate"].astype(str)
                           + "|" + df["GDP"].astype(str)).to_numpy()

    cols = [*catalog.UCI_ID_COLUMNS, *catalog.UCI_ADMISSION_FEATURES,
            *catalog.UCI_TARGET_COLUMNS, *catalog.UCI_AUDIT_COLUMNS]
    out = out[cols]
    supervised = out.loc[out["original_target"] != "Enrolled"].copy()
    supervised["target_dropout"] = supervised["target_dropout"].astype(int)
    enrolled = out.loc[out["original_target"] == "Enrolled"].copy()
    report = {
        "rows_total": int(len(df)),
        "rows_supervised": int(len(supervised)),
        "rows_enrolled_excluded": int(len(enrolled)),
        "features_included": len(catalog.UCI_ADMISSION_FEATURES),
        "quarantined": catalog.UCI_QUARANTINED,
        "prohibited": catalog.UCI_PROHIBITED,
    }
    return supervised.reset_index(drop=True), enrolled.reset_index(drop=True), report
