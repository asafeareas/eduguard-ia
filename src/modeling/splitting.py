"""Data splits for Phase 04 (approved decision Q15).

OULAD:
    train pool = presentations of 2013
    test (main) = presentations of 2014 EXCLUDING students that also appear in 2013
    test (secondary, "previously observed students") = 2014 rows of those recurrent students
Inside 2013, students are split (never shared) into:
    train (60%)        -> model fitting
    validation (20%)   -> model / hyperparameter selection, inactivity-rule N
    calibration (20%)  -> calibrator fitting, threshold / risk-band selection
The student id is used ONLY to build the split, never as a feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
TRAIN_YEAR, TEST_YEAR = 2013, 2014


@dataclass
class OuladSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    test_recurrent: pd.DataFrame
    audit: dict


def _class_info(df: pd.DataFrame, targets: list[str]) -> dict:
    return {t: {"n": int(len(df)), "positives": int(df[t].sum()),
                "rate": round(float(df[t].mean()), 4) if len(df) else None} for t in targets}


def split_by_student(df: pd.DataFrame, fraction: float, seed: int, group_col: str = "student_id"):
    """Split rows so that `fraction` of the students (not rows) go to the second part."""
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(gss.split(df, groups=df[group_col]))
    return df.iloc[np.sort(a)], df.iloc[np.sort(b)]


def oulad_split(df: pd.DataFrame, targets: list[str], seed: int = SEED) -> OuladSplit:
    pool = df.loc[df["presentation_year"] == TRAIN_YEAR]
    later = df.loc[df["presentation_year"] == TEST_YEAR]
    recurrent_ids = set(pool["student_id"]) & set(later["student_id"])
    test = later.loc[~later["student_id"].isin(recurrent_ids)]
    test_recurrent = later.loc[later["student_id"].isin(recurrent_ids)]

    train_val, calibration = split_by_student(pool, 0.20, seed)
    train, validation = split_by_student(train_val, 0.25, seed + 1)  # 0.25 * 0.8 = 0.2 of the pool

    parts = {"train": train, "validation": validation, "calibration": calibration,
             "test_main": test, "test_recurrent_secondary": test_recurrent}
    audit = {
        "rows_2013": int(len(pool)), "students_2013": int(pool["student_id"].nunique()),
        "rows_2014": int(len(later)), "students_2014": int(later["student_id"].nunique()),
        "students_in_both_years": len(recurrent_ids),
        "rows_excluded_from_main_test": int(len(test_recurrent)),
        "parts": {name: {"rows": int(len(p)), "students": int(p["student_id"].nunique()),
                         "classes": _class_info(p, targets)} for name, p in parts.items()},
        "class_rate_2014_before_exclusion": _class_info(later, targets),
        "seed": seed,
        "fractions_of_2013_students": {"train": 0.6, "validation": 0.2, "calibration": 0.2},
    }
    check_disjoint(parts)
    return OuladSplit(train, validation, calibration, test, test_recurrent, audit)


def check_disjoint(parts: dict[str, pd.DataFrame], group_col: str = "student_id") -> None:
    """Raise if any student appears in two of train/validation/calibration/test_main."""
    names = ["train", "validation", "calibration", "test_main"]
    ids = {n: set(parts[n][group_col]) for n in names if n in parts}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in ids and b in ids and ids[a] & ids[b]:
                raise AssertionError(f"students shared between {a} and {b}: {len(ids[a] & ids[b])}")
