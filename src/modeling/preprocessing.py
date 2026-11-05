"""Leakage-safe preprocessing (everything is fitted inside the sklearn Pipeline, on train only).

- Predictors = metadata feature columns MINUS sensitive attributes excluded by decision Q16
  (gender, disability), which stay available only for the fairness audit.
- Numeric/count/ordinal/binary: median imputation. Missing-indicators ONLY where missingness
  has a documented structural meaning (no assessment due / no submission yet, IMD missing).
  NO indicator for adm_registration_lead_days: its missingness was quarantined as
  outcome-revealing in Phase 03.
- Nominal: constant "missing" imputation + one-hot with handle_unknown="ignore".
- Logistic regression additionally standardises numeric columns; trees do not need it.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def to_text(X):
    """Nominal values as strings; missing stays missing (imputed next as 'missing')."""
    out = X.astype(str).astype(object)
    return out.mask(X.isna(), np.nan)

# Decision Q16: never predictors (OULAD names and UCI names).
EXCLUDED_SENSITIVE = {"adm_gender", "adm_disability", "Gender", "Educational special needs"}

MISSING_INDICATOR_COLUMNS = {"asm_submission_rate", "asm_mean_days_before_deadline", "adm_imd_band"}


def predictor_columns(meta: dict, drop: set[str] | None = None) -> list[str]:
    """Feature columns allowed as predictors (ids/targets/audit are never in meta['feature_columns'])."""
    drop = set(drop or set()) | EXCLUDED_SENSITIVE
    return [c for c in meta["feature_columns"] if c not in drop]


def column_groups(meta: dict, columns: list[str]) -> dict[str, list[str]]:
    ftypes = meta["feature_types"]
    nominal = [c for c in columns if ftypes[c] == "nominal"]
    numeric = [c for c in columns if ftypes[c] != "nominal"]
    return {
        "nominal": nominal,
        "numeric_with_indicator": [c for c in numeric if c in MISSING_INDICATOR_COLUMNS],
        "numeric": [c for c in numeric if c not in MISSING_INDICATOR_COLUMNS],
    }


def build_preprocessor(meta: dict, columns: list[str], scale: bool) -> ColumnTransformer:
    g = column_groups(meta, columns)

    def num(indicator: bool):
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=indicator, keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if g["numeric"]:
        transformers.append(("num", num(False), g["numeric"]))
    if g["numeric_with_indicator"]:
        transformers.append(("num_ind", num(True), g["numeric_with_indicator"]))
    if g["nominal"]:
        transformers.append(("cat", Pipeline([
            # Codes may be integers (UCI): cast to text so they are categories, never magnitudes.
            ("to_text", FunctionTransformer(to_text, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), g["nominal"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
