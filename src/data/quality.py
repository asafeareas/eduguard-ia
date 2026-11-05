"""Data-quality checks: key uniqueness, referential integrity, value consistency.

Each check returns plain numbers so results can be serialised and tested.
"""

from __future__ import annotations

import pandas as pd

from src.data.loaders import OULAD_KEY, PRESENTATION_KEY


def key_is_unique(df: pd.DataFrame, key: list[str]) -> dict:
    """Report whether `key` uniquely identifies rows of `df`."""
    dup = int(df.duplicated(key).sum())
    return {"key": key, "n_rows": int(len(df)), "n_duplicate_keys": dup, "unique": dup == 0}


def orphan_count(child: pd.DataFrame, parent: pd.DataFrame, on: list[str]) -> int:
    """Number of child rows whose key does not exist in parent."""
    parent_keys = parent[on].drop_duplicates()
    merged = child[on].merge(parent_keys, on=on, how="left", indicator=True)
    return int((merged["_merge"] == "left_only").sum())


def oulad_integrity(tables: dict[str, pd.DataFrame], student_vle: pd.DataFrame | None = None) -> dict:
    """Granularity and referential-integrity report for OULAD."""
    t = tables
    report: dict = {
        "granularity": {
            "courses": key_is_unique(t["courses"], PRESENTATION_KEY),
            "assessments": key_is_unique(t["assessments"], ["id_assessment"]),
            "vle": key_is_unique(t["vle"], ["id_site"]),
            "studentInfo": key_is_unique(t["studentInfo"], OULAD_KEY),
            "studentRegistration": key_is_unique(t["studentRegistration"], OULAD_KEY),
            "studentAssessment": key_is_unique(t["studentAssessment"], ["id_assessment", "id_student"]),
        },
        "orphans": {
            "assessments->courses": orphan_count(t["assessments"], t["courses"], PRESENTATION_KEY),
            "vle->courses": orphan_count(t["vle"], t["courses"], PRESENTATION_KEY),
            "studentInfo->courses": orphan_count(t["studentInfo"], t["courses"], PRESENTATION_KEY),
            "studentRegistration->studentInfo": orphan_count(t["studentRegistration"], t["studentInfo"], OULAD_KEY),
            "studentInfo->studentRegistration": orphan_count(t["studentInfo"], t["studentRegistration"], OULAD_KEY),
            "studentAssessment->assessments": orphan_count(t["studentAssessment"], t["assessments"], ["id_assessment"]),
        },
    }
    # studentAssessment has no presentation columns: attach them via assessments,
    # then check the student was registered in that presentation.
    sa = t["studentAssessment"].merge(
        t["assessments"][["id_assessment", *PRESENTATION_KEY]], on="id_assessment", how="left"
    )
    report["orphans"]["studentAssessment->studentInfo"] = orphan_count(sa, t["studentInfo"], OULAD_KEY)

    if student_vle is not None:
        sv_keys = student_vle[OULAD_KEY].drop_duplicates()
        report["orphans"]["studentVle->studentInfo"] = orphan_count(sv_keys, t["studentInfo"], OULAD_KEY)
        sites = student_vle[["id_site"]].drop_duplicates()
        report["orphans"]["studentVle->vle(id_site)"] = orphan_count(sites, t["vle"], ["id_site"])
        report["granularity"]["studentVle"] = {
            "key": [*OULAD_KEY, "id_site", "date"],
            "n_rows": int(len(student_vle)),
            "n_duplicate_keys": int(student_vle.duplicated([*OULAD_KEY, "id_site", "date"]).sum()),
        }

    # Value consistency between label and unregistration date.
    info = t["studentInfo"].merge(t["studentRegistration"], on=OULAD_KEY, how="left")
    withdrawn = info["final_result"] == "Withdrawn"
    unreg = info["date_unregistration"].notna()
    report["consistency"] = {
        "withdrawn_without_unregistration_date": int((withdrawn & ~unreg).sum()),
        "not_withdrawn_with_unregistration_date": int((~withdrawn & unreg).sum()),
        "unregistration_before_registration": int(
            (info["date_unregistration"] < info["date_registration"]).sum()
        ),
        "registration_missing": int(info["date_registration"].isna().sum()),
    }
    sa_dates = sa.merge(t["courses"], on=PRESENTATION_KEY, how="left")
    report["consistency"]["submissions_after_presentation_end"] = int(
        (sa_dates["date_submitted"] > sa_dates["module_presentation_length"]).sum()
    )
    report["consistency"]["submissions_after_end_that_are_banked"] = int(
        ((sa_dates["date_submitted"] > sa_dates["module_presentation_length"]) & (sa_dates["is_banked"] == 1)).sum()
    )
    return report


def uci_consistency(df: pd.DataFrame) -> dict:
    """Logical consistency checks for UCI 697 curricular-unit variables."""
    out: dict = {}
    for sem in ("1st", "2nd"):
        p = f"Curricular units {sem} sem "
        enrolled, approved = df[p + "(enrolled)"], df[p + "(approved)"]
        grade, evals = df[p + "(grade)"], df[p + "(evaluations)"]
        out[sem] = {
            "approved_gt_enrolled": int((approved > enrolled).sum()),
            "grade_zero_with_approvals": int(((grade == 0) & (approved > 0)).sum()),
            "grade_positive_without_approvals": int(((grade > 0) & (approved == 0)).sum()),
            "enrolled_zero": int((enrolled == 0).sum()),
            "enrolled_positive_zero_evaluations": int(((enrolled > 0) & (evals == 0)).sum()),
        }
    out["age_below_17"] = int((df["Age at enrollment"] < 17).sum())
    out["duplicate_rows"] = int(df.duplicated().sum())
    out["missing_cells"] = int(df.isna().sum().sum())
    return out


# ----------------------------------------------------------------------------- Phase 03
def dataset_quality_report(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    ids: list[str],
    ftypes: dict[str, str],
    domains: dict[str, dict] | None = None,
    corr_threshold: float = 0.9,
) -> dict:
    """Quality report for a modelling table. Purely diagnostic: drops nothing.

    Checks: missing values, dtypes, duplicated ids, out-of-domain values, constant
    features, target integrity/distribution and highly correlated feature pairs.
    """
    domains = domains or {}
    per_feature = {}
    for col in features:
        s = df[col]
        info = {"dtype": str(s.dtype), "ftype": ftypes.get(col), "n_missing": int(s.isna().sum()),
                "n_unique": int(s.nunique(dropna=True)), "constant": bool(s.nunique(dropna=True) <= 1)}
        dom = domains.get(col, {})
        valid = s.dropna()
        if "values" in dom:
            info["out_of_domain"] = int((~valid.isin(dom["values"])).sum())
        elif dom:
            low = (valid < dom["min"]).sum() if "min" in dom else 0
            high = (valid > dom["max"]).sum() if "max" in dom else 0
            info["out_of_domain"] = int(low + high)
        if pd.api.types.is_numeric_dtype(s):
            info.update(min=float(valid.min()) if len(valid) else None, max=float(valid.max()) if len(valid) else None)
        per_feature[col] = info

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    corr = df[numeric].corr(method="spearman")
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= corr_threshold:
                pairs.append({"a": a, "b": b, "spearman": round(float(v), 4)})

    target_info = {}
    for t in targets:
        vals = df[t]
        target_info[t] = {
            "n_missing": int(vals.isna().sum()),
            "values": sorted(map(int, vals.dropna().unique())),
            "counts": {str(k): int(v) for k, v in vals.value_counts().sort_index().items()},
            "positive_rate": round(float(vals.mean()), 4),
        }
    return {
        "n_rows": int(len(df)),
        "n_features": len(features),
        "duplicate_ids": int(df.duplicated(ids).sum()),
        "total_missing_feature_cells": int(df[features].isna().sum().sum()),
        "features_with_missing": {c: v["n_missing"] for c, v in per_feature.items() if v["n_missing"]},
        "constant_features": [c for c, v in per_feature.items() if v["constant"]],
        "out_of_domain": {c: v["out_of_domain"] for c, v in per_feature.items() if v.get("out_of_domain")},
        "high_correlation_pairs": sorted(pairs, key=lambda p: -abs(p["spearman"])),
        "targets": target_info,
        "features": per_feature,
    }
