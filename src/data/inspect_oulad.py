"""Preliminary, read-only inspection of OULAD with a focus on temporal availability.

Answers the Phase 01 question: is there enough information to simulate a
prediction during the first weeks of a module presentation (day 0 = start)?
No feature engineering, no modelling. Original files are never modified.

Usage:
    python -m src.data.inspect_oulad [--raw-dir data/raw/oulad_349/extracted]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.inspect_dataset import profile_dataframe

KEY = ["code_module", "code_presentation", "id_student"]
CUTOFF_DAYS = (14, 28, 42)
SMALL_TABLES = (
    "courses", "assessments", "vle", "studentInfo",
    "studentRegistration", "studentAssessment",
)


def load_small_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    # OULAD encodes missing values as "?" (verified in the raw files).
    return {name: pd.read_csv(raw_dir / f"{name}.csv", na_values="?") for name in SMALL_TABLES}


def aggregate_vle(path: Path, chunksize: int = 2_000_000) -> tuple[dict, pd.DataFrame]:
    """Stream studentVle.csv; return global stats and first-click day per student."""
    n_rows = 0
    min_date, max_date = float("inf"), float("-inf")
    n_missing = 0
    rows_before_start = 0
    first_click: list[pd.DataFrame] = []
    clicks_by_cutoff: dict[int, list[pd.DataFrame]] = {c: [] for c in CUTOFF_DAYS}
    for chunk in pd.read_csv(path, chunksize=chunksize, na_values="?"):
        n_rows += len(chunk)
        n_missing += int(chunk.isna().sum().sum())
        min_date = min(min_date, chunk["date"].min())
        max_date = max(max_date, chunk["date"].max())
        rows_before_start += int((chunk["date"] < 0).sum())
        first_click.append(chunk.groupby(KEY)["date"].min().reset_index())
        for c in CUTOFF_DAYS:
            sub = chunk[chunk["date"] <= c]
            clicks_by_cutoff[c].append(sub.groupby(KEY)["sum_click"].sum().reset_index())
    first = pd.concat(first_click).groupby(KEY)["date"].min().rename("first_click_day")
    per_student = first.to_frame()
    for c in CUTOFF_DAYS:
        s = pd.concat(clicks_by_cutoff[c]).groupby(KEY)["sum_click"].sum()
        per_student[f"clicks_until_day_{c}"] = s
    per_student = per_student.fillna(0).reset_index()
    stats = {
        "n_rows": n_rows,
        "total_missing_cells": n_missing,
        "date_min": int(min_date),
        "date_max": int(max_date),
        "rows_before_day_0": rows_before_start,
    }
    return stats, per_student


def temporal_analysis(t: dict[str, pd.DataFrame], vle_students: pd.DataFrame) -> dict:
    info = t["studentInfo"].merge(t["studentRegistration"], on=KEY, how="left")
    withdrawn = info["final_result"] == "Withdrawn"
    out: dict = {
        "students_unique_ids": int(info["id_student"].nunique()),
        "student_presentation_rows": int(len(info)),
        "duplicate_keys_studentInfo": int(info.duplicated(KEY).sum()),
        "withdrawn_without_unregistration_date": int((withdrawn & info["date_unregistration"].isna()).sum()),
        "non_withdrawn_with_unregistration_date": int((~withdrawn & info["date_unregistration"].notna()).sum()),
        "unregistration_date_quantiles": {
            str(q): float(v) for q, v in info.loc[withdrawn, "date_unregistration"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()
        },
        "unregistered_before_day_0": int((info["date_unregistration"] < 0).sum()),
    }

    # Population still registered at each cutoff and its label balance.
    with_vle = info.merge(vle_students, on=KEY, how="left")
    per_cutoff = {}
    for c in CUTOFF_DAYS:
        active = with_vle["date_unregistration"].isna() | (with_vle["date_unregistration"] > c)
        pop = with_vle[active]
        clicks = pop[f"clicks_until_day_{c}"].fillna(0)
        per_cutoff[f"day_{c}"] = {
            "already_unregistered_excluded": int((~active).sum()),
            "population": int(len(pop)),
            "final_result_counts": {k: int(v) for k, v in pop["final_result"].value_counts().items()},
            "students_with_zero_clicks_until_cutoff": int((clicks == 0).sum()),
            "median_clicks_until_cutoff_by_result": {
                k: float(v) for k, v in pop.assign(_c=clicks).groupby("final_result")["_c"].median().items()
            },
            "assessments_due_until_cutoff": int((t["assessments"]["date"] <= c).sum()),
        }
    out["per_cutoff"] = per_cutoff

    a = t["assessments"]
    first_due = (
        a[a["assessment_type"] != "Exam"].groupby(["code_module", "code_presentation"])["date"].min()
    )
    out["first_non_exam_assessment_due_day"] = {
        "min": float(first_due.min()), "median": float(first_due.median()), "max": float(first_due.max()),
    }
    out["assessments_missing_date"] = int(a["date"].isna().sum())
    out["students_without_any_vle_record"] = int(with_vle["first_click_day"].isna().sum())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OULAD preliminary inspection")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/oulad_349/extracted"))
    parser.add_argument("--out", type=Path, default=Path("research/inspection/oulad_349_profile.json"))
    args = parser.parse_args(argv)

    tables = load_small_tables(args.raw_dir)
    target_col = {"studentInfo": "final_result", "assessments": "assessment_type"}
    report = {
        "tables": {
            name: profile_dataframe(df, target=target_col.get(name)) for name, df in tables.items()
        }
    }
    vle_stats, vle_students = aggregate_vle(args.raw_dir / "studentVle.csv")
    report["tables"]["studentVle"] = vle_stats
    report["temporal"] = temporal_analysis(tables, vle_students)
    report["courses"] = tables["courses"].to_dict(orient="records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"profile -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
