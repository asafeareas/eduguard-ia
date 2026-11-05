"""Preliminary, read-only inspection of a tabular dataset.

Produces a JSON profile (shape, dtypes, missing values, duplicates, label
distribution, cardinality, numeric summaries). It never modifies the input
and performs no feature engineering.

Usage:
    python -m src.data.inspect_dataset data/raw/uci_dropout_697/extracted/data.csv \
        --sep ";" --target Target --out research/inspection/uci_dropout_697.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return a JSON-serialisable profile of `df`."""
    columns = []
    for name in df.columns:
        series = df[name]
        info = {
            "name": name,
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info.update(
                min=float(desc["min"]),
                max=float(desc["max"]),
                mean=round(float(desc["mean"]), 4),
                n_zeros=int((series == 0).sum()),
            )
        columns.append(info)

    profile = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns": columns,
    }
    if target is not None:
        counts = df[target].value_counts(dropna=False)
        profile["target"] = {
            "name": target,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): round(float(v) / len(df), 4) for k, v in counts.items()},
        }
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preliminary dataset inspection")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--target", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv_path, sep=args.sep, encoding="utf-8-sig")  # strips BOM
    df.columns = [c.strip() for c in df.columns]  # in-memory only; file untouched
    profile = profile_dataframe(df, target=args.target)
    text = json.dumps(profile, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"profile -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
