"""Build the Phase 03 modelling datasets (no model is trained here).

Usage:
    python -m src.feature_engineering.build_datasets            # OULAD weeks 2/4/6 + UCI admission
    python -m src.feature_engineering.build_datasets --only uci

Outputs (data/processed is NOT versioned, except *.metadata.json / *.manifest.json):
    data/processed/oulad/week_0k/oulad_week_0k.parquet
    data/processed/oulad/metadata/oulad_week_0k.{metadata,manifest}.json
    data/processed/uci/admission/uci_admission.parquet (+ uci_enrolled_unlabelled.parquet)
    data/processed/uci/admission/uci_admission.{metadata,manifest}.json
    research/inspection/phase03_quality_*.json  (versioned, aggregates only)

No imputation, scaling, encoding or balancing is applied: those must be fitted on
the training split only (Phase 04). Missing values are kept as missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.download import sha256_of
from src.data.loaders import PROJECT_ROOT, load_oulad, load_student_vle, load_uci
from src.data.quality import dataset_quality_report
from src.feature_engineering import catalog, oulad_features, uci_admission

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
INSPECTION = PROJECT_ROOT / "research" / "inspection"
PIPELINE_VERSION = "phase03-1.0"

TARGET_DEFINITIONS = {
    "target_withdrawn": "PRINCIPAL — Withdrawn = 1; Pass, Distinction, Fail = 0. Cancelamento do MÓDULO "
                        "(não é evasão definitiva da universidade).",
    "target_adverse_outcome": "SECUNDÁRIO — Withdrawn ou Fail = 1; Pass, Distinction = 0. Resultado acadêmico "
                              "adverso, não evasão institucional.",
    "target_dropout": "UCI — Dropout = 1; Graduate = 0; Enrolled excluído (desfecho censurado).",
}


def content_hash(df: pd.DataFrame) -> str:
    """Hash of values + column names/dtypes; independent of the parquet writer's metadata."""
    h = hashlib.sha256()
    h.update("|".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def code_version() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--", "src")
    return {"pipeline_version": PIPELINE_VERSION, "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "src_uncommitted_changes": bool(status)}


def verified_raw_files(source_id: str, used: list[str]) -> list[dict]:
    """Recompute SHA-256 of the raw files used and compare with the download manifest."""
    manifest = json.loads((RAW / source_id / f"{source_id}.source.json").read_text(encoding="utf-8"))
    expected = {f["path"]: f for f in manifest["files"]}
    out = []
    for rel in used:
        path = RAW / source_id / rel
        digest = sha256_of(path)
        if digest != expected[rel]["sha256"]:
            raise RuntimeError(f"Raw file changed since download: {path}")
        out.append({"path": f"data/raw/{source_id}/{rel}", "sha256": digest, "size_bytes": path.stat().st_size})
    return out


def _write(df: pd.DataFrame, path: Path) -> tuple[str, str]:
    """Write parquet; return (file sha256, content hash of the data AS READ BACK from disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return sha256_of(path), content_hash(pd.read_parquet(path))


def _json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def oulad_metadata(week: int) -> dict:
    feats = catalog.oulad_features_for_week(week)
    return {
        "dataset": f"oulad_week_{week:02d}",
        "id_columns": catalog.OULAD_ID_COLUMNS,
        "feature_columns": [f.name for f in feats],
        "target_columns": catalog.OULAD_TARGET_COLUMNS,
        "audit_columns": catalog.OULAD_AUDIT_COLUMNS,
        "feature_types": {f.name: f.ftype for f in feats},
        "feature_groups": {f.name: f.group for f in feats},
        "sensitive_features": [f.name for f in feats if f.sensitive],
        "quarantined_features": [f.name for f in catalog.OULAD_FEATURES if f.status == "quarantined"],
        "grouping_columns_for_validation": {"temporal": "presentation_year", "student": "student_id",
                                            "presentation": "presentation_id"},
        "note": "IDs, targets and audit columns must never be used as predictors.",
    }


def build_oulad() -> dict:
    tables = load_oulad()
    sv = oulad_features.prepare_vle(load_student_vle(), tables["vle"])
    raw_used = ["extracted/courses.csv", "extracted/assessments.csv", "extracted/vle.csv",
                "extracted/studentInfo.csv", "extracted/studentRegistration.csv",
                "extracted/studentAssessment.csv", "extracted/studentVle.csv"]
    raw_files = verified_raw_files("oulad_349", raw_used)
    summary = {}
    for week in catalog.WEEKS:
        df, exclusions = oulad_features.build_week(tables, sv, week)
        name = f"oulad_week_{week:02d}"
        out_path = PROCESSED / "oulad" / f"week_{week:02d}" / f"{name}.parquet"
        file_sha, stored_hash = _write(df, out_path)
        meta = oulad_metadata(week)
        feats = meta["feature_columns"]
        quality = dataset_quality_report(
            df, feats, catalog.OULAD_TARGET_COLUMNS, ["student_id", "presentation_id"], meta["feature_types"],
            {f.name: f.domain for f in catalog.oulad_features_for_week(week)})
        manifest = {
            "dataset": name,
            "source": {"source_id": "oulad_349", "name": "Open University Learning Analytics Dataset",
                       "version": "UCI ML Repository id 349 (DOI 10.24432/C5KK69)", "raw_files": raw_files},
            "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cutoff": {"week": week, "cutoff_day": week * 7 - 1,
                       "rule": "events with date <= cutoff_day; population registered and not unregistered by cutoff"},
            "n_rows": int(len(df)), "n_features": len(feats), "n_columns": int(df.shape[1]),
            "targets": {t: TARGET_DEFINITIONS[t] for t in catalog.OULAD_TARGET_COLUMNS},
            "class_distribution": quality["targets"],
            "population": exclusions,
            "output": {"path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                       "file_sha256": file_sha, "content_sha256": stored_hash},
            "code": code_version(),
            "transformations_applied": "none (no imputation/encoding/scaling/balancing)",
        }
        _json(meta, PROCESSED / "oulad" / "metadata" / f"{name}.metadata.json")
        _json(manifest, PROCESSED / "oulad" / "metadata" / f"{name}.manifest.json")
        _json(quality, INSPECTION / f"phase03_quality_{name}.json")
        summary[name] = {"rows": len(df), "features": len(feats), "targets": quality["targets"],
                         "exclusions": exclusions}
        print(f"[ok] {name}: {len(df)} rows, {len(feats)} features")
    return summary


def build_uci() -> dict:
    raw_files = verified_raw_files("uci_dropout_697", ["extracted/data.csv"])
    sup, enrolled, report = uci_admission.build_uci_admission(load_uci())
    base = PROCESSED / "uci" / "admission"
    sha_sup, hash_sup = _write(sup, base / "uci_admission.parquet")
    sha_enr, _ = _write(enrolled, base / "uci_enrolled_unlabelled.parquet")
    feats = catalog.UCI_ADMISSION_FEATURES
    ftypes = {**{c: "nominal" for c in catalog.UCI_NOMINAL}, **{c: "binary" for c in catalog.UCI_BINARY},
              **{c: "numeric" for c in catalog.UCI_NUMERIC}}
    quality = dataset_quality_report(sup, feats, catalog.UCI_TARGET_COLUMNS, catalog.UCI_ID_COLUMNS, ftypes,
                                     catalog.UCI_DOMAINS)
    meta = {"dataset": "uci_admission", "id_columns": catalog.UCI_ID_COLUMNS, "feature_columns": feats,
            "target_columns": catalog.UCI_TARGET_COLUMNS, "audit_columns": catalog.UCI_AUDIT_COLUMNS,
            "feature_types": ftypes, "sensitive_features": ["Gender", "Nacionality", "Marital status"],
            "quarantined_features": catalog.UCI_QUARANTINED, "prohibited_features": catalog.UCI_PROHIBITED,
            "grouping_columns_for_validation": {"cohort_proxy": "cohort_proxy"},
            "note": "Independent experiment. Never combine with OULAD."}
    manifest = {
        "dataset": "uci_admission",
        "source": {"source_id": "uci_dropout_697", "name": "Predict Students' Dropout and Academic Success",
                   "version": "UCI ML Repository id 697 (DOI 10.24432/C5MC89)", "raw_files": raw_files},
        "processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutoff": {"moment": "admission (enrollment)", "rule": "only variables documented as known at enrollment"},
        "n_rows": int(len(sup)), "n_features": len(feats), "n_columns": int(sup.shape[1]),
        "targets": {"target_dropout": TARGET_DEFINITIONS["target_dropout"]},
        "class_distribution": quality["targets"],
        "population": report,
        "output": {"path": "data/processed/uci/admission/uci_admission.parquet", "file_sha256": sha_sup,
                   "content_sha256": hash_sup,
                   "enrolled_unlabelled": {"path": "data/processed/uci/admission/uci_enrolled_unlabelled.parquet",
                                           "rows": int(len(enrolled)), "file_sha256": sha_enr}},
        "code": code_version(),
        "transformations_applied": "none (no imputation/encoding/scaling/balancing)",
    }
    _json(meta, base / "uci_admission.metadata.json")
    _json(manifest, base / "uci_admission.manifest.json")
    _json(quality, INSPECTION / "phase03_quality_uci_admission.json")
    print(f"[ok] uci_admission: {len(sup)} rows, {len(feats)} features ({len(enrolled)} Enrolled set apart)")
    return {"rows": len(sup), "features": len(feats), "targets": quality["targets"]}


def load_dataset(parquet_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a processed dataset and its column-role metadata."""
    return pd.read_parquet(parquet_path), json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def split_roles(df: pd.DataFrame, meta: dict, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y): X has ONLY feature columns; ids/targets/audit are excluded by construction."""
    if target not in meta["target_columns"]:
        raise KeyError(f"{target} is not a target of {meta['dataset']}")
    return df[meta["feature_columns"]].copy(), df[target].copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 03 modelling datasets")
    parser.add_argument("--only", choices=["oulad", "uci"], default=None)
    args = parser.parse_args(argv)
    summary = {}
    if args.only in (None, "oulad"):
        summary["oulad"] = build_oulad()
    if args.only in (None, "uci"):
        summary["uci"] = build_uci()
    _json(summary, INSPECTION / f"phase03_build_summary{'_' + args.only if args.only else ''}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
