"""Download registered datasets into data/raw/ while preserving originals.

Usage:
    python -m src.data.download uci_dropout_697
    python -m src.data.download oulad_349 --approved   # only after owner approval

Behaviour:
- The original archive is stored untouched in data/raw/<source_id>/.
- Archives are extracted to data/raw/<source_id>/extracted/ (never edited).
- A provenance manifest <source_id>.source.json records URL, licence,
  access timestamp, size and SHA-256 of every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.data.registry import DataSource, get_source

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
USER_AGENT = "EduGuard-IA-dataset-fetcher/0.1"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, base: Path) -> dict:
    """Return size/format/checksum metadata for one file."""
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lstrip(".").lower() or "unknown",
        "sha256": sha256_of(path),
    }


def safe_extract(archive: Path, target: Path) -> list[Path]:
    """Extract a zip archive, refusing entries that escape `target`."""
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    resolved_target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(resolved_target):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if member.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(destination)
    return extracted


def fetch(url: str, destination: Path, timeout: int = 120) -> None:
    """Stream a URL to disk (TLS verification left at library defaults)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)


def build_manifest(source: DataSource, source_dir: Path, files: list[Path]) -> dict:
    """Assemble the provenance manifest for a downloaded source."""
    return {
        "source_id": source.source_id,
        "name": source.name,
        "institution": source.institution,
        "country": source.country,
        "landing_url": source.landing_url,
        "download_url": source.download_url,
        "license": source.license,
        "citation": source.citation,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [describe_file(f, source_dir) for f in sorted(files)],
    }


def download_source(source_id: str, approved: bool = False, force: bool = False) -> Path:
    """Download, extract and document a registered source. Returns manifest path."""
    source = get_source(source_id)
    if not source.download_url:
        raise ValueError(f"'{source_id}' has no single download URL; see docs.")
    if not source.auto_download and not approved:
        raise PermissionError(
            f"'{source_id}' requires explicit owner approval "
            f"(size: {source.approx_download_size}). Re-run with --approved."
        )

    source_dir = RAW_DIR / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_name = Path(source.download_url.split("?")[0]).name.replace("+", "_")
    archive = source_dir / archive_name

    if archive.exists() and not force:
        print(f"[skip] {archive.name} already present (use --force to re-download)")
    else:
        print(f"[get ] {source.download_url}")
        fetch(source.download_url, archive)

    files = [archive]
    if zipfile.is_zipfile(archive):
        files += safe_extract(archive, source_dir / "extracted")

    manifest = build_manifest(source, source_dir, files)
    manifest_path = source_dir / f"{source_id}.source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok  ] manifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_id")
    parser.add_argument("--approved", action="store_true", help="owner approved this download")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    try:
        download_source(args.source_id, approved=args.approved, force=args.force)
    except (PermissionError, ValueError, KeyError) as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
