#!/usr/bin/env python3
"""
Helper to keep CITATION.cff aligned with releases.

Best practice:
1) Run dist/reserve_zenodo_DOI.py to create a draft and reserve a version DOI.
2) Run this script before committing/tagging:
   python dist/update_citation.py --version <vX.Y.Z> --version-doi <reserved> [--date YYYY-MM-DD]
   It sets version, date-released, identifiers, and top-level doi.
3) Commit and tag. The tag CI job can upload the archive to the draft.
   Publishing is manual unless you pass --publish to dist/publish_zenodo.py.

If you tagged without a draft DOI:
- Leave the tag untouched (no Zenodo deposition, no tag docs).
- Run dist/reserve_zenodo_DOI.py to create a new draft.
- Update CITATION.cff with the reserved DOI and incremented patch version.
- Commit and create a new tag (version +0.0.1).

Validation:
- If `cffconvert` is installed, the script runs `cffconvert --validate CITATION.cff`.
  Use --skip-validate to bypass. Requires ruamel.yaml (preferred) or PyYAML.
"""

from __future__ import annotations

import argparse
import subprocess
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFF = REPO_ROOT / "CITATION.cff"


def _load_cff(path: Path) -> tuple[Any, Callable[[Any], None]]:
    """
    Return (data, dumper) for CITATION.cff.
    Uses ruamel.yaml if available to preserve formatting, otherwise PyYAML.
    """
    try:
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh)

        def _dump(d: Any) -> None:
            with path.open("w", encoding="utf-8") as fh:
                yaml.dump(d, fh)

        return data, _dump
    except ImportError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Install either ruamel.yaml or PyYAML to run this script."
            ) from exc

        data = yaml.safe_load(path.read_text())

        def _dump(d: Any) -> None:
            path.write_text(yaml.safe_dump(d, sort_keys=False))

        return data, _dump


def _parse_date(user_date: str | None) -> str:
    if user_date:
        try:
            return date.fromisoformat(user_date).isoformat()
        except ValueError as exc:
            raise SystemExit("Invalid --date. Use YYYY-MM-DD.") from exc
    return date.today().isoformat()


def _upsert_identifier(
    identifiers: List[Dict[str, Any]], value: str, description: str
) -> None:
    for item in identifiers:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "doi" and item.get("value") == value:
            if description and not item.get("description"):
                item["description"] = description
            return
    entry: Dict[str, Any] = {"type": "doi", "value": value}
    if description:
        entry["description"] = description
    identifiers.append(entry)


def update_citation(
    version: str,
    release_date: str,
    citation_path: Path,
    version_doi: str | None,
) -> None:
    if not citation_path.exists():
        raise SystemExit(f"Missing CITATION file at {citation_path}")

    data, dump = _load_cff(citation_path)
    if not isinstance(data, dict):
        raise SystemExit("Unexpected CITATION.cff format.")

    data["version"] = version
    data["date-released"] = release_date

    identifiers = data.get("identifiers")
    if identifiers is None:
        identifiers_list: List[Dict[str, Any]] | None = None
    elif isinstance(identifiers, list):
        identifiers_list = identifiers
    else:
        raise SystemExit("Expected 'identifiers' to be a list in CITATION.cff.")

    if version_doi:
        if identifiers_list is None:
            identifiers_list = []
        _upsert_identifier(
            identifiers_list,
            version_doi,
            f"DOI for xsnow {version}",
        )
        data["doi"] = version_doi

    # Keep existing identifiers if present; drop empty lists to satisfy schema minItems=1.
    if identifiers_list is None:
        data.pop("identifiers", None)
    elif len(identifiers_list) == 0:
        data.pop("identifiers", None)
    else:
        data["identifiers"] = identifiers_list

    dump(data)
    print(f"Updated {citation_path.name} to version={version}, date={release_date}")
    if version_doi:
        print(f"  version DOI: {version_doi}")


def _maybe_validate(citation_path: Path, skip_validate: bool) -> None:
    if skip_validate:
        return
    cmd = shutil.which("cffconvert")
    if not cmd:
        print("cffconvert not installed; skipping validation. Install with `pip install cffconvert`.")
        return
    try:
        subprocess.run(
            [cmd, "--validate", "--infile", str(citation_path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(f"cffconvert validation failed with exit code {exc.returncode}.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update CITATION.cff with version/date/DOIs before tagging a release."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version string to write (required).",
    )
    parser.add_argument(
        "--version-doi",
        help="Reserved DOI for this specific release (sets top-level doi).",
    )
    parser.add_argument(
        "--date",
        help="Release date in YYYY-MM-DD (defaults to today).",
    )
    parser.add_argument(
        "--citation-path",
        type=Path,
        default=DEFAULT_CFF,
        help=f"Path to CITATION.cff (default: {DEFAULT_CFF})",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip running `cffconvert --validate` even if available.",
    )
    args = parser.parse_args()

    version = args.version
    release_date = _parse_date(args.date)
    update_citation(
        version=version,
        release_date=release_date,
        citation_path=args.citation_path,
        version_doi=args.version_doi,
    )
    _maybe_validate(args.citation_path, skip_validate=args.skip_validate)


if __name__ == "__main__":
    main()
