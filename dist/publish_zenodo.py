#!/usr/bin/env python3
"""
Upload a tagged release archive to an existing Zenodo draft deposition.

Intended workflow:
1) Run dist/reserve_zenodo_DOI.py before tagging to create a draft and reserve a DOI.
2) Update CITATION.cff with that version DOI via dist/update_citation.py.
3) Commit and tag the release.
4) Run this script (typically in CI on tags) to upload the tag archive.
   By default it does NOT publish; use --publish when you are ready.

Failure and recovery (tag created without a draft DOI):
- Leave the original tag untouched. It will have no Zenodo deposition and no tag docs.
- Create a new draft via dist/reserve_zenodo_DOI.py.
- Update CITATION.cff with the newly reserved version DOI.
- Increment the tag by +0.0.1 (patch bump), commit, and tag again.

Requirements:
- ZENODO_TOKEN in the environment.
- A draft deposition id or concept DOI must be available (from reserve_zenodo_DOI.py).

Usage examples:
  python dist/publish_zenodo.py --concept-doi 10.5281/zenodo.1234567 --tag v0.1.2
  python dist/publish_zenodo.py --draft-id 123456 --archive dist/xsnow-v0.1.2.tar.gz
  python dist/publish_zenodo.py --tag v0.1.2 --publish   # publish when ready
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, parse, request


DEFAULT_API_BASE = "https://zenodo.org/api"
SANDBOX_API_BASE = "https://sandbox.zenodo.org/api"


def _request_json(
    method: str,
    url: str,
    token: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    data = None
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else ""
        raise SystemExit(f"Zenodo API error {exc.code}: {detail or exc.reason}") from exc


def _request_raw(method: str, url: str, token: str, data: Optional[bytes] = None) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as resp:
            resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else ""
        raise SystemExit(f"Zenodo API error {exc.code}: {detail or exc.reason}") from exc


def _extract_id(url: str) -> Optional[int]:
    match = re.search(r"/(\\d+)(?:\\?.*)?$", url)
    if not match:
        return None
    return int(match.group(1))


def _find_draft_id(api_base: str, token: str, concept_doi: str) -> int:
    query = parse.quote(f'conceptdoi:"{concept_doi}"')
    url = f"{api_base}/deposit/depositions?size=20&sort=mostrecent&q={query}"
    data = _request_json("GET", url, token)
    depositions: List[Dict[str, Any]]
    if isinstance(data, list):
        depositions = data
    else:
        depositions = data.get("hits", {}).get("hits", [])

    for dep in depositions:
        state = dep.get("state")
        submitted = dep.get("submitted")
        if state == "draft" or submitted is False:
            return int(dep["id"])

    raise SystemExit(
        f"No draft deposition found for concept DOI {concept_doi}. "
        "Run dist/reserve_zenodo_DOI.py before tagging."
    )


def _get_deposition(api_base: str, token: str, draft_id: int) -> Dict[str, Any]:
    url = f"{api_base}/deposit/depositions/{draft_id}"
    data = _request_json("GET", url, token)
    if not isinstance(data, dict):
        raise SystemExit("Unexpected Zenodo response while fetching deposition.")
    return data


def _ensure_draft(deposition: Dict[str, Any]) -> None:
    if deposition.get("submitted") is True or deposition.get("state") == "published":
        raise SystemExit("Deposition is already published. Create a new draft first.")


def _delete_files(token: str, files: List[Dict[str, Any]]) -> None:
    for item in files:
        link = item.get("links", {}).get("self")
        if link:
            _request_json("DELETE", link, token)


def _build_archive(repo_root: Path, tag: str, out_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar.gz", "-o", str(out_path), tag],
        check=True,
    )


def _upload_file(bucket: str, token: str, file_path: Path) -> None:
    target = f"{bucket}/{file_path.name}"
    data = file_path.read_bytes()
    _request_raw("PUT", target, token, data=data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a tag archive to an existing Zenodo draft deposition."
    )
    parser.add_argument("--concept-doi", help="Concept DOI for the software.")
    parser.add_argument("--draft-id", type=int, help="Draft deposition id.")
    parser.add_argument("--tag", help="Git tag to archive (defaults to CI_COMMIT_TAG).")
    parser.add_argument("--archive", help="Path to a prebuilt archive to upload.")
    parser.add_argument(
        "--replace-files",
        action="store_true",
        help="Delete existing draft files before upload.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish after uploading (default is to leave as draft).",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Zenodo API base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use the Zenodo sandbox API.",
    )
    args = parser.parse_args()

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise SystemExit("ZENODO_TOKEN is required in the environment.")

    concept_doi = args.concept_doi or os.getenv("ZENODO_CONCEPT_DOI")
    draft_id = args.draft_id or os.getenv("ZENODO_DRAFT_ID")

    if not concept_doi and not draft_id:
        raise SystemExit("Provide --concept-doi or --draft-id (or set env vars).")

    api_base = SANDBOX_API_BASE if args.sandbox else args.api_base

    if draft_id:
        draft_id_int = int(draft_id)
    else:
        draft_id_int = _find_draft_id(api_base, token, concept_doi)

    deposition = _get_deposition(api_base, token, draft_id_int)
    _ensure_draft(deposition)

    files = deposition.get("files", []) or []
    if files and not args.replace_files:
        raise SystemExit(
            "Draft deposition already has files. Re-run with --replace-files if you want to overwrite."
        )
    if files and args.replace_files:
        _delete_files(token, files)

    bucket = deposition.get("links", {}).get("bucket")
    if not bucket:
        raise SystemExit("Zenodo draft has no bucket link; cannot upload files.")

    repo_root = Path(__file__).resolve().parents[1]
    archive_path = Path(args.archive) if args.archive else None
    if archive_path is None:
        tag = args.tag or os.getenv("CI_COMMIT_TAG")
        if not tag:
            raise SystemExit("Provide --tag or set CI_COMMIT_TAG to build an archive.")
        archive_path = repo_root / f"xsnow-{tag}.tar.gz"
        _build_archive(repo_root, tag, archive_path)

    if not archive_path.exists():
        raise SystemExit(f"Archive not found at {archive_path}")

    _upload_file(bucket, token, archive_path)
    print(f"Uploaded {archive_path.name} to draft {draft_id_int}.")

    if args.publish:
        url = f"{api_base}/deposit/depositions/{draft_id_int}/actions/publish"
        _request_json("POST", url, token)
        print(f"Published draft deposition {draft_id_int}.")
    else:
        print("Draft not published (use --publish when ready).")


if __name__ == "__main__":
    main()
