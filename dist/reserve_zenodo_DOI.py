#!/usr/bin/env python3
"""
Reserve the next Zenodo DOI for the xsnow concept DOI.

This script is intended to be run BEFORE you tag a release:
1) Reserve a DOI for the next version in Zenodo.
2) Update CITATION.cff with that version DOI (via dist/update_citation.py).
3) Commit and tag the release.
4) A tag-triggered CI job uploads the tag archive to the draft.
   Publishing is manual unless you pass --publish to dist/publish_zenodo.py.

Typical usage:
  python dist/reserve_zenodo_DOI.py

This will:
- look up the latest record for the xsnow concept DOI,
- create a new draft version via the Zenodo depositions API,
- reserve a DOI for that draft,
- print the reserved version DOI and draft deposition id.

Auth:
- Requires a Zenodo personal access token in ZENODO_TOKEN.
- Use --sandbox to target https://sandbox.zenodo.org for testing:
  * uses a separate Zenodo environment (test records only)
  * requires a sandbox token
  * DOIs are not real and are not indexed

Outputs:
- Prints the reserved DOI and draft id to stdout.
- Optionally writes the DOI to a file via --out, for use with update_citation.py.

Notes:
- You must have already published at least one version to obtain a concept DOI.
- CI publishing can use the draft id emitted by this script.

If you tagged without a draft DOI:
- Leave the tag untouched (no Zenodo deposition, no tag docs).
- Create a new draft with this script.
- Update CITATION.cff with the new reserved DOI and incremented patch version.
- Commit and tag again (version +0.0.1).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional
from urllib import parse, request, error


DEFAULT_API_BASE = "https://zenodo.org/api"
SANDBOX_API_BASE = "https://sandbox.zenodo.org/api"
CONCEPT_DOI = "10.5281/zenodo.18013539"


def _request_json(
    method: str,
    url: str,
    token: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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


def _extract_id(url: str) -> Optional[int]:
    match = re.search(r"/(\d+)(?:\\?.*)?$", url)
    if not match:
        return None
    return int(match.group(1))


def _latest_record_id(api_base: str, token: str, concept_doi: str) -> int:
    query = parse.quote(f'conceptdoi:"{concept_doi}"')
    url = f"{api_base}/records?size=1&sort=mostrecent&q={query}"
    data = _request_json("GET", url, token)
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        raise SystemExit(f"No records found for concept DOI {concept_doi}")

    record = hits[0]
    links = record.get("links", {})
    latest_url = links.get("latest") or links.get("self")
    record_id = _extract_id(latest_url) if latest_url else None
    if record_id is None:
        record_id = record.get("id")
    if record_id is None:
        raise SystemExit("Unable to determine latest record id from Zenodo response.")
    return int(record_id)


def _new_version(api_base: str, token: str, deposition_id: int) -> int:
    url = f"{api_base}/deposit/depositions/{deposition_id}/actions/newversion"
    data = _request_json("POST", url, token)
    links = data.get("links", {})
    draft_url = links.get("latest_draft")
    draft_id = _extract_id(draft_url) if draft_url else data.get("id")
    if draft_id is None:
        raise SystemExit("Unable to determine draft deposition id from Zenodo response.")
    return int(draft_id)


def _reserve_doi(api_base: str, token: str, draft_id: int) -> str:
    url = f"{api_base}/deposit/depositions/{draft_id}/actions/reserve_doi"
    data = _request_json("POST", url, token)
    doi = data.get("metadata", {}).get("doi") or data.get("doi")
    if not doi:
        raise SystemExit("Zenodo did not return a DOI for the draft deposition.")
    return doi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reserve the next Zenodo DOI for the xsnow concept DOI."
    )
    parser.add_argument(
        "--deposition-id",
        type=int,
        help="Existing deposition id (advanced; bypasses concept DOI lookup).",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Zenodo API base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help=(
            "Use the Zenodo sandbox API (test environment). "
            "Requires a sandbox token; DOIs are not real or indexed."
        ),
    )
    parser.add_argument(
        "--out",
        help="Optional path to write the reserved DOI.",
    )
    args = parser.parse_args()

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        raise SystemExit("ZENODO_TOKEN is required in the environment.")

    api_base = SANDBOX_API_BASE if args.sandbox else args.api_base

    if args.deposition_id:
        latest_id = args.deposition_id
    else:
        latest_id = _latest_record_id(api_base, token, CONCEPT_DOI)

    draft_id = _new_version(api_base, token, int(latest_id))
    doi = _reserve_doi(api_base, token, draft_id)

    print(f"draft_id={draft_id}")
    print(f"version_doi={doi}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(f"{doi}\n")


if __name__ == "__main__":
    main()
