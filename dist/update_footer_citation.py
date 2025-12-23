#!/usr/bin/env python3
"""
Update the website docs footer citation in docs/_config.yml.

This script renders the Copernicus-style citation via xsnow.cite("copernicus")
and injects it into the Sphinx Book Theme `extra_footer` block.
Run it before building docs (Makefile and CI call it automatically).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _render_citation(repo_root: Path) -> str:
    sys.path.insert(0, str(repo_root / "src"))
    import xsnow  # type: ignore

    citation = xsnow.cite("copernicus", display=False)
    if citation is None:
        raise SystemExit("xsnow.cite() returned no citation string.")
    return citation.strip()


def _update_extra_footer(config_path: Path, citation: str) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    i = 0
    replaced = False

    while i < len(lines):
        line = lines[i]
        if line.strip() == "extra_footer: |":
            out_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].startswith("        "):
                i += 1
            out_lines.append("        <div class=\"citation-footer\">")
            out_lines.append(f"          <p>{citation}</p>")
            out_lines.append("        </div>")
            replaced = True
            continue
        out_lines.append(line)
        i += 1

    if not replaced:
        raise SystemExit("extra_footer block not found in docs/_config.yml.")

    config_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "docs" / "_config.yml"
    citation = _render_citation(repo_root)
    _update_extra_footer(config_path, citation)


if __name__ == "__main__":
    main()
