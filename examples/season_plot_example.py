#!/usr/bin/env python
"""
Example: interactive seasonal snow-profile plots with the season_plot extension.

This reads data with ``xsnow.read()`` and selects what to plot from command-line
arguments (station / aspect / parameters), then writes an interactive HTML.

The defaults below match the bundled *gridded* sample (5 stations VIR1A..VIR5A,
slopes facing flat/N/E/S/W). With no arguments it just works:

    uv pip install plotly            # or: pip install 'xsnow[viz]'
    python examples/season_plot_example.py

Override anything from the CLI:

    # your own archive, a specific station + aspect + parameters
    python examples/season_plot_example.py \\
        --source path/to/archive.zarr \\
        --station VIR3A --aspect S \\
        --params grain_type density temperature \\
        --out vir3a_south.html --open

Notes
-----
* ``--source`` defaults to the bundled gridded sample (downloaded on first use).
* ``--aspect`` accepts a compass point ("N".."NW") or degrees (e.g. 90).
* Unknown station / aspect / parameter raise a clear error listing the available
  options (try ``--station NOPE`` or ``--aspect NW``).
"""
from __future__ import annotations

import argparse

import xsnow

# Importing the extension registers `plot_season` on every xsnowDataset.
# Extensions are opt-in: nothing is added until you import them.
import xsnow.extensions.season_plot  # noqa: F401  -> enables ds.plot_season(...)


# Defaults chosen to match the bundled gridded sample (see module docstring).
DEFAULT_STATION = "VIR1A"
DEFAULT_ASPECT = "E"                       # 90° slope; unambiguous in the sample
DEFAULT_PARAMS = ["grain_type", "lwc", "hand_hardness", "temperature"]


def _default_source() -> str:
    """Path to the bundled gridded sample (resolved lazily so --source wins)."""
    from xsnow.sample_data import snp_gridded_dir
    return f"{snp_gridded_dir()}/pros/gridded"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None,
                   help="Path readable by xsnow.read (dir, .zarr, .nc). "
                        "Defaults to the bundled gridded sample.")
    p.add_argument("--station", default=DEFAULT_STATION,
                   help=f"Station/location name (default: {DEFAULT_STATION}).")
    p.add_argument("--aspect", default=DEFAULT_ASPECT,
                   help=f'Aspect: compass ("NE") or degrees (default: {DEFAULT_ASPECT}).')
    p.add_argument("--params", nargs="+", default=DEFAULT_PARAMS,
                   help="Parameters to stack, top to bottom "
                        f"(default: {' '.join(DEFAULT_PARAMS)}).")
    p.add_argument("--out", default="season_example.html",
                   help="Output HTML path (default: season_example.html).")
    p.add_argument("--n-height", type=int, default=200,
                   help="Height-grid resolution (default: 200).")
    p.add_argument("--open", action="store_true",
                   help="Open the HTML in a browser when done.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Read the data. Falls back to the bundled sample when --source is omitted.
    source = args.source or _default_source()
    print(f"Reading: {source}")
    ds = xsnow.read(source)
    if ds is None:
        raise SystemExit(f"xsnow.read returned None for {source!r}")

    # 2. Select + plot. Selection happens inside plot_season via station/aspect;
    #    unknown/ambiguous values raise with the available options listed.
    fig = ds.plot_season(
        args.params,
        station=args.station,
        aspect=args.aspect,
        n_height=args.n_height,
        path=args.out,
        open_browser=args.open,
    )
    print(f"Wrote {args.out}  ({len(fig.data)} stacked panels) "
          f"for station={args.station!r} aspect={args.aspect!r}")


if __name__ == "__main__":
    main()
