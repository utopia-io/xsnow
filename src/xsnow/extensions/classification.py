"""
Summary
-------
Classify profiles, scalars, or layers in `xsnowDataset`s

This is a very generic extension that allows masking or classfiying any fields in an ``xsnowDataset``.
The module also acts as a **template for demonstrating how to develop a very simple extension**
that adds a few additional xsnowDataset methods. This coincides with **application scenario 1** from
the xsnow extension framework. 

In brief: To develop an additional ``xsnowDataset`` method, simply write a function that takes an
``xsnowDataset`` as first argument, and add the following line above the function signature:  
``@register_xsnowDataset_method``. Check out this module's source code.

Note
----
The extension includes a basic spatial classification helper that can attach a polygon/region label
per location (or per point grid) based on dataset ``latitude``/``longitude`` coordinates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method


# ---------------------- internal helpers (criteria parsing) -------------------

_OP_MAP = {
    ">":  np.greater,
    "<":  np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}
# Note: keep '==' ahead of '=' in the regex alternation. We accept common ops.
_COND_RE = re.compile(r"^\s*(\w+)\s*([<>=!]=?|==)\s*(-?\d+\.?\d*)\s*$")


def _parse_single_condition(ds_xr: xr.Dataset, cond: str) -> xr.DataArray:
    """
    Parse an atomic condition like 'density > 300' into a boolean mask DataArray.
    Aligns/broadcasts by named dims automatically (xarray semantics).
    """
    m = _COND_RE.match(cond)
    if not m:
        raise ValueError(f"Could not parse condition: {cond!r}")
    var, op, val_str = m.groups()

    if var not in ds_xr:
        raise ValueError(f"Variable {var!r} not found in dataset.")
    if op not in _OP_MAP:
        raise ValueError(f"Unsupported operator: {op!r}")

    val = float(val_str)
    da = ds_xr[var]
    op_func = _OP_MAP[op]

    mask = op_func(da, val).astype(bool)
    mask.name = f"mask_{var}_{op}_{val_str}"
    return mask


def _combine_masks(m1: xr.DataArray, logical: str, m2: xr.DataArray) -> xr.DataArray:
    """Combine boolean masks via '&' or '|' with xarray alignment."""
    if logical == "&":
        return (m1 & m2).astype(bool)
    elif logical == "|":
        return (m1 | m2).astype(bool)
    raise ValueError(f"Unsupported logical operator: {logical!r}")


def _boolean_criteria_to_mask(ds_xr: xr.Dataset, criteria: str) -> xr.DataArray:
    """
    Turn 'density > 300 & grain_size < 0.5' into a boolean mask.
    Left-to-right evaluation of '&' / '|' (no parentheses/precedence support).
    """
    parts = re.split(r"\s*([&|])\s*", criteria.strip())
    if not parts or not parts[0] or parts[0] in ("&", "|"):
        raise ValueError(f"Invalid criteria string: {criteria!r}")

    mask = None
    i = 0
    while i < len(parts):
        piece = parts[i].strip()
        this_mask = _parse_single_condition(ds_xr, piece)
        if mask is None:
            mask = this_mask
        else:
            logical = parts[i - 1].strip()
            mask = _combine_masks(mask, logical, this_mask)
        i += 2

    if mask is None:
        raise ValueError(f"Failed to build mask from criteria: {criteria!r}")
    return mask


def _mask_dataset_like(ds_xr: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """
    Apply where(mask) to variables that include all mask dims; leave others unchanged.
    """
    masked: Dict[str, xr.DataArray] = {}
    mask_dims = set(mask.dims)
    for name, var in ds_xr.data_vars.items():
        if mask_dims.issubset(set(var.dims)):
            masked[name] = var.where(mask)
        else:
            masked[name] = var
    return xr.Dataset(masked, coords=ds_xr.coords, attrs=ds_xr.attrs)


# ---------------------- internal helpers (spatial polygons) -------------------

@dataclass(frozen=True)
class _Region:
    name: str
    parts: Tuple[Tuple[np.ndarray, ...], ...]
    # parts: tuple of polygons; each polygon is tuple of rings; each ring is (N,2) float array [lon,lat]


def _as_path_maybe(obj: Any) -> Optional[Path]:
    if isinstance(obj, Path):
        return obj
    if isinstance(obj, str):
        p = Path(obj)
        if p.exists():
            return p
    return None


def _ensure_closed_ring(ring: np.ndarray) -> np.ndarray:
    if ring.shape[0] < 2:
        raise ValueError("Polygon ring must contain at least 2 points.")
    if not np.allclose(ring[0], ring[-1], equal_nan=False):
        ring = np.vstack([ring, ring[0]])
    return ring


def _ring_from_coords(coords: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Invalid ring coordinates; expected Nx2 or Nx3 array-like.")
    arr = arr[:, :2]  # drop Z if present
    return _ensure_closed_ring(arr)


def _extract_groups_at_depth(s: str, *, depth: int) -> list[str]:
    """
    Extract substrings contained in parentheses at a specific nesting depth.
    Returns the raw inner content (without the surrounding parentheses).
    """
    out: list[str] = []
    cur_depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(s):
        if ch == "(":
            cur_depth += 1
            if cur_depth == depth:
                start = i + 1
        elif ch == ")":
            if cur_depth == depth and start is not None:
                out.append(s[start:i])
                start = None
            cur_depth -= 1
    return out


def _parse_wkt_coords_list(s: str) -> np.ndarray:
    pts: list[Tuple[float, float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid WKT point token: {tok!r}")
        x = float(parts[0])
        y = float(parts[1])
        pts.append((x, y))
    if len(pts) < 3:
        raise ValueError("WKT ring must have at least 3 points.")
    return _ensure_closed_ring(np.asarray(pts, dtype=float))


def _regions_from_wkt(name: str, wkt: str) -> _Region:
    w = wkt.strip()
    upper = w.upper()
    if upper.startswith("POLYGON"):
        rings_txt = _extract_groups_at_depth(w, depth=2)
        if not rings_txt:
            raise ValueError(f"Invalid POLYGON WKT: {wkt!r}")
        rings = tuple(_parse_wkt_coords_list(r) for r in rings_txt)
        return _Region(name=str(name), parts=(rings,))
    if upper.startswith("MULTIPOLYGON"):
        poly_txts = _extract_groups_at_depth(w, depth=2)
        if not poly_txts:
            raise ValueError(f"Invalid MULTIPOLYGON WKT: {wkt!r}")
        parts: list[Tuple[np.ndarray, ...]] = []
        for poly_txt in poly_txts:
            rings_txt = _extract_groups_at_depth(poly_txt, depth=1)
            if not rings_txt:
                raise ValueError(f"Invalid MULTIPOLYGON part: {poly_txt!r}")
            parts.append(tuple(_parse_wkt_coords_list(r) for r in rings_txt))
        return _Region(name=str(name), parts=tuple(parts))
    raise ValueError("Only POLYGON and MULTIPOLYGON WKT are supported.")


def _feature_name(props: Mapping[str, Any], fallback: str) -> str:
    for key in ("name", "id", "title", "label"):
        if key in props and props[key] is not None:
            return str(props[key])
    return fallback


def _geojson_crs(obj: Mapping[str, Any]) -> Optional[str]:
    """
    Attempt to extract CRS from GeoJSON in the (deprecated) ``crs`` member.
    Returns a pyproj-compatible CRS string (e.g. "EPSG:4326") if found.
    """
    crs = obj.get("crs")
    if crs is None:
        return None
    if isinstance(crs, str):
        return crs
    if not isinstance(crs, Mapping):
        return None
    if crs.get("type") == "name":
        props = crs.get("properties") or {}
        if isinstance(props, Mapping) and props.get("name"):
            return str(props["name"])
    return None


def _feature_name_from_props(
    props: Mapping[str, Any],
    *,
    field: str,
    fallback: str,
) -> str:
    """
    Determine a feature name from GeoJSON properties.

    - Prefer ``props[field]`` if present.
    - Fall back to common alternatives (name/id/title/label).
    - Fall back to ``fallback`` as last resort.
    """
    if field in props and props[field] is not None:
        return str(props[field])
    return _feature_name(props, fallback=fallback)


def _regions_from_geojson(obj: Mapping[str, Any], *, feature_name_field: str) -> list[_Region]:
    def _iter_features(o: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        t = o.get("type")
        if t == "FeatureCollection":
            for f in o.get("features", []) or []:
                if isinstance(f, Mapping):
                    yield f
            return
        if t == "Feature":
            yield o
            return
        if t in {"Polygon", "MultiPolygon"}:
            yield {"type": "Feature", "geometry": o, "properties": {}}
            return
        raise ValueError(f"Unsupported GeoJSON type: {t!r}")

    regions: list[_Region] = []
    for i, feat in enumerate(_iter_features(obj)):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        props = feat.get("properties") or {}
        if not isinstance(props, Mapping):
            props = {}
        name = _feature_name_from_props(props, field=feature_name_field, fallback=f"poly_{i}")

        if gtype == "Polygon":
            if not isinstance(coords, list) or not coords:
                raise ValueError("GeoJSON Polygon coordinates must be a non-empty list.")
            rings = tuple(_ring_from_coords(r) for r in coords)
            regions.append(_Region(name=name, parts=(rings,)))
        elif gtype == "MultiPolygon":
            if not isinstance(coords, list) or not coords:
                raise ValueError("GeoJSON MultiPolygon coordinates must be a non-empty list.")
            parts: list[Tuple[np.ndarray, ...]] = []
            for poly in coords:
                if not isinstance(poly, list) or not poly:
                    raise ValueError("GeoJSON MultiPolygon part must be a non-empty list of rings.")
                parts.append(tuple(_ring_from_coords(r) for r in poly))
            regions.append(_Region(name=name, parts=tuple(parts)))
        else:
            raise ValueError(f"Unsupported GeoJSON geometry type: {gtype!r}")
    return regions


def _normalize_crs_str(crs: str) -> str:
    return str(crs).strip().upper().replace(" ", "")


def _load_regions(polygons: Any, *, feature_name_field: str) -> Tuple[list[_Region], Optional[str], Optional[str]]:
    """
    Load regions from:
      - GeoJSON file path (.geojson/.json)
      - GeoJSON mapping (FeatureCollection/Feature/Polygon/MultiPolygon)
      - WKT string (POLYGON/MULTIPOLYGON)
      - Mapping[str, str] interpreted as {name: wkt}
      - Sequence[str] of WKT (names auto-assigned poly_0, ...)
      - Sequence[Tuple[str, str]] of (name, WKT)
    Returns (regions, source_label, detected_polygons_crs).
    """
    p = _as_path_maybe(polygons)
    if p is not None:
        suffix = p.suffix.lower()
        text = p.read_text(encoding="utf-8")
        if suffix in {".geojson", ".json"}:
            obj = json.loads(text)
            if not isinstance(obj, Mapping):
                raise ValueError("GeoJSON file must decode to a JSON object.")
            detected = _geojson_crs(obj)
            return _regions_from_geojson(obj, feature_name_field=feature_name_field), str(p), detected
        if suffix in {".wkt", ".txt"}:
            # allow one region per non-empty line: "name<TAB>WKT" or just "WKT"
            regions: list[_Region] = []
            for i, line in enumerate(text.splitlines()):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    nm, wk = line.split("\t", 1)
                    nm = nm.strip() or f"poly_{i}"
                    wk = wk.strip()
                else:
                    nm, wk = f"poly_{i}", line
                regions.append(_regions_from_wkt(nm, wk))
            if not regions:
                raise ValueError("WKT file contained no polygons.")
            return regions, str(p), None
        raise ValueError(f"Unsupported polygon file type: {suffix!r} (expected .geojson/.json/.wkt/.txt)")

    if isinstance(polygons, Mapping):
        # GeoJSON mapping?
        if "type" in polygons:
            detected = _geojson_crs(polygons)
            return _regions_from_geojson(polygons, feature_name_field=feature_name_field), None, detected
        # Otherwise treat as mapping of names -> WKT
        regions = [_regions_from_wkt(str(k), str(v)) for k, v in polygons.items()]
        return regions, None, None

    if isinstance(polygons, str):
        # WKT string
        return [_regions_from_wkt("poly_0", polygons)], None, None

    if isinstance(polygons, Sequence):
        if not polygons:
            raise ValueError("polygons must not be empty.")
        if all(isinstance(x, str) for x in polygons):
            regions = [_regions_from_wkt(f"poly_{i}", str(w)) for i, w in enumerate(polygons)]
            return regions, None, None
        if all(isinstance(x, tuple) and len(x) == 2 for x in polygons):
            regions = [_regions_from_wkt(str(n), str(w)) for (n, w) in polygons]  # type: ignore[misc]
            return regions, None, None

    raise TypeError(
        "polygons must be a path/str WKT, a GeoJSON mapping, a mapping of {name: WKT}, "
        "or a sequence of WKT strings / (name, WKT) pairs."
    )


def _points_in_ring(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """
    Ray casting point-in-polygon for a single ring.
    x,y are 1D arrays. ring is (N,2) and must be closed.
    """
    if ring.shape[0] < 4:
        return np.zeros_like(x, dtype=bool)

    x0 = ring[:-1, 0]
    y0 = ring[:-1, 1]
    x1 = ring[1:, 0]
    y1 = ring[1:, 1]

    inside = np.zeros_like(x, dtype=bool)
    # Toggle parity for each edge intersection (memory-friendly; no (edges x points) allocation)
    for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
        cond_y = (yi0 > y) != (yi1 > y)
        den = (yi1 - yi0)
        if den == 0:
            x_int = np.inf
        else:
            x_int = (xi1 - xi0) * (y - yi0) / den + xi0
        inside ^= (cond_y & (x < x_int))
    return inside


def _points_in_region(x: np.ndarray, y: np.ndarray, region: _Region) -> np.ndarray:
    inside_region = np.zeros_like(x, dtype=bool)
    for rings in region.parts:
        if not rings:
            continue
        outer = rings[0]
        inside = _points_in_ring(x, y, outer)
        # holes
        for hole in rings[1:]:
            inside &= ~_points_in_ring(x, y, hole)
        inside_region |= inside
    return inside_region


def _points_in_region_block(
    x: np.ndarray,
    y: np.ndarray,
    *,
    region: _Region,
) -> np.ndarray:
    """Classify one eager coordinate block against a region."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        inside[valid] = _points_in_region(x[valid], y[valid], region)
    return inside


def _crs_equal(a: str, b: str) -> bool:
    """
    CRS equality test with a best-effort fallback if pyproj isn't available.
    """
    a_norm = _normalize_crs_str(a)
    b_norm = _normalize_crs_str(b)
    if a_norm == b_norm:
        return True
    # Best-effort EPSG normalization without pyproj
    m_a = re.search(r"EPSG:?(?P<code>\d+)", a_norm)
    m_b = re.search(r"EPSG:?(?P<code>\d+)", b_norm)
    if m_a and m_b and m_a.group("code") == m_b.group("code"):
        return True
    try:
        import pyproj  # type: ignore
    except Exception:
        return False
    try:
        return pyproj.CRS.from_user_input(a) == pyproj.CRS.from_user_input(b)
    except Exception:
        return False


def _is_epsg4326(crs: str) -> bool:
    if _normalize_crs_str(crs) in {"EPSG:4326", "EPSG4326", "4326"}:
        return True
    try:
        import pyproj  # type: ignore
    except Exception:
        return False
    try:
        return pyproj.CRS.from_user_input(crs).to_epsg() == 4326
    except Exception:
        return False


def _transform_regions(regions: Sequence[_Region], src_crs: str, dst_crs: str) -> list[_Region]:
    """
    Transform all region coordinates from src_crs to dst_crs using pyproj.
    """
    try:
        import pyproj  # type: ignore
    except Exception as e:
        raise ImportError(
            "Polygon CRS differs from dataset CRS and transformation is required, "
            "but 'pyproj' is not installed."
        ) from e

    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out: list[_Region] = []
    for r in regions:
        new_parts: list[Tuple[np.ndarray, ...]] = []
        for poly in r.parts:
            new_rings: list[np.ndarray] = []
            for ring in poly:
                xs, ys = transformer.transform(ring[:, 0], ring[:, 1])
                rr = np.stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)], axis=1)
                rr = _ensure_closed_ring(rr)
                new_rings.append(rr)
            new_parts.append(tuple(new_rings))
        out.append(_Region(name=r.name, parts=tuple(new_parts)))
    return out


def classify_by_bins(da: xr.DataArray, *, bins: np.ndarray, right: bool = True) -> xr.DataArray:
    """
    Bin a DataArray into integer classes using ``numpy.digitize``.

    This helper is designed to work hand-in-hand with :meth:`xsnowDataset.classify` and
    to act as a template for custom user-defined classification functions.

    Parameters
    ----------
    da : xr.DataArray
        Input values to bin. Output preserves dims/coords and aligns with ``da``.
    bins : array-like
        Bin edges (length B). Output classes are integers in ``[0, B]`` (B+1 classes).
    right : bool, default True
        If True, intervals are ``(-inf, b0]``, ``(b0, b1]``, ..., ``(b_{B-1}, +inf)``.
        If False, intervals are ``[-inf, b0)``, ``[b0, b1)``, ..., ``[b_{B-1}, +inf)``.

    Returns
    -------
    xr.DataArray
        Integer class labels with the same shape as ``da``.

    See Also
    --------
    classify : Attach a classifier result to an xsnowDataset.

    Examples
    --------
    >>> hs_bins = np.array([0.25, 0.50, 1.00, 1.50, 2.00])
    >>> class_attrs = {
    ...   "long_name": "Snow depth class (6 bins)",
    ...   "method": "np.digitize",
    ...   "bin_edges": hs_bins.tolist(),
    ...   "right_closed": True,
    ... }
    >>> ds2 = ds.classify(
    ...   classify_by_bins,
    ...   output_name="HS_class",
    ...   output_kind="coord",
    ...   input_var="HS",
    ...   attrs=class_attrs,
    ...   bins=hs_bins,
    ...   right=True,
    ... )
    >>> ds_deep = ds2.where(ds2.HS_class == 5, drop=True)
    """
    bins_arr = np.asarray(bins, dtype=float)

    def _digitize(x: np.ndarray, *, bins: np.ndarray, right: bool) -> np.ndarray:
        return np.digitize(x, bins, right=right)

    classes = xr.apply_ufunc(
        _digitize,
        da,
        dask="allowed",
        kwargs={"bins": bins_arr, "right": right},
        output_dtypes=[np.int64],
    )
    return classes.astype(np.int64)


# ---------------------- 1) general function-based classification --------------

@register_xsnowDataset_method
def classify(
    self: xsnowDataset,
    func: Callable[[xr.Dataset] | [xr.DataArray], xr.DataArray],
    *,
    output_name: str = "class_id",
    output_kind: Literal["coord", "data"] = "coord",
    input_var: Optional[str] = None,
    attrs: Optional[dict] = None,
    **func_kwargs
) -> xsnowDataset:
    """
    General classification: run a user function to compute a classification array,
    then attach it as a coordinate or data variable.

    Parameters
    ----------
    func : callable
        A custom classification function:
        Either f(xr.Dataset) -> xr.DataArray or f(xr.DataArray) -> xr.DataArray.
        The returned DataArray must align/broadcast within the dataset.
    output_name : str, default "class_id"
        Name of the resulting coord or data var.
    output_kind : {"coord","data"}, default "coord"
        How to store the classification result.
        * `"coord"`: attach as a coordinate (good for use with `where(...)`, e.g.
                     `ds.where(ds.hs_class == 2, drop=True)`, or for grouping).
        * `"data"`: attach as a data variable (good if you’ll plot it, compute stats on it, or 
                    don’t need to use it as a selector).
    input_var : str, optional
        If set, call func(self.data[input_var]); else func(self.data).
    attrs : dict, optional
        Metadata to attach to the resulting DataArray.
    **func_kwargs :
        Extra keyword arguments forwarded to `func`.

    Returns
    -------
    xsnowDataset
        A new dataset wrapper with the classification attached.

    See Also
    --------
    classify_citeria : A generic, boolean classification function.
    classify_by_bins : Bin a variable into integer classes (template classifier).

    Examples
    --------
    >>> def by_hs_tertiles(hs: xr.DataArray) -> xr.DataArray:
    ...     c = xr.zeros_like(hs, dtype=np.int64)
    ...     c = c.where(hs <= 0.5, 1)     # > 0.5 --> at least 1
    ...     c = c.where(hs <= 1.5, 2)     # > 1.5 --> 2
    ...     c.attrs["mapping"] = {0: "shallow", 1: "medium", 2: "deep"}
    ...     return c
    >>> ds2 = ds.classify(by_hs, output_name="hs_class", output_kind="coord", input_var="HS")
    >>> ds2.where(ds2.hs_class == 2, drop=True)

    >>> from xsnow.extensions.classification import classify_by_bins
    >>> hs_bins = np.array([0.25, 0.50, 1.00, 1.50, 2.00])
    >>> hs_mapping = {
            0: "(-inf, 0.25]",
            1: "(0.25, 0.50]",
            2: "(0.50, 1.00]",
            3: "(1.00, 1.50]",
            4: "(1.50, 2.00]",
            5: "(2.00, +inf)"
        }
    >>> class_attrs = {
            "mapping": hs_mapping,
            "long_name": "Snow depth class (6 bins)",
            "method": "np.digitize",
            "bin_edges": hs_bins.tolist(),
            "right_closed": True,
        }
    >>> ds2 = ds.classify(
            classify_by_bins,
            output_name="HS_class",
            output_kind="coord",
            input_var="HS",          # classifier gets ds["HS"]
            attrs=class_attrs,
            bins=hs_bins,            # forwarded to classify_by_bins()
            right=True,
        )
    >>> ds_extremely_deep = ds2.where(ds2.HS_class == 5, drop=True)
    """
    if self.data is None:
        raise ValueError("Cannot classify an empty dataset.")

    xds = self.data
    target = xds[input_var] if input_var else xds
    out = func(target, **func_kwargs)
    if not isinstance(out, xr.DataArray):
        raise TypeError(f"classifier must return xr.DataArray, got {type(out).__name__}")
    if out.name is None:
        out = out.rename(output_name)
    if attrs:
        out = out.copy()
        out.attrs.update(attrs)

    if output_kind == "coord":
        new_xr = xds.assign_coords({output_name: out})
    elif output_kind == "data":
        new_xr = xds.assign({output_name: out})
    else:
        raise ValueError("output_kind must be 'coord' or 'data'.")

    return self.__class__(new_xr)


# ---------------------- 2) criteria-based classification ---------------------

@register_xsnowDataset_method
def classify_criteria(
    self: xsnowDataset,
    criteria: str,
    *,
    name: str = "classification_mask",
    kind: Literal["mask", "coord", "data"] = "mask",
    attrs: Optional[dict] = None,
) -> xr.DataArray | xsnowDataset:
    """
    Criteria-based boolean classification: build a boolean mask from a string expression.
    The mask can be returned, or attached as a coordinate/data variable.

    Parameters
    ----------
    criteria : str
        Boolean conditions chained with '&' and/or '|', e.g.
        "density > 300 & grain_size < 0.5".
        (Minimal parser: left-to-right; no parentheses/precedence.)
    name : str, default "classification_mask"
        Name for the attached output.
    kind : {"mask","coord","data"}, default "mask"
        - "mask": return the mask (xr.DataArray)
        - "coord":  attach as a coordinate and return xsnowDataset
        - "data":   attach as a data var and return xsnowDataset
    attrs : dict, optional
        Attributes to add to the mask array.

    Returns
    -------
    xr.DataArray | xsnowDataset

    See Also
    --------
    classify : Generic, customizable classification into multiple classes

    Examples
    --------
    >>> mask = ds.classify_criteria("density < 200 & grain_size > 0.8", kind="mask")
    >>> ds_sel = ds.where(mask, drop=True)

    >>> ds2 = ds.classify_criteria("HS > 1.5", name="deep_snow", kind="coord")
    >>> ds2.where(ds2.deep_snow, drop=True)
    """
    if self.data is None:
        raise ValueError("Cannot classify an empty dataset.")

    # If you support GPU, you might want: xds = self.to_cpu().data
    xds = self.data
    mask = _boolean_criteria_to_mask(xds, criteria)
    if attrs:
        mask = mask.copy()
        mask.attrs.update(attrs)

    if kind == "mask":
        return mask
    elif kind == "coord":
        new_xr = xds.assign_coords({name: mask})
        return self.__class__(new_xr)
    elif kind == "data":
        new_xr = xds.assign({name: mask})
        return self.__class__(new_xr)
    else:
        raise ValueError("kind must be one of {'mask','coord','data'}.")


# ---------------------- optional convenience: mask and keep shape ------------

@register_xsnowDataset_method
def mask_by_criteria(self: xsnowDataset, criteria: str) -> xsnowDataset:
    """
    Convenience wrapper around `classify_criteria(..., kind='mask')`:
    masks non-matching layers (i.e., NaN) while keeping shape/dimensions identical. 
    If the criteria are layer-based, scalars are untouched, and vice versa.

    Returns
    -------
    Masked xsnowDataset with identical shape/dimensions as the input, but NaNs occur where 
    the criteria are not True.
    """
    if self.data is None:
        raise ValueError("Cannot filter an empty dataset.")
    mask = self.classify_criteria(criteria, kind="mask")
    return self.__class__(_mask_dataset_like(self.data, mask))


# ---------------------- 3) spatial classification ----------------------------

@register_xsnowDataset_method
def classify_spatially(
    self: xsnowDataset,
    polygons: Any,
    *,
    name: str = "polygon",
    kind: Literal["mask", "coord", "data"] = "coord",
    attrs: Optional[dict] = None,
    outside_label: str = "outside",
    polygons_crs: Optional[str] = None,
    feature_name_field: str = "id",
) -> xr.DataArray | xsnowDataset:
    """
    Spatial classification of points into named polygons/regions.

    This attaches a region label (or returns a per-polygon mask) based on the dataset's
    horizontal coordinates. Typical use-cases:
      - select all data from a specific region
      - aggregate over regions via ``groupby``
    
    The dataset CRS is taken from ``self.attrs['crs']``. If it is EPSG:4326,
    this method uses the ``longitude``/``latitude`` coordinates. Otherwise it expects
    projected coordinates ``easting``/``northing``.
    
    If the polygon CRS differs from the dataset CRS, polygons are transformed to the
    dataset CRS using ``pyproj`` (optional dependency).

    Parameters
    ----------
    polygons
        One of:
          - path to a GeoJSON file (.geojson/.json)
          - GeoJSON mapping (FeatureCollection/Feature/Polygon/MultiPolygon)
          - WKT string (POLYGON/MULTIPOLYGON)
          - mapping of ``{name: wkt}``
          - sequence of WKT strings (names auto: poly_0, poly_1, ...)
          - sequence of ``(name, wkt)`` pairs
        Coordinate order is assumed to be ``(x, y)``. For EPSG:4326 this means
        ``(longitude, latitude)``; for projected CRSs, ``(easting, northing)``.
    name : str, default "polygon"
        Name for the attached label coordinate/data var (or the returned mask name).
    kind : {"mask","coord","data"}, default "coord"
        - "mask": return a boolean mask with dims ``('polygon', <point-dims...>)``
        - "coord": attach a string label per point as a coordinate and return xsnowDataset
        - "data": attach a string label per point as a data variable and return xsnowDataset
    attrs : dict, optional
        Metadata to attach to the output.
    outside_label : str, default "outside"
        Label assigned to points not contained in any provided polygon.
    polygons_crs : str, optional
        CRS of the input polygons (e.g. ``"EPSG:4326"``). If not provided, the CRS is
        only auto-detected for GeoJSON that contains an explicit (deprecated) top-level
        ``"crs"`` member. If the polygon CRS cannot be identified unambiguously, this
        method raises.
    feature_name_field : str, default "id"
        For GeoJSON inputs, this field is read from each feature's ``properties`` to
        determine the polygon/region name. If absent, it falls back to common alternatives
        (``name``, ``id``, ``title``, ``label``) and finally ``poly_<i>``.

    Returns
    -------
    xr.DataArray | xsnowDataset

    Notes
    -----
    Dask-backed horizontal coordinates are classified lazily and blockwise.
    The polygon geometries remain small eager metadata, while each coordinate
    block is evaluated only when the returned labels or masks are computed.

    Examples
    --------
    >>> import xarray as xr
    >>> import xsnow
    >>> xs = xsnow.sample_data.snp_gridded_ds().compute()
    >>> xs.longitude.values
    array([11.191828, 11.286182, 11.2973  , 11.392236, 11.5008  ])
    >>> xs.latitude.values
    array([47.146392, 47.436152, 47.148326, 47.437999, 47.367788])
    >>> xs.altitude.values
    array([2372., 1749., 1860., 1801., 2066.])

    Create two regions (rectangles) and leave one location outside:

    >>> regions = {
    ...   "valley": "POLYGON ((11.15 47.12, 11.15 47.17, 11.33 47.17, 11.33 47.12, 11.15 47.12))",
    ...   "ridge":  "POLYGON ((11.25 47.42, 11.25 47.46, 11.43 47.46, 11.43 47.42, 11.25 47.42))",
    ... }
    >>> xs2 = xs.classify_spatially(regions, name="region", kind="coord", polygons_crs="EPSG:4326")
    >>> xs_valley = xs2.where(xs2.region == "valley", drop=True)

    Group by region and slope (slope is already an existing coordinate):

    >>> xs_stats = xs2.groupby(["region", "slope"]).mean()

    Group by region, slope, and an elevation band (>=2000 m) created via ``classify_criteria``:

    >>> xs3 = xs2.classify_criteria(
    ...   "altitude >= 2000",
    ...   name="band",
    ...   kind="coord",
    ...   attrs={"mapping": {False: "<2000m", True: ">=2000m"}},
    ... )
    >>> xs_stats2 = xs3.groupby(["region", "slope", "band"]).mean()
    """
    if self.data is None:
        raise ValueError("Cannot classify an empty dataset.")

    xds = self.data
    dataset_crs = xds.attrs.get("crs")
    if not dataset_crs:
        raise ValueError(
            "Dataset CRS is required for spatial classification; expected `xsnowDataset.data.attrs['crs']` "
            "(e.g. 'EPSG:4326')."
        )

    use_lon_lat = _is_epsg4326(str(dataset_crs))
    if use_lon_lat:
        if "latitude" not in xds.coords or "longitude" not in xds.coords:
            raise ValueError("Dataset CRS is EPSG:4326 but 'latitude'/'longitude' coordinates are missing.")
        x_da = xds.coords["longitude"]
        y_da = xds.coords["latitude"]
    else:
        if "easting" not in xds.coords or "northing" not in xds.coords:
            raise ValueError(
                "Dataset CRS is not EPSG:4326; expected projected coordinates 'easting' and 'northing' as coords."
            )
        x_da = xds.coords["easting"]
        y_da = xds.coords["northing"]

    regions, source_label, detected_polygons_crs = _load_regions(polygons, feature_name_field=feature_name_field)
    if not regions:
        raise ValueError("No polygons/regions were loaded.")

    poly_crs = polygons_crs or detected_polygons_crs
    if not poly_crs:
        raise ValueError(
            "Polygon CRS could not be identified unambiguously. "
            "Pass `polygons_crs='EPSG:xxxx'` (or include an explicit GeoJSON 'crs' member)."
        )

    if not _crs_equal(str(poly_crs), str(dataset_crs)):
        try:
            regions = _transform_regions(regions, str(poly_crs), str(dataset_crs))
        except Exception as e:
            raise ValueError(
                f"Polygon CRS ({poly_crs!r}) does not match dataset CRS ({dataset_crs!r}) and could not be "
                "transformed; pass a valid CRS via `polygons_crs=...` and ensure 'pyproj' is installed."
            ) from e

    if x_da.dims != y_da.dims or x_da.shape != y_da.shape:
        raise ValueError("Dataset horizontal coordinate arrays must have matching dims and shape.")

    region_names = [r.name for r in regions]
    region_masks = [
        xr.apply_ufunc(
            _points_in_region_block,
            x_da,
            y_da,
            kwargs={"region": region},
            dask="parallelized",
            vectorize=False,
            output_dtypes=[bool],
        )
        for region in regions
    ]
    mask_da = xr.concat(
        region_masks,
        dim=xr.IndexVariable(
            "polygon", np.asarray(region_names, dtype=object)
        ),
    ).rename(name)

    if kind == "mask":
        if attrs:
            mask_da = mask_da.copy()
            mask_da.attrs.update(attrs)
        return mask_da

    # Apply in reverse order so an earlier polygon overwrites every later
    # match, preserving the documented "first polygon match wins" rule.
    label_da = xr.full_like(
        x_da, fill_value=str(outside_label), dtype=object
    )
    for i in range(len(region_names) - 1, -1, -1):
        label_da = xr.where(
            mask_da.isel(polygon=i, drop=True),
            region_names[i],
            label_da,
        )
    label_da = label_da.rename(name)
    label_da.attrs.update(
        {
            "outside_label": str(outside_label),
            "regions": [str(n) for n in region_names],
            "source": source_label,
            "method": "point-in-polygon (ray casting), first match wins",
            "dataset_crs": str(dataset_crs),
            "polygons_crs": str(poly_crs),
            "x_coord": str(x_da.name),
            "y_coord": str(y_da.name),
        }
    )
    if attrs:
        label_da = label_da.copy()
        label_da.attrs.update(attrs)

    if kind == "coord":
        return self.__class__(xds.assign_coords({name: label_da}))
    if kind == "data":
        return self.__class__(xds.assign({name: label_da}))
    raise ValueError("kind must be one of {'mask','coord','data'}.")
