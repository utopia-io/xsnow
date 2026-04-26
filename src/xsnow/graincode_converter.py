import numpy as np
import xarray as xr
from typing import Dict, Tuple, Iterable, Union

SLF_GRAIN_LABEL = {0:'PPgp', 1:'PP', 2:'DF', 3:'RG', 4:'FC', 5:'DH', 6:'SH', 7:'MF', 8:'IF', 9:'FCxr'}
SLF_CYCLE_LABEL = {0:'dry', 1:'mel', 2:'rfr'}

MISSING_TRIPLE: Tuple[Union[str, None], ...] = (None, None, None)

def _triplet_for_SLF_digits(digs: Iterable[int]) -> Tuple[str, str, str]:
    digs = list(digs)
    if len(digs) == 1:
        prim = sec = SLF_GRAIN_LABEL[digs[0]]
        cyc = 'dry'
    elif len(digs) == 2:
        prim, sec = (SLF_GRAIN_LABEL[digs[0]], SLF_GRAIN_LABEL[digs[1]])
        cyc = 'dry'
    elif len(digs) == 3:
        prim, sec = (SLF_GRAIN_LABEL[digs[0]], SLF_GRAIN_LABEL[digs[1]])
        cyc = SLF_CYCLE_LABEL[digs[2]]
        if cyc == 'rfr':
            prim = 'MFcr'  # special case
    else:
        raise ValueError("Code must have 1, 2, or 3 digits.")
    return (prim, sec, cyc)

def _build_static_map_SLF() -> Dict[str, Tuple[str, str, str]]:
    d: Dict[str, Tuple[str, str, str]] = {}
    # 1-digit
    for a in range(10):
        d[str(a)] = _triplet_for_SLF_digits([a])
    # 2-digit
    for a in range(10):
        for b in range(10):
            d[f"{a}{b}"] = _triplet_for_SLF_digits([a, b])
    # 3-digit (cycle 0..2)
    for a in range(10):
        for b in range(10):
            for c in (0, 1, 2):
                d[f"{a}{b}{c}"] = _triplet_for_SLF_digits([a, b, c])
    return d

SLF_TO_ICSSG: Dict[str, Tuple[str, str, str]] = _build_static_map_SLF()

def _primary_labels_SLF(codes: np.ndarray) -> np.ndarray:
    """
    Vectorized primary ICSSG label computation from 1/2/3-digit SLF codes.
    codes: numpy array of float/int; NaN allowed
    returns: numpy object array of labels or nan for missing/invalid
    """
    a = codes.astype("float64", copy=False)  # for NaN support
    out = np.empty(a.shape, dtype=object)
    out[:] = np.nan  # default

    valid = ~np.isnan(a)
    if not np.any(valid):
        return out

    ci = np.zeros_like(a, dtype=np.int64)
    ci[valid] = a[valid].astype(np.int64)

    len1 = valid & (ci < 10) & (ci >= 0)
    len2 = valid & (ci >= 10) & (ci < 100)
    len3 = valid & (ci >= 100) & (ci < 1000)

    d0 = np.full_like(ci, -1)  # initialize with -1 becomes important below
    d1 = np.full_like(ci, -1)
    d2 = np.full_like(ci,  0)  # default cycle=dry if not provided

    # 1-digit: (d0, d1) are the same; cycle=dry
    d0[len1] = ci[len1]
    d1[len1] = ci[len1]
    d2[len1] = 0

    # 2-digit: (d0, d1) from tens/ones; cycle=dry
    d0[len2] = ci[len2] // 10
    d1[len2] = ci[len2] % 10
    d2[len2] = 0

    # 3-digit: hundreds/tens/ones
    d0[len3] = (ci[len3] // 100) % 10
    d1[len3] = (ci[len3] // 10) % 10
    d2[len3] =  ci[len3] % 10

    # map primary, apply rfr->MFcr rule (cycle==2)
    grain_labels = np.asarray(list(SLF_GRAIN_LABEL.values()), dtype="object")
    prim = grain_labels[d0.clip(0, 9)]
    prim = np.where(d2 == 2, "MFcr", prim)

    # assign only on positions that had valid lengths
    valid_len = len1 | len2 | len3
    out[valid_len] = prim[valid_len]

    return out


def _secondary_labels_SLF(codes: np.ndarray) -> np.ndarray:
    """
    Vectorized secondary label computation from 1/2/3-digit SLF codes.
    For 1-digit codes, secondary == primary; for 2/3-digit, secondary = second digit.
    """
    a = codes.astype("float64", copy=False)
    out = np.empty(a.shape, dtype=object)
    out[:] = np.nan

    valid = ~np.isnan(a)
    if not np.any(valid):
        return out

    ci = np.zeros_like(a, dtype=np.int64)
    ci[valid] = a[valid].astype(np.int64)

    len1 = valid & (ci < 10) & (ci >= 0)
    len2 = valid & (ci >= 10) & (ci < 100)
    len3 = valid & (ci >= 100) & (ci < 1000)

    d1 = np.full_like(ci, -1)

    # 1-digit: secondary == that digit
    d1[len1] = ci[len1]
    # 2/3-digit: second digit (tens place)
    d1[len2] = ci[len2] % 10
    d1[len3] = (ci[len3] // 10) % 10

    grain_labels = np.asarray(list(SLF_GRAIN_LABEL.values()), dtype="object")
    sec = grain_labels[d1.clip(0, 9)]

    valid_len = len1 | len2 | len3
    out[valid_len] = sec[valid_len]

    return out


def slf_graincode_to_ICSSG_primary(
    graincodes: Union[xr.DataArray, np.ndarray]
) -> xr.DataArray:
    """
    Return primary ICSSG labels for SLF grain codes.

    Parameters
    ----------
    graincodes : xarray.DataArray or array-like (int/float)
        1-, 2-, or 3-digit SLF grain codes. NaNs allowed.

    Returns
    -------
    xarray.DataArray (dtype=object)
        Primary grain labels.

    Notes
    -----
    - 1-digit code: primary == that digit's label; secondary == same; cycle="dry".
    - 2-digit code: primary, secondary from digits; cycle="dry".
    - 3-digit code: third digit is cycle (0="dry", 1="mel", 2="rfr").
    - If cycle == 2: primary = "MFcr"

    Examples
    --------
    >>> import xarray as xr
    >>> codes = xr.DataArray([3, 33, 332, np.nan])
    >>> slf_graincode_to_ICSSG_primary(codes).values
    array(['RG', 'RG', 'MFcr', nan], dtype=object)
    """
    if isinstance(graincodes, xr.DataArray):
        labels = xr.apply_ufunc(
            _primary_labels_SLF,
            graincodes,
            input_core_dims=[[]],
            output_core_dims=[[]],
            dask="allowed",
            vectorize=False,
            output_dtypes=[object],
        )
        return labels
    else:
        arr = np.asarray(graincodes)
        return xr.DataArray(_primary_labels_SLF(arr))


def slf_graincode_to_ICSSG_secondary(
    graincodes: Union[xr.DataArray, np.ndarray]
) -> xr.DataArray:
    """
    Return secondary ICSSG labels for SLF grain codes.

    Parameters
    ----------
    graincodes : xarray.DataArray or array-like (int/float)
        1-, 2-, or 3-digit SLF grain codes. NaNs allowed.

    Returns
    -------
    xarray.DataArray (dtype=object)
        Secondary grain labels.

    Notes
    -----
    See `slf_graincode_to_ICSSG_primary()` for more details.

    Examples
    --------
    >>> import xarray as xr
    >>> codes = xr.DataArray([3, 33, 332, np.nan])
    >>> slf_graincode_to_ICSSG_secondary(codes).values
    array(['RG', 'RG', 'RG', nan], dtype=object)
    """
    if isinstance(graincodes, xr.DataArray):
        labels = xr.apply_ufunc(
            _secondary_labels_SLF,
            graincodes,
            input_core_dims=[[]],
            output_core_dims=[[]],
            dask="allowed",
            vectorize=False,
            output_dtypes=[object],
        )
        return labels
    else:
        arr = np.asarray(graincodes)
        return xr.DataArray(_secondary_labels_SLF(arr))
