import numpy as np
import xarray as xr
from xsnow import registry
from xsnow.plugin import register_xsnowDataset_method
from xsnow.extensions.assign import assign_to_dataset

@register_xsnowDataset_method
def compute_rta(ds: xr.Dataset, config, rta_threshold=0.8) -> xr.Dataset:
    """
    Compute Relative Threshold Area (RTA) for each profile in an xsnow Dataset.
    This is a simplified but faithful translation of sarp.snowprofile::computeRTA().
    """
    dataset_cls = ds.__class__
    depth      = ds["height"].values
    hardness   = ds["hand_hardness"].values
    grain_size = ds["grain_size"].values
    grain_type = ds["grain_type"].values
    grain_type_str = ds["grain_type"].icssg.primary().values
    hard_diff  = ds["hardness_difference"].values
    gs_diff    = ds["gs_difference"].values

    *outer_shape, L = depth.shape
       
    thick = _thickness(depth)
    weibull = _weibull_depth(depth)
    crust_coeff = _crust_correction(depth, hardness, thick)
    weibull_adj = _adjust_weibull(weibull, crust_coeff)
    dhard_upper, dhard_lower = _interface_diffs(hard_diff)
    dgsize_upper, dgsize_lower = _interface_diffs(gs_diff)
    hbar, gbar, dhbar, dgbar, wbar, sd_h, sd_g, sd_dh, sd_dg, sd_w = _profile_stats(
        hardness, grain_size, dhard_upper, dgsize_upper, weibull_adj
    )
    
    lem_sum_upper = _lemons_sum(
        hardness, grain_size, grain_type_str,
        dhard_upper, dgsize_upper, weibull_adj,
        hbar, gbar, dhbar, dgbar, wbar,
        sd_h, sd_g, sd_dh, sd_dg, sd_w
    )

    lem_sum_lower = _lemons_sum(
        hardness, grain_size, grain_type_str,
        dhard_lower, dgsize_lower, weibull_adj,
        hbar, gbar, dhbar, dgbar, wbar,
        sd_h, sd_g, sd_dh, sd_dg, sd_w
    )
    rta = np.nanmax(np.stack([lem_sum_upper, lem_sum_lower]), axis=0)
    rta_max = np.nanmax(rta, axis=-1)
    rta = rta / np.where(rta_max[..., None] == 0, 1.0, rta_max[..., None])

    dims = ds["height"].dims
    outer_dims = dims[:-1]
    coords = {d: ds[d].values for d in outer_dims}
    coords["layer"] = ds["height"].coords["layer"].values
    dsArray = xr.DataArray(rta, dims=dims, coords=coords, name="rta")
    ds = assign_to_dataset(ds, dsArray, "rta", "Relative Threshold Area", "dimensionless")
    sk38_masked = ds['sk38'].where(ds['rta'] > rta_threshold)
    ds = assign_to_dataset(ds, sk38_masked, "sk38_masked", f"Sk38 Hardness Masked by RTA > {rta_threshold}", "dimensionless")

    ds = _rta_interface(
        ds, hardness, grain_size, grain_type_str, dhard_lower, dgsize_lower, weibull_adj,
        hbar, gbar, dhbar, dgbar, wbar,
        sd_h, sd_g, sd_dh, sd_dg, sd_w,
        dims, coords
    )
    
    return dataset_cls(ds)

#region rta helpers
def _rta_interface(ds, hardness, grain_size, grain_type, dhard_lower, dgsize_lower, weibull_adj, hbar, gbar, dhbar, dgbar, wbar, sd_h, sd_g, sd_dh, sd_dg, sd_w, dims, coords) -> xr.DataArray:
    
    # --- INTERFACE RTA (as in R) ---
    def shift(arr, n):
        if n == 0:
            return arr
        result = np.full_like(arr, np.nan)
        if n > 0:
            result[..., n:] = arr[..., :-n]
        else:
            result[..., :n] = arr[..., -n:]
        return result

    hardness_above = shift(hardness, 1)
    grain_size_above = shift(grain_size, 1)
    weibull_adj_above = shift(weibull_adj, 1)
    grain_type_above = shift(grain_type, 1)

    pwl_types = np.array(["FC", "FCxr", "DH", "SH"])
    def is_pwl(gt):
        return np.isin(gt, pwl_types).astype(float)

    bolmat_above = np.empty(hardness.shape + (6,), dtype=float)
    bolmat_above[..., 0] = is_pwl(grain_type_above)
    bolmat_above[..., 1] = (hardness_above - hbar[..., None]) / sd_h[..., None]
    bolmat_above[..., 2] = (grain_size_above - gbar[..., None]) / sd_g[..., None]
    bolmat_above[..., 3] = (dhard_lower - dhbar[..., None]) / sd_dh[..., None]
    bolmat_above[..., 4] = (dgsize_lower - dgbar[..., None]) / sd_dg[..., None]
    bolmat_above[..., 5] = (weibull_adj_above - wbar[..., None]) / sd_w[..., None]

    bolmat_below = np.empty(hardness.shape + (6,), dtype=float)
    bolmat_below[..., 0] = is_pwl(grain_type)
    bolmat_below[..., 1] = (hardness - hbar[..., None]) / sd_h[..., None]
    bolmat_below[..., 2] = (grain_size - gbar[..., None]) / sd_g[..., None]
    bolmat_below[..., 3] = (dhard_lower - dhbar[..., None]) / sd_dh[..., None]
    bolmat_below[..., 4] = (dgsize_lower - dgbar[..., None]) / sd_dg[..., None]
    bolmat_below[..., 5] = (weibull_adj - wbar[..., None]) / sd_w[..., None]

    def minmax_scale(mat):
        minv = np.nanmin(mat, axis=-2, keepdims=True)
        maxv = np.nanmax(mat, axis=-2, keepdims=True)
        diff = np.maximum(maxv - minv, 1.0)
        return (mat - minv) / diff

    def scale_hardness(arr):
        return 1.0 - arr

    for i in range(6):
        if i == 1:
            bolmat_above[..., i] = scale_hardness(minmax_scale(bolmat_above[..., i]))
            bolmat_below[..., i] = scale_hardness(minmax_scale(bolmat_below[..., i]))
        else:
            bolmat_above[..., i] = minmax_scale(bolmat_above[..., i])
            bolmat_below[..., i] = minmax_scale(bolmat_below[..., i])

    rta_interface = np.maximum(
        np.nansum(bolmat_above, axis=-1),
        np.nansum(bolmat_below, axis=-1)
    )
    rta_interface_max = np.nanmax(rta_interface, axis=-1)
    rta_interface = rta_interface / np.where(rta_interface_max[..., None] == 0, 1.0, rta_interface_max[..., None])

    dsArray_interface = xr.DataArray(rta_interface, dims=dims, coords=coords, name="rta_interface")
    ds = assign_to_dataset(ds, dsArray_interface, "rta_interface", "Relative Threshold Area (interface)", "dimensionless")
    return ds

def _thickness(depth):
    thickness = np.zeros_like(depth, dtype=float)
    thickness[..., 1:] = np.diff(depth, axis=-1)
    return thickness

def _weibull_depth(depth):
    w1 = 2.5
    w2 = 50
    return (w1 / w2) * depth**(w1 - 1) * np.exp(-(depth / w2)**w1)

def _crust_correction(depth, hardness, thickness):
    # Compute crust mask
    crust_mask = (hardness >= 3) & (thickness > 1)
    crust_value = np.where(crust_mask, np.exp(-depth / 20), 0.0)
    crust_coeff = crust_value[..., ::-1].cumsum(axis=-1)[..., ::-1]
    return crust_coeff

def _adjust_weibull(weibull, crust_coeff):
    return weibull - crust_coeff

def _interface_diffs(diff: xr.DataArray):
    # diff: (..., layer) already
    upper = np.full_like(diff, np.nan, dtype=float)
    lower = np.full_like(diff, np.nan, dtype=float)
    upper[..., 1:] = diff[..., :-1]
    lower[..., :-1] = diff[..., 1:]
    return upper, lower

def mean_last(a):
    return np.nanmean(a, axis=-1)

def sd_last(a, m):
    sd = np.sqrt(np.nanmean((a - m[..., None])**2, axis=-1))
    return np.maximum(sd, 1.0)

def _profile_stats(hardness, grain_size, dhard_upper, dgsize_upper, weibull_adj):
    hbar  = mean_last(hardness)
    gbar  = mean_last(grain_size)
    dhbar = mean_last(dhard_upper)
    dgbar = mean_last(dgsize_upper)
    wbar  = mean_last(weibull_adj)

    sd_h  = sd_last(hardness,     hbar)
    sd_g  = sd_last(grain_size,   gbar)
    sd_dh = sd_last(dhard_upper,  dhbar)
    sd_dg = sd_last(dgsize_upper, dgbar)
    sd_w  = sd_last(weibull_adj,  wbar)

    return hbar, gbar, dhbar, dgbar, wbar, sd_h, sd_g, sd_dh, sd_dg, sd_w

def _scale01(a):
    mn = np.nanmin(a, axis=-1)
    mx = np.nanmax(a, axis=-1)
    diff = np.maximum(mx - mn, 1.0)
    return (a - mn[..., None]) / diff[..., None]

def _lemons_sum(hardness, grain_size, grain_type,
                dhard, dgsize, weibull_adj,
                hbar, gbar, dhbar, dgbar, wbar,
                sd_h, sd_g, sd_dh, sd_dg, sd_w) -> xr.DataArray:
    weak_types = np.array(["FC", "FCxr", "DH", "SH"])
    lem_gtype = np.isin(grain_type, weak_types).astype(float)

    lem_hard  = (hardness    - hbar[..., None])  / sd_h[..., None]
    lem_gsize = (grain_size  - gbar[..., None])  / sd_g[..., None]
    lem_dhard = (dhard - dhbar[..., None]) / sd_dh[..., None]
    lem_dgsize= (dgsize- dgbar[..., None]) / sd_dg[..., None]
    lem_depth = (weibull_adj - wbar[..., None])  / sd_w[..., None]

    lem_hard  = 1.0 - _scale01(lem_hard)
    lem_gsize = _scale01(lem_gsize)
    lem_dhard = _scale01(lem_dhard)
    lem_dgsize= _scale01(lem_dgsize)
    lem_depth = _scale01(lem_depth)

    return (lem_gtype +
            lem_hard +
            lem_gsize +
            lem_dhard +
            lem_dgsize +
            lem_depth)

#endregion
