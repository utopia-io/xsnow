"""
Avalanche hazard computation for the operational pipeline.

Implements the full 1stations.R post-processing chain:
  - Weather aggregation (aggregate24)
  - Precipitation event detection (derive_potential_datetags)
  - Layer date assignment (assign_datetags, derive_bdate)
  - Probability of instability (compute_punstable) via RF model
  - Persistent weak layer extraction (extract_pwls)
  - Hazard index computation (calc_hazard_indices)
  - Cross-slope hazard aggregation (aggregate_hazard)

References:
  - Richter et al. (2019) for critical crack length
  - Bellaire (2006), Jamieson & Johnston (1998) for penetration depth
  - Mayer et al. (2022) for RF instability model
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Union

from ..plugin import register_xsnowDataset_method
from ..utils import get_backend


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
P_UNSTABLE_THRESHOLD = 0.77   # Critical instability threshold
PRECIP_THRESHOLD_CM = 2.5     # cm, for datetag derivation
SEVERITY_DEPTH_CAP_CM = 50    # cm, max depth contribution to severity
MAX_PWLS = 3                  # Max persistent weak layers to extract
NEW_SNOW_CAP_24H = 50         # cm
NEW_SNOW_CAP_72H = 75         # cm
LWC_CAP_PCT = 10              # %
WIND_MAX_CA = 10              # m/s, Canada domain
WIND_MAX_DEFAULT = 16         # m/s, other domains
SKI_PEN_CAP_PCT = 50          # %

PERSISTENT_GRAIN_TYPES = {"SH", "DH", "FC", "FCxr", "MFcr", "IF"}
# SLF grain type codes for persistent types
PERSISTENT_GRAIN_CODES = {660, 670, 680, 682, 772, 780}

RHO_ICE = 917.0
MIN_THICK_CRUST_CM = 3
CRUST_DENSITY_THRESHOLD = 500.0
CRUST_GRAIN_CODE = 772  # MFcr


# ===========================================================================
# Phase A: Weather & Datetags
# ===========================================================================

def aggregate24(wx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly SMET weather to daily values.

    Parameters
    ----------
    wx : pd.DataFrame
        Weather timeseries with columns: timestamp, HN24, HNW24, MS_Rain,
        VW, DW, HS_mod. Timestamps should be UTC.

    Returns
    -------
    pd.DataFrame
        Daily values with columns: date, qpf24, hn24, hs, rain24, wind24,
        wdir, qpf72, hn72, rain72
    """
    wx = wx.copy()
    wx['timestamp'] = pd.to_datetime(wx['timestamp'])
    wx['date'] = wx['timestamp'].dt.date

    # Snow values at 0 UTC (first occurrence per date)
    psum_snow = wx.drop_duplicates(subset='date', keep='first')[
        ['date', 'HNW24', 'HN24', 'HS_mod']
    ].copy()
    psum_snow.columns = ['date', 'qpf24', 'hn24', 'hs']

    # Rain: sum per date, shifted +1 day
    rain = wx.groupby('date')['MS_Rain'].sum().reset_index()
    rain.columns = ['date', 'rain24']
    rain['date'] = rain['date'] + pd.Timedelta(days=1)

    # Wind speed: mean per date, shifted +1 day
    vw = wx.groupby('date')['VW'].mean().reset_index()
    vw.columns = ['date', 'wind24']
    vw['date'] = vw['date'] + pd.Timedelta(days=1)

    # Wind direction: circular mean per date, shifted +1 day
    def circular_mean_deg(angles):
        angles_rad = np.deg2rad(angles.dropna())
        if len(angles_rad) == 0:
            return np.nan
        return (np.rad2deg(np.arctan2(
            np.mean(np.sin(angles_rad)),
            np.mean(np.cos(angles_rad))
        )) + 360) % 360

    dw = wx.groupby('date')['DW'].apply(circular_mean_deg).reset_index()
    dw.columns = ['date', 'wdir']
    dw['date'] = dw['date'] + pd.Timedelta(days=1)

    # Merge all
    result = psum_snow
    for df in [rain, vw, dw]:
        result = result.merge(df, on='date', how='outer')

    # Combine rain + snow for QPF
    result['qpf24'] = result['qpf24'].fillna(0) + result['rain24'].fillna(0)

    # 3-day running sums (past 3 days including current)
    result = result.sort_values('date').reset_index(drop=True)
    result['qpf72'] = result['qpf24'].rolling(3, min_periods=1).sum()
    result['hn72'] = result['hn24'].rolling(3, min_periods=1).sum()
    result['rain72'] = result['rain24'].rolling(3, min_periods=1).sum()

    return result[['date', 'qpf24', 'hn24', 'hs', 'rain24', 'wind24',
                    'wdir', 'qpf72', 'hn72', 'rain72']]


def derive_potential_datetags(wx24: pd.DataFrame) -> np.ndarray:
    """Identify precipitation events from daily weather data.

    A new precipitation event starts when:
    - Previous 72h snow accumulation was < 2.5 cm (dry period)
    - Current day new snow > 2.5 cm

    Parameters
    ----------
    wx24 : pd.DataFrame
        Daily weather from aggregate24(), must have 'date', 'hn72', 'hn24'.

    Returns
    -------
    np.ndarray of datetime64
        Dates marking the start of precipitation events (shifted +1 day).
    """
    wx24 = wx24.sort_values('date').reset_index(drop=True)
    hn72_prev = wx24['hn72'].shift(1).fillna(0)
    mask = (hn72_prev < PRECIP_THRESHOLD_CM) & (wx24['hn24'] > PRECIP_THRESHOLD_CM)
    dates = pd.to_datetime(wx24.loc[mask, 'date']) + pd.Timedelta(days=1)
    return dates.values


def derive_bdate(layers: pd.DataFrame) -> pd.DataFrame:
    """Compute burial dates for snow layers.

    The burial date of a layer is the deposition date of the overlying layer.
    The surface layer (topmost) has no burial date (NaT).

    Parameters
    ----------
    layers : pd.DataFrame
        Must have 'ddate' column (deposition dates as datetime).

    Returns
    -------
    pd.DataFrame
        Input with 'bdate' column added.
    """
    layers = layers.copy()
    ddate = pd.to_datetime(layers['ddate'])
    # bdate[i] = ddate[i+1] (the layer above buries this one)
    bdate = ddate.shift(-1)
    layers['bdate'] = bdate
    return layers


def assign_datetags(
    layers: pd.DataFrame,
    potential_datetags: np.ndarray,
    vdate: pd.Timestamp,
) -> pd.DataFrame:
    """Assign precipitation event dates to snow layers.

    For each layer, finds the first potential datetag >= layer's deposition
    date. Also computes burial dates.

    Parameters
    ----------
    layers : pd.DataFrame
        Must have 'ddate' column.
    potential_datetags : np.ndarray
        Sorted array of datetag dates.
    vdate : pd.Timestamp
        Profile validity date (used for surface layers with no burial date).

    Returns
    -------
    pd.DataFrame
        Input with 'datetag' and 'bdate' columns added.
    """
    layers = derive_bdate(layers)

    potential_datetags = np.sort(pd.to_datetime(potential_datetags))
    n_layers = len(layers)

    if n_layers == 0 or len(potential_datetags) == 0:
        layers['datetag'] = pd.NaT
        return layers

    ddate = pd.to_datetime(layers['ddate'])
    datetags = pd.Series(pd.NaT, index=layers.index)

    for i in range(n_layers):
        if pd.isna(ddate.iloc[i]):
            continue
        # First datetag >= ddate
        candidates = potential_datetags[potential_datetags >= ddate.iloc[i]]
        if len(candidates) > 0:
            datetags.iloc[i] = candidates[0]

    layers['datetag'] = datetags
    return layers


# ===========================================================================
# Phase B: Instability Computation
# ===========================================================================

def compute_slab_rhogs(density, thickness, grain_size, xp=np):
    """Compute slab density/grainsize ratio above each layer.

    For layer i: sum(density[i+1:] * thickness[i+1:] / grain_size[i+1:]) /
                 sum(thickness[i+1:])

    Parameters
    ----------
    density : array, shape (..., L)
    thickness : array, shape (..., L)
    grain_size : array, shape (..., L)
        Grain size in mm.
    xp : module
        numpy or cupy.

    Returns
    -------
    array, shape (..., L)
        Slab density/grainsize ratio. Surface layer is NaN.
    """
    valid = xp.isfinite(density) & xp.isfinite(thickness) & xp.isfinite(grain_size)
    valid = valid & (thickness > 0) & (grain_size > 0)

    rho_over_gs = xp.where(valid, density * thickness / grain_size, 0.0)
    thick_clean = xp.where(valid, thickness, 0.0)

    # Cumsum from top down (reversed)
    rho_gs_cum = xp.cumsum(rho_over_gs[..., ::-1], axis=-1)[..., ::-1]
    thick_cum = xp.cumsum(thick_clean[..., ::-1], axis=-1)[..., ::-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        result = rho_gs_cum / thick_cum

    # Shift so layer i gets slab strictly above
    tail_nan = xp.full_like(result[..., :1], xp.nan)
    result = xp.concatenate([result[..., 1:], tail_nan], axis=-1)

    return result


def compute_pendepth(height, density, grain_type, slope_angle_deg, xp=np):
    """Compute skier penetration depth.

    Based on Jamieson & Johnston (1998) and Bellaire (2006).
    Searches top 30cm for MFcr crusts, computes average density of top 30cm.

    Parameters
    ----------
    height : array, shape (..., L)
        Layer heights from ground [cm], bottom to top.
    density : array, shape (..., L)
    grain_type : array, shape (..., L)
        SLF grain type codes.
    slope_angle_deg : float
        Slope angle in degrees.

    Returns
    -------
    float or array
        Penetration depth [m]. Scalar per profile.
    """
    # This operates per-profile, so we handle the last two dims
    # For now, a vectorized implementation over the layer dimension
    hs = xp.nanmax(height, axis=-1)  # snow height per profile
    thickness = xp.diff(
        xp.concatenate([xp.zeros_like(height[..., :1]), height], axis=-1),
        axis=-1
    )

    L = height.shape[-1]
    rho_pk = xp.zeros_like(hs)
    dz_pk = xp.full_like(hs, 1e-12)
    top_crust = xp.zeros_like(hs)

    # Iterate from top layer downward within top 30cm
    for ee in range(L - 1, -1, -1):
        in_top_30 = (hs - height[..., ee]) < 30
        rho_pk = xp.where(in_top_30, rho_pk + density[..., ee] * thickness[..., ee], rho_pk)
        dz_pk = xp.where(in_top_30, dz_pk + thickness[..., ee], dz_pk)

        # Check for MFcr crust (code 772, density > 500)
        is_crust = (
            in_top_30 &
            (grain_type[..., ee] == CRUST_GRAIN_CODE) &
            (density[..., ee] > CRUST_DENSITY_THRESHOLD)
        )
        crust_thick_enough = thickness[..., ee] * xp.cos(xp.deg2rad(slope_angle_deg)) > MIN_THICK_CRUST_CM
        top_crust = xp.where(is_crust & crust_thick_enough & (top_crust == 0),
                             height[..., ee], top_crust)

    with np.errstate(divide='ignore', invalid='ignore'):
        rho_pk = rho_pk / dz_pk

    # Penetration depth = min(0.8 * 43.3 / rho_30, HS - top_crust)
    pen = xp.minimum(0.8 * 43.3 / rho_pk, hs - top_crust)
    # Convert to meters
    pen = pen / 100.0
    return pen


def compute_punstable_features(ds_data, ski_pen=None, slope_angle_deg=0.0):
    """Compute the 6 RF model features from an xsnow dataset.

    Parameters
    ----------
    ds_data : xr.Dataset
        Must contain: viscous_deformation_rate, grain_size, density,
        shear_strength, sphericity, height.
    ski_pen : array-like or None
        Skier penetration depth in meters, shape (...) matching the
        non-layer dimensions of the dataset. If provided (e.g. from SMET
        'ski_pen' variable), used directly. If None, computed from the
        profile using Jamieson & Johnston (1998) / Bellaire (2006).
    slope_angle_deg : float
        Slope angle for penetration depth calculation (only used when
        ski_pen is None).

    Returns
    -------
    dict of xr.DataArray
        Keys: viscdefrate, rcflat, sphericity, grainsize, penetrationdepth, slab_rhogs
    """
    from .critical_crack_length import compute_rho_slab, compute_rc_flat_richter2019

    xp = np
    height = xp.asarray(ds_data['height'].data)
    density = xp.asarray(ds_data['density'].data)
    grain_size = xp.asarray(ds_data['grain_size'].data)
    shear_strength = xp.asarray(ds_data['shear_strength'].data)
    sphericity = xp.asarray(ds_data['sphericity'].data)
    viscdefrate = xp.asarray(ds_data['viscous_deformation_rate'].data)
    grain_type = xp.asarray(ds_data['grain_type'].data)

    thickness = xp.diff(
        xp.concatenate([xp.zeros_like(height[..., :1]), height], axis=-1),
        axis=-1
    )

    # 1. viscous deformation rate (already in dataset)

    # 2. critical crack length
    # R rounds slab_rho to integers before computing rc_flat, and rounds
    # rc_flat to 2 decimals. Match this for identical results.
    rho_slab = compute_rho_slab(density, thickness, xp)
    rho_slab_rounded = xp.round(rho_slab)
    rcflat = compute_rc_flat_richter2019(
        density, grain_size, shear_strength, rho_slab_rounded, xp
    )
    rcflat = xp.round(rcflat, decimals=2)

    # 3. sphericity (already in dataset)
    # 4. grain size (already in dataset)

    # 5. penetration depth
    if ski_pen is not None:
        # Use pre-computed ski_pen from SMET (SNOWPACK output)
        pendepth = xp.asarray(ski_pen)
    else:
        # Fallback: compute from profile (less accurate)
        pendepth = compute_pendepth(height, density, grain_type, slope_angle_deg, xp)

    # Broadcast to layer dimension
    if pendepth.ndim < viscdefrate.ndim:
        pendepth_broadcast = xp.broadcast_to(
            pendepth[..., xp.newaxis],
            viscdefrate.shape
        )
    else:
        pendepth_broadcast = xp.broadcast_to(pendepth, viscdefrate.shape)

    # 6. slab density / grain size ratio
    # R rounds slab_rhogs to integers
    slab_rhogs = compute_slab_rhogs(density, thickness, grain_size, xp)
    slab_rhogs = xp.round(slab_rhogs)

    return {
        'viscdefrate': viscdefrate,
        'rcflat': rcflat,
        'sphericity': sphericity,
        'grainsize': grain_size,
        'penetrationdepth': pendepth_broadcast,
        'slab_rhogs': slab_rhogs,
    }


def predict_punstable(features: dict, model, xp=np):
    """Run RF model prediction on precomputed features.

    Parameters
    ----------
    features : dict
        Output of compute_punstable_features(). Each value is array (..., L).
    model : sklearn RandomForestClassifier
        Loaded via joblib.load().

    Returns
    -------
    array, shape (..., L)
        Probability of instability per layer (0-1). NaN where features are missing.
    """
    original_shape = features['viscdefrate'].shape

    # Stack all features into (N, 6) for sklearn
    feature_names = ['viscdefrate', 'rcflat', 'sphericity', 'grainsize',
                     'penetrationdepth', 'slab_rhogs']
    feat_arrays = [features[name].ravel() for name in feature_names]
    X = xp.stack(feat_arrays, axis=-1)  # (N, 6)

    # Mask: only predict where all features are finite
    valid_mask = xp.all(xp.isfinite(X), axis=-1)
    p_unstable = xp.full(X.shape[0], xp.nan)

    if xp.any(valid_mask):
        X_valid = X[valid_mask]
        # Convert to numpy for sklearn
        if hasattr(X_valid, 'get'):  # cupy
            X_valid = X_valid.get()
        proba = model.predict_proba(X_valid)[:, 0]
        p_unstable[valid_mask] = xp.round(xp.asarray(proba), decimals=2)

    return p_unstable.reshape(original_shape)


# ===========================================================================
# Phase C: Hazard Extraction
# ===========================================================================

def extract_pwls(
    layers: pd.DataFrame,
    p_unstable_col: str = 'p_unstable',
    depth_col: str = 'depth',
    gtype_col: str = 'grain_type',
    gsize_col: str = 'grain_size',
    datetag_col: str = 'datetag',
    bdate_col: str = 'bdate',
) -> pd.DataFrame:
    """Extract persistent weak layers from a snow profile.

    Filters to persistent grain types, ranks by severity, returns top 3.

    Parameters
    ----------
    layers : pd.DataFrame
        Snow profile layers with computed p_unstable, datetag, etc.

    Returns
    -------
    pd.DataFrame
        Up to 3 most severe weak layers with columns:
        label, datetag, bdate, gtype, gsize, depth, p_unstable, severity, critical
    """
    if layers.empty or p_unstable_col not in layers.columns:
        return pd.DataFrame()

    # Filter to persistent grain types
    if layers[gtype_col].dtype == object or layers[gtype_col].dtype.kind == 'U':
        pwl = layers[layers[gtype_col].isin(PERSISTENT_GRAIN_TYPES)].copy()
    else:
        pwl = layers[layers[gtype_col].isin(PERSISTENT_GRAIN_CODES)].copy()

    if pwl.empty:
        return pd.DataFrame()

    # Per datetag: keep layer with max p_unstable
    if datetag_col in pwl.columns:
        pwl_valid = pwl.dropna(subset=[datetag_col])
        if not pwl_valid.empty:
            idx = pwl_valid.groupby(datetag_col)[p_unstable_col].idxmax()
            pwl = pwl.loc[idx]

    # Mark critical
    pwl['critical'] = pwl[p_unstable_col] > P_UNSTABLE_THRESHOLD

    # Compute severity
    pwl['severity'] = (
        pwl[p_unstable_col] *
        np.minimum(pwl[depth_col].fillna(0), SEVERITY_DEPTH_CAP_CM) /
        SEVERITY_DEPTH_CAP_CM
    )

    # Keep critical layers, or single most severe if none critical
    if pwl['critical'].any():
        pwl = pwl[pwl['critical']]
    else:
        pwl = pwl.nlargest(1, 'severity')

    # Top 3 by severity
    pwl = pwl.nlargest(MAX_PWLS, 'severity')

    # Generate labels: "{month_letter}{day}-{gtype}-{depth}"
    if bdate_col in pwl.columns:
        bdate = pd.to_datetime(pwl[bdate_col])
        month_letters = bdate.dt.strftime('%b').str[0]
        days = bdate.dt.day.astype(str)
        gtype_str = pwl[gtype_col].astype(str)
        depth_str = pwl[depth_col].round(0).astype(int).astype(str)
        pwl['label'] = month_letters + days + '-' + gtype_str + '-' + depth_str
    else:
        pwl['label'] = ''

    cols = ['label', datetag_col, bdate_col, gtype_col, gsize_col,
            depth_col, p_unstable_col, 'severity', 'critical']
    return pwl[[c for c in cols if c in pwl.columns]].reset_index(drop=True)


def calc_hazard_indices(
    wx24_row: dict,
    pwl_severity: float = 0.0,
    lwc_profile: float = 0.0,
    domain: str = "ca",
) -> dict:
    """Compute 5 hazard indices for a single profile/day.

    Parameters
    ----------
    wx24_row : dict
        Daily weather with keys: hn24, hn72, wind24, wdir, ski_pen.
    pwl_severity : float
        Max severity from extract_pwls().
    lwc_profile : float
        Volume-weighted mean liquid water content (%).
    domain : str
        "ca" for Canada (wind max 10), otherwise 16.

    Returns
    -------
    dict
        Keys: new, wind, wet, pwl, haz
    """
    hn24 = wx24_row.get('hn24', 0) or 0
    hn72 = wx24_row.get('hn72', 0) or 0
    wind_speed = wx24_row.get('wind24', 0) or 0
    ski_pen_pct = wx24_row.get('ski_pen', 0) or 0

    # New snow index
    new = 0.5 * (min(hn24, NEW_SNOW_CAP_24H) / NEW_SNOW_CAP_24H +
                 min(hn72, NEW_SNOW_CAP_72H) / NEW_SNOW_CAP_72H)

    # Wind slab index
    max_wind = WIND_MAX_CA if domain == "ca" else WIND_MAX_DEFAULT
    wind = min(wind_speed * min(ski_pen_pct, SKI_PEN_CAP_PCT) / SKI_PEN_CAP_PCT,
               max_wind) / max_wind

    # Wet snow index
    wet = min(lwc_profile, LWC_CAP_PCT) / LWC_CAP_PCT

    # PWL index
    pwl = pwl_severity

    # Overall hazard: max of all pairwise sums / 2
    haz = max(
        new + pwl, new + wind, new + wet,
        pwl + wind, pwl + wet, wind + wet
    ) / 2

    return {'new': new, 'wind': wind, 'wet': wet, 'pwl': pwl, 'haz': haz}


def aggregate_hazard(haz_list: list[dict]) -> dict:
    """Aggregate hazard indices across multiple slopes.

    Parameters
    ----------
    haz_list : list of dict
        Hazard dicts from calc_hazard_indices(), one per slope.

    Returns
    -------
    dict
        Mean values across slopes. Wind direction uses circular mean.
    """
    if not haz_list:
        return {}

    df = pd.DataFrame(haz_list)
    result = {}

    for col in df.columns:
        if col == 'wdir':
            # Circular mean for wind direction
            angles = df[col].dropna()
            if len(angles) > 0:
                rad = np.deg2rad(angles)
                result[col] = (np.rad2deg(np.arctan2(
                    np.mean(np.sin(rad)), np.mean(np.cos(rad))
                )) + 360) % 360
            else:
                result[col] = np.nan
        elif df[col].dtype in [np.float64, np.float32, np.int64, float, int]:
            result[col] = df[col].mean()
        else:
            result[col] = df[col].iloc[0]

    return result
