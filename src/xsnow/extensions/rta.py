"""
Structural instability index RTA (relative threshold sum approach)

Per-layer structural weakness after Monti & Schweizer (2013): six
structural properties -- grain size, hand hardness, grain type, the
interface differences of grain size and hardness, and a crust-corrected
Weibull depth weighting -- are expressed as standardized anomalies
relative to the profile, min-max scaled, and summed. The sum is
normalized by the profile maximum, so RTA ranges in [0, 1] with the
structurally weakest layer at 1; values >= 0.8 flag weak-layer
candidates.

SNOWPACK's own RTA output (.pro parameter code 0607) follows the same
approach but differs in the details: interface differences only count
against a harder neighbour (maximum of both sides), the grain type is
scored in three levels from the primary and secondary form, properties
are min-max scaled without standardization or floors, the depth
weighting acts at the layer bottom with the crust correction clamped at
zero, and a single sum replaces the two interface variants. Agreement
between the two is therefore only approximate (the weak-layer
classification at 0.8 agrees for roughly 9 out of 10 layers).

References
----------
- Monti, F., & Schweizer, J. (2013). A relative difference approach to
  detect potential weak layers within a snow profile. Proc. ISSW 2013.
"""
import numpy as np

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import (
    finite_max,
    registry_attrs,
    require_vars,
)
from .profile_block_computation import (
    apply_profile_block,
    float_result_dtype,
    get_array_backend,
)

WEIBULL_SHAPE = 2.5          # shape of the depth weighting
WEIBULL_SCALE = 50.0         # cm, scale of the depth weighting
CRUST_MIN_HARDNESS = 3.0     # hand hardness index of a supporting crust
CRUST_MIN_THICKNESS = 1.0    # cm, thickness of a supporting crust
CRUST_DECAY = 20.0           # cm, e-folding depth of the crust effect
WEAK_GRAIN_CLASSES = (4.0, 5.0, 6.0, 9.0)   # FC, DH, SH, FCxr


def _profile_mean(x, valid, xp):
    """Mean over the valid layers (layer axis last)."""
    n = xp.maximum(valid.sum(axis=-1), 1).astype(x.dtype)
    return xp.where(valid, x, 0.0).sum(axis=-1) / n


def _population_sd(x, mean, valid, xp):
    """Population standard deviation over valid layers, floored at 1;
    NaN when the mask has no valid entries."""
    n = valid.sum(axis=-1)
    var = xp.where(valid, (x - mean[..., None]) ** 2, 0.0).sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = xp.sqrt(xp.where(
            n > 0, var / xp.maximum(n, 1).astype(x.dtype), xp.nan
        ))
    return xp.maximum(sd, 1.0)


def _minmax_span(z, valid, xp):
    """Per-profile minimum and (max - min) floored at 1."""
    mn = xp.min(xp.where(valid, z, xp.inf), axis=-1)
    mx = xp.max(xp.where(valid, z, -xp.inf), axis=-1)
    mn = xp.where(xp.isfinite(mn), mn, xp.nan)
    span = xp.maximum(mx - mn, 1.0)
    return mn, span


def _compute_rta_block(height_m, hardness, gsize_m, gtype):
    """Compute RTA for an eager block containing complete profiles."""
    xp = get_array_backend(height_m)
    height_m = xp.asarray(height_m)
    hardness = xp.asarray(hardness)
    gsize_m = xp.asarray(gsize_m)
    gtype = xp.asarray(gtype)

    valid = (xp.isfinite(height_m) & xp.isfinite(hardness)
             & xp.isfinite(gsize_m) & xp.isfinite(gtype))

    h = xp.where(valid, hardness, xp.nan)
    g = xp.where(valid, gsize_m * 1000.0, xp.nan)            # mm
    height = xp.where(valid, height_m * 100.0, xp.nan)       # cm
    surface = finite_max(height, xp)
    depth = surface[..., None] - height                      # layer top
    bottom = xp.concatenate(
        [xp.zeros_like(height[..., :1]), height[..., :-1]], axis=-1
    )
    thick = height - bottom

    # interface differences towards the layer below / above
    def _diffs(x):
        below = xp.full_like(x, xp.nan)
        above = xp.full_like(x, xp.nan)
        below[..., 1:] = xp.abs(x[..., 1:] - x[..., :-1])
        above[..., :-1] = below[..., 1:]
        return below, above

    dh_b, dh_a = _diffs(h)
    dg_b, dg_a = _diffs(g)
    iface_a = valid & xp.isfinite(dh_a)
    iface_b = valid & xp.isfinite(dh_b)

    # crust-corrected Weibull depth weighting
    with np.errstate(invalid='ignore'):
        weibull = (
            (WEIBULL_SHAPE / WEIBULL_SCALE)
            * depth ** (WEIBULL_SHAPE - 1.0)
            * xp.exp(-(depth / WEIBULL_SCALE) ** WEIBULL_SHAPE)
        )
    crust = (valid & (h >= CRUST_MIN_HARDNESS)
             & (thick > CRUST_MIN_THICKNESS))
    crust_value = xp.where(crust, xp.exp(-depth / CRUST_DECAY), 0.0)
    crust_coeff = crust_value[..., ::-1].cumsum(axis=-1)[..., ::-1]
    wadj = xp.where(valid, weibull - crust_coeff, xp.nan)

    # profile statistics (differences: mean over all interfaces)
    hbar = _profile_mean(h, valid, xp)
    gbar = _profile_mean(g, valid, xp)
    dhbar = _profile_mean(dh_a, iface_a, xp)
    dgbar = _profile_mean(dg_a, iface_a, xp)
    wbar = _profile_mean(wadj, valid, xp)

    sd_h = _population_sd(h, hbar, valid, xp)
    sd_g = _population_sd(g, gbar, valid, xp)
    sd_dg_b = _population_sd(dg_b, dgbar, iface_b, xp)
    sd_dh_a = _population_sd(dh_a, dhbar, iface_a, xp)
    sd_dg_a = _population_sd(dg_a, dgbar, iface_a, xp)
    sd_w = _population_sd(wadj, wbar, valid, xp)

    # standardized anomalies
    f1 = xp.floor(gtype / 100.0)
    lem_gt = xp.isin(f1, xp.asarray(WEAK_GRAIN_CLASSES)).astype(f1.dtype)
    z_h = (h - hbar[..., None]) / sd_h[..., None]
    z_g = (g - gbar[..., None]) / sd_g[..., None]

    # Vectorized transcription of sarp.snowprofile::computeRTA()
    # (layer target), reproducing its output exactly.
    # SUSPECTED SLIP in computeRTA.R (kept for exact agreement): the
    # standard deviation of this anomaly is taken from a non-existent
    # column and thereby floored to 1, so the below-interface hardness
    # difference is centred but not standardized. Corrected version:
    # sd_dh_b = _population_sd(dh_b, dhbar, iface_b, xp)
    # z_dh_b = (dh_b - dhbar[..., None]) / sd_dh_b[..., None]
    z_dh_b = dh_b - dhbar[..., None]
    z_dg_b = (dg_b - dgbar[..., None]) / sd_dg_b[..., None]
    z_dh_a = (dh_a - dhbar[..., None]) / sd_dh_a[..., None]
    z_dg_a = (dg_a - dgbar[..., None]) / sd_dg_a[..., None]
    z_w = (wadj - wbar[..., None]) / sd_w[..., None]

    # min-max scaling, shared between both interface variants
    mn_h, span_h = _minmax_span(z_h, valid, xp)
    mn_g, span_g = _minmax_span(z_g, valid, xp)
    mn_dh, span_dh = _minmax_span(z_dh_b, iface_b, xp)
    mn_dg, span_dg = _minmax_span(z_dg_b, iface_b, xp)
    mn_w, span_w = _minmax_span(z_w, valid, xp)

    def _scale(z, mn, span):
        return (z - mn[..., None]) / span[..., None]

    s_h = 1.0 - _scale(z_h, mn_h, span_h)
    s_g = _scale(z_g, mn_g, span_g)
    s_w = _scale(z_w, mn_w, span_w)
    # SUSPECTED SLIP in computeRTA.R (kept for exact agreement): the
    # depth property of the above-interface variant is min-max scaled a
    # second time. Corrected version:
    # s_w_above = s_w
    s_w_above = _scale(s_w, mn_w, span_w)

    sum_b = (lem_gt + s_h + s_g + _scale(z_dh_b, mn_dh, span_dh)
             + _scale(z_dg_b, mn_dg, span_dg) + s_w)
    sum_a = (lem_gt + s_h + s_g + _scale(z_dh_a, mn_dh, span_dh)
             + _scale(z_dg_a, mn_dg, span_dg) + s_w_above)

    with np.errstate(invalid='ignore'):
        rta = xp.fmax(sum_b, sum_a)
        rta = xp.where(valid, rta, xp.nan)
        rta = rta / finite_max(rta, xp)[..., None]

    single = valid.sum(axis=-1) < 2
    return xp.where(single[..., None] & valid, 0.0, rta)


@register_xsnowDataset_method
def compute_rta(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Compute the structural instability index ``rta`` for every layer.

    Returns
    -------
    xsnowDataset
        New dataset with the ``rta`` variable attached ([0, 1], weakest
        layer = 1; NaN on padding, 0 for single-layer profiles).

    Notes
    -----
    Requires ``height`` [m], ``hand_hardness`` (index steps),
    ``grain_size`` [m] and ``grain_type`` (Swiss Code F1F2F3). Each of
    the six structural properties is standardized by the profile mean
    and population standard deviation (floored at 1; grain size in mm
    and depths in cm, as the floors imply these units), min-max scaled
    across the profile (span floored at 1, hardness inverted), and
    summed; interface differences enter as the weaker of the
    upper-/lower-interface variants (per-layer maximum). The depth
    property is a Weibull weighting of the layer-top depth, reduced by
    the accumulated influence of overlying supporting crusts (hand
    hardness >= 3, thicker than 1 cm). A valid layer whose interface
    differences are undefined on both sides (isolated between invalid
    layers) is NaN as well; the single-layer zero applies only to
    profiles with fewer than two valid layers.
    """
    require_vars(
        ds, ['height', 'hand_hardness', 'grain_size', 'grain_type'],
        'rta calculation',
    )

    height_da = ds.data['height']
    hardness_da = ds.data['hand_hardness']
    gsize_da = ds.data['grain_size']
    gtype_da = ds.data['grain_type']
    rta = apply_profile_block(
        _compute_rta_block,
        height_da,
        hardness_da,
        gsize_da,
        gtype_da,
        output_dtype=float_result_dtype(
            height_da, hardness_da, gsize_da, gtype_da
        ),
    )

    attrs = registry_attrs(
        'rta', 'Relative Threshold Sum Approach (RTA)', '1'
    )
    rta.attrs = attrs
    new_xr = ds.data.assign(rta=rta)
    return ds.__class__(new_xr)
