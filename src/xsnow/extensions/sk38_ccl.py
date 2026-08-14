"""
Physical stability index sk38_ccl and its RTA-gated variant

``sk38_ccl = ccl / l_sk`` compares the critical crack length (Richter et
al. 2019, on a virtual 38 degree slope) to the length of the crack a
skier can induce (Foehn 1987); values below 1 indicate that a skier can
initiate a self-propagating crack.

References
----------
- Gaume, J., & Reuter, B. (2017). Assessing snow instability in skier-
  triggered snow slab avalanches by combining failure initiation and
  crack propagation. Cold Reg. Sci. Technol., 144, 6-15.
- Monti, F., & Schweizer, J. (2013). A relative difference approach to
  detect potential weak layers within a snow profile. Proc. ISSW 2013.
- Richter, B., Schweizer, J., Rotach, M. W., & van Herwijnen, A. (2019).
  Validating modeled critical crack length for crack propagation in the
  snow cover model SNOWPACK. The Cryosphere, 13, 3353-3366.
"""
import math

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import get_backend
from .critical_crack_length import _errstate, compute_rc_slope_richter2019
from .rta import compute_rta
from .sk38 import (
    _prepare_skier_mechanics,
    _registry_attrs,
    _wrap_like_height,
    alpha_max_skier_stress,
    skier_delta_tau,
)

INDEX_CAP = 5.0
RTA_THRESHOLD = 0.8       # weak-layer candidate threshold
_BISECT_ITERATIONS = 60   # bisection depth for the crack-tip angles


def compute_skier_crack_length(mech, xp):
    """
    Skier-induced crack length ``l_sk`` [m].

    Returns ``(l_sk, crack_possible)``; ``l_sk`` is NaN where the skier
    stress never reaches the strength excess (``crack_possible`` False).

    Notes
    -----
    Bisection finds the two angles at which the line-load stress equals
    the strength excess ``tau_p - tau_g``.
    """
    theta = mech['theta']
    line_load = mech['line_load']
    h_equiv = mech['h_equiv_sk']
    strength_excess = mech['tau_p'] - mech['tau_g']

    alpha_star = alpha_max_skier_stress(theta)

    def stress_excess(alpha):
        return (
            skier_delta_tau(alpha, h_equiv, theta, line_load, xp)
            - strength_excess
        )

    with _errstate(xp, divide='ignore', invalid='ignore'):
        crack_possible = stress_excess(xp.asarray(alpha_star)) > 0.0

        # bisection towards the lower and the upper crack-tip angle
        lo1 = xp.full_like(h_equiv, 1.0e-9)
        hi1 = xp.full_like(h_equiv, alpha_star)
        lo2 = xp.full_like(h_equiv, alpha_star)
        hi2 = xp.full_like(h_equiv, math.pi / 2.0 - 1.0e-9)
        for _ in range(_BISECT_ITERATIONS):
            mid1 = 0.5 * (lo1 + hi1)
            below = stress_excess(mid1) < 0.0
            lo1 = xp.where(below, mid1, lo1)
            hi1 = xp.where(below, hi1, mid1)

            mid2 = 0.5 * (lo2 + hi2)
            above = stress_excess(mid2) > 0.0
            lo2 = xp.where(above, mid2, lo2)
            hi2 = xp.where(above, hi2, mid2)
        alpha_1 = 0.5 * (lo1 + hi1)
        alpha_2 = 0.5 * (lo2 + hi2)

        l_sk = h_equiv * (
            1.0 / xp.tan(alpha_1) - 1.0 / xp.tan(alpha_2)
        )
    l_sk = xp.where(crack_possible, l_sk, xp.nan)
    return l_sk, crack_possible


@register_xsnowDataset_method
def compute_sk38_ccl(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Compute the stability index ``sk38_ccl = ccl / l_sk``.

    Returns
    -------
    xsnowDataset
        New dataset with the ``sk38_ccl`` variable attached.

    Notes
    -----
    Requires ``density``, ``shear_strength``, ``height``,
    ``grain_type`` and ``grain_size``.
    ``0`` = spontaneous failure expected (slab load exceeds strength);
    ``NaN`` = not computable (topmost layer, skier-penetrated layers,
    padding).
    """
    if 'grain_size' not in ds.data:
        raise ValueError(
            "Missing required variable for sk38_ccl calculation: "
            "'grain_size'"
        )

    mech = _prepare_skier_mechanics(ds)
    xp = mech['xp']
    grain_size = xp.asarray(ds.data['grain_size'].data)

    ccl = compute_rc_slope_richter2019(
        rho_layer=mech['density'],
        grain_size_m=grain_size,
        shear_strength_pa=mech['tau_p'],
        tau_g=mech['tau_g'],
        sigma_n=mech['sigma_n'],
        h_equiv=mech['h_equiv'],
        xp=xp,
    )
    l_sk, crack_possible = compute_skier_crack_length(mech, xp)

    with _errstate(xp, divide='ignore', invalid='ignore'):
        index = xp.minimum(ccl / l_sk, INDEX_CAP)

    # sentinel precedence: topmost layer, then spontaneous failure, then
    # the skier-penetrated layers, then the actual (capped) index
    spontaneous = mech['tau_p'] < mech['tau_g']
    result = xp.where(crack_possible, index, INDEX_CAP)
    result = xp.where(spontaneous, 0.0, result)
    result = xp.where(mech['penetrated'] & ~spontaneous, xp.nan, result)
    result = xp.where(mech['top_layer'] | ~mech['valid'], xp.nan, result)

    attrs = _registry_attrs(
        'sk38_ccl',
        'stability index SK38-CCL '
        '(critical crack length / skier crack length)',
        '1',
    )
    attrs['index_cap'] = INDEX_CAP
    new_xr = ds.data.assign(
        sk38_ccl=_wrap_like_height(result, mech['height_da'], attrs)
    )
    return ds.__class__(new_xr)


@register_xsnowDataset_method
def compute_sk38_ccl_rta(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Compute the RTA-gated stability index ``sk38_ccl_rta``.

    Returns
    -------
    xsnowDataset
        New dataset with the ``sk38_ccl_rta`` variable attached.

    Notes
    -----
    ``sk38_ccl`` where ``rta >= RTA_THRESHOLD``, ``INDEX_CAP`` where
    below (NaN stays NaN). Uses the per-layer variable ``rta`` if
    present, otherwise computes it via
    :func:`xsnow.extensions.rta.compute_rta`.
    """
    if 'rta' not in ds.data:
        ds = compute_rta(ds)
    rta_da = ds.data['rta']

    pure = compute_sk38_ccl(ds)
    index_da = pure['sk38_ccl']

    xp = get_backend(pure)
    index = xp.asarray(index_da.data)
    rta_arr = xp.asarray(rta_da.broadcast_like(index_da).data)
    gated = xp.where(
        xp.isfinite(index) & (rta_arr < RTA_THRESHOLD),
        INDEX_CAP, index,
    )

    attrs = _registry_attrs(
        'sk38_ccl_rta',
        'RTA-gated stability index SK38-CCL',
        '1',
    )
    attrs['index_cap'] = INDEX_CAP
    attrs['rta_threshold'] = RTA_THRESHOLD

    new_xr = pure.data.assign(
        sk38_ccl_rta=_wrap_like_height(gated, index_da, attrs)
    )
    new_xr = new_xr.drop_vars('sk38_ccl')
    return ds.__class__(new_xr)
