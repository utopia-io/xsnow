"""
Skier stability index SK38 and shared skier-loading mechanics

The skier-specific building blocks (penetration depth, equivalent slab
excluding the penetrated layers, line-load shear stress) plus the SK38
index on a virtual 38 degree slope. The underlying slab mechanics come
from :mod:`xsnow.extensions.critical_crack_length`; computation runs on
CPU or GPU depending on the dataset backend.

References
----------
- Foehn, P. M. B. (1987). The stability index and various triggering
  mechanisms. IAHS Publ., 162, 195-214.
- Jamieson, J. B., & Johnston, C. D. (1998). Refinements to the stability
  index for skier-triggered dry-slab avalanches. Ann. Glaciol., 26,
  296-302.
"""
import math

import numpy as np

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import (
    finite_max,
    get_logger,
    registry_attrs,
    require_vars,
)
from .profile_block_computation import (
    apply_profile_block,
    float_result_dtype,
    get_array_backend,
)
from .critical_crack_length import (
    compute_slab_mechanics,
    slab_weighted_mean,
)
from .rta import compute_rta  # noqa: F401 (registers rta)

DEFAULT_INCLINATION = 38.0   # deg, virtual slope angle of the index
# N m^-1, effective skier line load (~85 kg, 1.7 m skis, dynamic factor 2.5)
SKIER_LINE_LOAD = 1226.0
# kg m^-2, penetration depth P = 34.6 / rho_slab (Jamieson & Johnston 1998)
SKIER_PEN_COEFF = 34.6
# m, vertical averaging depth for the penetration slab density
PEN_SLAB_DEPTH = 0.30
CRUST_GRAIN_CODE = 772       # Swiss Code F1F2F3 of a melt-freeze crust
CRUST_MIN_DENSITY = 500.0    # kg m^-3, supporting crust density
CRUST_MIN_THICKNESS = 0.03   # m, supporting crust thickness
# treatment of the partially penetrated layer ('full' or 'residual')
PENETRATED_LAYER = 'full'


def compute_skier_penetration(
    density,
    thickness_vertical,
    grain_type,
    xp,
    use_crust_limit=True,
):
    """
    Skier penetration depth [m, vertical].

    Notes
    -----
    ``P = 34.6 / rho_top`` (Jamieson & Johnston 1998), with ``rho_top``
    the mean density of the top 30 cm of the snowpack (the boundary
    layer enters with its partial thickness; layers of unknown density
    are excluded from the average). With ``use_crust_limit=True`` a
    supporting melt-freeze crust (grain type 772, density >= 500
    kg m^-3, contiguous thickness >= 3 cm) caps the depth at the top of
    that crust.
    """
    valid = xp.isfinite(thickness_vertical) & (thickness_vertical > 0)
    thick = xp.where(valid, thickness_vertical, 0.0)

    # depth below the surface of each layer's top
    cum_incl = xp.cumsum(thick[..., ::-1], axis=-1)[..., ::-1]
    cum_above = cum_incl - thick

    weights = xp.clip(
        xp.minimum(thick, PEN_SLAB_DEPTH - cum_above), 0.0, None
    )
    known = valid & xp.isfinite(density)
    weights = xp.where(known, weights, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        rho_top = (
            xp.sum(xp.where(known, density, 0.0) * weights, axis=-1)
            / xp.sum(weights, axis=-1)
        )
        pen = SKIER_PEN_COEFF / rho_top

    if not use_crust_limit:
        return pen

    crust = valid & (grain_type == CRUST_GRAIN_CODE) & (
        xp.where(xp.isfinite(density), density, 0.0) >= CRUST_MIN_DENSITY
    )
    # accumulated thickness of each contiguous crust stack, from its base
    crust_cum = xp.cumsum(xp.where(crust, thick, 0.0), axis=-1)
    baseline = xp.where(crust, -xp.inf, crust_cum)
    baseline = xp.maximum.accumulate(baseline, axis=-1)
    baseline = xp.maximum(baseline, 0.0)
    run_thickness = xp.where(crust, crust_cum - baseline, 0.0)

    qualifies = (
        crust
        & (run_thickness >= CRUST_MIN_THICKNESS)
        & (cum_above < pen[..., None])
    )
    depth_to_crust = xp.where(qualifies, cum_above, xp.inf)
    pen_crust = xp.min(depth_to_crust, axis=-1)

    return xp.minimum(pen, pen_crust)


def alpha_max_skier_stress(inclination_rad):
    """
    Angle at which the skier-induced shear stress is maximal for a slope
    angle ``inclination_rad``, found by bisection (host-side float).
    """
    phi = float(inclination_rad)

    def derivative(x):
        return (
            2.0 * math.sin(x) * math.cos(x) ** 2 * math.sin(x + phi)
            - math.sin(x) ** 3 * math.sin(x + phi)
            + math.sin(x) ** 2 * math.cos(x) * math.cos(x + phi)
        )

    lo, hi = 0.05, math.pi / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if derivative(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def skier_delta_tau(alpha, h_equiv, inclination_rad, line_load, xp):
    """
    Skier-induced shear stress of a line load (Foehn 1987).

    Notes
    -----
    ``delta_tau = 2 R cos(a) sin^2(a) sin(a + phi) / (pi h)`` with the
    line load ``R`` and the (equivalent) slab thickness ``h`` above the
    weak layer.
    """
    return (
        2.0 * line_load
        * xp.cos(alpha) * xp.sin(alpha) ** 2
        * xp.sin(alpha + inclination_rad)
        / (xp.pi * h_equiv)
    )


def _prepare_skier_mechanics(
    density,
    shear_strength,
    height,
    grain_type,
    xp,
    penetrated_layer=PENETRATED_LAYER,
):
    """
    Slab mechanics plus the skier-specific quantities for eager arrays.

    Extends :func:`~xsnow.extensions.critical_crack_length.
    compute_slab_mechanics` with the skier penetration, the mask of
    penetrated layers, the topmost layer, and the equivalent slab
    thickness ``h_equiv_sk`` excluding the skier-penetrated part.

    Notes
    -----
    ``penetrated_layer`` selects how the partially penetrated layer
    enters ``h_equiv_sk``: ``'full'`` counts its full thickness (slab
    surface at its top), ``'residual'`` only its thickness below the
    penetration front (slab surface at the front).
    """
    if penetrated_layer not in ('full', 'residual'):
        raise ValueError(
            "penetrated_layer must be 'full' or 'residual', got "
            f"{penetrated_layer!r}"
        )
    theta = math.radians(DEFAULT_INCLINATION)
    cos_t = math.cos(theta)

    density = xp.asarray(density)
    shear_strength = xp.asarray(shear_strength)
    height = xp.asarray(height)
    grain_type = xp.asarray(grain_type)
    bottom = xp.concatenate(
        [xp.zeros_like(height[..., :1]), height[..., :-1]], axis=-1
    )
    thick_vert = height - bottom

    mech = compute_slab_mechanics(
        density, thick_vert, height, grain_type, shear_strength,
        DEFAULT_INCLINATION, xp,
    )
    valid = mech['valid']
    height_n = mech['height_n']
    thick_n = mech['thick_n']
    e_layer = mech['e_layer']

    # skier penetration and the layers it consumes
    penetration = compute_skier_penetration(
        mech['density'], thick_vert, grain_type, xp,
    )
    min_height_n = mech['surface_n'] - penetration * cos_t
    penetrated = valid & (height_n > min_height_n[..., None])
    first_penetrated = penetrated & (
        xp.cumsum(penetrated.astype(xp.int64), axis=-1) == 1
    )

    # equivalent slab above each layer, excluding the skier-penetrated
    # part ('full' vs 'residual', see docstring)
    if penetrated_layer == 'full':
        in_slab = valid & (~penetrated | first_penetrated)
        thick_sk = xp.where(in_slab, thick_n, xp.nan)
        height_sk = xp.where(in_slab, height_n, -xp.inf)
        surface_sk = finite_max(height_sk, xp)
    else:
        bottom_n = height_n - thick_n
        residual = min_height_n[..., None] - bottom_n
        thick_sk = xp.where(first_penetrated, residual, thick_n)
        thick_sk = xp.where(
            penetrated & ~first_penetrated, xp.nan, thick_sk
        )
        thick_sk = xp.where(valid, thick_sk, xp.nan)
        surface_sk = min_height_n

    d_slab_sk = surface_sk[..., None] - height_n
    e_slab_sk = (
        slab_weighted_mean(e_layer ** (1.0 / 3.0), thick_sk, xp) ** 3
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        h_equiv_sk = d_slab_sk * (e_slab_sk / e_layer) ** (1.0 / 3.0)

    # topmost valid layer (never evaluated)
    reversed_valid = valid[..., ::-1]
    top_layer = (
        xp.cumsum(reversed_valid.astype(xp.int64), axis=-1) == 1
    )[..., ::-1] & valid

    mech.update({
        'xp': xp,
        'theta': theta,
        'line_load': SKIER_LINE_LOAD,
        'penetration': penetration,
        'penetrated': penetrated,
        'h_equiv_sk': h_equiv_sk,
        'top_layer': top_layer,
    })
    return mech


def _compute_sk38_block(density, shear_strength, height, grain_type):
    """Compute SK38 for an eager block containing complete profiles."""
    xp = get_array_backend(density)
    mech = _prepare_skier_mechanics(
        density, shear_strength, height, grain_type, xp
    )

    alpha_star = alpha_max_skier_stress(mech['theta'])
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_tau_max = skier_delta_tau(
            xp.asarray(alpha_star, dtype=mech['h_equiv_sk'].dtype),
            mech['h_equiv_sk'], mech['theta'], mech['line_load'], xp
        )
        sk38 = mech['tau_p'] / (mech['tau_g'] + delta_tau_max)

    computable = (
        mech['valid'] & ~mech['penetrated'] & ~mech['top_layer']
    )
    return xp.where(computable, sk38, xp.nan)


@register_xsnowDataset_method
def compute_sk38(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Compute the skier stability index ``sk38`` for every layer.

    Returns
    -------
    xsnowDataset
        New dataset with the ``sk38`` variable attached (an existing
        ``sk38`` is overwritten).

    Notes
    -----
    ``SK38 = tau_p / (tau_g + delta_tau_max)`` with the corrected shear
    strength ``tau_p``, the slab shear stress ``tau_g`` on the virtual
    38 degree slope, and the maximum skier-induced stress
    ``delta_tau_max`` (Foehn 1987), evaluated with the equivalent slab
    thickness excluding the skier-penetrated layers. Requires
    ``density``, ``shear_strength``, ``height`` and ``grain_type``; the
    topmost and skier-penetrated layers are NaN.
    """
    require_vars(
        ds, ['density', 'shear_strength', 'height', 'grain_type'],
        'skier stability',
    )
    density_da = ds.data['density']
    shear_da = ds.data['shear_strength']
    height_da = ds.data['height']
    grain_type_da = ds.data['grain_type']
    sk38 = apply_profile_block(
        _compute_sk38_block,
        density_da,
        shear_da,
        height_da,
        grain_type_da,
        output_dtype=float_result_dtype(
            density_da, shear_da, height_da
        ),
    )

    logger = get_logger(__name__)
    if 'sk38' in ds.data:
        logger.info("Overwriting existing variable 'sk38'.")

    sk38.attrs = registry_attrs('sk38', 'skier stability index SK38', '1')
    result = ds.data.assign(sk38=sk38)
    return ds.__class__(result)


STABLE_SK38 = 6.0   # SNOWPACK's SK38 cap, reported where the RTA gate closes


@register_xsnowDataset_method
def compute_sk38_rta(
    ds: 'xsnowDataset', rta_threshold: float = 0.8
) -> 'xsnowDataset':
    """
    RTA-gated native SK38: layers that are no weak-layer candidates
    (``rta`` < ``rta_threshold``) report the stable cap 6.0 instead of
    their SK38.

    Requires ``sk38``; a missing ``rta`` is computed and kept. NaN rta
    leaves the SK38 value ungated.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    require_vars(xs, ('sk38',), 'compute_sk38_rta')
    pure = xs.compute_rta() if 'rta' not in xs.data else xs

    index_da = pure.data['sk38']
    rta_da = pure.data['rta'].broadcast_like(index_da)
    gated = index_da.where(
        ~(np.isfinite(index_da) & (rta_da < rta_threshold)),
        STABLE_SK38,
    )

    attrs = registry_attrs('sk38_rta', 'RTA-gated stability index SK38', '1')
    attrs['rta_threshold'] = rta_threshold
    gated.attrs = attrs
    new_xr = pure.data.assign(sk38_rta=gated)
    return ds.__class__(new_xr)
