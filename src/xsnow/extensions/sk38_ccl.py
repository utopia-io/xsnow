"""
Physical stability index sk38_ccl and its RTA-gated variant

``sk38_ccl = ccl / l_sk`` compares the critical crack length (Richter et
al. 2019, on a virtual 38 degree slope) to the length of the crack a
skier can induce (Foehn 1987); values below 1 indicate that a skier can
initiate a self-propagating crack.

Both ``sk38_ccl`` and its RTA-gated variant ``sk38_ccl_rta`` follow the
combined failure-initiation and crack-propagation framework described by
Borsotti et al. (in preparation).

References
----------
- Borsotti, J., Mariani, A., Villa, G., Metzger, M., Mayer, S.,
  Schweizer, J., Haegeli, P., & Monti, F. Assessing skier-induced
  snowpack instability from simulated snow stratigraphy by combining
  failure initiation and crack propagation. In preparation.
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

import numpy as np

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import registry_attrs, require_vars
from .profile_block_computation import (
    apply_profile_block,
    float_result_dtype,
    get_backend,
)
from .critical_crack_length import compute_rc_slope_richter2019
from .rta import compute_rta
from .sk38 import (
    _compute_skier_mechanics,
    alpha_max_skier_stress,
    skier_delta_tau,
)

INDEX_CAP = 5.0
RTA_THRESHOLD = 0.8       # weak-layer candidate threshold
_BISECT_ITERATIONS = 60   # bisection depth for the crack-tip angles


def compute_skier_crack_length(
    inclination_rad,
    line_load,
    h_equiv,
    strength_excess,
    xp,
):
    """
    Skier-induced crack length ``l_sk`` [m].

    Returns ``(l_sk, crack_possible)``; ``l_sk`` is NaN where the skier
    stress never reaches the strength excess (``crack_possible`` False).

    Parameters
    ----------
    inclination_rad : float
        Virtual slope inclination [rad].
    line_load : float
        Effective skier line load [N m^-1].
    h_equiv : xp.ndarray
        Equivalent slab thickness above each candidate weak layer [m].
    strength_excess : xp.ndarray
        Corrected shear strength minus gravitational shear stress [Pa].
    xp : module
        Eager array backend (NumPy or CuPy).

    Returns
    -------
    l_sk : xp.ndarray
        Skier-induced crack length [m], or NaN where a crack cannot form.
    crack_possible : xp.ndarray
        Boolean mask indicating where skier stress exceeds the strength
        excess.

    Notes
    -----
    Bisection finds the two angles at which the line-load stress equals
    ``strength_excess``.
    """
    alpha_star = alpha_max_skier_stress(inclination_rad)

    def stress_excess(alpha):
        return (
            skier_delta_tau(
                alpha, h_equiv, inclination_rad, line_load, xp
            )
            - strength_excess
        )

    with np.errstate(divide='ignore', invalid='ignore'):
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


def _compute_sk38_ccl_block(
    density,
    shear_strength,
    height,
    grain_type,
    grain_size,
):
    """Compute SK38-CCL for an eager block of complete profiles."""
    xp = get_backend(density)
    mech = _compute_skier_mechanics(
        density, shear_strength, height, grain_type, xp
    )
    grain_size = xp.asarray(grain_size)

    ccl = compute_rc_slope_richter2019(
        rho_layer=mech.density,
        grain_size_m=grain_size,
        shear_strength_pa=mech.tau_p,
        tau_g=mech.tau_g,
        sigma_n=mech.sigma_n,
        h_equiv=mech.h_equiv,
        xp=xp,
    )
    l_sk, crack_possible = compute_skier_crack_length(
        inclination_rad=mech.inclination_rad,
        line_load=mech.line_load,
        h_equiv=mech.h_equiv_sk,
        strength_excess=mech.tau_p - mech.tau_g,
        xp=xp,
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        index = xp.minimum(ccl / l_sk, INDEX_CAP)

    # sentinel precedence: topmost layer, then undefined mechanics, then
    # the skier-penetrated layers, then spontaneous failure, then the
    # actual (capped) index. The spontaneous verdict needs only the
    # strength excess; the crack computation additionally needs the
    # equivalent slab thickness.
    spontaneous = mech.tau_p < mech.tau_g
    known = xp.isfinite(mech.tau_p - mech.tau_g) & (
        spontaneous | xp.isfinite(mech.h_equiv_sk)
    )
    result = xp.where(crack_possible, index, INDEX_CAP)
    result = xp.where(spontaneous, 0.0, result)
    result = xp.where(mech.penetrated & ~spontaneous, xp.nan, result)
    result = xp.where(known, result, xp.nan)
    return xp.where(
        mech.top_layer | ~mech.valid, xp.nan, result
    )


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
    padding, and layers whose slab mechanics are undefined because of
    missing input data -- a missing value is never reported as stable).
    """
    require_vars(
        ds,
        [
            'density',
            'shear_strength',
            'height',
            'grain_type',
            'grain_size',
        ],
        'sk38_ccl calculation',
    )
    density_da = ds.data['density']
    shear_da = ds.data['shear_strength']
    height_da = ds.data['height']
    grain_type_da = ds.data['grain_type']
    grain_size_da = ds.data['grain_size']
    result = apply_profile_block(
        _compute_sk38_ccl_block,
        density_da,
        shear_da,
        height_da,
        grain_type_da,
        grain_size_da,
        output_dtype=float_result_dtype(
            density_da, shear_da, height_da, grain_size_da
        ),
    )

    attrs = registry_attrs(
        'sk38_ccl',
        'stability index SK38-CCL '
        '(critical crack length / skier crack length)',
        '1',
    )
    attrs['index_cap'] = INDEX_CAP
    result.attrs = attrs
    new_xr = ds.data.assign(sk38_ccl=result)
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
    below (NaN stays NaN). Layers with a finite index but undefined
    (NaN) ``rta`` are not gated and keep their ``sk38_ccl`` value, i.e.
    they conservatively count as weak-layer candidates. Uses the
    per-layer variable ``rta`` if present, otherwise computes it via
    :func:`xsnow.extensions.rta.compute_rta`; a computed ``rta`` stays
    attached to the returned dataset. An existing ``sk38_ccl`` variable is
    used directly and retained. Otherwise, ``sk38_ccl`` is computed only as
    an intermediate and is not attached to the returned dataset.
    """
    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")
    if 'rta' not in ds.data:
        ds = compute_rta(ds)
    rta_da = ds.data['rta']

    keep_index = 'sk38_ccl' in ds.data
    if keep_index:
        ungated = ds
        index_da = ds.data['sk38_ccl']
    else:
        ungated = compute_sk38_ccl(ds)
        index_da = ungated['sk38_ccl']

    rta_da = rta_da.broadcast_like(index_da)
    gated = index_da.where(
        ~(np.isfinite(index_da) & (rta_da < RTA_THRESHOLD)),
        INDEX_CAP,
    )

    attrs = registry_attrs(
        'sk38_ccl_rta',
        'RTA-gated stability index SK38-CCL',
        '1',
    )
    attrs['index_cap'] = INDEX_CAP
    attrs['rta_threshold'] = RTA_THRESHOLD

    gated.attrs = attrs
    new_xr = ungated.data.assign(sk38_ccl_rta=gated)
    if not keep_index:
        new_xr = new_xr.drop_vars('sk38_ccl')
    return ds.__class__(new_xr)
