"""
Analytical parametrizations for the critical crack length

This module implements the parametrization by Richter et al (2019) for the
critical crack length, in two formulations: flat terrain (the default) and
a virtual slope. The slope formulation is part of the sk38_ccl index; it
adds the slope-parallel load and uses the weak layer's own modulus with an
equivalent slab thickness. Computation runs on CPU or GPU depending on
user environment and install.

Other parametrizations are welcome to be added.

"""
import math
from dataclasses import dataclass
from typing import Any, Optional

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
    get_backend,
)

# ---------------------------------------------------------------------------
# Physical constants for the Richter et al. (2019) crack length formulation
# ---------------------------------------------------------------------------
RHO_ICE = 917.0          # kg m^-3, density of ice
G = 9.81                 # m s^-2, gravitational acceleration
POISSON = 0.2            # [-], assumed Poisson's ratio for snow slab
GS_0 = 0.00125           # m, reference grain size scale in Richter param.
A_RICHTER = 4.6e-9       # Richter et al. (2019) parameter "a"
B_RICHTER = -2.0         # Richter et al. (2019) parameter "b"
E0 = 5.07e9              # Pa, modulus prefactor in Richter et al. (2019)
E_EXP = 5.13             # exponent on (rho_slab / rho_ice)


@dataclass(frozen=True, slots=True)
class _SlabMechanics:
    """Reusable eager-array results from the virtual-slope calculation.

    ``height_n``, ``thick_n``, and ``surface_n`` use slope-normal geometry.
    Only values consumed by the crack-length or skier calculations are kept;
    full-array intermediates used solely inside ``compute_slab_mechanics`` are
    released when that function returns.
    """

    valid: Any
    density: Any
    height_n: Any
    thick_n: Any
    surface_n: Any
    e_layer: Any
    h_equiv: Any
    tau_g: Any
    sigma_n: Any
    tau_p: Any


# ---------------------------------------------------------------------------
# Slab and crack-length physics
# ---------------------------------------------------------------------------


def layer_thickness(height_da):
    """Per-layer vertical thickness [m] from the cumulative ``height``
    (layer 0 = bottom)."""
    return height_da - height_da.shift(layer=1, fill_value=0)


def _require_contiguous_layers(height, xp):
    """Raise if a layer below the surface has an undefined (NaN)
    height: the cumulative slab sums would silently skip its mass. The
    check runs on the heights -- in the derived thicknesses such a gap
    right under the surface is indistinguishable from legitimate
    padding. Zero-thickness layers are fine (they carry no mass)."""
    finite = xp.isfinite(height)
    n_finite_above = (
        xp.cumsum(finite[..., ::-1].astype(xp.int64), axis=-1)[..., ::-1]
        - finite.astype(xp.int64)
    )
    if bool(xp.any(~finite & (n_finite_above > 0))):
        raise ValueError(
            "Profile contains an interior layer of undefined height "
            "(non-finite height below the surface); layers must be "
            "contiguous, with padding only above the surface."
        )


def _mean_strictly_above(numer, denom, xp):
    """
    Cumulative ``sum(numer) / sum(denom)`` over all layers strictly above
    each layer (towards the surface). The surface layer, with nothing
    above it, yields NaN. The layer axis is last; layer 0 is the deepest.
    """
    numer_cum = xp.cumsum(numer[..., ::-1], axis=-1)[..., ::-1]
    denom_cum = xp.cumsum(denom[..., ::-1], axis=-1)[..., ::-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_incl = numer_cum / denom_cum

    # shift so that layer i only sees layers strictly above itself
    tail_nan = xp.full_like(mean_incl[..., :1], xp.nan)
    return xp.concatenate([mean_incl[..., 1:], tail_nan], axis=-1)


def compute_rho_slab(density, thickness, xp):
    """
    Compute the average slab density above each physical layer.

    For each layer i, we define the "slab" as all layers *strictly above* i
    (shallower, closer to the snow surface). We return the mean density of
    that overlying slab. The topmost physical layer has no slab above, so it
    receives NaN. Padded layers above the real surface are also NaN.

    Assumptions
    -----------
    - Axis -1 is the 'layer' dimension.
    - layer=0 is the deepest layer (near ground).
    - layer increases upward toward the snow surface.
    - After the true surface layer, remaining entries are padding with NaN
      in both `density` and `thickness`.
    - `thickness` is the physical thickness [m] of each layer. Zero/NaN
      thickness means "not a real layer" (padding).

    Parameters
    ----------
    density : xp.ndarray
        Shape (..., L). Layer density [kg m^-3], ordered bottom (0) -> surface.
    thickness : xp.ndarray
        Shape (..., L). Layer thickness [m], same ordering and shape.
    xp : module
        Array backend (e.g. numpy or cupy).

    Returns
    -------
    rho_slab : xp.ndarray
        Shape (..., L). For each layer i, rho_slab[i] is the mean density of
        all VALID layers above i. The surface layer itself (no overlying slab)
        is NaN. Padded layers beyond the surface remain NaN.

    Notes
    -----
    We mask out any padded layers (thickness <= 0 or NaN) so they don't
    contaminate the cumulative sums; the slab mean itself is
    ``sum(mass) / sum(thickness)`` over the layers strictly above.
    """
    # Identify physically valid layers
    valid = xp.isfinite(thickness) & (thickness > 0)

    # Zero-out invalid (padding) so they don't affect sums
    thickness_clean = xp.where(valid, thickness, 0.0)          # (..., L)
    mass_per_area = xp.where(valid, density * thickness, 0.0)  # kg m^-2

    return _mean_strictly_above(mass_per_area, thickness_clean, xp)


def slab_weighted_mean(values, thickness, xp):
    """
    Thickness-weighted mean of ``values`` over all valid layers strictly
    above each layer (towards the surface). The surface layer and padded
    layers yield NaN. The layer axis is last; layer 0 is the deepest.
    """
    valid = xp.isfinite(thickness) & (thickness > 0) & xp.isfinite(values)
    weights = xp.where(valid, thickness, 0.0)
    weighted = xp.where(valid, values * thickness, 0.0)

    return _mean_strictly_above(weighted, weights, xp)


def strength_correction(grain_type, sigma_n, shear_strength, xp):
    """
    Additional shear strength from the slope-normal load [Pa].

    An unknown (NaN) grain type yields NaN -- the correction depends on
    the grain form, so a missing value cannot default to a form.
    """
    f1 = xp.floor(grain_type / 100.0)
    tau_add_other = sigma_n * (8.0e-5 * shear_strength + 0.224)
    tau_add_pp = sigma_n * (
        8.0e-5 * shear_strength + 2.2e-5 * sigma_n + 0.056
    )
    tau_add = xp.where(f1 == 1, tau_add_pp, tau_add_other)
    tau_add = xp.where(f1 == 6, 0.0, tau_add)
    return xp.where(xp.isfinite(f1), tau_add, xp.nan)


def compute_slab_mechanics(density, thickness_vertical, height, grain_type,
                           shear_strength, inclination, xp) -> _SlabMechanics:
    """
    Reusable slab quantities above each layer on a virtual slope.

    This calculation accepts eager array blocks. Dask-backed datasets must
    use the registered dataset methods, which schedule it blockwise with a
    complete layer axis. It returns a frozen record of backend arrays with the
    layer axis last.
    """
    inputs = (density, thickness_vertical, height, grain_type, shear_strength)
    if any(hasattr(array, '__dask_graph__') for array in inputs):
        raise TypeError(
            "compute_slab_mechanics expects eager array blocks; use the "
            "xsnowDataset stability methods for Dask-backed data."
        )

    theta = math.radians(inclination)
    density = xp.asarray(density)
    thick_vert = xp.asarray(thickness_vertical)
    height = xp.asarray(height)
    grain_type = xp.asarray(grain_type)
    shear = xp.asarray(shear_strength)

    _require_contiguous_layers(height, xp)
    valid = xp.isfinite(thick_vert) & (thick_vert > 0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    height_n = xp.where(valid, height * cos_t, xp.nan)
    thick_n = xp.where(valid, thick_vert * cos_t, xp.nan)
    surface_n = finite_max(height_n, xp)
    d_slab = surface_n[..., None] - height_n

    rho_slab = compute_rho_slab(density, thick_n, xp)
    e_layer = E0 * (density / RHO_ICE) ** E_EXP
    e_slab = slab_weighted_mean(e_layer ** (1.0 / 3.0), thick_n, xp) ** 3
    with np.errstate(divide='ignore', invalid='ignore'):
        h_equiv = d_slab * (e_slab / e_layer) ** (1.0 / 3.0)

    tau_g = rho_slab * G * d_slab * sin_t
    sigma_n = rho_slab * G * d_slab * cos_t
    tau_p = shear + strength_correction(grain_type, sigma_n, shear, xp)

    return _SlabMechanics(
        valid=valid,
        density=density,
        height_n=height_n,
        thick_n=thick_n,
        surface_n=surface_n,
        e_layer=e_layer,
        h_equiv=h_equiv,
        tau_g=tau_g,
        sigma_n=sigma_n,
        tau_p=tau_p,
    )


def compute_rc_flat_richter2019(
    rho_layer,
    grain_size_m,
    shear_strength_pa,
    rho_slab,
    xp,
):
    """
    Compute critical crack length rc_flat [m] following Richter et al. (2019).

    This returns the critical self-propagating crack length for a weak layer
    under a slab on *flat terrain* (i.e. slope effects are NOT included).
    Layers with no overlying slab (surface layer, padding) get NaN.

    Parameters
    ----------
    rho_layer : xp.ndarray
        Layer density [kg m^-3], shape (..., L).
        This is the candidate failure layer density.
    grain_size_m : xp.ndarray
        Grain size [m] for each layer, shape (..., L).
    shear_strength_pa : xp.ndarray
        Shear strength [Pa] for each layer, shape (..., L).
    rho_slab : xp.ndarray
        Mean density [kg m^-3] of the slab ABOVE each layer
        (output of compute_rho_slab), shape (..., L).
        Should be NaN for the surface layer and padded layers.
    xp : module
        Array backend (numpy or cupy).

    Returns
    -------
    rc_flat : xp.ndarray
        Shape (..., L). Critical crack length [m] for each layer.
        NaN where no slab overlies the layer.

    Notes
    -----
    This uses:
        E' = E0 * (rho_slab / RHO_ICE)**E_EXP / (1 - POISSON^2)
        D_sl/sigma_n = 1 / (g * rho_slab)
        term1 ~ sqrt( A * (...)**B )
        term2 ~ sqrt( 2 * tau_p * E' * D_sl/sigma_n )
        rc_flat = term1 * term2

    where tau_p is shear strength in Pa and grain size is in m, matching
    xsnow's internal unit convention.

    No slope angle correction is applied here; this is for flat terrain
    (normal loading only).
    """
    # E' (effective modulus)
    eprime = (
        E0
        * (rho_slab / RHO_ICE) ** E_EXP
        / (1.0 - POISSON ** 2)
    )

    # D_sl / sigma_n  = 1 / (g * rho_slab)
    dsl_over_sigman = 1.0 / (G * rho_slab)

    # Richter et al. parametrization. A layer with zero grain size or
    # density (nonphysical fill values) makes this power infinite and
    # rc_flat comes out +inf (with a numpy divide warning) --
    # deliberately kept for now rather than masked to NaN.
    term1_under = A_RICHTER * (
        (rho_layer / RHO_ICE) * (grain_size_m / GS_0)
    ) ** B_RICHTER

    term2_under = 2.0 * shear_strength_pa * eprime * dsl_over_sigman

    with np.errstate(invalid='ignore'):
        term1 = xp.sqrt(xp.where(term1_under > 0, term1_under, xp.nan))
        term2 = xp.sqrt(xp.where(term2_under > 0, term2_under, xp.nan))

    rc_flat = term1 * term2

    # Ensure that layers without an overlying slab (rho_slab = NaN) remain NaN
    rc_flat = xp.where(xp.isnan(rho_slab), xp.nan, rc_flat)

    return rc_flat


def compute_rc_slope_richter2019(
    rho_layer,
    grain_size_m,
    shear_strength_pa,
    tau_g,
    sigma_n,
    h_equiv,
    xp,
):
    """
    Compute critical crack length rc_slope [m] on a virtual slope.

    Notes
    -----
    ``rc = sqrt(F_wl) sqrt(h_equiv E_wl') (-tau_g + sqrt(tau_g^2 +
    2 sigma_n (tau_p - tau_g))) / sigma_n`` (Richter et al. 2019).
    Unlike rc_flat, rc_slope includes the slope-parallel load and uses
    the weak layer's own modulus; the two formulations do not converge
    into each other for vanishing slope angles. Layers whose
    slope-parallel load exceeds the (corrected) strength -- the
    spontaneous failure regime -- are NaN.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        f_wl = A_RICHTER * (
            rho_layer / RHO_ICE * grain_size_m / GS_0
        ) ** B_RICHTER
        e_wl = E0 * (rho_layer / RHO_ICE) ** E_EXP
        eprime_wl = e_wl / (1.0 - POISSON ** 2)
        excess = (
            tau_g ** 2 + 2.0 * sigma_n * (shear_strength_pa - tau_g)
        )
        rc = (
            xp.sqrt(f_wl)
            * xp.sqrt(h_equiv * eprime_wl)
            * (-tau_g + xp.sqrt(excess))
            / sigma_n
        )
        rc = xp.where(shear_strength_pa >= tau_g, rc, xp.nan)
    return rc


# ---------------------------------------------------------------------------
# Array-execution adapter
# ---------------------------------------------------------------------------


def _compute_critical_crack_length_block(
    density,
    grain_size,
    shear_strength,
    height,
    thickness,
    grain_type=None,
    *,
    inclination=None,
):
    """Connect the xarray gateway to the eager Richter physics functions.

    Each argument is one eager block whose last axis contains every layer.
    The outer axes may contain any number of profiles from chunks of times,
    locations, slopes, or other dimensions.
    """
    xp = get_backend(density)

    if inclination is None:
        density = xp.asarray(density)
        grain_size = xp.asarray(grain_size)
        shear_strength = xp.asarray(shear_strength)
        height = xp.asarray(height)
        thickness = xp.asarray(thickness)

        _require_contiguous_layers(height, xp)
        rho_slab = compute_rho_slab(density, thickness, xp)
        return compute_rc_flat_richter2019(
            rho_layer=density,
            grain_size_m=grain_size,
            shear_strength_pa=shear_strength,
            rho_slab=rho_slab,
            xp=xp,
        )

    if grain_type is None:
        raise ValueError(
            "The slope formulation requires a grain-type array."
        )
    mech = compute_slab_mechanics(
        density,
        thickness,
        height,
        grain_type,
        shear_strength,
        inclination,
        xp,
    )
    return compute_rc_slope_richter2019(
        rho_layer=mech.density,
        grain_size_m=xp.asarray(grain_size),
        shear_strength_pa=mech.tau_p,
        tau_g=mech.tau_g,
        sigma_n=mech.sigma_n,
        h_equiv=mech.h_equiv,
        xp=xp,
    )


# ---------------------------------------------------------------------------
# Registered dataset gateway
# ---------------------------------------------------------------------------


@register_xsnowDataset_method
def compute_critical_crack_length(
    ds: 'xsnowDataset',
    inclination: Optional[float] = None,
) -> 'xsnowDataset':
    """
    Compute and attach the critical crack length to this xsnowDataset:
    `rc_flat` [m] for flat terrain (`inclination=None`, the default,
    `compute_rc_flat_richter2019()`), or `rc_slope` [m] on a virtual
    slope of the given angle (`compute_rc_slope_richter2019()`).

    Requirements
    ------------
    The dataset must contain:
      - `density`            [kg m^-3]
      - `grain_size`         [m]
      - `shear_strength`     [Pa]
      - `height`             [m], cumulative height from ground upward
      - `grain_type`         (Swiss Code F1F2F3), slope formulation only

    We assume:
      - Dimension `layer` exists and indexes from ground (layer=0) upward.
      - `height` is monotonically increasing along `layer`.
      - Layers above the true snow surface are padded with NaN thickness.

    Parameters
    ----------
    inclination : float, optional
        Virtual slope angle in degrees. None (default) selects the
        flat-terrain formulation.

    Returns
    -------
    new_ds : xsnowDataset
        A new xsnowDataset with a new data variable `rc_flat` [m] or
        `rc_slope` [m].
    """

    required = ['density', 'grain_size', 'shear_strength', 'height']
    if inclination is not None:
        required.append('grain_type')
    require_vars(ds, required, 'critical crack length calculation')

    density_da = ds.data['density']
    gsize_da = ds.data['grain_size']
    shear_da = ds.data['shear_strength']
    height_da = ds.data['height']

    thick_da = layer_thickness(height_da)  # [m]
    output_dtype = float_result_dtype(
        density_da, gsize_da, shear_da, height_da
    )

    inputs = [density_da, gsize_da, shear_da, height_da, thick_da]
    if inclination is None:
        out_name = 'rc_flat'
        default_long_name = 'critical crack length on flat terrain'
    else:
        inputs.append(ds.data['grain_type'])
        out_name = 'rc_slope'
        default_long_name = 'critical crack length on the slope'

    rc_da = apply_profile_block(
        _compute_critical_crack_length_block,
        *inputs,
        output_dtype=output_dtype,
        kwargs={'inclination': inclination},
    )

    rc_da.attrs = registry_attrs(out_name, default_long_name, 'm')
    new_xr = ds.data.assign(**{out_name: rc_da})
    return ds.__class__(new_xr)
