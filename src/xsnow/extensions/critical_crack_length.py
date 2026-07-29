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
import contextlib
import math
from typing import Optional

import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..utils import get_backend
from ..registry import registry

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


def _errstate(xp, **kwargs):
    """Floating-point error-state context; a no-op on backends without
    one (CuPy does not implement ``errstate``)."""
    if hasattr(xp, 'errstate'):
        return xp.errstate(**kwargs)
    return contextlib.nullcontext()


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
    - We mask out any padded layers (thickness <= 0 or NaN) so they don't
      contaminate the cumulative sums.
    - We reverse along the layer axis, do cumsums from the top down,
      then reverse back. Finally, we shift so that layer i gets only the
      slab *strictly above* i.
    """

    # Identify physically valid layers
    valid = xp.isfinite(thickness) & (thickness > 0)

    # Zero-out invalid (padding) so they don't affect sums
    thickness_clean = xp.where(valid, thickness, 0.0)          # (..., L)
    mass_per_area = xp.where(valid, density * thickness, 0.0)  # kg m^-2

    # Accumulate "from surface downward":
    # Reverse layer axis so index -1 is now bottom, then cumsum.
    mass_cum_from_top = xp.cumsum(mass_per_area[..., ::-1], axis=-1)
    thick_cum_from_top = xp.cumsum(thickness_clean[..., ::-1], axis=-1)

    # Mean density of (layer >= current in reversed coords)
    with _errstate(xp, divide='ignore', invalid='ignore'):
        rho_rev = mass_cum_from_top / thick_cum_from_top  # (..., L), reversed axis

    # Flip back to bottom->top ordering
    rho_above_inclusive = rho_rev[..., ::-1]  # (..., L)

    # Shift by 1 so that layer i gets only layers ABOVE i (strictly above)
    # After shifting, the last entry (true surface) becomes NaN.
    tail_nan = xp.full_like(rho_above_inclusive[..., :1], xp.nan)
    rho_strict_above = xp.concatenate(
        [rho_above_inclusive[..., 1:], tail_nan],
        axis=-1
    )

    return rho_strict_above


def slab_weighted_mean(values, thickness, xp):
    """
    Thickness-weighted mean of ``values`` over all valid layers strictly
    above each layer (towards the surface). The surface layer and padded
    layers yield NaN. The layer axis is last; layer 0 is the deepest.
    """
    valid = xp.isfinite(thickness) & (thickness > 0) & xp.isfinite(values)
    weights = xp.where(valid, thickness, 0.0)
    weighted = xp.where(valid, values * thickness, 0.0)

    sum_from_top = xp.cumsum(weighted[..., ::-1], axis=-1)[..., ::-1]
    weight_from_top = xp.cumsum(weights[..., ::-1], axis=-1)[..., ::-1]

    with _errstate(xp, divide='ignore', invalid='ignore'):
        mean_incl = sum_from_top / weight_from_top

    # shift so that layer i only sees layers strictly above itself
    tail_nan = xp.full_like(mean_incl[..., :1], xp.nan)
    return xp.concatenate([mean_incl[..., 1:], tail_nan], axis=-1)


def strength_correction(grain_type, sigma_n, shear_strength, xp):
    """
    Additional shear strength from the slope-normal load [Pa].

    """
    f1 = xp.floor(grain_type / 100.0)
    tau_add_other = sigma_n * (8.0e-5 * shear_strength + 0.224)
    tau_add_pp = sigma_n * (
        8.0e-5 * shear_strength + 2.2e-5 * sigma_n + 0.056
    )
    tau_add = xp.where(f1 == 1, tau_add_pp, tau_add_other)
    return xp.where(f1 == 6, 0.0, tau_add)


def compute_slab_mechanics(density, thickness_vertical, height, grain_type,
                           shear_strength, inclination, xp):
    """
    Slab quantities above each layer on a virtual slope, as a dict of
    backend arrays (slope-normal geometry, layer axis last).
    """
    theta = math.radians(inclination)
    density = xp.asarray(density)
    thick_vert = xp.asarray(thickness_vertical)
    height = xp.asarray(height)
    grain_type = xp.asarray(grain_type)
    shear = xp.asarray(shear_strength)

    valid = xp.isfinite(thick_vert) & (thick_vert > 0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    height_n = xp.where(valid, height * cos_t, xp.nan)
    thick_n = xp.where(valid, thick_vert * cos_t, xp.nan)
    surface_n = xp.nanmax(xp.where(valid, height_n, -xp.inf), axis=-1)
    surface_n = xp.where(xp.isfinite(surface_n), surface_n, xp.nan)
    d_slab = surface_n[..., None] - height_n

    rho_slab = compute_rho_slab(density, thick_n, xp)
    e_layer = E0 * (density / RHO_ICE) ** E_EXP
    e_slab = slab_weighted_mean(e_layer ** (1.0 / 3.0), thick_n, xp) ** 3
    with _errstate(xp, divide='ignore', invalid='ignore'):
        h_equiv = d_slab * (e_slab / e_layer) ** (1.0 / 3.0)

    tau_g = rho_slab * G * d_slab * sin_t
    sigma_n = rho_slab * G * d_slab * cos_t
    tau_p = shear + strength_correction(grain_type, sigma_n, shear, xp)

    return {
        'valid': valid,
        'density': density,
        'height_n': height_n,
        'thick_n': thick_n,
        'surface_n': surface_n,
        'd_slab': d_slab,
        'rho_slab': rho_slab,
        'e_layer': e_layer,
        'e_slab': e_slab,
        'h_equiv': h_equiv,
        'tau_g': tau_g,
        'sigma_n': sigma_n,
        'tau_p': tau_p,
    }


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

    # Richter et al. parametrization
    term1_under = A_RICHTER * (
        (rho_layer / RHO_ICE) * (grain_size_m / GS_0)
    ) ** B_RICHTER

    term2_under = 2.0 * shear_strength_pa * eprime * dsl_over_sigman

    with _errstate(xp, invalid='ignore'):
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
    into each other for vanishing slope angles.
    """
    with _errstate(xp, divide='ignore', invalid='ignore'):
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
    return rc


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

    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")

    required = ['density', 'grain_size', 'shear_strength', 'height']
    if inclination is not None:
        required.append('grain_type')
    for v in required:
        if v not in ds.data:
            raise ValueError(
                f"Missing required variable for critical crack length "
                f"calculation: '{v}'"
            )

    # Pick NumPy or CuPy depending on where ds.data lives
    xp = get_backend(ds)

    density_da = ds.data['density']
    gsize_da = ds.data['grain_size']
    shear_da = ds.data['shear_strength']
    height_da = ds.data['height']

    # Derive per-layer thickness from cumulative height along 'layer'.
    # layer=0 is bottom (height ~ 0), layer increases upward.
    height_bottom = height_da.shift(layer=1, fill_value=0)
    thick_da = height_da - height_bottom  # [m]

    if inclination is None:
        # Move data to backend arrays (NumPy or CuPy)
        rho_layer_arr = xp.asarray(density_da.data)
        gsize_arr = xp.asarray(gsize_da.data)
        shear_arr = xp.asarray(shear_da.data)
        thick_arr = xp.asarray(thick_da.data)

        # Compute density of slab above each layer
        rho_slab_arr = compute_rho_slab(
            density=rho_layer_arr,
            thickness=thick_arr,
            xp=xp,
        )

        # Richter et al. (2019) flat-terrain rc
        rc_arr = compute_rc_flat_richter2019(
            rho_layer=rho_layer_arr,
            grain_size_m=gsize_arr,
            shear_strength_pa=shear_arr,
            rho_slab=rho_slab_arr,
            xp=xp,
        )
        out_name = 'rc_flat'
        default_long_name = 'critical crack length on flat terrain'
    else:
        mech = compute_slab_mechanics(
            density_da.data, thick_da.data, height_da.data,
            ds.data['grain_type'].data, shear_da.data, inclination, xp,
        )
        rc_arr = compute_rc_slope_richter2019(
            rho_layer=mech['density'],
            grain_size_m=xp.asarray(gsize_da.data),
            shear_strength_pa=mech['tau_p'],
            tau_g=mech['tau_g'],
            sigma_n=mech['sigma_n'],
            h_equiv=mech['h_equiv'],
            xp=xp,
        )
        out_name = 'rc_slope'
        default_long_name = 'critical crack length on the slope'

    rc_da = xr.DataArray(
        rc_arr,
        dims=height_da.dims,
        coords=height_da.coords,
    )

    if out_name in registry:
        reg_entry = registry[out_name]
        rc_da.attrs = {
            'long_name': reg_entry.get('long_name', default_long_name),
            'units': reg_entry.get('xsnow_units', 'm'),
            'standard_name': reg_entry.get('standard_name', out_name),
        }
    else:
        rc_da.attrs = {
            'long_name': default_long_name,
            'units': 'm',
            'standard_name': out_name,
        }

    new_xr = ds.data.assign(**{out_name: rc_da})
    return ds.__class__(new_xr)
