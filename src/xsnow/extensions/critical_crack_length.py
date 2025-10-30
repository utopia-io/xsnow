"""
Analytical parametrizations for the critical crack length

This module implements the parametrization by Richter et al (2019) for the critical crack length
(flat terrain). It allows computation on CPU or GPU depending on user environment and install.

Other parametrizations are welcome to be added.

"""
import contextlib
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

@register_xsnowDataset_method
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
    with xp.errstate(divide='ignore', invalid='ignore'):
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


def compute_rc_flat_richter2019(
    rho_layer,
    grain_size_mm,
    shear_strength_kpa,
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
    grain_size_mm : xp.ndarray
        Grain size [mm] for each layer, shape (..., L).
    shear_strength_kpa : xp.ndarray
        Shear strength [kPa] for each layer, shape (..., L).
        (kPa = kN/m^2; will be converted to Pa.)
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

    where tau_p is shear strength converted from kPa to Pa,
    and grain size is converted from mm to m.

    No slope angle correction is applied here; this is for flat terrain
    (normal loading only), like the original xsnow rc_flat calculation.
    """

    # Unit conversions
    grain_size_m = grain_size_mm * 0.001     # mm -> m
    tau_p_pa = shear_strength_kpa * 1000.0  # kPa -> Pa

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

    term2_under = 2.0 * tau_p_pa * eprime * dsl_over_sigman

    with xp.errstate(invalid='ignore'):
        term1 = xp.sqrt(xp.where(term1_under > 0, term1_under, xp.nan))
        term2 = xp.sqrt(xp.where(term2_under > 0, term2_under, xp.nan))

    rc_flat = term1 * term2

    # Ensure that layers without an overlying slab (rho_slab = NaN) remain NaN
    rc_flat = xp.where(xp.isnan(rho_slab), xp.nan, rc_flat)

    return rc_flat


@register_xsnowDataset_method
def compute_critical_crack_length(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Compute and attach the variable `rc_flat` (critical crack length on flat
    terrain, in meters) to this xsnowDataset.

    This wraps the physics in `compute_rho_slab()` and
    `compute_rc_flat_richter2019()` and handles:
    - pulling data out of the xarray-backed dataset
    - computing layer thickness from cumulative height
    - dispatching to NumPy or CuPy depending on backend
    - wrapping the result back into xarray with metadata

    Requirements
    ------------
    The dataset must contain:
      - `density`            [kg m^-3]
      - `grain_size`         [mm]
      - `shear_strength`     [kPa]
      - `height`             [m], cumulative height from ground upward

    We assume:
      - Dimension `layer` exists and indexes from ground (layer=0) upward.
      - `height` is monotonically increasing along `layer`.
      - Layers above the true snow surface are padded with NaN thickness.

    Slope angle / aspect:
      - Slope angle (`slope_inclination`) and aspect (`slope_azimuth`) are NOT used here.
        This is a flat-terrain formulation (same as in earlier xsnow versions).

    Returns
    -------
    new_ds : xsnowDataset
        A new xsnowDataset with a new data variable `rc_flat` [m].
    """

    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")

    required = ['density', 'grain_size', 'shear_strength', 'height']
    for v in required:
        if v not in ds.data:
            raise ValueError(
                f"Missing required variable for rc_flat calculation: '{v}'"
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

    # Move data to backend arrays (xp.ndarray or cupy.ndarray)
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

    # Physics core: Richter et al. (2019) flat-terrain rc
    rc_flat_arr = compute_rc_flat_richter2019(
        rho_layer=rho_layer_arr,
        grain_size_mm=gsize_arr,
        shear_strength_kpa=shear_arr,
        rho_slab=rho_slab_arr,
        xp=xp,
    )

    # Wrap result back into xarray using same dims/coords as height
    rc_flat_da = xr.DataArray(
        rc_flat_arr,
        dims=height_da.dims,
        coords=height_da.coords,
    )

    # Attach registry metadata if available
    if 'rc_flat' in registry:
        reg_entry = registry['rc_flat']
        rc_flat_da.attrs = {
            'long_name': reg_entry.get('long_name', 'critical crack length (flat terrain)'),
            'units': reg_entry.get('si_units', 'm'),
            'standard_name': reg_entry.get('standard_name', 'rc_flat'),
        }
    else:
        rc_flat_da.attrs = {
            'long_name': 'critical crack length on flat terrain',
            'units': 'm',
            'standard_name': 'rc_flat',
        }

    new_xr = ds.data.assign(rc_flat=rc_flat_da)
    return ds.__class__(new_xr)
