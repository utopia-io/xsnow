# src/xsnow/extensions.py
"""
This module contains all functions registered as methods on the xsnowDataset object.

Functions decorated with `@register_xsnowDataset_method` are defined here and
become available as methods (e.g., `ds.compute_critical_cut_length()`). This
keeps the core `xsnowDataset` class lean while allowing for a rich, extensible
set of features.

"""

import contextlib
import re
from typing import TYPE_CHECKING
import numpy as np
import re
import pandas as pd
import xarray as xr

from .plugin import register_xsnowDataset_method
from .utils import get_backend
from .registry import registry

# Use a type-checking block to avoid a circular import at runtime,
# while still allowing type hints for the xsnowDataset class.
if TYPE_CHECKING:
    from .dataset import xsnowDataset

# ================================================================= #
#                  SCIENTIFIC ANALYSIS EXTENSIONS                   #
# ================================================================= #

@register_xsnowDataset_method
def find_layer_by_criteria(ds: 'xsnowDataset', criteria: str) -> 'xsnowDataset':
    """
    Finds and masks layers that do not match a string of boolean criteria.

    This function parses a criteria string and applies it as a filter to the
    dataset. Data in layers that do not meet the specified conditions is
    replaced with NaN, effectively isolating the layers of interest. The
    shape and dimensions of the dataset are preserved.

    For robustness, this operation is performed on a CPU-based copy of the data.

    Args:
        ds: The input dataset to filter.
        criteria: A string containing the filtering conditions. Conditions can be
            combined using '&' (and) and '|' (or).
            Example: "density > 300 & grain_size < 0.5"

    Returns:
        A new xsnowDataset instance where data in layers that do not match the
        criteria is replaced with NaN.

    Raises:
        ValueError: If the criteria string is invalid, or if a variable
            in the criteria is not found in the dataset.
    """
    if ds.data is None:
        raise ValueError("Cannot filter an empty dataset.")

    # This complex filtering is more robust on the CPU with pandas
    ds_cpu = ds.to_cpu()

    conditions = re.split(r'\s*([&|])\s*', criteria)
    final_mask = None
    
    i = 0
    while i < len(conditions):
        condition_str = conditions[i]
        
        match = re.match(r'^\s*(\w+)\s*([<>=!]+)\s*(-?\d+\.?\d*)\s*$', condition_str)
        if not match:
            raise ValueError(f"Could not parse condition: '{condition_str}'")
        
        var, op, val = match.groups()
        val = float(val)

        if var not in ds_cpu.data:
            raise ValueError(f"Variable '{var}' not found in dataset.")

        variable_data = ds_cpu.data[var]
        if op == '>': mask = variable_data > val
        elif op == '<': mask = variable_data < val
        elif op == '>=': mask = variable_data >= val
        elif op == '<=': mask = variable_data <= val
        elif op == '==': mask = variable_data == val
        elif op == '!=': mask = variable_data != val
        else: raise ValueError(f"Unsupported operator: '{op}'")

        if final_mask is None:
            final_mask = mask
        else:
            logical_op = conditions[i-1]
            if logical_op == '&': final_mask = final_mask & mask
            elif logical_op == '|': final_mask = final_mask | mask
        
        i += 2

    return ds.data if final_mask is None else ds_cpu.data.where(final_mask)

@register_xsnowDataset_method
def compute_critical_cut_length(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Computes the critical cut length (rc_flat) for each layer.

    This function calculates the critical crack length required for
    self-propagating fracture in a snow slab on flat terrain, a key parameter
    in avalanche forecasting. The calculation is hardware-aware and will use
    CuPy for GPU acceleration if the dataset's arrays are on the GPU.

    The surface layer of the snowpack will have a result of NaN, as it has no
    slab above it.

    Args:
        ds: The input dataset. Must contain the following data variables:
            'density', 'grain_size', 'shear_strength', and 'height'.

    Returns:
        A new xsnowDataset instance with the 'rc_flat' variable (in meters) added.

    Raises:
        ValueError: If the dataset is empty or if any of the required
            data variables are missing.
    """
    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")

    required_vars = ['density', 'grain_size', 'shear_strength', 'height']
    for var in required_vars:
        if var not in ds.data:
            raise ValueError(f"Missing required variable for rc_flat calculation: '{var}'")

    # --- Use the utility function to get the correct backend ---
    xp = get_backend(ds)
    is_gpu = (xp != np)

    # --- Physical Constants ---
    RHO_ICE, GS_0, G, A, B = 917.0, 0.00125, 9.81, 4.6e-9, -2.0

    height, density, grain_size, shear_strength = (
        ds.data['height'], ds.data['density'],
        ds.data['grain_size'], ds.data['shear_strength']
    )

    # --- Calculations ---
    height_of_bottom = height.shift(layer=1, fill_value=0)
    thick = height - height_of_bottom
    layer_load = (density * thick * G)
    load = layer_load.reindex(layer=layer_load.layer[::-1]).cumsum(dim='layer').reindex(layer=layer_load.layer)

    total_thick_above = ds.data['height'].max(dim='layer', skipna=True) - height
    rho_sl_raw = load / (total_thick_above * G)
    
    if is_gpu:
        rho_sl_numpy = rho_sl_raw.as_numpy()
        rho_sl_cpu_da = xr.DataArray(rho_sl_numpy, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
        rho_sl_filled_cpu = rho_sl_cpu_da.bfill(dim='layer').ffill(dim='layer')
        rho_sl = xr.DataArray(xp.asarray(rho_sl_filled_cpu.values), dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
    else:
        rho_sl = rho_sl_raw.bfill(dim='layer').ffill(dim='layer')
    
    # Ensure all raw arrays are on the correct backend (xp)
    # before performing calculations. This prevents type errors if xarray
    # unexpectedly returns a numpy array from a GPU-backed dataset.
    tau_p_data = xp.asarray(shear_strength.data) * 1000.0
    gs_data = xp.asarray(grain_size.data) * 0.001
    rho_sl_data = xp.asarray(rho_sl.data)
    density_data = xp.asarray(density.data)

    e_prime_data = 5.07e9 * (rho_sl_data / RHO_ICE)**5.13 / (1 - 0.2**2)
    dsl_over_sigman_data = 1.0 / (G * rho_sl_data)
    term1_under_data = A * (density_data / RHO_ICE * gs_data / GS_0)**B
    term2_under_data = 2 * tau_p_data * e_prime_data * dsl_over_sigman_data

    context = contextlib.nullcontext() if is_gpu else xp.errstate(invalid='ignore')

    with context:
        term1 = xp.sqrt(xp.where(term1_under_data > 0, term1_under_data, xp.nan))
        term2 = xp.sqrt(xp.where(term2_under_data > 0, term2_under_data, xp.nan))
    
    rc_flat_combined = term1 * term2
    rc_flat_combined[..., -1] = xp.nan
    
    rc_flat_final = xr.DataArray(
        rc_flat_combined, 
        dims=height.dims, 
        coords=height.coords
    )
    
    if 'rc_flat' in registry:
        reg_entry = registry['rc_flat']
        rc_flat_final.attrs = {
            'long_name': reg_entry.get('long_name'),
            'units': reg_entry.get('si_units'),
            'standard_name': reg_entry.get('standard_name')
        }

    return ds.data.assign(rc_flat=rc_flat_final)


@register_xsnowDataset_method
def get_wet_slab_LOC(ds: 'xsnowDataset') -> xr.DataArray:
    """
    Finds the height of the wet slab Layer of Concern (LOC) for each timestamp.

    The LOC is defined as the layer in the bottom half of the snowpack with a
    weak grain type (Facets or Depth Hoar) that has the largest grain size
    increase from the layer immediately above it.

    Args:
        ds (xsnowDataset): The dataset to analyze.

    Returns:
        xr.DataArray: A DataArray of the LOC height for each timestamp.
        Values will be NaN for timestamps where no LOC is found.
    """
    # Ensure HS is available for the calculation
    ds = ds.compute_hs() if 'HS' not in ds.data else ds
    hs = ds.data['HS']

    # Criteria 1 & 2: Mask for candidate layers
    is_in_bottom_half = ds.data['height'] <= (hs / 2)
    is_weak_grain_type = (ds.data['grain_type'] >= 400) & (ds.data['grain_type'] <= 599)
    is_candidate = is_in_bottom_half & is_weak_grain_type

    # Criteria 3: Calculate the grain size increase from the layer above
    grain_size_gradient = ds.data['grain_size'] - ds.data['grain_size'].shift(layer=1)
    
    # Consider only positive gradients for our candidate layers
    candidate_gradients = grain_size_gradient.where(is_candidate & (grain_size_gradient > 0))
    
    # Find the layer index that has the maximum gradient for each timestamp
    loc_layer_indices = candidate_gradients.idxmax(dim='layer')

    # FIX: Use a robust "fill-and-mask" approach for selection
    # 1. Create a boolean mask of where LOCs were actually found (where the index is not NaN)
    was_loc_found = loc_layer_indices.notnull()

    # 2. Create a clean indexer: fill NaNs with a placeholder (like 0) and convert to integer
    clean_indices = loc_layer_indices.fillna(0).astype(int)

    # 3. Perform the selection using the clean indexer. This will select the correct
    #    layer where an LOC was found, and a placeholder (layer 0) otherwise.
    selected_heights = ds.data.height.sel(layer=clean_indices)
    
    # 4. Use the boolean mask to nullify the placeholder values, leaving only the
    #    correctly identified heights and filling the rest with NaN.
    return selected_heights.where(was_loc_found)
