# analysis.py
"""
This module contains functions for advanced snowpack analysis and the
computation of post-processing variables.

The functions in this module are designed to be hardware-aware, leveraging
GPU acceleration with CuPy when available, and falling back to NumPy for
CPU-based computation. They typically take an `xsnowDataset` object as input,
perform a calculation, and return a new `xsnowDataset` instance with the
results added as new data variables.
"""

import contextlib
from typing import TYPE_CHECKING
import numpy as np
import re
import pandas as pd
import xarray as xr

# Import the new utility function
from .utils import get_backend

# Use a type-checking block to avoid a circular import at runtime,
# while still allowing type hints for the xsnowDataset class.
if TYPE_CHECKING:
    from .dataset import xsnowDataset

# --- Analysis Functions ---

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

    if final_mask is None:
        return ds

    filtered_data = ds_cpu.data.where(final_mask)
    
    from .dataset import xsnowDataset
    return xsnowDataset(filtered_data)


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
    
    # FIX: Explicitly ensure all raw arrays are on the correct backend (xp)
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
    
    rc_flat_final.attrs = {
        'long_name': 'Critical Cut Length (Flat Terrain)',
        'units': 'm'
    }

    from .dataset import xsnowDataset
    return xsnowDataset(ds.data.assign(rc_flat=rc_flat_final))