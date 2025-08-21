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

from typing import TYPE_CHECKING
import numpy as np
import re
import pandas as pd
import xarray as xr

# Use a type-checking block to avoid a circular import at runtime,
# while still allowing type hints for the xsnowDataset class.
if TYPE_CHECKING:
    from .dataset import xsnowDataset

# --- Analysis Functions ---

def find_layer_by_criteria(ds: 'xsnowDataset', criteria: str) -> 'xsnowDataset':
    """
    Finds layers that match a string of boolean criteria.

    This function parses a criteria string (e.g., 'density > 300 & grain_size < 0.5')
    and applies it as a filter to the dataset, returning only the layers that
    meet the specified conditions.

    Args:
        ds (xsnowDataset): The input dataset to filter.
        criteria (str): A string containing the filtering conditions.
            - Supported operators: >, <, >=, <=, ==, !=
            - Supported logical operators: & (and), | (or)

    Returns:
        xsnowDataset: A new xsnowDataset instance containing only the layers
        that match the criteria. Other layers will be filled with NaN.
    """
    if ds.data is None:
        raise ValueError("Cannot filter an empty dataset.")

    # --- Hardware-Aware Computation Setup ---
    is_gpu = hasattr(ds.data.variables[list(ds.data.data_vars)[0]].data, 'device')
    
    # For this complex filtering, it's more robust to work on the CPU with pandas
    ds_cpu = ds.to_cpu() if is_gpu else ds

    # Split the criteria by logical operators (& or |)
    conditions = re.split(r'\s*([&|])\s*', criteria)
    
    final_mask = None
    
    # Process the list of conditions and operators
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
    filtered_ds_obj = xsnowDataset(filtered_data)
    
    # If the original was on GPU, move the result back
    return filtered_ds_obj.to_gpu() if is_gpu else filtered_ds_obj


def compute_critical_cut_length(ds: 'xsnowDataset') -> 'xsnowDataset':
    """
    Computes the critical cut length (rc_flat) for each layer.

    This function is adapted from the original snowpack_reader.py and calculates
    the critical crack length required for self-propagating fracture. It is
    hardware-aware and will use CuPy for GPU-accelerated computation if the
    dataset's arrays are CuPy arrays.

    Args:
        ds (xsnowDataset): The input dataset. Must contain 'density',
            'grain_size', 'shear_strength', and 'height' variables.

    Returns:
        xsnowDataset: A new xsnowDataset instance with the 'rc_flat'
        variable added.
    """
    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")

    required_vars = ['density', 'grain_size', 'shear_strength', 'height']
    for var in required_vars:
        if var not in ds.data:
            raise ValueError(f"Missing required variable for rc_flat calculation: '{var}'")

    # --- Hardware-Aware Computation Setup ---
    xp = np
    is_gpu = False
    if hasattr(ds.data['density'].data, 'device'):
        try:
            import cupy as cp
            xp = cp
            is_gpu = True
            print("INFO: Performing critical cut length computation on GPU.")
        except ImportError:
            print("WARNING: Data is on GPU but CuPy is not found. Computation may fail.")

    # --- Physical Constants ---
    RHO_ICE, GS_0, G, A, B = 917.0, 0.00125, 9.81, 4.6e-9, -2.0

    # --- Data Preparation ---
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
    
    # Handle filling NaN values in a hardware-aware way
    if is_gpu:
        # xarray's bfill/ffill are not GPU-compatible, so we must move to CPU and back
        rho_sl_numpy = rho_sl_raw.as_numpy()
        rho_sl_cpu_da = xr.DataArray(rho_sl_numpy, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
        rho_sl_filled_cpu = rho_sl_cpu_da.bfill(dim='layer').ffill(dim='layer')
        rho_sl_gpu = xp.asarray(rho_sl_filled_cpu.values)
        rho_sl = xr.DataArray(rho_sl_gpu, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
    else:
        rho_sl = rho_sl_raw.bfill(dim='layer').ffill(dim='layer')

    tau_p = shear_strength * 1000.0
    gs = grain_size * 0.001
    e_prime = 5.07e9 * (rho_sl / RHO_ICE)**5.13 / (1 - 0.2**2)
    dsl_over_sigman = 1.0 / (G * rho_sl)
    term1_under = A * (density / RHO_ICE * gs / GS_0)**B
    term2_under = 2 * tau_p * e_prime * dsl_over_sigman

    # The .where() method on a CuPy-backed DataArray can be buggy.
    # This more explicit logic operates on the raw array data to avoid the issue.
    
    # Term 1
    term1_data = term1_under.data.copy()
    term1_data[term1_data <= 0] = xp.nan
    with np.errstate(invalid='ignore'):
        term1_sqrt = xp.sqrt(term1_data)
    term1 = xr.DataArray(term1_sqrt, dims=term1_under.dims, coords=term1_under.coords)

    # Term 2
    term2_data = term2_under.data.copy()
    term2_data[term2_data <= 0] = xp.nan
    with np.errstate(invalid='ignore'):
        term2_sqrt = xp.sqrt(term2_data)
    term2 = xr.DataArray(term2_sqrt, dims=term2_under.dims, coords=term2_under.coords)
    
    rc_flat_combined = term1 * term2
    
    # Use xp.nan_to_num on the raw data array instead of .fillna()
    # to avoid the CuPy incompatibility.
    rc_flat_filled_data = xp.nan_to_num(rc_flat_combined.data, nan=9999.0)
    rc_flat_da = xr.DataArray(rc_flat_filled_data, dims=rc_flat_combined.dims, coords=rc_flat_combined.coords)

    # Set rc_flat for the surface layer to 9999.0 as it is not physically meaningful
    max_heights = height.max(dim='layer', skipna=True)
    is_not_surface = height < max_heights
    
    # Use the hardware-aware `xp.where` on the raw data arrays to avoid
    # the bug in xarray's `.where()` with CuPy.
    final_data = xp.where(is_not_surface.data, rc_flat_da.data, 9999.0)
    rc_flat_final = xr.DataArray(final_data, dims=rc_flat_da.dims, coords=rc_flat_da.coords)
    
    rc_flat_final.attrs = {
        'long_name': 'Critical Cut Length (Flat Terrain)',
        'units': 'm'
    }

    # Add the final calculated data back to the main dataset
    new_xarray_ds = ds.data.assign(rc_flat=rc_flat_final)
    
    from .dataset import xsnowDataset
    return xsnowDataset(new_xarray_ds)
