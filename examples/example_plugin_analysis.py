# example_plugin_analysis.py
"""
This file defines one or more custom analysis functions and registers them
as plugins for the xsnow package.
"""
import xsnow
import xarray as xr
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports at runtime, while allowing
# type checkers to understand the xsnowDataset type.
if TYPE_CHECKING:
    from xsnow.dataset import xsnowDataset

@xsnow.register_xsnowDataset_method
def calculate_top_slab_density(ds: 'xsnowDataset', depth_cm: float = 20.0) -> xr.DataArray:
    """
    Calculates the mean density of the top slab of the snowpack.

    This function identifies all snow layers within a specified depth from the
    snow surface and computes their mean density.

    Args:
        ds (xsnowDataset): The dataset to analyze. The `self` instance is
            passed automatically by the plugin system.
        depth_cm (float): The depth of the top slab in centimeters.
            Defaults to 20.0.

    Returns:
        xr.DataArray: A DataArray containing the mean density of the top slab
        for each profile (each location and time). The result is indexed by
        the original dimensions (e.g., location, time, slope).
    """
    if ds.data is None:
        raise ValueError("Cannot perform calculation on an empty dataset.")

    # Convert slab depth from cm to meters to match the 'height' variable units
    depth_m = depth_cm / 100.0

    # 1. Find the total snow height (HS) for each profile
    total_height = ds.data['height'].max(dim='layer', skipna=True)

    # 2. Determine the height threshold for the top slab
    # This is the height below which we are no longer in the top slab.
    slab_bottom_height = total_height - depth_m

    # 3. Create a boolean mask to identify layers within the top slab.
    # A layer is included if its height is above the calculated bottom threshold.
    top_slab_mask = ds.data['height'] >= slab_bottom_height

    # 4. Apply the mask to the density data.
    # This operation will keep the density values for layers within the top
    # slab and replace all other values with NaN.
    top_slab_densities = ds.data['density'].where(top_slab_mask)

    # 5. Calculate the mean of the remaining density values along the 'layer' dimension.
    # The skipna=True argument is crucial, as it tells xarray to ignore the
    # NaN values when computing the mean.
    mean_density = top_slab_densities.mean(dim='layer', skipna=True)

    # Add descriptive attributes to the resulting DataArray for clarity
    mean_density.attrs = {
        'long_name': f'Mean density of the top {depth_cm} cm slab',
        'units': 'kg m-3',
        'analysis_function': 'calculate_top_slab_density'
    }

    return mean_density
