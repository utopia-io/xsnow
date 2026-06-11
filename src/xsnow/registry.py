# registry.py
# -*- coding: utf-8 -*-
"""
================================================================================
Snowpack Variable Registry
================================================================================

This file defines a comprehensive registry for snowpack variables,
mapping common aliases to standardized names and their corresponding
SI units and SNOWPACK parameter codes.
"""

registry = {
    "height": {
        "param_code": "0501",
        "aliases": ["height", "height [> 0: top, < 0: bottom of elem.]"],
        "standard_name": "height_of_snow_element",
        "si_units": "m"
    },
    "element_ID": {
        "param_code": "0504",
        "aliases": ["element id", "id"],
        "standard_name": "snow_element_identifier",
        "si_units": "1"
    },
    "density": {
        "param_code": "0502",
        "aliases": ["density", "rho", "rho_snow", "element density"],
        "standard_name": "mass_density_of_snow",
        "si_units": "kg m-3"
    },
    "temperature": {
        "param_code": "0503",
        # SNOWPACK .pro element temperature is in °C and xsnow does not convert
        # it, so the unit must read "degC" (was incorrectly "K").
        "aliases": ["temperature", "temp", "element temperature"],
        "standard_name": "temperature_of_snow",
        "si_units": "degC"
    },
    "lwc": {
        "param_code": "0506",
        "aliases": ["liquid water content", "lwc", "liquid water content by volume"],
        "standard_name": "liquid_water_content_of_snow",
        "si_units": "%"
    },
    "dendricity": {
        "param_code": "0508",
        "aliases": ["dendricity"],
        "standard_name": "dendricity_of_snow_grains",
        "si_units": "1"
    },
    "sphericity": {
        "param_code": "0509",
        "aliases": ["sphericity"],
        "standard_name": "sphericity_of_snow_grains",
        "si_units": "1"
    },
    "coord_number": {
        "param_code": "0510",
        "aliases": ["coordination number"],
        "standard_name": "coordination_number_of_snow_grains",
        "si_units": "1"
    },
    "bond_size": {
        "param_code": "0511",
        "aliases": ["bond size"],
        "standard_name": "bond_size_of_snow_grains",
        "si_units": "mm"
    },
    "grain_size": {
        "param_code": "0512",
        "aliases": ["grain size"],
        "standard_name": "grain_size_of_snow",
        "si_units": "mm"
    },
    "grain_type": {
        "param_code": "0513",
        "aliases": ["grain type"],
        "standard_name": "snow_grain_type",
        "si_units": "Swiss Code F1F2F3"
    },
    "SH_at_surface": {
        "param_code": "0514",
        "aliases": ["Surface Hoar layer is present at the surface", "SH at surface", "SH_surface"],
        "standard_name": "SH at surface",
        "si_units": ["mm", "m2 kg-1"] 
    },
    "ice_content": {
        "param_code": "0515",
        "aliases": ["ice volume fraction"],
        "standard_name": "ice_volume_fraction_of_snow",
        "si_units": "%"
    },
    "air_content": {
        "param_code": "0516",
        "aliases": ["air volume fraction"],
        "standard_name": "air_volume_fraction_of_snow",
        "si_units": "%"
    },
    "stress": {
        "param_code": "0517",
        "aliases": ["stress", "stress in"],
        "standard_name": "stress_in_snow",
        "si_units": "kPa"
    },
    "viscosity": {
        "param_code": "0518",
        "aliases": ["viscosity"],
        "standard_name": "viscosity_of_snow",
        "si_units": "GPa s"
    },
    "temperature_gradient": {
        "param_code": "0520",
        "aliases": ["temperature gradient"],
        "standard_name": "temperature_gradient_in_snow",
        "si_units": "K m-1"
    },
    "viscous_deformation_rate": {
        "param_code": "0523",
        "aliases": ["viscous deformation rate"],
        "standard_name": "viscous_deformation_rate_of_snow",
        "si_units": "1.e-6 s-1"
    },
    "stab_deformation_rate": {
        "param_code": "0531",
        "aliases": ["deformation rate stability index sdef", "sdef"],
        "standard_name": "deformation_rate_stability_index_sdef",
        "si_units": "1"
    },
    "sn38": {
        "param_code": "0532",
        "aliases": ["natural stability index sn38", "sn38"],
        "standard_name": "natural_stability_index_sn38",
        "si_units": "1"
    },
    "sk38": {
        "param_code": "0533",
        "aliases": ["stability index sk38", "sk38"],
        "standard_name": "stability_index_sk38",
        "si_units": "1"
    },
    "hand_hardness": {
        "param_code": "0534",
        "aliases": ["hand hardness", "hand hardness in index steps"],
        "standard_name": "hand_hardness_of_snow",
        "si_units": "1"
    },
    "opt_equ_grain_size": {
        "param_code": "0535",
        "aliases": ["optical equivalent grain size"],
        "standard_name": "optical_equivalent_grain_size_of_snow",
        "si_units": "mm"
    },
    "shear_strength": {
        "param_code": "0601",
        "aliases": ["snow shear strength"],
        "standard_name": "snow_shear_strength",
        "si_units": "kPa"
    },
    "gs_difference": {
        "param_code": "0602",
        "aliases": ["grain size difference"],
        "standard_name": "grain_size_difference_in_snow",
        "si_units": "mm"
    },
    "hardness_difference": {
        "param_code": "0603",
        "aliases": ["hardness difference"],
        "standard_name": "hardness_difference_in_snow",
        "si_units": "1"
    },
    "ssi": {
        "param_code": "0604",
        "aliases": ["ssi"],
        "standard_name": "stability_index_ssi",
        "si_units": "1"
    },
    "height_nodes": {
        "param_code": "1501",
        "aliases": [],
        "standard_name": "height_of_snow_nodes",
        "si_units": "m"
    },
    "sn38_nodes": {
        "param_code": "1532",
        "aliases": [],
        "standard_name": "natural_stability_index_sn38_at_nodes",
        "si_units": "1"
    },
    "sk38_nodes": {
        "param_code": "1533",
        "aliases": [],
        "standard_name": "stability_index_sk38_at_nodes",
        "si_units": "1"
    },
    "date_of_birth": {
        "param_code": "0540",
        "aliases": ["element deposition date", "deposition date"],
        "standard_name": "deposition_date_of_snow_element",
        "si_units": "ISO"
    },
    "accumulated_temperature_gradient": {
        "param_code": "0607",
        "aliases": [],
        "standard_name": "accumulated_temperature_gradient",
        "si_units": "K day m-1"
    },
    'z': {
        "param_code": None,
        "aliases": ['Depth Below Snow Surface'],
        'standard_name': 'depth',
        "long_name": "Depth Below Snow Surface",
        'si_units': 'm',
    },
    'rc_flat': {
        "param_code": "0606",
        "aliases": ['critical_cut_length', 'critical_crack_length', 'rc_flat', 'rc'],
        'standard_name': 'critical_cut_length_for_flat_terrain',
        'long_name': 'Critical Cut Length (Flat Terrain)',
        'si_units': 'm',
    },
    'HS': {
        "param_code": None,
        "aliases": ['HS', 'snow_height', 'snow_depth'],
        'standard_name': 'snow_height',
        'long_name': 'Total snow height',
        'si_units': 'm',
    },

}