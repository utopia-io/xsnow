import xarray as xr
from xsnow import registry

PRO_DATACODE_MAP = {
    "p_unstable": "1130",
    "p_unstable_proportion": "1131", # proportion of layers with p_unstable > 0.77
    "rta":"0606",
}

def assign_to_dataset(ds,dataArray, name, longName, units, data_code=None):
    base_ds = ds.data if hasattr(ds, "data") else ds
    if not isinstance(base_ds, xr.Dataset):
        raise TypeError(f"Unsupported dataset type for assignment: {type(ds)!r}")

    if not isinstance(dataArray, xr.DataArray):
        dataArray = xr.DataArray(dataArray, name=name)

    # Attach registry metadata if available.
    reg_entry = dict(registry.get(name, {}))
    if data_code is None:
        data_code = PRO_DATACODE_MAP.get(name)

    if data_code is not None:
        reg_entry.setdefault("data_code", data_code)
        reg_entry.setdefault("pro_code", data_code)
        reg_entry.setdefault("code", data_code)
        reg_entry.setdefault("param_code", data_code)
        registry[name] = reg_entry

    dataArray.attrs = {
        'long_name': reg_entry.get('long_name', longName),
        'units': reg_entry.get('si_units', units),
        'standard_name': reg_entry.get('standard_name', name),
    }

    new_xr = base_ds.assign(**{name: dataArray})
    return new_xr