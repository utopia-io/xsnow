"""Runtime contract checks for the curated xsnowDataset type stub."""

from __future__ import annotations

import ast
import inspect
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

import xsnow


STUB_PATH = Path(__file__).parents[2] / "src" / "xsnow" / "core.pyi"


def _class_node(name: str) -> ast.ClassDef:
    tree = ast.parse(STUB_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {STUB_PATH}")


def _is_property(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in node.decorator_list
    )


def _stub_method_nodes() -> dict[str, ast.FunctionDef]:
    methods: dict[str, ast.FunctionDef] = {}
    for node in _class_node("xsnowDataset").body:
        if isinstance(node, ast.FunctionDef) and not _is_property(node):
            methods.setdefault(node.name, node)
    return methods


def _property_names() -> set[str]:
    props: set[str] = set()
    for node in _class_node("xsnowDataset").body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if _is_property(node):
            props.add(node.name)
    return props


def _explicit_stub_parameters(node: ast.FunctionDef) -> list[str]:
    args = node.args
    params = [
        arg.arg
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if arg.arg != "self"
    ]
    return params

@pytest.mark.parametrize("name", sorted(_property_names()))
def test_stubbed_properties_are_available_at_runtime(xsnow_dataset: xsnow.xsnowDataset, name: str) -> None:
    """Properties advertised in the stub must be forwarded by xsnowDataset."""
    assert hasattr(xsnow_dataset, name)


@pytest.mark.parametrize("name", sorted(_stub_method_nodes()))
def test_stubbed_methods_are_available_at_runtime(xsnow_dataset: xsnow.xsnowDataset, name: str) -> None:
    """Every method advertised in core.pyi must be reachable at runtime."""
    target = xsnow.xsnowDataset if name == "__init__" else xsnow_dataset
    assert hasattr(target, name)
    assert callable(getattr(target, name))


@pytest.mark.parametrize("name,node", sorted(_stub_method_nodes().items()))
def test_explicit_stub_parameters_exist_at_runtime(
    xsnow_dataset: xsnow.xsnowDataset,
    name: str,
    node: ast.FunctionDef,
) -> None:
    """
    Any parameter named explicitly in the curated stub must exist at runtime.

    This deliberately does not require exact signature equality: registered
    xsnow methods and forwarded xarray methods are kept coarse in the stub to
    reduce drift. If the stub names a durable leading parameter such as
    ``source``, ``path``, or ``other``, this test verifies that the runtime
    method still accepts it.
    """
    target = xsnow.xsnowDataset if name == "__init__" else getattr(xsnow_dataset, name)
    runtime_params = inspect.signature(target).parameters
    accepts_arbitrary_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in runtime_params.values()
    )
    for param_name in _explicit_stub_parameters(node):
        assert param_name in runtime_params or accepts_arbitrary_kwargs, (
            f"{name} advertises parameter {param_name!r} in core.pyi, "
            "but the runtime signature does not accept it"
        )


def _empty_dir(tmp_path: Path) -> Path:
    path = tmp_path / "empty"
    path.mkdir()
    return path


def _swap_dims(xs: xsnow.xsnowDataset, _tmp_path: Path) -> xsnow.xsnowDataset:
    labels = ["lower", "middle", "upper"]
    return xs.assign_coords(layer_label=("layer", labels)).swap_dims({"layer": "layer_label"})


def _to_zarr(xs: xsnow.xsnowDataset, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["du"], returncode=0, stdout="0K stub_contract.zarr\n")
    with (
        patch.object(xr.Dataset, "to_zarr", return_value=None),
        patch("xsnow.io.zarr.subprocess.run", return_value=completed),
    ):
        return xs.to_zarr(tmp_path / "stub_contract.zarr")


XSNOW_RETURN_CASES: dict[str, tuple[Callable[[xsnow.xsnowDataset, Path], Any], type[Any]]] = {
    "append_latest": (lambda xs, tmp: xs.append_latest(_empty_dir(tmp)), xsnow.xsnowDataset),
    "compute_hs": (lambda xs, _tmp: xs.compute_hs(), xsnow.xsnowDataset),
    "compute_z_depth": (lambda xs, _tmp: xs.compute_z_depth(), xsnow.xsnowDataset),
    "copy": (lambda xs, _tmp: xs.copy(), xsnow.xsnowDataset),
    "fill_gaps_with": (lambda xs, _tmp: xs.fill_gaps_with(xs), xsnow.xsnowDataset),
    "merge_smet": (lambda xs, tmp: xs.merge_smet(_empty_dir(tmp)), xsnow.xsnowDataset),
    "overwrite_with": (lambda xs, _tmp: xs.overwrite_with(xs), xsnow.xsnowDataset),
    "print": (lambda xs, _tmp: xs.print(force_details=False), type(None)),
    "profile_status_summary": (lambda xs, _tmp: xs.profile_status_summary(), type(None)),
    "reorder_dims": (lambda xs, _tmp: xs.reorder_dims(), xsnow.xsnowDataset),
    "to_cpu": (lambda xs, _tmp: xs.to_cpu(), xsnow.xsnowDataset),
    "to_gpu": (lambda xs, _tmp: xs.to_gpu(), xsnow.xsnowDataset),
    "to_netcdf": (lambda xs, tmp: xs.to_netcdf(tmp / "stub_contract.nc"), type(None)),
    "to_pro": (lambda xs, tmp: xs.to_pro(tmp / "stub_contract.pro"), type(None)),
    "to_smet": (lambda xs, tmp: xs.to_smet(tmp / "stub_contract.smet"), type(None)),
    "to_xarray": (lambda xs, _tmp: xs.to_xarray(), xr.Dataset),
    "to_zarr": (_to_zarr, type(None)),
    "trim_padding": (lambda xs, _tmp: xs.trim_padding(), xsnow.xsnowDataset),
    "update_profile_status": (lambda xs, _tmp: xs.update_profile_status(), xsnow.xsnowDataset),
}

XARRAY_RETURN_CASES: dict[str, tuple[Callable[[xsnow.xsnowDataset, Path], Any], type[Any]]] = {
    "all": (lambda xs, _tmp: xs.all(dim="time"), xsnow.xsnowDataset),
    "any": (lambda xs, _tmp: xs.any(dim="time"), xsnow.xsnowDataset),
    "assign": (lambda xs, _tmp: xs.assign(extra_scalar=1), xsnow.xsnowDataset),
    "assign_coords": (lambda xs, _tmp: xs.assign_coords(extra_coord=1), xsnow.xsnowDataset),
    "broadcast_equals": (lambda xs, _tmp: xs.broadcast_equals(xs.to_xarray()), bool),
    "chunk": (lambda xs, _tmp: xs.chunk({"time": 1}), xsnow.xsnowDataset),
    "compute": (lambda xs, _tmp: xs.compute(), xsnow.xsnowDataset),
    "count": (lambda xs, _tmp: xs.count(dim="time"), xsnow.xsnowDataset),
    "drop_dims": (lambda xs, _tmp: xs.drop_dims("slope"), xsnow.xsnowDataset),
    "drop_sel": (lambda xs, _tmp: xs.drop_sel(layer=[2]), xsnow.xsnowDataset),
    "drop_vars": (lambda xs, _tmp: xs.drop_vars("density"), xsnow.xsnowDataset),
    "dropna": (lambda xs, _tmp: xs.dropna(dim="layer", how="all"), xsnow.xsnowDataset),
    "equals": (lambda xs, _tmp: xs.equals(xs.to_xarray()), bool),
    "expand_dims": (lambda xs, _tmp: xs.expand_dims(member=[0]), xsnow.xsnowDataset),
    "fillna": (lambda xs, _tmp: xs.fillna(0), xsnow.xsnowDataset),
    "identical": (lambda xs, _tmp: xs.identical(xs.to_xarray()), bool),
    "isel": (lambda xs, _tmp: xs.isel(layer=0), xsnow.xsnowDataset),
    "load": (lambda xs, _tmp: xs.load(), xsnow.xsnowDataset),
    "max": (lambda xs, _tmp: xs.max(dim="layer"), xsnow.xsnowDataset),
    "mean": (lambda xs, _tmp: xs.mean(dim="layer"), xsnow.xsnowDataset),
    "min": (lambda xs, _tmp: xs.min(dim="layer"), xsnow.xsnowDataset),
    "persist": (lambda xs, _tmp: xs.persist(), xsnow.xsnowDataset),
    "rename": (lambda xs, _tmp: xs.rename({"density": "density_renamed"}), xsnow.xsnowDataset),
    "sel": (lambda xs, _tmp: xs.sel(location="TestStation"), xsnow.xsnowDataset),
    "sortby": (lambda xs, _tmp: xs.sortby("time"), xsnow.xsnowDataset),
    "squeeze": (lambda xs, _tmp: xs.squeeze(), xsnow.xsnowDataset),
    "sum": (lambda xs, _tmp: xs.sum(dim="layer"), xsnow.xsnowDataset),
    "swap_dims": (_swap_dims, xsnow.xsnowDataset),
    "to_array": (lambda xs, _tmp: xs.to_array(), xr.DataArray),
    "to_dataframe": (lambda xs, _tmp: xs.to_dataframe(), pd.DataFrame),
    "transpose": (lambda xs, _tmp: xs.transpose("slope", "location", "time", "layer"), xsnow.xsnowDataset),
    "unify_chunks": (lambda xs, _tmp: xs.unify_chunks(), xsnow.xsnowDataset),
    "where": (lambda xs, _tmp: xs.where(xs["density"] > 0), xsnow.xsnowDataset),
}

RETURN_CASES = XSNOW_RETURN_CASES | XARRAY_RETURN_CASES
RAISE_CASES = {"combine_first", "update"}


def _runtime_public_method_names() -> set[str]:
    names = {
        name
        for name, value in vars(xsnow.xsnowDataset).items()
        if not name.startswith("_") and callable(value)
    }
    names.update(
        name
        for name in dir(xsnow.xsnowDataset)
        if not name.startswith("_") and callable(getattr(xsnow.xsnowDataset, name, None))
    )
    return names


def test_public_stub_methods_have_runtime_return_checks() -> None:
    """Keep this module in sync when adding public methods to core.pyi."""
    public_methods = {name for name in _stub_method_nodes() if not name.startswith("__")}
    missing = public_methods - set(RETURN_CASES) - RAISE_CASES
    assert not missing


def test_runtime_public_methods_missing_from_stub_emit_warning() -> None:
    """Warn when the runtime exposes public xsnowDataset methods absent from core.pyi."""
    stubbed_methods = {name for name in _stub_method_nodes() if not name.startswith("__")}
    runtime_only = sorted(_runtime_public_method_names() - stubbed_methods)
    if runtime_only:
        warnings.warn(
            "Public xsnowDataset runtime methods are not listed in core.pyi: "
            + ", ".join(runtime_only),
            UserWarning,
            stacklevel=2,
        )


@pytest.mark.parametrize("name", sorted(XSNOW_RETURN_CASES))
def test_xsnow_methods_return_advertised_broad_type(
    xsnow_dataset: xsnow.xsnowDataset,
    tmp_path: Path,
    name: str,
) -> None:
    """Representative xsnow-native calls should match the broad stub return type."""
    call, expected_type = XSNOW_RETURN_CASES[name]
    result = call(xsnow_dataset, tmp_path)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize("name", sorted(XARRAY_RETURN_CASES))
def test_xarray_forwarded_methods_return_advertised_broad_type(
    xsnow_dataset: xsnow.xsnowDataset,
    tmp_path: Path,
    name: str,
) -> None:
    """Representative forwarded xarray calls should match the broad stub return type."""
    call, expected_type = XARRAY_RETURN_CASES[name]
    result = call(xsnow_dataset, tmp_path)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize("name", sorted(RAISE_CASES))
def test_stubbed_blocked_xarray_methods_raise(
    xsnow_dataset: xsnow.xsnowDataset,
    name: str,
) -> None:
    """Methods typed as NoReturn are deliberate runtime blockers."""
    with pytest.raises(NotImplementedError):
        getattr(xsnow_dataset, name)()


def test_dunder_stubbed_methods_return_advertised_broad_type(xsnow_dataset: xsnow.xsnowDataset) -> None:
    """Basic dunder behavior in the stub should match the runtime wrapper."""
    assert "density" in xsnow_dataset
    assert isinstance(dir(xsnow_dataset), list)
    assert isinstance(xsnow_dataset["density"], xr.DataArray)
    assert isinstance(xsnow_dataset[["density"]], xr.Dataset)
    assert isinstance(repr(xsnow_dataset), str)

    xsnow_dataset["new_var"] = xr.DataArray([1, 2, 3], dims=["layer"])
    assert "new_var" in xsnow_dataset
