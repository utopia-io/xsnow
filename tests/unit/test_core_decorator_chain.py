# tests/core/test_decorator_chain.py
"""
Unit tests for the DatasetDecorator chain behavior.

These tests focus on extension-agnostic invariants:
  - Rewrapping is non-destructive (no mutation of existing instances).
  - The decorator chain is rebuilt around the new xr.Dataset.
  - Wrapper synchronization: base.data is the same object as outer.data after rewrap.

We intentionally use trivial in-test decorators rather than real extensions
to keep these tests stable while extensions evolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow import DatasetDecorator, xsnowDataset


@pytest.fixture
def sample_xr_dataset() -> xr.Dataset:
    layer_coords = [0, 1, 2]
    data = {
        "height": (("location", "time", "layer"), np.array([[[0.1, 0.2, 0.3]]])),
        "density": (("location", "time", "layer"), np.array([[[100, 200, 300]]])),
        "profile_status": (("location", "time"), np.array([[1]], dtype="i1")),
        "HS": (("location", "time"), np.array([[0.3]])),
    }
    coords = {
        "location": ["STATION_A"],
        "time": [pd.to_datetime("2025-01-15")],
        "layer": layer_coords,
    }
    return xr.Dataset(data, coords)


class AddADim(DatasetDecorator):
    """Trivial decorator that adds an `a` dimension when requested."""

    def add_a_dim(self) -> "AddADim":
        out = self.data.expand_dims(a=[0])
        return self._rewrap(out)

    def subset_raw_xr(self) -> xr.Dataset:
        """Return a raw xarray.Dataset on purpose (should not be auto-rewrapped)."""
        return self.data.isel(layer=0)


class AddBDim(DatasetDecorator):
    """Trivial decorator that adds a `b` dimension when requested."""

    def add_b_dim(self) -> "AddBDim":
        out = self.data.expand_dims(b=[0, 1])
        return self._rewrap(out)


class AddCDim(DatasetDecorator):
    """Trivial decorator that adds a `c` dimension when requested."""

    def add_c_dim(self) -> "AddCDim":
        out = self.data.expand_dims(c=[42])
        return self._rewrap(out)


def test_rewrap_is_non_destructive_and_rebuilds_base(sample_xr_dataset: xr.Dataset) -> None:
    xs = xsnowDataset(sample_xr_dataset)
    decorated = AddADim(xs)

    modified_xr = decorated.data.assign_coords(extra=1)
    xs_decorated = decorated._rewrap(modified_xr)

    assert xs_decorated.data is not xs.data  # non-destructive: new dataset object
    assert xs_decorated.base is not xs  # new base wrapper rebuilt
    assert xs_decorated.base.data is xs_decorated.data  # synchronized

    # Original objects remain unchanged.
    assert xs.data is sample_xr_dataset
    assert "extra" not in xs.data.coords


def test_rewrap_accepts_xsnowdataset_input(sample_xr_dataset: xr.Dataset) -> None:
    xs = xsnowDataset(sample_xr_dataset)
    decorated = AddADim(xs)

    modified_xr = sample_xr_dataset.assign_coords(extra=123)
    modified_xs = xsnowDataset(modified_xr)
    out = decorated._rewrap(modified_xs)

    assert isinstance(out, AddADim)
    assert out.data is modified_xr
    assert out.base.data is out.data


def test_xarray_ops_preserve_outermost_decorator_type(sample_xr_dataset: xr.Dataset) -> None:
    xs = xsnowDataset(sample_xr_dataset)
    outer = AddCDim(AddBDim(AddADim(xs)))

    out = outer.isel(layer=0)

    assert isinstance(out, AddCDim)
    assert "layer" not in out.dims
    assert isinstance(out.base, AddBDim)
    assert isinstance(out.base.base, AddADim)
    assert out.base.base.base.data is out.data


def test_calling_inner_decorator_method_keeps_outermost(sample_xr_dataset: xr.Dataset) -> None:
    xs = xsnowDataset(sample_xr_dataset)
    outer = AddCDim(AddBDim(AddADim(xs)))

    # `add_a_dim` is defined only on AddADim, but should be callable via the chain
    # while still returning the outermost decorator type.
    out = outer.add_a_dim()

    assert isinstance(out, AddCDim)
    assert "a" in out.dims
    assert isinstance(out.base, AddBDim)
    assert isinstance(out.base.base, AddADim)
    assert out.base.data is out.data


def test_inner_method_can_return_raw_xarray_without_rewrap(sample_xr_dataset: xr.Dataset) -> None:
    xs = xsnowDataset(sample_xr_dataset)
    outer = AddCDim(AddBDim(AddADim(xs)))

    out = outer.subset_raw_xr()

    assert isinstance(out, xr.Dataset)
    assert "layer" not in out.dims
