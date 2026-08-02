# tests/unit/io/test_memory.py
"""
Tests for the pre-flight memory guard (xsnow.io.memory).

The guard exists because a kernel OOM kill is uncatchable, so the size check
must happen before the eager load is issued. These tests cover budget parsing,
the size estimate, and the guard's behaviour on both archive read paths.
"""
import numpy as np
import pytest
import xarray as xr

import xsnow
from xsnow import read
from xsnow.io.memory import (
    DEFAULT_MEMORY_FRACTION,
    MemoryLimitExceeded,
    MemorySizeUnknown,
    available_memory_bytes,
    check_memory_budget,
    estimate_decoded_nbytes,
    resolve_memory_limit,
)


@pytest.fixture
def archive_paths(tmp_path):
    """A small dataset written out as both NetCDF and Zarr."""
    ds = xsnow.single_profile_timeseries()
    nc_path = tmp_path / "archive.nc"
    zarr_path = tmp_path / "archive.zarr"
    ds.to_netcdf(nc_path)
    ds.to_zarr(zarr_path)
    return {"nc": nc_path, "zarr": zarr_path, "nbytes": int(ds.data.nbytes)}


class TestResolveMemoryLimit:
    """Budget parsing and the three-state on/explicit/off contract."""

    def test_none_and_false_disable_the_guard(self):
        assert resolve_memory_limit(None) is None
        assert resolve_memory_limit(False) is None

    def test_auto_is_a_fraction_of_available_memory(self):
        available = available_memory_bytes()
        if available is None:
            pytest.skip("available memory could not be detected on this platform")
        budget = resolve_memory_limit("auto")
        assert budget == int(available * DEFAULT_MEMORY_FRACTION)

    def test_auto_is_the_default(self):
        assert resolve_memory_limit() == resolve_memory_limit("auto")

    def test_true_means_auto(self):
        assert resolve_memory_limit(True) == resolve_memory_limit("auto")

    def test_auto_disables_guard_when_memory_undetectable(self, monkeypatch):
        monkeypatch.setattr(
            "xsnow.io.memory.available_memory_bytes", lambda: None
        )
        assert resolve_memory_limit("auto") is None

    @pytest.mark.parametrize("value,expected", [
        (2048, 2048),
        (1024.0, 1024),
        ("1B", 1),
        ("1kb", 1000),
        ("8GB", 8 * 1000**3),
        ("512MiB", 512 * 1024**2),
        ("1GiB", 1024**3),
        ("1g", 1024**3),
        ("  4 GB  ", 4 * 1000**3),
        ("1.5GiB", int(1.5 * 1024**3)),
    ])
    def test_explicit_sizes(self, value, expected):
        assert resolve_memory_limit(value) == expected

    def test_decimal_and_binary_suffixes_differ(self):
        assert resolve_memory_limit("1GB") != resolve_memory_limit("1GiB")

    @pytest.mark.parametrize("bad", ["banana", "8ZB", "GB", ""])
    def test_unparseable_sizes_raise(self, bad):
        with pytest.raises(ValueError):
            resolve_memory_limit(bad)

    @pytest.mark.parametrize("bad", [0, -1, "0GB"])
    def test_non_positive_budgets_raise(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_memory_limit(bad)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported memory_limit type"):
            resolve_memory_limit([])


class TestEstimateDecodedNbytes:
    """The size estimate must be trustworthy or absent, never a wrong number."""

    def test_matches_nbytes_for_numeric_data(self):
        ds = xr.Dataset({"x": ("time", np.zeros(100, dtype="float64"))})
        assert estimate_decoded_nbytes(ds) == int(ds.nbytes)

    def test_object_dtype_returns_none(self):
        ds = xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        assert estimate_decoded_nbytes(ds) is None

    def test_does_not_materialize_lazy_data(self, archive_paths):
        opened = xr.open_dataset(archive_paths["nc"])
        try:
            estimate_decoded_nbytes(opened)
            assert not any(
                var._in_memory for var in opened.data_vars.values()
            ), "estimating the size must not pull data into memory"
        finally:
            opened.close()

    def test_returns_none_for_non_dataset(self):
        assert estimate_decoded_nbytes(object()) is None


class TestCheckMemoryBudget:
    """The guard itself, independent of any read path."""

    def test_none_budget_is_a_no_op(self):
        ds = xr.Dataset({"x": ("time", np.zeros(1000))})
        assert check_memory_budget(ds, None) is True

    def test_returns_true_when_within_budget(self):
        ds = xr.Dataset({"x": ("time", np.zeros(10))})
        assert check_memory_budget(ds, 10 * 1024**2) is True

    def test_returns_false_when_over_budget(self):
        ds = xr.Dataset({"x": ("time", np.zeros(1000, dtype="float64"))})
        assert check_memory_budget(ds, 16) is False

    def test_does_not_raise_by_default(self):
        """The bool is the default contract; raising is opt-in."""
        ds = xr.Dataset({"x": ("time", np.zeros(1000, dtype="float64"))})
        check_memory_budget(ds, 16)  # must not raise

    def test_raises_when_asked(self):
        ds = xr.Dataset({"x": ("time", np.zeros(1000, dtype="float64"))})
        with pytest.raises(MemoryLimitExceeded):
            check_memory_budget(ds, 16, raise_on_exceed=True)

    def test_unknown_size_returns_none(self):
        """
        An unmeasurable size is distinct from fitting or exceeding the budget.

        Object arrays report only xarray's pointer-based nbytes, so there is
        no trustworthy number to compare with the budget.
        """
        ds = xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        assert check_memory_budget(ds, 1) is None
        assert check_memory_budget(ds, 1, raise_on_exceed=True) is None

    def test_unknown_size_raises_when_asked(self):
        ds = xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        with pytest.raises(MemorySizeUnknown):
            check_memory_budget(ds, 1, raise_on_unknown=True)

    def test_unknown_size_is_catchable_as_memory_error(self):
        ds = xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        with pytest.raises(MemoryError):
            check_memory_budget(ds, 1, raise_on_unknown=True)

    def test_disabled_guard_ignores_raise_on_unknown(self):
        ds = xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        assert check_memory_budget(ds, None, raise_on_unknown=True) is True

    def test_accepts_size_strings_directly(self):
        """It resolves memory_limit itself rather than taking raw bytes."""
        ds = xr.Dataset({"x": ("time", np.zeros(10, dtype="float64"))})
        assert check_memory_budget(ds, "8GB") is True
        assert check_memory_budget(ds, "1B") is False

    def test_defaults_to_auto(self):
        ds = xr.Dataset({"x": ("time", np.zeros(10, dtype="float64"))})
        assert check_memory_budget(ds) is True

    def test_is_catchable_as_memory_error(self):
        ds = xr.Dataset({"x": ("time", np.zeros(1000, dtype="float64"))})
        with pytest.raises(MemoryError):
            check_memory_budget(ds, 16, raise_on_exceed=True)

    def test_message_is_actionable(self):
        ds = xr.Dataset({"x": ("time", np.zeros(1000, dtype="float64"))})
        with pytest.raises(MemoryLimitExceeded) as excinfo:
            check_memory_budget(ds, 16, source="test archive", raise_on_exceed=True)
        message = str(excinfo.value)
        assert "test archive" in message
        assert "memory_limit" in message

    def test_boundary_equal_to_budget_passes(self):
        ds = xr.Dataset({"x": ("time", np.zeros(10, dtype="float64"))})
        assert check_memory_budget(ds, int(ds.nbytes)) is True


class TestCheckMemoryBudgetAsMethod:
    """Registered on xsnowDataset and exported at package level."""

    def test_exported_at_package_level(self):
        assert xsnow.check_memory_budget is check_memory_budget
        assert xsnow.MemorySizeUnknown is MemorySizeUnknown

    def test_available_as_a_dataset_method(self, archive_paths):
        ds = read(archive_paths["nc"], lazy=True)
        assert ds.check_memory_budget("8GB") is True

    def test_method_returns_false_on_tight_budget(self, archive_paths):
        ds = read(archive_paths["nc"], lazy=True)
        assert ds.check_memory_budget(16) is False

    def test_method_can_raise(self, archive_paths):
        ds = read(archive_paths["nc"], lazy=True)
        with pytest.raises(MemoryLimitExceeded):
            ds.check_memory_budget(16, raise_on_exceed=True)

    def test_method_returns_none_for_unknown_size(self):
        ds = xsnow.xsnowDataset(
            xr.Dataset({"x": ("time", np.array(["a", "bb"], dtype=object))})
        )
        assert ds.check_memory_budget(1) is None

    def test_method_checks_without_materializing(self, archive_paths):
        """The whole point: ask before computing, not after."""
        opened = xr.open_dataset(archive_paths["nc"])
        try:
            check_memory_budget(opened, "8GB")
            assert not any(var._in_memory for var in opened.data_vars.values())
        finally:
            opened.close()


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
class TestReadIntegration:
    """The guard as wired into read() for both archive formats."""

    def test_generous_budget_loads_normally(self, archive_paths, fmt):
        ds = read(archive_paths[fmt], lazy=False, memory_limit="8GB")
        assert ds is not None
        assert ds.data is not None

    def test_tight_budget_raises(self, archive_paths, fmt):
        with pytest.raises(MemoryLimitExceeded):
            read(archive_paths[fmt], lazy=False, memory_limit=16)

    def test_guard_can_be_disabled(self, archive_paths, fmt):
        ds = read(archive_paths[fmt], lazy=False, memory_limit=None)
        assert ds is not None

    def test_lazy_reads_are_not_blocked(self, archive_paths, fmt):
        """The guard protects the eager allocation, not lazy access."""
        ds = read(archive_paths[fmt], lazy=True, memory_limit=16)
        assert ds is not None

    def test_default_read_is_unaffected(self, archive_paths, fmt):
        """A small archive under the default budget reads as it always did."""
        ds = read(archive_paths[fmt])
        assert ds is not None

    def test_refusal_is_not_swallowed_as_none(self, archive_paths, fmt):
        """read() returns None for unreadable sources; a refusal must raise."""
        with pytest.raises(MemoryLimitExceeded):
            read(archive_paths[fmt], lazy=False, memory_limit=1)

    def test_unknown_size_refusal_is_not_swallowed(
        self, archive_paths, fmt, monkeypatch
    ):
        """read() refuses an eager load when its size cannot be verified."""
        monkeypatch.setattr(
            "xsnow.io.memory.estimate_decoded_nbytes",
            lambda ds: None,
        )
        with pytest.raises(MemorySizeUnknown):
            read(archive_paths[fmt], lazy=False, memory_limit="8GB")

    def test_malformed_budget_raises_valueerror(self, archive_paths, fmt):
        with pytest.raises(ValueError):
            read(archive_paths[fmt], memory_limit="banana")

    def test_selectors_narrow_the_checked_size(self, archive_paths, fmt):
        """
        The check runs after selectors, so a budget too small for the whole
        archive can still admit a sufficiently narrow selection.
        """
        full = archive_paths["nbytes"]
        with pytest.raises(MemoryLimitExceeded):
            read(archive_paths[fmt], lazy=False, memory_limit=full // 2)

        narrowed = read(
            archive_paths[fmt],
            lazy=False,
            memory_limit=full // 2,
            time=slice(None, None, 4),
        )
        assert narrowed is not None
