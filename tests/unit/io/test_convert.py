# tests/unit/io/test_convert.py
"""
Tests for the batch-append archive converter (xsnow.io.convert).

The converter builds a Zarr season archive from a PRO/SMET collection in
eager batches, appending along ``location``. The behaviour that matters is
that a batched build produces the same archive as a single-shot read, and
that the pre-scan makes the layer-dimension mismatch impossible rather than
latent.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xsnow
from xsnow import build_archive
from xsnow.io.convert import (
    ArchiveAppendError,
    _ArchiveSchema,
    _FileRecord,
    _as_timestamp,
    _group_into_batches,
    _prescan,
    _validate_appendable,
)


class TestTimestampParsing:
    """
    PRO timestamps are day-first.

    A month-first reading is silently plausible for the first twelve days of
    any month, and inflates the archive's time axis rather than erroring.
    """

    def test_day_first_is_not_read_as_month_first(self):
        assert _as_timestamp("05.01.2025 00:00:00") == pd.Timestamp("2025-01-05")

    def test_unambiguous_day_still_parses(self):
        assert _as_timestamp("31.12.2024 06:00:00") == pd.Timestamp(
            "2024-12-31 06:00:00"
        )

    def test_none_and_garbage_are_tolerated(self):
        assert _as_timestamp(None) is None
        assert _as_timestamp("not a date") is None

    def test_existing_timestamp_passes_through(self):
        stamp = pd.Timestamp("2025-02-03")
        assert _as_timestamp(stamp) == stamp


def _clone_stations(source_dir, dest_dir, indices):
    """
    Copy one station's file under new names.

    Every clone keeps the source's slope metadata, so the resulting batch is
    one where all stations share a slope -- the case that makes read()'s
    norm_slopes='auto' normalize the slope coordinate.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    template = sorted(source_dir.glob("*.pro"))[0]
    content = template.read_text()
    for i in indices:
        name = f"STATION_{i:03d}"
        (dest_dir / f"{name}.pro").write_text(
            content.replace(template.stem, name)
        )
    return dest_dir


@pytest.fixture
def station_dir(synthetic_pro, tmp_path):
    """A directory of per-station .pro files spanning the same time range."""
    data_dir = tmp_path / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    return synthetic_pro.multi_station(data_dir, n_stations=6, n_profiles_each=5)


class TestPrescan:
    """The pass that fixes the archive's canonical grid before any write."""

    def test_finds_stations_and_layers(self, station_dir):
        pro_files = sorted(station_dir.glob("*.pro"))
        result = _prescan(pro_files, [])

        assert len(result.records) == len(pro_files)
        assert len(result.stations) == len(pro_files)
        assert result.max_layers > 0

    def test_records_the_global_time_span(self, station_dir):
        pro_files = sorted(station_dir.glob("*.pro"))
        result = _prescan(pro_files, [])

        assert result.time_start is not None
        assert result.time_end is not None
        assert result.time_start <= result.time_end

    def test_max_layers_is_the_deepest_across_all_files(self, station_dir):
        pro_files = sorted(station_dir.glob("*.pro"))
        result = _prescan(pro_files, [])
        per_file = [r.max_layers for r in result.records if r.kind == "pro"]

        assert result.max_layers == max(per_file)

    def test_groups_records_by_station(self, station_dir):
        pro_files = sorted(station_dir.glob("*.pro"))
        grouped = _prescan(pro_files, []).by_station()

        assert all(len(records) >= 1 for records in grouped.values())


class TestBatchGrouping:
    """Batches are packed by estimated bytes, never splitting a station."""

    @staticmethod
    def _station(name, cells):
        record = _FileRecord(
            path=f"{name}.pro", station=name, kind="pro",
            n_timesteps=cells, max_layers=1,
        )
        return (name, [record])

    def test_packs_multiple_stations_into_one_batch(self):
        stations = [self._station(f"S{i}", 10) for i in range(5)]
        batches = _group_into_batches(stations, target_bytes=10_000, bytes_per_cell=1)

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_splits_when_target_is_exceeded(self):
        stations = [self._station(f"S{i}", 100) for i in range(4)]
        batches = _group_into_batches(stations, target_bytes=250, bytes_per_cell=1)

        assert len(batches) > 1
        assert sum(len(b) for b in batches) == 4

    def test_oversized_station_gets_its_own_batch(self):
        stations = [self._station("BIG", 1000), self._station("SMALL", 1)]
        batches = _group_into_batches(stations, target_bytes=10, bytes_per_cell=1)

        assert len(batches) == 2
        assert [s for s, _ in batches[0]] == ["BIG"]

    def test_every_station_appears_exactly_once(self):
        stations = [self._station(f"S{i}", 7) for i in range(11)]
        batches = _group_into_batches(stations, target_bytes=20, bytes_per_cell=1)
        seen = [name for batch in batches for name, _ in batch]

        assert sorted(seen) == sorted(name for name, _ in stations)


class TestValidateAppendable:
    """Shape mismatches are caught before the Zarr writer sees them."""

    @staticmethod
    def _schema(location=9, time=5, x_dims=("location", "time")):
        return _ArchiveSchema(
            sizes={"location": location, "time": time},
            var_dims={"x": x_dims},
        )

    def test_matching_shape_passes(self):
        ds = xr.Dataset({"x": (("location", "time"), np.zeros((2, 5)))})
        _validate_appendable(self._schema(), ds, "location", "batch 1")

    def test_mismatched_non_append_dim_raises(self):
        ds = xr.Dataset({"x": (("location", "time"), np.zeros((2, 4)))})
        with pytest.raises(ArchiveAppendError, match="time"):
            _validate_appendable(self._schema(), ds, "location", "batch 1")

    def test_missing_dim_raises(self):
        ds = xr.Dataset({"y": ("location", np.zeros(2))})
        with pytest.raises(ArchiveAppendError, match="missing dimension"):
            _validate_appendable(self._schema(), ds, "location", "batch 1")

    def test_append_dim_size_is_allowed_to_differ(self):
        ds = xr.Dataset({"x": (("location", "time"), np.zeros((3, 5)))})
        _validate_appendable(self._schema(), ds, "location", "batch 1")

    def test_differing_variable_dims_raise(self):
        """The slope-coordinate case, caught by name rather than by size."""
        schema = _ArchiveSchema(
            sizes={"location": 9, "slope": 2},
            var_dims={"azimuth": ("location", "slope")},
        )
        ds = xr.Dataset(coords={"azimuth": ("slope", np.zeros(2))})
        with pytest.raises(ArchiveAppendError, match="azimuth"):
            _validate_appendable(schema, ds, "location", "batch 1")

    def test_mismatch_message_names_norm_slopes(self):
        ds = xr.Dataset({"x": (("location", "time"), np.zeros((2, 4)))})
        with pytest.raises(ArchiveAppendError, match="norm_slopes"):
            _validate_appendable(self._schema(), ds, "location", "batch 1")


class TestBuildArchive:
    """End-to-end: a batched build must equal a single-shot read."""

    def test_single_batch_build(self, station_dir, tmp_path):
        target = build_archive(station_dir, tmp_path / "season.zarr")

        assert target.exists()
        with xr.open_zarr(target) as ds:
            assert ds.sizes["location"] == len(sorted(station_dir.glob("*.pro")))

    def test_batched_build_matches_single_shot_read(self, station_dir, tmp_path):
        """The whole point: batching must not change the result."""
        target = build_archive(
            station_dir, tmp_path / "batched.zarr", batch_bytes=1
        )
        reference = xsnow.read(station_dir, lazy=False)

        with xr.open_zarr(target) as built:
            assert sorted(map(str, built.location.values)) == sorted(
                map(str, reference.data.location.values)
            )
            assert built.sizes["time"] == reference.data.sizes["time"]
            assert built.sizes["layer"] == reference.data.sizes["layer"]
            assert set(built.data_vars) == set(reference.data.data_vars)

    def test_forced_multi_batch_writes_every_station(self, station_dir, tmp_path):
        n_stations = len(sorted(station_dir.glob("*.pro")))
        target = build_archive(
            station_dir, tmp_path / "multi.zarr", batch_bytes=1
        )

        with xr.open_zarr(target) as ds:
            assert ds.sizes["location"] == n_stations
            assert len(set(map(str, ds.location.values))) == n_stations

    def test_layer_dim_is_the_prescanned_max(self, station_dir, tmp_path):
        pro_files = sorted(station_dir.glob("*.pro"))
        expected = _prescan(pro_files, []).max_layers
        target = build_archive(station_dir, tmp_path / "layers.zarr", batch_bytes=1)

        with xr.open_zarr(target) as ds:
            assert ds.sizes["layer"] == expected

    def test_explicit_max_layers_below_prescan_raises(self, station_dir, tmp_path):
        """The mid-winter shape-mismatch, surfaced up front instead of latent."""
        with pytest.raises(ArchiveAppendError, match="smaller than the deepest"):
            build_archive(station_dir, tmp_path / "x.zarr", max_layers=1)

    def test_explicit_max_layers_above_prescan_is_honoured(self, station_dir, tmp_path):
        pro_files = sorted(station_dir.glob("*.pro"))
        headroom = _prescan(pro_files, []).max_layers + 10
        target = build_archive(
            station_dir, tmp_path / "headroom.zarr", max_layers=headroom
        )

        with xr.open_zarr(target) as ds:
            assert ds.sizes["layer"] == headroom

    def test_chunks_are_applied_to_the_store(self, station_dir, tmp_path):
        target = build_archive(
            station_dir,
            tmp_path / "chunked.zarr",
            batch_bytes=1,
            chunks={"location": 2, "time": -1, "layer": -1},
        )

        with xr.open_zarr(target) as ds:
            var = next(iter(ds.data_vars.values()))
            assert var.encoding.get("chunks") is not None

    def test_zarr_extension_is_added(self, station_dir, tmp_path):
        target = build_archive(station_dir, tmp_path / "noext")

        assert target.name.endswith(".zarr")
        assert target.exists()

    def test_append_mode_extends_an_existing_archive(
        self, synthetic_pro, tmp_path
    ):
        first = tmp_path / "first"
        first.mkdir(parents=True, exist_ok=True)
        synthetic_pro.multi_station(first, n_stations=3, n_profiles_each=5)
        target = build_archive(first, tmp_path / "grow.zarr")

        with xr.open_zarr(target) as ds:
            assert ds.sizes["location"] == 3

        second = _clone_stations(first, tmp_path / "second", range(3, 6))
        build_archive(second, target, mode="a")

        with xr.open_zarr(target) as ds:
            assert ds.sizes["location"] == 6
            assert len(set(map(str, ds.location.values))) == 6

    def test_appending_same_slope_batch_to_differing_slope_archive(
        self, synthetic_pro, tmp_path
    ):
        """
        Regression: read()'s norm_slopes='auto' normalizes slope coords to
        ('slope',) when a batch's stations happen to share a slope, and keeps
        ('location', 'slope') when they differ. Batches must not change shape
        based on which stations they happen to contain.
        """
        first = tmp_path / "differing"
        first.mkdir(parents=True, exist_ok=True)
        synthetic_pro.multi_station(first, n_stations=3, n_profiles_each=4)
        target = build_archive(first, tmp_path / "slopes.zarr")

        # Every station in this batch shares one slope angle.
        same = _clone_stations(first, tmp_path / "identical", range(3, 6))
        build_archive(same, target, mode="a")

        with xr.open_zarr(target) as ds:
            assert ds.sizes["location"] == 6
            assert ds["azimuth"].dims == ("location", "slope")

    def test_missing_source_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            build_archive(empty, tmp_path / "none.zarr")

    def test_invalid_on_time_mismatch_raises(self, station_dir, tmp_path):
        with pytest.raises(ValueError, match="on_time_mismatch"):
            build_archive(
                station_dir, tmp_path / "x.zarr", on_time_mismatch="explode"
            )

    def test_existing_target_is_not_overwritten_by_default(
        self, station_dir, tmp_path
    ):
        target = build_archive(station_dir, tmp_path / "once.zarr")
        with pytest.raises(FileExistsError):
            build_archive(station_dir, target)

    def test_mode_w_overwrites(self, station_dir, tmp_path):
        target = build_archive(station_dir, tmp_path / "twice.zarr")
        rebuilt = build_archive(station_dir, target, mode="w")

        with xr.open_zarr(rebuilt) as ds:
            assert ds.sizes["location"] == len(sorted(station_dir.glob("*.pro")))

    def test_batch_bytes_accepts_size_strings(self, station_dir, tmp_path):
        target = build_archive(
            station_dir, tmp_path / "sized.zarr", batch_bytes="4GB"
        )

        assert target.exists()
