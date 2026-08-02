import logging
import uuid

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from xsnow.utils import (
    detect_slope_coord_issues,
    get_logger,
    normalize_slope_coords,
)


def _cleanup_logger(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        log.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    log.propagate = True


def test_get_logger_adds_console_handler_and_sets_level():
    name = f"xsnow.test.console.{uuid.uuid4()}"
    log = get_logger(name)

    stream_handlers = [
        h for h in log.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]

    assert len(stream_handlers) == 1
    assert stream_handlers[0].level == logging.INFO
    assert log.level == logging.INFO
    assert log.propagate is False

    _cleanup_logger(log)


def test_get_logger_requires_logfile_when_to_file_true_without_handler():
    name = f"xsnow.test.nofile.{uuid.uuid4()}"
    existing = logging.getLogger(name)
    _cleanup_logger(existing)

    with pytest.raises(ValueError):
        get_logger(name, to_file=True, logger=existing)

    _cleanup_logger(existing)


def test_get_logger_supports_separate_console_and_file_levels(tmp_path):
    name = f"xsnow.test.file.{uuid.uuid4()}"
    logfile = tmp_path / "log.txt"

    log = get_logger(
        name,
        to_console=True,
        level_console=logging.WARNING,
        to_file=True,
        logfile=str(logfile),
        level_file=logging.DEBUG,
    )

    stream_handler = next(
        h for h in log.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    file_handler = next(h for h in log.handlers if isinstance(h, logging.FileHandler))

    assert stream_handler.level == logging.WARNING
    assert file_handler.level == logging.DEBUG
    assert log.level == logging.DEBUG

    log.debug("debug-line")
    log.warning("warn-line")
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

    contents = logfile.read_text()
    assert "debug-line" in contents
    assert "warn-line" in contents

    _cleanup_logger(log)


def test_get_logger_does_not_alter_provided_logger_without_flag():
    name = f"xsnow.test.noalter.{uuid.uuid4()}"
    existing = logging.getLogger(name)
    _cleanup_logger(existing)
    existing.propagate = True

    returned = get_logger(name, logger=existing, alter_logger=False)

    assert returned is existing
    assert returned.handlers == []
    assert returned.propagate is True

    _cleanup_logger(existing)


def test_get_logger_uses_level_prefixes(capsys):
    name = f"xsnow.test.prefix.{uuid.uuid4()}"
    log = get_logger(name)

    log.info("hello")
    log.warning("warn")
    for handler in log.handlers:
        handler.flush()

    captured = capsys.readouterr().err
    assert "[i]" in captured
    assert "[w]" in captured


def _slope_dataset(values, *, lazy=False):
    values = np.asarray(values, dtype=float)
    coord_values = (
        da.from_array(values, chunks=(1, values.shape[1]))
        if lazy
        else values
    )
    marker_values = (
        da.ones(values.shape, chunks=(1, values.shape[1]))
        if lazy
        else np.ones(values.shape)
    )
    return xr.Dataset(
        {"marker": (("location", "slope"), marker_values)},
        coords={
            "location": np.arange(values.shape[0]),
            "slope": np.arange(values.shape[1]),
            "inclination": (("location", "slope"), coord_values),
        },
    )


def test_normalize_slope_coords_default_auto_normalizes_consistent_values():
    ds = _slope_dataset(
        [
            [0.0, 35.0],
            [0.0, 35.0],
            [0.0, 35.0],
        ]
    )

    normalized = normalize_slope_coords(ds)

    assert normalized["inclination"].dims == ("slope",)
    np.testing.assert_allclose(normalized["inclination"], [0.0, 35.0])


def test_normalize_slope_coords_default_auto_preserves_union_that_would_grow():
    values = np.array(
        [
            [0.0, 35.0],
            [0.0, 40.0],
        ]
    )
    ds = _slope_dataset(values)

    normalized = normalize_slope_coords(
        ds,
        slope_coord_names=["inclination"],
    )

    assert normalized["inclination"].dims == ("location", "slope")
    np.testing.assert_allclose(normalized["inclination"], values)


def test_normalize_slope_coords_materializes_lazy_metadata_once():
    ds = _slope_dataset(
        np.tile([0.0, 35.0], (100, 1)),
        lazy=True,
    )
    computations = []

    with dask.config.set(scheduler="synchronous"), Callback(
        start=lambda dask_graph: computations.append(dask_graph)
    ):
        normalized = normalize_slope_coords(
            ds,
            slope_coord_names=["inclination"],
        )

    assert len(computations) == 1
    assert normalized["inclination"].dims == ("slope",)
    assert isinstance(normalized["inclination"].data, np.ndarray)
    assert isinstance(normalized["marker"].data, da.Array)


def test_normalize_slope_coords_only_normalizes_recognized_metadata():
    ds = _slope_dataset(
        [
            [0.0, 35.0],
            [0.0, 35.0],
        ]
    ).assign_coords(
        azimuth=(
            ("location", "slope"),
            [[0.0, 180.0], [0.0, 180.0]],
        ),
        z=(
            ("location", "slope", "layer"),
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.1, 1.2], [1.3, 1.4]],
            ],
        ),
    )

    diagnostics = detect_slope_coord_issues(ds)
    normalized = normalize_slope_coords(ds)

    assert set(diagnostics["multidim_coords"]) == {"azimuth", "inclination"}
    assert "z" not in diagnostics["multidim_coords"]
    assert normalized["inclination"].dims == ("slope",)
    assert normalized["azimuth"].dims == ("slope",)
    assert normalized["z"].dims == ("location", "slope", "layer")
    np.testing.assert_allclose(normalized["z"], ds["z"])


def test_normalize_slope_coords_accepts_custom_metadata_names():
    ds = _slope_dataset(
        [
            [0.0, 35.0],
            [0.0, 35.0],
        ]
    ).rename(inclination="custom_slope_metadata")

    normalized = normalize_slope_coords(
        ds,
        slope_coord_names=("custom_slope_metadata",),
    )

    assert normalized["custom_slope_metadata"].dims == ("slope",)


def _slope_union_dataset(
    inclination,
    azimuth,
    marker,
    *,
    lazy=False,
):
    inclination = np.asarray(inclination, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    marker = np.asarray(marker, dtype=float)
    if lazy:
        inclination = da.from_array(inclination, chunks=(1, inclination.shape[1]))
        azimuth = da.from_array(azimuth, chunks=(1, azimuth.shape[1]))
        marker = da.from_array(marker, chunks=(1, marker.shape[1]))

    return xr.Dataset(
        {
            "marker": (("location", "slope"), marker),
            "location_marker": ("location", np.arange(marker.shape[0])),
        },
        coords={
            "location": [f"L{index}" for index in range(marker.shape[0])],
            "slope": np.arange(marker.shape[1]),
            "inclination": (("location", "slope"), inclination),
            "azimuth": (("location", "slope"), azimuth),
            "z": (("location", "slope"), marker + 0.5),
        },
    )


def test_normalize_slope_coords_union_globalizes_distinct_local_slopes():
    ds = _slope_union_dataset(
        [[35.0], [20.0]],
        [[180.0], [0.0]],
        [[3.5], [2.0]],
    )

    normalized = normalize_slope_coords(ds, strategy="union")

    assert normalized.sizes["slope"] == 2
    assert normalized["inclination"].dims == ("slope",)
    assert normalized["azimuth"].dims == ("slope",)
    np.testing.assert_allclose(normalized["inclination"], [20.0, 35.0])
    np.testing.assert_allclose(normalized["azimuth"], [0.0, 180.0])
    np.testing.assert_allclose(
        normalized["marker"],
        [[np.nan, 3.5], [2.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        normalized["z"],
        [[np.nan, 4.0], [2.5, np.nan]],
        equal_nan=True,
    )
    assert normalized["location_marker"].dims == ("location",)


def test_normalize_slope_coords_auto_remaps_compatible_subset():
    ds = _slope_union_dataset(
        [
            [10.0, 20.0, 30.0],
            [20.0, np.nan, np.nan],
            [10.0, 20.0, 30.0],
        ],
        [
            [0.0, 90.0, 180.0],
            [90.0, np.nan, np.nan],
            [0.0, 90.0, 180.0],
        ],
        [
            [1.0, 2.0, 3.0],
            [20.0, np.nan, np.nan],
            [10.0, 20.0, 30.0],
        ],
    )

    normalized = normalize_slope_coords(ds, strategy="auto")

    assert normalized.sizes["slope"] == ds.sizes["slope"]
    assert normalized["inclination"].dims == ("slope",)
    np.testing.assert_allclose(normalized["inclination"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(normalized["azimuth"], [0.0, 90.0, 180.0])
    np.testing.assert_allclose(
        normalized["marker"],
        [
            [1.0, 2.0, 3.0],
            [np.nan, 20.0, np.nan],
            [10.0, 20.0, 30.0],
        ],
        equal_nan=True,
    )


def test_normalize_slope_coords_auto_preserves_union_that_would_grow():
    ds = _slope_union_dataset(
        [[20.0], [35.0]],
        [[0.0], [180.0]],
        [[2.0], [3.5]],
    )

    normalized = normalize_slope_coords(ds, strategy="auto")

    assert normalized is ds
    assert normalized.sizes["slope"] == 1
    assert normalized["inclination"].dims == ("location", "slope")


def test_normalize_slope_coords_union_rejects_duplicate_local_signature():
    ds = _slope_union_dataset(
        [[20.0, 20.0]],
        [[90.0, 90.0]],
        [[1.0, 2.0]],
    )

    with pytest.raises(ValueError, match="duplicate slope metadata"):
        normalize_slope_coords(ds, strategy="union")


def test_normalize_slope_coords_union_keeps_profile_data_lazy():
    ds = _slope_union_dataset(
        [[20.0], [35.0]],
        [[0.0], [180.0]],
        [[2.0], [3.5]],
        lazy=True,
    )
    computations = []

    with dask.config.set(scheduler="synchronous"), Callback(
        start=lambda dask_graph: computations.append(dask_graph)
    ):
        normalized = normalize_slope_coords(ds, strategy="union")

    assert len(computations) == 2
    assert isinstance(normalized["marker"].data, da.Array)
    assert normalized.sizes["slope"] == 2


def test_normalize_slope_coords_union_preserves_profile_status_semantics():
    ds = _slope_union_dataset(
        [[20.0], [35.0]],
        [[0.0], [180.0]],
        [[2.0], [3.5]],
    ).assign(
        profile_status=(
            ("location", "slope"),
            np.array([[1], [-1]], dtype="i1"),
        )
    )

    normalized = normalize_slope_coords(ds, strategy="union")

    assert normalized["profile_status"].dtype == np.dtype("i1")
    np.testing.assert_array_equal(
        normalized["profile_status"],
        [[1, 0], [0, -1]],
    )


@pytest.mark.parametrize("strategy", ("auto", "union"))
def test_normalize_slope_coords_preserves_all_missing_metadata(strategy):
    ds = _slope_union_dataset(
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[2.0, 2.5], [3.5, 4.0]],
    )

    with pytest.warns(UserWarning, match="contain no usable values"):
        normalized = normalize_slope_coords(ds, strategy=strategy)

    assert normalized is ds
    assert normalized["inclination"].dims == ("location", "slope")

@pytest.mark.parametrize("strategy", ("auto", "union"))
def test_normalize_slope_coords_preserves_absent_metadata(strategy):
    ds = _slope_union_dataset(
        [[10.0, 20.0], [10.0, 20.0]],
        [[0.0, 90.0], [0.0, 90.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ).drop_vars(["inclination", "azimuth"])

    with pytest.warns(UserWarning, match="no recognized slope metadata"):
        normalized = normalize_slope_coords(ds, strategy=strategy)

    assert normalized is ds


@pytest.mark.parametrize("strategy", ("auto", "union"))
def test_normalize_slope_coords_preserves_unidentified_location(strategy):
    ds = _slope_union_dataset(
        [[10.0, 20.0], [np.nan, np.nan]],
        [[0.0, 90.0], [np.nan, np.nan]],
        [[1.0, 2.0], [3.0, np.nan]],
    )

    with pytest.warns(UserWarning, match=r"do not identify any slope for 1 location"):
        normalized = normalize_slope_coords(ds, strategy=strategy)

    assert normalized is ds
    assert normalized["inclination"].dims == ("location", "slope")
