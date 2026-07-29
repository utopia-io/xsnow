import logging
import uuid

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from xsnow.utils import get_logger, normalize_slope_coords


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


def test_normalize_slope_coords_collapses_consistent_values():
    ds = _slope_dataset(
        [
            [0.0, 35.0],
            [0.0, 35.0],
            [0.0, 35.0],
        ]
    )

    normalized = normalize_slope_coords(
        ds,
        slope_coord_names=["inclination"],
        allow_per_location=True,
    )

    assert normalized["inclination"].dims == ("slope",)
    np.testing.assert_allclose(normalized["inclination"], [0.0, 35.0])


def test_normalize_slope_coords_preserves_inconsistent_values_when_allowed():
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
        allow_per_location=True,
    )

    assert normalized["inclination"].dims == ("location", "slope")
    np.testing.assert_allclose(normalized["inclination"], values)


def test_normalize_slope_coords_reports_first_inconsistent_location():
    ds = _slope_dataset(
        [
            [0.0, 35.0],
            [0.0, 35.0],
            [0.0, 40.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Location 0 vs Location 2 differ at slope indices: \[1\]",
    ):
        normalize_slope_coords(
            ds,
            slope_coord_names=["inclination"],
        )


def test_normalize_slope_coords_computes_lazy_coordinate_once():
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
            allow_per_location=True,
        )

    assert len(computations) == 1
    assert normalized["inclination"].dims == ("slope",)
    assert isinstance(normalized["inclination"].data, da.Array)
    assert isinstance(normalized["marker"].data, da.Array)
