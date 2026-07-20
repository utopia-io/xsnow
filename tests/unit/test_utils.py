import logging
import uuid

import pytest

from xsnow.utils import get_logger


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
