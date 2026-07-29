# io/memory.py
"""
Pre-flight memory guard for xsnow's eager loads.

A process that is killed by the kernel OOM killer receives SIGKILL, which is
uncatchable: there is no exception, no ``finally``, no ``atexit`` hook, and no
log line. Memory safety therefore cannot be handled reactively from inside the
process; it has to be checked *before* the allocation happens.

This module provides that check. It compares a dataset's decoded in-memory
size against a byte budget and raises before the eager load is issued, so the
caller gets an actionable error instead of a dead interpreter.

The size estimate comes from ``Dataset.nbytes``, which is derived from shape
and dtype and does not materialize data on any of xsnow's read paths (lazy
indexing or Dask). It reports the *uncompressed in-memory* size, which is
exactly the quantity that matters for RAM, and it is checked after selectors
have narrowed the dataset, so it reflects the subset actually being read.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional, Union

from ..utils import get_logger


#: Fraction of detected available memory used as the budget in ``'auto'`` mode.
DEFAULT_MEMORY_FRACTION = 0.7

#: Accepted by ``memory_limit`` to request the automatic RAM-derived budget.
MemoryLimitLike = Union[str, int, float, bool, None]

_SIZE_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
    "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4,
}


class MemoryLimitExceeded(MemoryError):
    """
    Raised pre-flight when an eager load would exceed the memory budget.

    Subclasses ``MemoryError`` so that callers already guarding against
    allocation failures catch this too.
    """


def _format_bytes(num_bytes: float) -> str:
    """Render a byte count using binary units, for error messages."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes:.0f} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TiB"  # pragma: no cover - loop always returns


def _parse_size_string(value: str) -> int:
    """
    Parse a human-readable size such as ``'8GB'`` or ``'512 MiB'`` into bytes.

    Decimal suffixes (kb/mb/gb/tb) are powers of 1000; binary suffixes
    (kib/mib/gib/tib) and bare single letters (k/m/g/t) are powers of 1024.
    """
    match = _SIZE_PATTERN.match(value)
    if not match:
        raise ValueError(
            f"Could not parse memory limit {value!r}. Expected a size like "
            f"'8GB', '512MiB', a number of bytes, 'auto', or None."
        )
    magnitude, suffix = match.groups()
    factor = _SIZE_UNITS.get(suffix.lower())
    if factor is None:
        raise ValueError(
            f"Unknown size unit {suffix!r} in memory limit {value!r}. "
            f"Supported units: {', '.join(sorted(u for u in _SIZE_UNITS if u))}."
        )
    return int(float(magnitude) * factor)


def available_memory_bytes() -> Optional[int]:
    """
    Best-effort estimate of memory available to this process, in bytes.

    Uses ``psutil`` when it is importable (it is an optional dependency, not a
    core one), falling back to POSIX ``sysconf``. Returns None when neither
    source is usable, in which case the caller should skip the check rather
    than guess.

    Note:
        A container or scheduled HPC job may have a cgroup limit or scheduler
        setting that caps its assigned memory, but this function does not read
        that limit directly. The detected value may instead describe the host
        or whole compute node, and can include reclaimable file-system cache
        shared with other work. Do not rely on ``'auto'`` in these environments
        unless you have verified what it detects; pass an explicit cap based on
        the assigned memory.
    """
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - psutil platform quirks
            pass

    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def resolve_memory_limit(
    memory_limit: MemoryLimitLike = "auto",
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """
    Resolve a ``memory_limit`` argument into an absolute byte budget.

    Args:
        memory_limit: ``'auto'`` (default) derives the budget from detected
            available memory; an int, float, or size string sets an absolute
            cap; None or False disables the guard.
        logger: Optional logger for diagnostics.

    Returns:
        The budget in bytes, or None when the guard is disabled or no budget
        could be determined.

    Raises:
        ValueError: If the value is not a recognized size or is not positive.
    """
    if memory_limit is None or memory_limit is False:
        return None

    if memory_limit is True:
        memory_limit = "auto"

    if isinstance(memory_limit, str):
        if memory_limit.strip().lower() == "auto":
            available = available_memory_bytes()
            if available is None:
                get_logger(__name__, logger=logger).debug(
                    "Memory guard disabled: available memory could not be "
                    "detected (install psutil or pass an explicit memory_limit)."
                )
                return None
            return int(available * DEFAULT_MEMORY_FRACTION)
        budget = _parse_size_string(memory_limit)
    elif isinstance(memory_limit, (int, float)) and not isinstance(memory_limit, bool):
        budget = int(memory_limit)
    else:
        raise ValueError(
            f"Unsupported memory_limit type {type(memory_limit).__name__}. "
            f"Expected 'auto', a size string, a number of bytes, or None."
        )

    if budget <= 0:
        raise ValueError(
            f"memory_limit must be positive, got {memory_limit!r}. "
            f"Pass None to disable the guard."
        )
    return budget


def estimate_decoded_nbytes(ds: Any) -> Optional[int]:
    """
    Estimate a dataset's decoded in-memory size without materializing it.

    Returns None when the size cannot be trusted: object and variable-length
    string arrays report only xarray's pointer-based estimate, which does not
    represent the real payload.
    """
    try:
        if any(
            variable.dtype.kind in {"O", "T"}
            for variable in ds.variables.values()
        ):
            return None
        return int(ds.nbytes)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def check_memory_budget(
    ds: Any,
    budget_bytes: Optional[int],
    *,
    source: str = "dataset",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Raise before an eager load that would exceed the memory budget.

    Args:
        ds: The (still lazy) dataset about to be loaded.
        budget_bytes: Budget from :func:`resolve_memory_limit`. None skips
            the check.
        source: Human-readable description of what is being read, used in the
            error message.
        logger: Optional logger for diagnostics.

    Raises:
        MemoryLimitExceeded: If the estimated size exceeds the budget.
    """
    if budget_bytes is None:
        return

    log = get_logger(__name__, logger=logger)
    estimated = estimate_decoded_nbytes(ds)

    if estimated is None:
        log.debug(
            f"Memory guard skipped for {source}: decoded size could not be "
            f"estimated (object or variable-length string data)."
        )
        return

    if estimated <= budget_bytes:
        log.debug(
            f"Memory guard passed for {source}: "
            f"{_format_bytes(estimated)} within {_format_bytes(budget_bytes)}."
        )
        return

    raise MemoryLimitExceeded(
        f"Reading {source} eagerly would materialize about "
        f"{_format_bytes(estimated)}, which exceeds the memory limit of "
        f"{_format_bytes(budget_bytes)}.\n"
        f"Options:\n"
        f"  - narrow the read with time/location/slope/realization selectors\n"
        f"  - pass lazy=True to keep the data Dask-backed\n"
        f"  - raise or disable the budget with memory_limit='32GB' or "
        f"memory_limit=None"
    )
