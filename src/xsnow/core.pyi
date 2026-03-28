from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, Self, overload

import xarray as xr

class xsnowDataset(xr.Dataset):
    """Typed wrapper around xr.Dataset for snow profile data. While the actual implementation does
    not inherit from xr.Dataset, typing it as such allows for the various methods that are forwarded to
    xr.Dataset to be visible to type checkers. There is no change to runtime behavior.

    The fictious inheritance is a workaround to keep the stub simple. One subtle issue is that type checkers
    will infer isinstance(xs, xr.Dataset) to be always True when xs is an xsnowDataset, while at runtime it will
    be False. So any code that uses this construct may be incorrectly analyzed by the type checker. As an
    example, consider this function:

    def get_xarray(xs: Union[xsnowDataset, xr.Dataset]) -> xr.Dataset:
        if isinstance(xs, xr.Dataset):
            return xs
        else:
            return xs.data

    While the else branch is reachable at runtime, the type checker will see it as unreachable and
    may analyze the code incorrectly.

    A fix is to use isinstance(xs, xsnowDataset) instead, and update logic as needed:

    def get_xarray_fixed(xs: Union[xsnowDataset, xr.Dataset]) -> xr.Dataset:
        if isinstance(xs, xsnowDataset):
            return xs.data
        else:
            return xs
    """

    data: xr.Dataset

    def __init__(self, source: xr.Dataset | Self) -> None: ...

    @overload
    def __getitem__(self, key: Hashable) -> xr.DataArray: ...
    @overload
    def __getitem__(self, key: list[Hashable]) -> xr.Dataset: ...
    @overload
    def __getitem__(self, key: Mapping[Any, Any]) -> xr.Dataset: ...  # type: ignore[override]

    def __setitem__(self, key: Hashable | list[Hashable] | Mapping, value: Any) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __dir__(self) -> list[str]: ...
    def __repr__(self, *, force_details: bool = False) -> str: ...

    def copy(self, deep: bool = False) -> "xsnowDataset": ...  # type: ignore[override]
    def merge_smet(
        self,
        source: str | list[str] | Path | list[Path],
        var_prefix: str | None = None,
        **kwargs: Any,
    ) -> "xsnowDataset": ...
    def to_xarray(self) -> xr.Dataset: ...
    def compute_hs(self) -> "xsnowDataset": ...
    def compute_z_depth(self) -> "xsnowDataset": ...
    def update_profile_status(self) -> "xsnowDataset": ...
    def profile_status_summary(self) -> None: ...
    def to_gpu(self) -> "xsnowDataset": ...
    def to_cpu(self) -> "xsnowDataset": ...

class DatasetDecorator(xsnowDataset):
    @property
    def base(self) -> xsnowDataset: ...
