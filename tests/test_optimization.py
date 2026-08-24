from typing import Any

from array_api.latest import ArrayNamespace

from biem_helmholtz_2d.optimization._example import example_optimization


def test_optimization(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    example_optimization(xp=xp, dtype=dtype, device=device)
