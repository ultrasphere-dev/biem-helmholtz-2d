from pathlib import Path
from typing import Any

from array_api.latest import ArrayNamespace

from biem_helmholtz_2d.optimization._example import example_optimization


def test_optimization(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    path = Path(__file__).parent / "optimization_test"
    example_optimization(xp=xp, dtype=dtype, device=device, alpha_reg=0, path=path / "alpha0")
    example_optimization(xp=xp, dtype=dtype, device=device, path=path / "alpha")
