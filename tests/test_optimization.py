from pathlib import Path
from typing import Any

from array_api.latest import ArrayNamespace

from biem_helmholtz_2d.optimization._example import example_optimization


def test_optimization(xp: ArrayNamespace, dtype: Any, device: Any, tmp_path: Path) -> None:
    for alpha_reg in (0, 0.1):
        path = tmp_path / str(alpha_reg)
        example_optimization(
            xp=xp,
            dtype=dtype,
            device=device,
            k=1,
            alpha=1,
            eta=1,
            n=16,
            alpha_reg=alpha_reg,
            path=path,
        )
        assert (path / "optimization_history.json").is_file()
        assert (path / "optimized_shape.json").is_file()
        assert (path / "optimized_near_field.json").is_file()
