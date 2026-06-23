from typing import Any

import pytest
from array_api.latest import ArrayNamespace

from biem_helmholtz_2d.optimization._example import example_optimization


@pytest.mark.skip(reason="Too slow")
def test_optimization(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    example_optimization(xp=xp, dtype=dtype, device=device)
