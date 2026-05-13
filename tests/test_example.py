from typing import Any

import pytest
from array_api.latest import ArrayNamespaceFull

from biem_helmholtz_2d._example import example_3_1, example_3_1_answer


def test_example_3_1(xp: ArrayNamespaceFull, device: Any, dtype: Any) -> None:
    actual = example_3_1(3, xp=xp, dtype=dtype, device=device)
    expected = example_3_1_answer()
    assert actual == pytest.approx(expected, rel=1e-6)
