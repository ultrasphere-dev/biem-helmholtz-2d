from typing import Any

from array_api.latest import ArrayNamespace

from biem_helmholtz_2d._quadrature import log_cot_power_quadrature, trapezoidal_quadrature


def test_quadrature_cache(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    trapezoidal_quadrature.cache_clear()
    log_cot_power_quadrature.cache_clear()
    first = trapezoidal_quadrature(8, xp=xp, device=device, dtype=dtype)
    second = trapezoidal_quadrature(8, xp=xp, device=device, dtype=dtype)
    assert first is second
    assert trapezoidal_quadrature.cache_info().hits == 1
    assert trapezoidal_quadrature(9, xp=xp, device=device, dtype=dtype)[0].shape != first[0].shape
    first_log = log_cot_power_quadrature(8, 0, xp=xp, device=device, dtype=dtype)
    second_log = log_cot_power_quadrature(8, 0, xp=xp, device=device, dtype=dtype)
    assert first_log is second_log
    assert log_cot_power_quadrature.cache_info().hits == 1
