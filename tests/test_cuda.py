import importlib
from typing import Any

import pytest
from scipy import special

from biem_helmholtz_2d import adjoint_central_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._potential_inner import dlp, slp
from biem_helmholtz_2d._scipy_wrapper import scipy_hankel1, scipy_jv, scipy_yv
from biem_helmholtz_2d.optimization._shape import ParameterShape

torch = pytest.importorskip("torch")
torch_xp = importlib.import_module("array_api_compat.torch")


@pytest.fixture
def xp() -> Any:
    return torch_xp


@pytest.fixture
def dtype() -> Any:
    return torch.float64


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_bessel_orders(xp: Any, dtype: Any, device: str, order: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = xp.asarray([1e-8, 1e-4, 0.01, 0.5, 1, 2, 5, 10, 50, 100], device=device, dtype=dtype)
    x_cpu = xp.asarray(x, device="cpu")
    for actual, reference in (
        (scipy_jv(order, x), special.jv(order, x_cpu)),
        (scipy_yv(order, x), special.yv(order, x_cpu)),
        (scipy_hankel1(order, x), special.hankel1(order, x_cpu)),
    ):
        expected = xp.asarray(reference, device=device, dtype=actual.dtype)
        tolerance = 3e-12 if dtype == torch.float64 else 2e-6
        assert bool(xp.all(xp.abs(actual - expected) <= tolerance * (1 + xp.abs(expected))))
    if order == 2:
        zero = xp.asarray([0], device=device, dtype=dtype)
        assert float(scipy_jv(2, zero)[0]) == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("complex_wave_number", [False, True])
def test_torch_adjoint_and_different_quadratures(
    xp: Any, dtype: Any, device: str, complex_wave_number: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    centers = xp.asarray([[0, 0], [6, 0]], device=device, dtype=dtype)
    one = xp.asarray([1], device=device, dtype=dtype)
    zero = xp.asarray([0], device=device, dtype=dtype)
    empty = xp.zeros(0, device=device, dtype=dtype)
    shapes = [ParameterShape(center=centers[i], cos_coefs=one, sin_coefs=empty) for i in range(2)]
    perturbations = [
        ParameterShape(
            center=xp.zeros(2, device=device, dtype=dtype), cos_coefs=one, sin_coefs=empty
        ),
        ParameterShape(
            center=xp.zeros(2, device=device, dtype=dtype), cos_coefs=zero, sin_coefs=empty
        ),
    ]
    k = xp.asarray(
        1 + 0.1j if complex_wave_number else 1,
        device=device,
        dtype=xp.complex128 if complex_wave_number else dtype,
    )
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    point = xp.asarray([3, 3], device=device, dtype=dtype)
    result = adjoint_central_derivative(
        shape=shapes,
        h_shape=perturbations,
        point=point,
        target=xp.asarray(0, device=device, dtype=dtype),
        k=k,
        alpha=k,
        eta=k,
        incident_field=plane_wave(k, direction),
        incident_field_grad=plane_wave_grad(k, direction),
        n_solve=8,
        n_quadrature=12,
        steps=[1e-5],
    )
    assert result.field.device.type == device
    assert float(result.adjoint) == pytest.approx(float(result.central_objective[0]), abs=1e-7)
    if complex_wave_number:
        for potential in (slp, dlp):
            value = potential(
                point,
                lambda t: xp.ones_like(t),
                shape_x=shapes[0].x,
                shape_dx=shapes[0].dx,
                k=k,
                n=8,
            )
            assert value.device.type == device
            assert bool(xp.isfinite(value))
