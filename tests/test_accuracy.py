import json
from typing import Any

import pytest
from array_api.latest import ArrayNamespace

from biem_helmholtz_2d import accuracy, adjoint_central_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d.optimization._shape import ParameterShape


def test_accuracy_general_configuration(xp: ArrayNamespace, device: Any, dtype: Any) -> None:
    """Study noncircular bodies with a different incident wave and perturbation."""
    centers = xp.asarray([[0, 0], [4, 0]], dtype=dtype, device=device)
    zero = xp.asarray([0], dtype=dtype, device=device)
    shape = [
        ParameterShape(
            center=centers[0],
            cos_coefs=xp.asarray([1, 0.12], dtype=dtype, device=device),
            sin_coefs=xp.asarray([0.07], dtype=dtype, device=device),
        ),
        ParameterShape(
            center=centers[1],
            cos_coefs=xp.asarray([0.7], dtype=dtype, device=device),
            sin_coefs=zero,
        ),
    ]
    perturbations = [
        ParameterShape(
            center=xp.zeros(2, dtype=dtype, device=device), cos_coefs=zero, sin_coefs=zero
        ),
        ParameterShape(
            center=xp.zeros(2, dtype=dtype, device=device),
            cos_coefs=zero,
            sin_coefs=xp.asarray([0.2], dtype=dtype, device=device),
        ),
    ]
    complex_dtype = xp.result_type(dtype, 1j)
    k = xp.asarray(1.3 + 0.05j, dtype=complex_dtype, device=device)
    alpha = xp.asarray(1 + 0.1j, dtype=complex_dtype, device=device)
    eta = xp.asarray(0.8 - 0.1j, dtype=complex_dtype, device=device)
    point = xp.asarray([1, 3], dtype=dtype, device=device)
    target = xp.asarray(0.2 + 0.1j, dtype=complex_dtype, device=device)
    direction = xp.asarray([0, 1], dtype=dtype, device=device)
    incident = plane_wave(k, direction)
    incident_grad = plane_wave_grad(k, direction)
    parameters = {
        "shape": shape,
        "h_shape": perturbations,
        "point": point,
        "target": target,
        "k": k,
        "alpha": alpha,
        "eta": eta,
        "incident_field": incident,
        "incident_field_grad": incident_grad,
    }
    data = accuracy(
        **parameters,
        solve_orders=[6],
        quadrature_orders=[5, 7],
        reference_order=8,
        step=1e-5,
    )
    json.dumps(data)
    assert data["settings"]["quadrature_orders"] == [5, 7]
    assert [(row["n_solve"], row["n_quadrature"]) for row in data["records"]] == [(6, 5), (6, 7)]
    reference = adjoint_central_derivative(**parameters, n_solve=8, n_quadrature=8, steps=[1e-5])
    sample = adjoint_central_derivative(**parameters, n_solve=6, n_quadrature=5, steps=[1e-5])
    row = data["records"][0]
    assert row["field"]["real"] == pytest.approx(float(xp.real(sample.field)))
    assert row["field"]["imag"] == pytest.approx(float(xp.imag(sample.field)))
    assert row["objective_derivative"] == pytest.approx(float(sample.adjoint))
    assert row["objective_central"] == pytest.approx(float(sample.central_objective[0]))
    assert row["field_error"] == pytest.approx(float(xp.abs(sample.field - reference.field)))
    assert row["field_derivative_error"] == pytest.approx(
        float(xp.abs(sample.central_field[0] - reference.central_field[0]))
    )
    assert row["objective_derivative_error"] == pytest.approx(
        float(xp.abs(sample.adjoint - reference.adjoint))
    )
