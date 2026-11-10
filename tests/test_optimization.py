import json
from pathlib import Path
from typing import Any

import pytest
from array_api.latest import ArrayNamespace

from biem_helmholtz_2d.optimization._example import (
    example_optimization,
    example_optimization_plot,
    project_coefficients,
)
from biem_helmholtz_2d.optimization._shape import ParameterShape


@pytest.mark.parametrize("n_scatterers", [1, 2])
def test_optimization(
    xp: ArrayNamespace, dtype: Any, device: Any, tmp_path: Path, n_scatterers: int
) -> None:
    parameters = []
    centers = xp.asarray([[0, 0], [5, 0]], dtype=dtype, device=device)[:n_scatterers]
    for alpha_reg in (0, 0.1):
        path = tmp_path / str(alpha_reg)
        example_optimization(
            k=1,
            centers=centers,
            n_modes=3,
            n_steps=2,
            alpha=1,
            eta=1,
            n=16,
            alpha_reg=alpha_reg,
            path=path,
        )
        assert (path / "optimization_history.json").is_file()
        assert (path / "optimized_shape.json").is_file()
        assert (path / "optimized_near_field.json").is_file()
        history = json.loads((path / "optimization_history.json").read_text())
        assert history["alpha_reg"] == alpha_reg
        assert all(
            b <= a for a, b in zip(history["val_hist"], history["val_hist"][1:], strict=False)
        )
        cos = xp.asarray(history["final_parameters"]["cos_coefs"], dtype=dtype, device=device)[
            :, 1:
        ]
        sin = xp.asarray(history["final_parameters"]["sin_coefs"], dtype=dtype, device=device)
        assert cos.shape == (n_scatterers, 3)
        assert bool(xp.all(xp.sum(xp.sqrt(cos**2 + sin**2), axis=-1) <= 1 + 1e-12))
        parameters.append(xp.stack([cos, sin]))
        example_optimization_plot(path)
        assert (path / f"optimized_shape_{alpha_reg}.svg").is_file()
    assert bool(xp.any(xp.abs(parameters[0] - parameters[1]) > 1e-8))


def test_optimization_complex_couplings(
    xp: ArrayNamespace, dtype: Any, device: Any, tmp_path: Path
) -> None:
    example_optimization(
        path=tmp_path,
        centers=xp.asarray([[0, 0]], dtype=dtype, device=device),
        k=1,
        alpha=1 + 0.1j,
        eta=1 - 0.2j,
        n=4,
        n_modes=1,
        n_steps=1,
    )
    history = json.loads((tmp_path / "optimization_history.json").read_text())
    assert history["alpha"]["imag"] == pytest.approx(0.1)
    assert history["eta"]["imag"] == pytest.approx(-0.2)
    assert all(b <= a for a, b in zip(history["val_hist"], history["val_hist"][1:], strict=False))


def test_optimization_without_modes(
    xp: ArrayNamespace, dtype: Any, device: Any, tmp_path: Path
) -> None:
    example_optimization(
        path=tmp_path,
        centers=xp.asarray([[0, 0]], dtype=dtype, device=device),
        k=1,
        alpha=1,
        eta=1,
        n=4,
        n_modes=0,
        n_steps=1,
    )
    history = json.loads((tmp_path / "optimization_history.json").read_text())
    assert history["final_parameters"]["cos_coefs"] == [[1]]
    assert history["final_parameters"]["sin_coefs"] == [[]]


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
        ([[0.3, 0.4], [0, 0]], [[0.3, 0.4], [0, 0]]),
        ([[3, 4], [0, 1]], [[0.6, 0.8], [0, 0]]),
        ([[2, 0], [0, 2]], [[0.5, 0], [0, 0.5]]),
    ],
)
def test_projection(
    xp: ArrayNamespace, dtype: Any, device: Any, coefficients: Any, expected: Any
) -> None:
    coefficients = xp.asarray(coefficients, dtype=dtype, device=device)
    expected = xp.asarray(expected, dtype=dtype, device=device)
    result = project_coefficients(coefficients)
    tolerance = 20 * xp.finfo(dtype).eps
    assert bool(xp.all(xp.abs(result - expected) < tolerance))
    assert bool(xp.all(xp.abs(project_coefficients(result) - result) < tolerance))


def test_reject_singleton_coordinate_axes(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    with pytest.raises(ValueError, match="cosine/sine pairs"):
        project_coefficients(xp.zeros((2, 1), dtype=dtype, device=device))
    with pytest.raises(ValueError, match="two coordinates"):
        ParameterShape(
            cos_coefs=xp.ones(1, dtype=dtype, device=device),
            sin_coefs=xp.zeros(0, dtype=dtype, device=device),
            center=xp.zeros(1, dtype=dtype, device=device),
        )


def test_weighted_projection(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    coefficients = xp.asarray([[[1, 0], [0, 1]], [[0, 0], [0.1, 0]]], dtype=dtype, device=device)
    weights = xp.asarray([1, 4], dtype=dtype, device=device)
    result = project_coefficients(coefficients, weights=weights)
    expected = xp.asarray([[0.2, 0], [0, 0.8]], dtype=dtype, device=device)
    tolerance = 20 * xp.finfo(dtype).eps
    assert bool(xp.all(xp.abs(result[0] - expected) < tolerance))
    assert bool(xp.all(xp.abs(result[1] - coefficients[1]) < tolerance))
    assert bool(xp.all(xp.abs(project_coefficients(result, weights=weights) - result) < tolerance))
    # At an active boundary, metric projection of a Riesz step is descent.
    derivative = xp.asarray([[1, 0], [-1, 0]], dtype=dtype, device=device)
    boundary = xp.asarray([[0.5, 0], [0.5, 0]], dtype=dtype, device=device)
    next_step = project_coefficients(boundary - derivative / weights[:, None], weights=weights)
    assert float(xp.sum(derivative * (next_step - boundary))) < 0
    candidate = xp.asarray([[0.3, 0.2], [0.1, -0.1]], dtype=dtype, device=device)
    normal = weights[:, None] * (coefficients[0] - result[0])
    assert float(xp.sum(normal * (candidate - result[0]))) <= tolerance
