"""Quadrature convergence studies for point-field shape sensitivities."""

import math
from collections.abc import Callable, Sequence
from typing import Any

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import Shape
from tqdm import tqdm

from ._sensitivity import SensitivityResult, adjoint_central_derivative


def accuracy(
    *,
    shape: Sequence[Shape],
    h_shape: Sequence[Shape],
    point: Array,
    target: Array,
    k: Array,
    alpha: Array,
    eta: Array,
    incident_field: Callable[[Array], Array],
    incident_field_grad: Callable[[Array], Array],
    solve_orders: Sequence[int],
    quadrature_orders: Sequence[int],
    reference_order: int,
    step: float,
) -> dict[str, Any]:
    r"""
    Compare field and shape-sensitivity results at different quadrature orders.

    The objective is $|u_{\mathrm{scat}}(x_0)-c|^2$. The field derivative is
    computed by central differences, while the objective derivative is computed
    by the adjoint and also by central differences. Errors are absolute
    differences from the numerical reference, not rigorous error bounds.

    Parameters
    ----------
    shape, h_shape : Sequence[Shape]
        Boundaries and perturbations, each of (...) -> (..., 2).
    point : Array
        Observation point $x_0$ of shape (2,).
    target : Array
        Desired scattered field $c$ of shape ().
    k, alpha, eta : Array
        Scalar wave number and coupling parameters of shape ().
    incident_field : Callable[[Array], Array]
        Incident field of (..., 2) -> (...,).
    incident_field_grad : Callable[[Array], Array]
        Spatial incident-field gradient of (..., 2) -> (..., 2).
    solve_orders, quadrature_orders : Sequence[int]
        Positive orders for the Nyström solves and the integration, respectively.
        Each order $n$ uses $2n-1$ nodes; all combinations are evaluated.
    reference_order : int
        Order used for both stages of the reference, greater than every study order.
    step : float
        Positive, finite central-difference step.

    Returns
    -------
    dict[str, Any]
        JSON-compatible mapping with ``settings``, ``reference``, and one
        ``records`` entry per solve/integration order pair. Complex numbers are
        mappings with ``real`` and ``imag`` keys. This routine currently accepts
        scalar wave parameters and a single observation point.

    Examples
    --------
    >>> import array_api_compat.numpy as xp
    >>> from ie_circle import CircleShape
    >>> from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
    >>> k = xp.asarray(1.0)
    >>> direction = xp.asarray([1.0, 0.0])
    >>> study = accuracy(
    ...     shape=[CircleShape(1)], h_shape=[CircleShape(1)],
    ...     point=xp.asarray([3.0, 3.0]), target=xp.asarray(0.0),
    ...     k=k, alpha=k, eta=k, incident_field=plane_wave(k, direction),
    ...     incident_field_grad=plane_wave_grad(k, direction),
    ...     solve_orders=[3], quadrature_orders=[3], reference_order=4, step=1e-5,
    ... )
    >>> [(row['n_solve'], row['n_quadrature']) for row in study['records']]
    [(3, 3)]

    """
    check_shapes("D,D", point, (2,), names="point,coordinate_dimension")
    if point.shape != (2,) or any(value.ndim != 0 for value in (target, k, alpha, eta)):
        raise ValueError("point must have shape (2,) and wave parameters and target must be scalar")
    array_namespace(point, target, k, alpha, eta)
    if not shape or len(shape) != len(h_shape):
        raise ValueError("shape and h_shape must have the same nonzero length")
    if not solve_orders or not quadrature_orders:
        raise ValueError("solve_orders and quadrature_orders must be nonempty")
    if min(*solve_orders, *quadrature_orders) < 1 or reference_order <= max(
        *solve_orders, *quadrature_orders
    ):
        raise ValueError("reference_order must exceed all positive study orders")
    if step <= 0 or not math.isfinite(step):
        raise ValueError("step must be positive and finite")

    def evaluate(n_solve: int, n_quad: int, /) -> SensitivityResult:
        return adjoint_central_derivative(
            shape=shape,
            h_shape=h_shape,
            point=point,
            target=target,
            k=k,
            alpha=alpha,
            eta=eta,
            incident_field=incident_field,
            incident_field_grad=incident_field_grad,
            n_solve=n_solve,
            n_quadrature=n_quad,
            steps=[step],
        )

    def complex_json(value: Array, /) -> dict[str, float]:
        number = complex(value)
        return {"real": number.real, "imag": number.imag}

    configurations = [(n, q) for n in solve_orders for q in quadrature_orders]
    with tqdm(total=len(configurations) + 1, desc="Quadrature accuracy", unit="case") as progress:
        progress.set_postfix_str(f"reference n={reference_order}")
        reference = evaluate(reference_order, reference_order)
        progress.update()
        ref_field = complex(reference.field)
        ref_field_derivative = complex(reference.central_field[0])
        ref_adjoint = float(reference.adjoint)
        records = []
        for n_solve, n_quad in configurations:
            progress.set_postfix_str(f"solve={n_solve}, quadrature={n_quad}")
            result = evaluate(n_solve, n_quad)
            field = complex(result.field)
            field_derivative = complex(result.central_field[0])
            adjoint = float(result.adjoint)
            records.append({
                "n_solve": n_solve,
                "n_quadrature": n_quad,
                "field": complex_json(result.field),
                "field_derivative": complex_json(result.central_field[0]),
                "objective_derivative": adjoint,
                "objective_central": float(result.central_objective[0]),
                "field_error": abs(field - ref_field),
                "field_derivative_error": abs(field_derivative - ref_field_derivative),
                "objective_derivative_error": abs(adjoint - ref_adjoint),
            })
            progress.update()
    return {
        "settings": {
            "step": step,
            "reference_order": reference_order,
            "solve_orders": list(solve_orders),
            "quadrature_orders": list(quadrature_orders),
        },
        "reference": {
            "field": complex_json(reference.field),
            "field_derivative": complex_json(reference.central_field[0]),
            "objective_derivative": ref_adjoint,
            "objective_central": float(reference.central_objective[0]),
        },
        "records": records,
    }
