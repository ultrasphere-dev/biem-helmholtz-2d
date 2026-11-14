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
    n_solve: Sequence[int],
    n_quad: Sequence[int],
    step: float,
    n_quad_ref: int | None = None,
) -> dict[str, Any]:
    r"""
    Compare field and shape-sensitivity results at different discretizations.

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
    n_solve, n_quad : Sequence[int]
        Positive discretization parameters for the Nyström solves and integration.
        Each parameter $n$ uses $2n-1$ nodes; all combinations are evaluated.
    n_quad_ref : int or None
        Parameter used for both reference stages, greater than all study parameters.
        None uses 1.5 times their maximum, rounded up.
    step : float
        Positive, finite central-difference step.

    Returns
    -------
    dict[str, Any]
        JSON-compatible mapping with ``settings``, ``reference``, and one
        ``records`` entry per solve/integration parameter pair. Complex numbers are
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
    ...     n_solve=[3], n_quad=[3], n_quad_ref=4, step=1e-5,
    ... )
    >>> [(row['n_solve'], row['n_quad']) for row in study['records']]
    [(3, 3)]

    """
    check_shapes("D,D", point, (2,), names="point,coordinate_dimension")
    if point.shape != (2,) or any(value.ndim != 0 for value in (target, k, alpha, eta)):
        raise ValueError("point must have shape (2,) and wave parameters and target must be scalar")
    array_namespace(point, target, k, alpha, eta)
    if not shape or len(shape) != len(h_shape):
        raise ValueError("shape and h_shape must have the same nonzero length")
    if not n_solve or not n_quad:
        raise ValueError("n_solve and n_quad must be nonempty")
    n_max = max(*n_solve, *n_quad)
    if n_quad_ref is None:
        n_quad_ref = (3 * n_max + 1) // 2
    if min(*n_solve, *n_quad) < 1 or n_quad_ref <= n_max:
        raise ValueError("n_quad_ref must exceed all positive study parameters")
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
            n_quad=n_quad,
            steps=[step],
        )

    def complex_json(value: Array, /) -> dict[str, float]:
        number = complex(value)
        return {"real": number.real, "imag": number.imag}

    configurations = [(n, q) for n in n_solve for q in n_quad]
    with tqdm(total=len(configurations) + 1, desc="Quadrature accuracy", unit="case") as pbar:
        pbar.set_postfix_str(f"reference n={n_quad_ref}")
        reference = evaluate(n_quad_ref, n_quad_ref)
        pbar.update()
        ref_field = complex(reference.field)
        ref_field_derivative = complex(reference.central_field[0])
        ref_adjoint = float(reference.adjoint)
        records = []
        for solve, quad in configurations:
            pbar.set_postfix_str(f"solve={solve}, quadrature={quad}")
            result = evaluate(solve, quad)
            field = complex(result.field)
            field_derivative = complex(result.central_field[0])
            adjoint = float(result.adjoint)
            records.append({
                "n_solve": solve,
                "n_quad": quad,
                "field": complex_json(result.field),
                "field_derivative": complex_json(result.central_field[0]),
                "objective_derivative": adjoint,
                "objective_central": float(result.central_objective[0]),
                "field_error": abs(field - ref_field),
                "field_derivative_error": abs(field_derivative - ref_field_derivative),
                "objective_derivative_error": abs(adjoint - ref_adjoint),
            })
            pbar.update()
    return {
        "settings": {
            "step": step,
            "n_quad_ref": n_quad_ref,
            "n_solve": list(n_solve),
            "n_quad": list(n_quad),
        },
        "reference": {
            "field": complex_json(reference.field),
            "field_derivative": complex_json(reference.central_field[0]),
            "objective_derivative": ref_adjoint,
            "objective_central": float(reference.central_objective[0]),
        },
        "records": records,
    }
