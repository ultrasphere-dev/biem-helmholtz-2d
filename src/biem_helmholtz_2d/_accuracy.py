"""Convergence studies for independently chosen numerical quantities."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from array_api.latest import Array
from array_api_compat import array_namespace


def accuracy(
    evaluate: Callable[[int, int], Mapping[str, Array]],
    /,
    *,
    n_solve: Sequence[int],
    n_quad: Sequence[int],
    n_quad_ref: int | None = None,
) -> dict[str, Any]:
    """
    Compare quantities returned by ``evaluate(n_solve, n_quad)`` with a numerical reference.

    Each parameter $n$ uses $2n-1$ nodes. ``n_quad_ref`` is used for both
    reference stages; by default it is 1.5 times the largest study parameter,
    rounded up. Errors are absolute differences, not rigorous bounds.
    """
    if not n_solve or not n_quad:
        raise ValueError("n_solve and n_quad must be nonempty")
    n_max = max(*n_solve, *n_quad)
    if n_quad_ref is None:
        n_quad_ref = (3 * n_max + 1) // 2
    if min(*n_solve, *n_quad) < 1 or n_quad_ref <= n_max:
        raise ValueError("n_quad_ref must exceed all positive study parameters")

    def serialize(value: Array, /) -> Any:
        xp = array_namespace(value)
        value = xp.asarray(value, device="cpu")
        if xp.isdtype(value.dtype, "complex floating"):
            return {"real": serialize(xp.real(value)), "imag": serialize(xp.imag(value))}
        if value.ndim == 0:
            return float(value)
        return [serialize(value[i]) for i in range(value.shape[0])]

    reference = evaluate(n_quad_ref, n_quad_ref)
    names = reference.keys()
    if {"n_solve", "n_quad"} & names or any(f"{name}_error" in names for name in names):
        raise ValueError("quantity names must not conflict with record or error names")
    records = []
    for solve in n_solve:
        for quad in n_quad:
            result = evaluate(solve, quad)
            if result.keys() != reference.keys():
                raise ValueError(
                    "evaluate must return the same quantities for every discretization"
                )
            if any(value.shape != reference[name].shape for name, value in result.items()):
                raise ValueError("quantity shapes must be the same for every discretization")
            records.append({
                "n_solve": solve,
                "n_quad": quad,
                **{name: serialize(value) for name, value in result.items()},
                **{
                    f"{name}_error": serialize(
                        array_namespace(value, reference[name]).abs(value - reference[name])
                    )
                    for name, value in result.items()
                },
            })
    return {
        "settings": {"n_quad_ref": n_quad_ref, "n_solve": list(n_solve), "n_quad": list(n_quad)},
        "reference": {name: serialize(value) for name, value in reference.items()},
        "records": records,
    }
