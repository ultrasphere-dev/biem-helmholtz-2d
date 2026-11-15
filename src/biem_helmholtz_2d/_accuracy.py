"""Convergence studies for independently chosen numerical quantities."""

from collections.abc import Callable, Sequence

from array_api.latest import Array
from array_api_compat import array_namespace


def accuracy(
    evaluate: Callable[[int, int], Array],
    /,
    *,
    n_solve: Sequence[int],
    n_quad: Sequence[int],
    n_reference: int | None = None,
) -> tuple[Array, Array]:
    """Evaluate one quantity on the solve/integration grid and at a reference order."""
    if not n_solve or not n_quad or min(*n_solve, *n_quad) < 1:
        raise ValueError("n_solve and n_quad must contain positive parameters")
    n_max = max(*n_solve, *n_quad)
    if n_reference is None:
        n_reference = (3 * n_max + 1) // 2
    if n_reference <= n_max:
        raise ValueError("n_reference must exceed all study parameters")
    reference = evaluate(n_reference, n_reference)
    xp = array_namespace(reference)
    values = xp.stack([xp.stack([evaluate(solve, quad) for quad in n_quad]) for solve in n_solve])
    if values.shape[2:] != reference.shape:
        raise ValueError("quantity shapes must be the same at every discretization")
    return reference, values
