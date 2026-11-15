from typing import Any

from array_api.latest import Array, ArrayNamespace
from array_api_compat import array_namespace


def off_diagonal_indices(n: int, /, *, xp: ArrayNamespace, device: Any) -> tuple[Array, Array]:
    """Return row and column indices for the off-diagonal entries of an n by n matrix."""
    if n < 1:
        raise ValueError("n must be positive")
    indices = xp.arange(n * (n - 1), device=device, dtype=xp.int64)
    if n == 1:
        return indices, indices
    rows = indices // (n - 1)
    columns = indices % (n - 1)
    return rows, columns + xp.astype(columns >= rows, xp.int64)


def off_diagonal_matrix(values: Array, n: int, /) -> Array:
    """Restore flattened off-diagonal entries, filling the diagonal with zero."""
    if n < 1 or values.ndim == 0 or values.shape[-1] != n * (n - 1):
        raise ValueError("values must contain n * (n - 1) off-diagonal entries")
    xp = array_namespace(values)
    indices = xp.arange(n, device=values.device, dtype=xp.int64)
    rows, columns = indices[:, None], indices[None, :]
    lookup = xp.where(
        rows == columns, n * (n - 1), rows * (n - 1) + columns - xp.astype(columns > rows, xp.int64)
    )
    padded = xp.concat(
        [values, xp.zeros((*values.shape[:-1], 1), dtype=values.dtype, device=values.device)],
        axis=-1,
    )
    return xp.reshape(
        xp.take(padded, xp.reshape(lookup, (-1,)), axis=-1), (*values.shape[:-1], n, n)
    )
