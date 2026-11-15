from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes


def kernel(
    direction: Array,
    /,
    *,
    shape_x: Array,
    shape_dx: Array,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    r"""Far-field kernel of $\alpha\mathrm{DL}-i\eta\mathrm{SL}$."""
    check_shapes("*AD,*BD,*CD,*K,*L,*M,D", direction, shape_x, shape_dx, k, alpha, eta, (2,))
    if any(value.shape[-1] != 2 for value in (direction, shape_x, shape_dx)):
        raise ValueError("direction and boundary samples must have two coordinates")
    xp = array_namespace(direction, shape_x, shape_dx, k, alpha, eta)
    direction = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)
    normal = xp.stack([shape_dx[..., 1], -shape_dx[..., 0]], axis=-1)
    jacobian = xp.linalg.vector_norm(shape_dx, axis=-1)
    phase = xp.exp(-1j * k * xp.sum(direction * shape_x, axis=-1))
    coefficient = (1 - 1j) / xp.sqrt(16 * xp.pi * k)
    return coefficient * phase * (alpha * k * xp.sum(direction * normal, axis=-1) + eta * jacobian)
