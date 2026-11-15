from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes


def kernel(
    direction: Array,
    /,
    *,
    shape_x: Array,
    shape_dx: Array,
    h: Array,
    dh: Array,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    r"""Shape derivative of the far-field kernel in direction $h$."""
    check_shapes(
        "*AD,*BD,*CD,*ED,*FD,*K,*L,*M,D",
        direction,
        shape_x,
        shape_dx,
        h,
        dh,
        k,
        alpha,
        eta,
        (2,),
    )
    if any(value.shape[-1] != 2 for value in (direction, shape_x, shape_dx, h, dh)):
        raise ValueError("direction, boundary samples and perturbations must have two coordinates")
    xp = array_namespace(direction, shape_x, shape_dx, h, dh, k, alpha, eta)
    direction = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)
    normal = xp.stack([shape_dx[..., 1], -shape_dx[..., 0]], axis=-1)
    dnormal = xp.stack([dh[..., 1], -dh[..., 0]], axis=-1)
    jacobian = xp.linalg.vector_norm(shape_dx, axis=-1)
    djacobian = xp.sum(shape_dx * dh, axis=-1) / jacobian
    phase = xp.exp(-1j * k * xp.sum(direction * shape_x, axis=-1))
    coefficient = (1 - 1j) / xp.sqrt(16 * xp.pi * k)
    value = alpha * k * xp.sum(direction * normal, axis=-1) + eta * jacobian
    derivative = alpha * k * xp.sum(direction * dnormal, axis=-1) + eta * djacobian
    return coefficient * phase * (derivative - 1j * k * xp.sum(direction * h, axis=-1) * value)
