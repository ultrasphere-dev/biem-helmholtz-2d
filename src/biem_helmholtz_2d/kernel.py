from array_api._2024_12 import Array
from array_api_compat import array_namespace
from scipy.special import hankel1

from .bie import Kernel, KernelResultImpl
from .shape import Shape


def _is_diagonal(x: Array, y: Array, eps: float = 1e-6) -> Array:
    xp = array_namespace(x, y)
    absFmod = xp.abs(xp.fmod(x - y, 2 * xp.pi))
    return (absFmod < eps) | (absFmod > 2 * xp.pi - eps)


def double_layer(k: float, shapex: Shape, shapey: Shape) -> Kernel:
    def inner(x: Array, y: Array, /) -> KernelResultImpl:
        xp = array_namespace(x, y)
        r = xp.linalg.vector_norm(shapex.x(x) - shapey.x(y), axis=-1)
        n_dot_diff = (
            shapey.dx(y)[..., 1] * (shapey.x(y) - shapex.x(x))[..., 0]
            - shapey.dx(y)[..., 0] * (shapey.x(y) - shapex.x(x))[..., 1]
        )
        L = 1j * k / 2.0 * n_dot_diff * hankel1(1, k * r) / r
        L1_nondiag = k / (2.0 * xp.pi) * n_dot_diff * bessel_j1(k * r) / r
        L1_diag = 0.0
        L1 = xp.where(_is_diagonal(x, y), L1_diag, L1_nondiag)
        L2_diag = (
            shapex.dx(x)[..., 0] * shapex.ddx(x)[..., 1]
            - shapex.dx(x)[..., 1] * shapex.ddx(x)[..., 0]
        ) / (2.0 * xp.pi * shapex.jacobian(x) ** 2)
        L2_nondiag = get_kernel_2(L, L1_nondiag, x, y)
        L2 = xp.where(_is_diagonal(x, y), L2_diag, L2_nondiag)
        return KernelResultImpl(
            analytic=xp.where(shapex == shapey, L1, L),
            singular_log=xp.where(
                shapex == shapey, L2, xp.asarray(0.0, dtype=x.dtype, device=x.device)
            ),
            singular_cauchy=xp.asarray(0.0, dtype=x.dtype, device=x.device),
        )

    return inner


def single_layer(k: float, shapex: Shape, shapey: Shape) -> Kernel:
    def inner(x: Array, y: Array, /) -> KernelResultImpl:
        xp = array_namespace(x, y)
        r = xp.linalg.vector_norm(shapex.x(x) - shapey.x(y), axis=-1)
        M = 1j / 2.0 * hankel1(0, k * r) * shapey.jacobian(y)
        M1 = -1 / (2.0 * xp.pi) * bessel_j0(k * r) * shapey.jacobian(y)
        M2_diag = shapex.jacobian(x) * (
            1j / 2.0
            - 0.5772156649015329 / xp.pi
            - 1.0 / (2.0 * xp.pi) * xp.log((k * shapex.jacobian(x)) ** 2 / 4.0)
        )
        M2_nondiag = get_kernel_2(M, M1_diag_nondiag, x, y)
        M2 = xp.where(_is_diagonal(x, y), M2_diag, M2_nondiag)
        return KernelResultImpl(
            analytic=xp.where(shapex == shapey, M1, M),
            singular_log=xp.where(
                shapex == shapey, M2, xp.asarray(0.0, dtype=x.dtype, device=x.device)
            ),
            singular_cauchy=xp.asarray(0.0, dtype=x.dtype, device=x.device),
        )

    return inner


def get_kernel_2(kernel_original: Array, kernel1: Array, x: Array, y: Array) -> Array:
    xp = array_namespace(x, y)
    return kernel_original - kernel1 * xp.log(4 * xp.sin((x - y) / 2.0) ** 2)
