from __future__ import annotations

from array_api._2024_12 import Array
from array_api_compat import array_namespace, is_torch_array
from scipy.special import hankel1, jvp, yv

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    triton = None

if triton is not None:

    @triton.jit
    def _bessel_kernel(  # type: ignore[no-untyped-def]
        x, result, size, ORDER: tl.constexpr, KIND: tl.constexpr, BLOCK: tl.constexpr
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + index, mask=index < size, other=1)
        if KIND == 0:
            if ORDER == 0:
                evaluated = libdevice.j0(value)
            elif ORDER == 1:
                evaluated = libdevice.j1(value)
            else:
                evaluated = tl.where(value == 0, 0.0, libdevice.jn(2, value))
        elif ORDER == 0:
            evaluated = libdevice.y0(value)
        elif ORDER == 1:
            evaluated = libdevice.y1(value)
        else:
            evaluated = libdevice.yn(2, value)
        tl.store(result + index, evaluated, mask=index < size)


def _torch_bessel(order: int, x: Array, /, *, kind: str) -> Array:
    if triton is None:
        raise RuntimeError("CUDA Bessel functions require the 'cuda' extra: uv sync --extra cuda")
    xp = array_namespace(x)
    contiguous = x.contiguous()
    result = xp.empty_like(contiguous)
    size = contiguous.numel()
    if size:
        with xp.cuda.device(x.device):
            _bessel_kernel[triton.cdiv(size, 256),](
                contiguous, result, size, order, 0 if kind == "j" else 1, 256
            )
    return result


def scipy_jv(order: int, x: Array, n: int = 0, /) -> Array:
    """
    Wrapper around scipy.special.jv that handles derivatives.

    Parameters
    ----------
    order : int
        Order of the Bessel function.
    x : Array
        Input array of shape (...,).
    n : int
        Order of derivative (default 0 for function value).

    Returns
    -------
    Array
        Bessel function J_order(x) or its n-th derivative of shape (...,).

    """
    xp = array_namespace(x)
    if is_torch_array(x) and x.device.type == "cuda" and xp.isdtype(x.dtype, "real floating"):
        if n == 0 and order in (0, 1, 2):
            return _torch_bessel(order, x, kind="j")
        if n == 1 and order == 0:
            return -_torch_bessel(1, x, kind="j")
        if n == 1 and order == 1:
            return (_torch_bessel(0, x, kind="j") - _torch_bessel(2, x, kind="j")) / 2
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = jvp(order, x_cpu, n)
    return xp.asarray(result_cpu, device=x.device, dtype=x.dtype)


def scipy_yv(order: int, x: Array, /) -> Array:
    """
    Wrapper around scipy.special.yv that handles dtype conversion.

    Parameters
    ----------
    order : int
        Order of the Bessel function.
    x : Array
        Input array of shape (...,).

    Returns
    -------
    Array
        Bessel function Y_order(x) of shape (...,).

    """
    xp = array_namespace(x)
    if (
        is_torch_array(x)
        and x.device.type == "cuda"
        and xp.isdtype(x.dtype, "real floating")
        and order in (0, 1, 2)
    ):
        return _torch_bessel(order, x, kind="y")
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = yv(order, x_cpu)
    return xp.asarray(result_cpu, device=x.device, dtype=x.dtype)


def scipy_hankel1(order: int, x: Array, /) -> Array:
    """
    Wrapper around scipy.special.hankel1 that handles complex dtype conversion.

    Parameters
    ----------
    order : int
        Order of the Hankel function.
    x : Array
        Input array of shape (...,).

    Returns
    -------
    Array
        Hankel function H^(1)_order(x) of shape (...,).

    """
    xp = array_namespace(x)
    promoted_dtype = xp.promote_types(x.dtype, xp.complex128)
    if (
        is_torch_array(x)
        and x.device.type == "cuda"
        and xp.isdtype(x.dtype, "real floating")
        and order in (0, 1, 2)
    ):
        positive = xp.abs(x)
        result = xp.astype(
            _torch_bessel(order, positive, kind="j"), promoted_dtype
        ) + 1j * xp.astype(_torch_bessel(order, positive, kind="y"), promoted_dtype)
        return xp.where(x < 0, (-1) ** (order + 1) * xp.conj(result), result)
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = hankel1(order, x_cpu)
    return xp.asarray(result_cpu, device=x.device, dtype=promoted_dtype)
