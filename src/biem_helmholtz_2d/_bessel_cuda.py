"""Real Bessel functions on CUDA via NVIDIA's double-precision libdevice."""

import triton
import triton.language as tl
from array_api.latest import Array
from array_api_compat import array_namespace
from triton.language.extra.cuda import libdevice


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


def bessel_cuda(order: int, x: Array, /, *, kind: str) -> Array:
    """Evaluate J or Y of order 0, 1, or 2 on a CUDA tensor."""
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
