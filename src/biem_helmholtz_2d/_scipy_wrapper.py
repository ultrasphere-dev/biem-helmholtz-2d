from __future__ import annotations

from array_api._2024_12 import Array
from array_api_compat import array_namespace


def scipy_jv(order: int, x: Array, n: int = 0, /, *, device: str, dtype: Array.dtype) -> Array:
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
    device : str
        Device for output array.
    dtype : Array.dtype
        Output dtype (real for real inputs).
    
    Returns
    -------
    Array
        Bessel function J_order(x) or its n-th derivative of shape (...,).
    """
    from scipy.special import jvp
    
    xp = array_namespace(x)
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = jvp(order, x_cpu, n)
    return xp.asarray(result_cpu, device=device, dtype=dtype)


def scipy_yv(order: int, x: Array, /, *, device: str, dtype: Array.dtype) -> Array:
    """
    Wrapper around scipy.special.yv that handles dtype conversion.
    
    Parameters
    ----------
    order : int
        Order of the Bessel function.
    x : Array
        Input array of shape (...,).
    device : str
        Device for output array.
    dtype : Array.dtype
        Output dtype (real for real inputs).
    
    Returns
    -------
    Array
        Bessel function Y_order(x) of shape (...,).
    """
    from scipy.special import yv
    
    xp = array_namespace(x)
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = yv(order, x_cpu)
    return xp.asarray(result_cpu, device=device, dtype=dtype)


def scipy_hankel1(order: int, x: Array, /, *, device: str, dtype: Array.dtype) -> Array:
    """
    Wrapper around scipy.special.hankel1 that handles complex dtype conversion.
    
    Parameters
    ----------
    order : int
        Order of the Hankel function.
    x : Array
        Input array of shape (...,).
    device : str
        Device for output array.
    dtype : Array.dtype
        Output dtype. Will be promoted to complex.
    
    Returns
    -------
    Array
        Hankel function H^(1)_order(x) of shape (...,).
    """
    from scipy.special import hankel1
    
    xp = array_namespace(x)
    x_cpu = xp.asarray(x, device="cpu")
    result_cpu = hankel1(order, x_cpu)
    
    # Promote dtype to complex to avoid casting complex to float
    promoted_dtype = xp.promote_types(dtype, xp.complex128)
    return xp.asarray(result_cpu, device=device, dtype=promoted_dtype)

