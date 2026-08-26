from __future__ import annotations

from collections.abc import Callable

from array_api.latest import Array
from array_api_compat import array_namespace


def plane_wave(k: Array, direction: Array, /) -> Callable[[Array], Array]:
    r"""
    Return a plane wave $e^{i k \hat{d} \cdot x}$ propagating in direction $d$.

    Parameters
    ----------
    k : Array
        Wave number $k$ of shape (...,).
    direction : Array
        Direction vector $d$ of shape (..., 2). Normalized internally.

    Returns
    -------
    Callable[[Array], Array]
        Plane wave function of (..., 2) -> (...,).

    Examples
    --------
    >>> import numpy as np
    >>> k = np.asarray(1)
    >>> pw = plane_wave(k, np.asarray([1, 0]))
    >>> x = np.asarray([[0, 0], [1, 2]])
    >>> pw(x)
    array([1.        +0.j        , 0.54030231+0.84147098j])

    """
    xp = array_namespace(k, direction)
    d_hat = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)

    def _pw(x: Array) -> Array:
        xp = array_namespace(k, x, direction)
        return xp.exp(1j * k * xp.sum(d_hat * x, axis=-1))

    return _pw


def plane_wave_grad(k: Array, direction: Array, /) -> Callable[[Array], Array]:
    r"""
    Return the gradient of the plane wave $e^{i k \hat{d} \cdot x}$.

    $$
    \nabla_x e^{i k \hat{d} \cdot x} = i k \hat{d} \, e^{i k \hat{d} \cdot x}
    $$

    Parameters
    ----------
    k : Array
        Wave number $k$ of shape (...,).
    direction : Array
        Direction vector $d$ of shape (..., 2). Normalized internally.

    Returns
    -------
    Callable[[Array], Array]
        Gradient function of (..., 2) -> (..., 2).

    Examples
    --------
    >>> import numpy as np
    >>> k = np.asarray(1)
    >>> pwg = plane_wave_grad(k, np.asarray([1, 0]))
    >>> x = np.asarray([[0, 0], [1, 2]])
    >>> pwg(x)
    array([[ 0.        +1.j        ,  0.        +0.j        ],
           [-0.84147098+0.54030231j,  0.        +0.j        ]])
    >>> eps = 1e-5
    >>> pw = plane_wave(k, np.asarray([1, 0]))
    >>> pw_plus = pw(x[:, None, :] + eps * np.asarray([[1, 0], [0, 1]]))
    >>> pw_minus = pw(x[:, None, :] - eps * np.asarray([[1, 0], [0, 1]]))
    >>> pwg_cd = (pw_plus - pw_minus) / (2 * eps)
    >>> pwg_cd
    array([[ 0.        +1.j        ,  0.        +0.j        ],
           [-0.84147098+0.54030231j,  0.        +0.j        ]])
    >>> np.allclose(pwg(x), pwg_cd)
    True

    """
    xp = array_namespace(k, direction)
    d_hat = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)

    def _pwg(x: Array) -> Array:
        xp = array_namespace(k, x, direction)
        pw = xp.exp(1j * k * xp.sum(d_hat * x, axis=-1))
        return 1j * k * d_hat * pw[..., None]

    return _pwg
