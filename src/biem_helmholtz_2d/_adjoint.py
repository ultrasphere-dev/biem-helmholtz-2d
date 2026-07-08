from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import (
    NystromInterpolant,
    QuadratureType,
    Shape,
    log_cot_power_quadrature,
    nystrom,
    trapezoidal_quadrature,
)

from ._potential import dlp_kernel_split, slp_kernel_split
from ._potential_shape_derivative import (
    dlp_shape_derivative_split,
    slp_shape_derivative_split,
)


def _solve_adjoint(
    *,
    k: Array,
    shape: Shape,
    alpha: Array,
    eta: Array,
    n: int,
    rhs: Callable[[Array], Array],
    eps: float = 0,
) -> NystromInterpolant:
    r"""
    Solve $A^* \psi = \mathrm{rhs}$ under the $L^2$ sesquilinear form.

    $A^* = \frac\alpha2 I + \overline{\mathcal D(\tau,t)}
    + i\eta\,\overline{\mathcal S(\tau,t)}$.
    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device

    def k_log(t: Array, tau: Array) -> Array:
        dlp_log, _ = dlp_kernel_split(
            t=tau, tau=t, k=k[..., None, None], x=shape.x, dx=shape.dx, ddx=shape.ddx, eps=eps
        )
        slp_log, _ = slp_kernel_split(
            t=tau, tau=t, k=k[..., None, None], x=shape.x, dx=shape.dx, eps=eps
        )
        return (alpha * xp.conj(dlp_log) + 1j * eta * xp.conj(slp_log))[..., None, None]

    def k_cont(t: Array, tau: Array) -> Array:
        _, dlp_cont = dlp_kernel_split(
            t=tau, tau=t, k=k[..., None, None], x=shape.x, dx=shape.dx, ddx=shape.ddx, eps=eps
        )
        _, slp_cont = slp_kernel_split(
            t=tau, tau=t, k=k[..., None, None], x=shape.x, dx=shape.dx, eps=eps
        )
        return (alpha * xp.conj(dlp_cont) + 1j * eta * xp.conj(slp_cont))[..., None, None]

    def a(t: Array) -> Array:
        xp = array_namespace(t)
        return xp.ones_like(t)[..., None] * (alpha / 2)

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): k_cont,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }

    return nystrom(a, kernels, rhs, n=n, xp=xp, device=device, dtype=dtype)


def objective_derivative(
    *,
    k: Array,
    shape: Shape,
    alpha: Array,
    eta: Array,
    n: int,
    phi: NystromInterpolant,
    grad_phi_j: Callable[[Array], Array],
    dr_j: Array,
    dr_g: Array,
    h_shape: Shape,
    eps: float = 0,
) -> Array:
    r"""
    Shape derivative $D_r J(r)[h]$ under the $L^2$ sesquilinear form.

    $$
    \begin{aligned}
    (\tfrac\alpha2 I + \mathcal D - i\eta\,\mathcal S)\phi_r &= g_r,\\[2mm]
    (\tfrac\alpha2 I + \mathcal D^* + i\eta\,\mathcal S^*)\psi_r
        &= -\operatorname{grad}_\phi j,\\[2mm]
    D_r J(r)[h] &= D_r j(r,\phi_r)[h]
        + \operatorname{Re}\bigl\langle
            \psi_r,\;
            D_r\mathcal D[h]\phi_r
            - i\eta\,D_r\mathcal S[h]\phi_r
            - D_r g[h]
          \bigr\rangle .
    \end{aligned}
    $$

    Parameters
    ----------
    k : Array
        Wave number of shape (...,).
    shape : Shape
        Boundary parametrisation of (...,) -> (..., 2).
    alpha : Array
        Coupling parameter $\alpha$ of shape (...,).
    eta : Array
        Coupling parameter $\eta$ of shape (...,).
    n : int
        Maximum order minus $1$, number of quadrature nodes $2n-1$.
    phi : NystromInterpolant
        Forward density $\phi_r$.
    grad_phi_j : Callable[[Array], Array]
        Riesz representation w.r.t. $\langle f,g\rangle=\int f\overline g$.
        Signature (...,) -> (...,).
    dr_j : Array
        Value $D_r j(r,\phi_r)[h]$ (scalar).
    dr_g : Array
        Shape derivative $D_r g[h]$ at quadrature nodes of shape ``(Q,)``.
    h_shape : Shape
        Shape perturbation $h$, providing $h(t)$, $h'(t)$, $h''(t)$
        via ``h_shape.x``, ``h_shape.dx``, ``h_shape.ddx``
        of (...,) -> (..., 2).
    eps : float
        Tolerance for switching to diagonal limit in singular kernel evaluations.

    Returns
    -------
    Array
        Shape derivative $D_r J(r)[h]$ as a scalar of shape (...,).

    Examples
    --------
    >>> import numpy as np
    >>> from ie_circle import CircleShape
    >>> from biem_helmholtz_2d._acoustic import scattering_dirichlet
    >>> xp = np; k = xp.asarray(1.0); rho = 1.0
    >>> shape = CircleShape(rho)
    >>> a = xp.asarray(1.0); e = xp.asarray(0.0)
    >>> def inc(x): return xp.exp(1j * k * x[..., 0])
    >>> phi = scattering_dirichlet(k=k, shape=shape, incident_field=inc, alpha=a, eta=e, n=8)
    >>> class _ZeroShape:
    ...     def x(self, t, /): return xp.zeros((*t.shape, 2))
    ...     def dx(self, t, /): return xp.zeros((*t.shape, 2))
    ...     def ddx(self, t, /): return xp.zeros((*t.shape, 2))
    >>> from ie_circle import trapezoidal_quadrature
    >>> t, _ = trapezoidal_quadrature(8, xp=xp, device='cpu', dtype=xp.float64)
    >>> dr_g = xp.zeros(2 * 8 - 1)
    >>> def gj(t): return xp.zeros_like(t)
    >>> objective_derivative(
    ...     k=k, shape=shape, alpha=a, eta=e, n=8,
    ...     phi=phi, grad_phi_j=gj, dr_j=xp.asarray(0.0),
    ...     dr_g=dr_g, h_shape=_ZeroShape(),
    ... )
    np.float64(0.0)

    """
    from array_api_shape_check import check_shapes

    xp = array_namespace(k, alpha, eta, dr_j, dr_g)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device
    t, w_trap = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    n_nodes = 2 * n - 1
    w_trap_val = w_trap[0]

    # dr_g may be complex (depends on incident field gradient);
    # only check its first dimension matches
    check_shapes("Q", dr_g, names="dr_g")

    h = h_shape.x
    dh = h_shape.dx
    ddh = h_shape.ddx

    psi = _solve_adjoint(
        k=k,
        shape=shape,
        alpha=alpha,
        eta=eta,
        n=n,
        rhs=lambda t_in: -grad_phi_j(t_in)[..., None],
        eps=eps,
    )
    psi_t = psi(t)
    phi_t = phi(t)

    _, w_log_cot_raw = log_cot_power_quadrature(n, 0, xp=xp, device=device, dtype=dtype)
    roll_idx = (
        -xp.arange(n_nodes, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_nodes, device=device, dtype=xp.int64)[None, :]
    ) % n_nodes
    w_log_cot = w_log_cot_raw[roll_idx]

    t_row = t[:, None]
    tau_col = t[None, :]

    slp_log, slp_cont = slp_shape_derivative_split(
        t=t_row,
        tau=tau_col,
        k=k[..., None, None],
        x=shape.x,
        dx=shape.dx,
        h=h,
        dh=dh,
        eps=eps,
    )
    dlp_log, dlp_cont = dlp_shape_derivative_split(
        t=t_row,
        tau=tau_col,
        k=k[..., None, None],
        x=shape.x,
        dx=shape.dx,
        ddx=shape.ddx,
        h=h,
        dh=dh,
        ddh=ddh,
        eps=eps,
    )

    k_log_sd = alpha * dlp_log - 1j * eta * slp_log
    k_cont_sd = alpha * dlp_cont - 1j * eta * slp_cont

    shape_deriv_phi = xp.sum(k_log_sd * phi_t * w_log_cot, axis=1) + xp.sum(
        k_cont_sd * phi_t * w_trap_val, axis=1
    )

    # sesquilinear inner product
    inner = xp.sum(psi_t * xp.conj(shape_deriv_phi - dr_g) * w_trap_val)

    return dr_j + xp.real(inner)
