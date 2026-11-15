from __future__ import annotations

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import (
    NystromInterpolant,
    QuadratureType,
    Shape,
    log_cot_power_quadrature,
    nystrom,
    trapezoidal_quadrature,
)

from ._acoustic import _coupled_kernels
from ._pairs import off_diagonal_indices, off_diagonal_matrix
from ._potential_inner_derivative import dlp_shape_derivative_kernel, slp_shape_derivative_kernel
from ._potential_shape_derivative import dlp_shape_derivative_split, slp_shape_derivative_split


def solve_adjoint(
    *,
    k: Array,
    shape: Shape,
    alpha: Array,
    eta: Array,
    n: int,
    grad_phi_j: Array,
) -> NystromInterpolant:
    r"""
    Solve $A^*\psi=-\operatorname{grad}_\phi J$ for the coupled system.

    The conjugate-transposed blocks include the coupling parameters. For one
    boundary, $A^* = \overline\alpha I/2 + \overline\alpha D^*
    + i\overline\eta S^*$.

    Parameters
    ----------
    k, alpha, eta : Array
        Wave number and coupling parameters of broadcastable shape (...(B),).
    shape : Shape
        Disjoint boundaries of (...) -> (..., C, 2).
    n : int
        Quadrature parameter, with $Q=2n-1$ nodes.
    grad_phi_j : Array
        Density gradient at the solve nodes of shape (Q, ...(B), C),
        where $C$ is the number of boundaries.

    Returns
    -------
    NystromInterpolant
        Adjoint density of (...) -> (..., ...(B), C).

    """
    xp = array_namespace(k, alpha, eta, grad_phi_j)
    dtype = xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    count = shape.x(xp.asarray(0, dtype=dtype, device=k.device)).shape[-2]
    if count == 0:
        raise ValueError("At least one boundary is required")
    check_shapes("Q*BC,QC", grad_phi_j, (2 * n - 1, count), names="grad_phi_j,expected")
    if grad_phi_j.shape[0] != 2 * n - 1 or grad_phi_j.shape[-1] != count:
        raise ValueError("grad_phi_j must have one value per parameter and scatterer")
    k, alpha, eta, _ = xp.broadcast_arrays(k, alpha, eta, grad_phi_j[0, ..., 0])
    batch_shape = k.shape
    missing = k.ndim - (grad_phi_j.ndim - 2)
    grad_phi_j = grad_phi_j[(slice(None),) + (None,) * missing + (...,)]
    forward = _coupled_kernels(shape=shape, k=k, alpha=alpha, eta=eta)

    def k_log(t: Array, tau: Array) -> Array:
        return xp.conj(xp.matrix_transpose(forward[QuadratureType.LOG_COT_POWER, 0](tau, t)))

    def k_cont(t: Array, tau: Array) -> Array:
        return xp.conj(xp.matrix_transpose(forward[QuadratureType.NO_SINGULARITY, 0](tau, t)))

    def coefficient(t: Array) -> Array:
        value = xp.broadcast_to(xp.conj(alpha) / 2, (*t.shape, *batch_shape))
        return value[..., None] * xp.ones(count, device=t.device, dtype=t.dtype)

    def rhs(t: Array) -> Array:
        return -xp.broadcast_to(grad_phi_j, (*t.shape, *batch_shape, count))

    return nystrom(
        coefficient,
        {(QuadratureType.NO_SINGULARITY, 0): k_cont, (QuadratureType.LOG_COT_POWER, 0): k_log},
        rhs,
        n=n,
        xp=xp,
        device=k.device,
        dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta)),
    )


def objective_derivative(
    *,
    k: Array,
    shape: Shape,
    alpha: Array,
    eta: Array,
    n: int,
    phi: NystromInterpolant,
    adjoint: NystromInterpolant,
    dr_j: Array,
    dr_g: Array,
    h_shape: Shape,
    eps: float = 0,
) -> Array:
    r"""
    Shape derivative $D_r J(r)[h]$ under the $L^2$ sesquilinear form.

    $$
    \begin{aligned}
    (\tfrac\alpha2 I + \alpha\mathcal D - i\eta\,\mathcal S)\phi_r &= g_r,\\[2mm]
    (\tfrac{\overline\alpha}{2} I + \overline\alpha\mathcal D^*
        + i\overline\eta\,\mathcal S^*)\psi_r &= -\operatorname{grad}_\phi j,\\[2mm]
    D_r J(r)[h] &= D_r j(r,\phi_r)[h]
        + \operatorname{Re}\bigl\langle
            \psi_r,\;
            \alpha D_r\mathcal D[h]\phi_r
            - i\eta\,D_r\mathcal S[h]\phi_r
            - D_r g[h]
          \bigr\rangle .
    \end{aligned}
    $$

    Parameters
    ----------
    k : Array
        Wave number of shape (...(B),).
    shape : Shape
        Disjoint boundary parametrisations of (...) -> (..., C, 2).
    alpha : Array
        Coupling parameter $\alpha$ of shape (...(B),).
    eta : Array
        Coupling parameter $\eta$ of shape (...(B),).
    n : int
        Integration parameter, with $2n-1$ nodes. It may differ from the
        Nyström parameter used to compute ``phi`` and ``adjoint``.
    phi : NystromInterpolant
        Forward density $\phi_r$ of (...) -> (..., ...(B), C).
    adjoint : NystromInterpolant
        Solution returned by :func:`solve_adjoint` of (...) -> (..., ...(B), C).
        Reuse it for all perturbations of the same objective and geometry.
    dr_j : Array
        Value $D_r j(r,\phi_r)[h]$ of shape (...(B),).
    dr_g : Array
        Shape derivative $D_r g[h]$ at quadrature nodes of shape ``(Q, ...(B), C)``.
    h_shape : Shape
        Shape perturbations $h$ of (...) -> (..., C, 2), in boundary order.
    eps : float
        Tolerance for switching to diagonal limit in singular kernel evaluations.

    Returns
    -------
    Array
        Shape derivative $D_r J(r)[h]$ of shape (...(B),).

    Examples
    --------
    >>> import numpy as np
    >>> from ie_circle import CircleShape
    >>> from biem_helmholtz_2d._acoustic import scattering_dirichlet
    >>> xp = np; k = xp.asarray(1.0); rho = 1
    >>> shape = CircleShape(rho)
    >>> a = xp.asarray(1.0); e = xp.asarray(0.0)
    >>> def inc(x): return xp.exp(1j * k * x[..., 0])
    >>> phi = scattering_dirichlet(k=k, shape=shape, incident_field=inc, alpha=a, eta=e, n=8)
    >>> class _ZeroShape:
    ...     def x(self, t, /): return xp.zeros((*t.shape, 1, 2))
    ...     def dx(self, t, /): return xp.zeros((*t.shape, 1, 2))
    ...     def ddx(self, t, /): return xp.zeros((*t.shape, 1, 2))
    >>> from ie_circle import trapezoidal_quadrature
    >>> t, _ = trapezoidal_quadrature(8, xp=xp, device='cpu', dtype=xp.float64)
    >>> dr_g = xp.zeros((2 * 8 - 1, 1))
    >>> gj = xp.zeros_like(t)[..., None]
    >>> psi = solve_adjoint(k=k, shape=shape, alpha=a, eta=e, n=8, grad_phi_j=gj)
    >>> objective_derivative(
    ...     k=k, shape=shape, alpha=a, eta=e, n=8,
    ...     phi=phi, adjoint=psi, dr_j=xp.asarray(0),
    ...     dr_g=dr_g, h_shape=_ZeroShape(),
    ... )
    np.float64(0.0)

    """
    xp = array_namespace(k, alpha, eta, dr_j, dr_g)
    t, w = trapezoidal_quadrature(
        n, xp=xp, device=k.device, dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    )
    n_nodes = 2 * n - 1
    count = shape.x(t[:1]).shape[-2]
    if count == 0 or h_shape.x(t[:1]).shape[-2] != count:
        raise ValueError("shape and h_shape must have the same nonzero length")
    phi_t, psi_t = phi(t), adjoint(t)
    values_and_names = ((phi_t, "phi"), (psi_t, "adjoint"), (dr_g, "dr_g"))
    infos = [
        check_shapes("Q*BC,QC", values, (n_nodes, count), names=f"{name},expected")
        for values, name in values_and_names
    ]
    for values, name in values_and_names:
        if values.shape[0] != n_nodes or values.shape[-1] != count:
            raise ValueError(f"{name} must have one value per quadrature node and scatterer")
    k, alpha, eta, _, _, _, dr_j = xp.broadcast_arrays(
        k, alpha, eta, phi_t[0, ..., 0], psi_t[0, ..., 0], dr_g[0, ..., 0], dr_j
    )
    batch_shape = k.shape
    batch_ndim = len(batch_shape)
    broadcasted = []
    for (values, _), info in zip(values_and_names, infos, strict=True):
        missing = batch_ndim - len(info.unique["B"].shape_broadcasted)
        values = values[(slice(None),) + (None,) * missing + (...,)]
        broadcasted.append(xp.broadcast_to(values, (n_nodes, *batch_shape, count)))
    phi_t, psi_t, dr_g = broadcasted
    check_shapes(
        "Q*BC,Q*BC,Q*BC,*B,*B,*B,*B,QC",
        phi_t,
        psi_t,
        dr_g,
        k,
        alpha,
        eta,
        dr_j,
        (n_nodes, count),
        names="phi,adjoint,dr_g,k,alpha,eta,dr_j,expected",
    )
    _, weights = log_cot_power_quadrature(n, 0, xp=xp, device=k.device, dtype=t.dtype)
    indices = (
        xp.arange(n_nodes, device=k.device, dtype=xp.int64)[None, :]
        - xp.arange(n_nodes, device=k.device, dtype=xp.int64)[:, None]
    ) % n_nodes
    weights = weights[indices][(...,) + (None,) * (batch_ndim + 1)]
    t_row = t[(slice(None), None) + (None,) * batch_ndim]
    tau = t[(None, slice(None)) + (None,) * batch_ndim]
    x_t, x_tau = shape.x(t_row), shape.x(tau)
    dx_tau, ddx_tau = shape.dx(tau), shape.ddx(tau)
    h_t, h_tau = h_shape.x(t_row), h_shape.x(tau)
    dh_tau, ddh_tau = h_shape.dx(tau), h_shape.ddx(tau)
    sl = slp_shape_derivative_split(
        t=t_row[..., None],
        tau=tau[..., None],
        k=k[..., None],
        x_t=x_t,
        x_tau=x_tau,
        dx_tau=dx_tau,
        h_t=h_t,
        h_tau=h_tau,
        dh_tau=dh_tau,
        eps=eps,
    )
    dl = dlp_shape_derivative_split(
        t=t_row[..., None],
        tau=tau[..., None],
        k=k[..., None],
        x_t=x_t,
        x_tau=x_tau,
        dx_tau=dx_tau,
        ddx_tau=ddx_tau,
        h_t=h_t,
        h_tau=h_tau,
        dh_tau=dh_tau,
        ddh_tau=ddh_tau,
        eps=eps,
    )
    diagonal = (alpha[..., None] * dl[0] - 1j * eta[..., None] * sl[0]) * weights
    diagonal += (alpha[..., None] * dl[1] - 1j * eta[..., None] * sl[1]) * w[0]
    residual = xp.sum(diagonal * phi_t[None, ...], axis=1)
    if count > 1:
        rows, cols = off_diagonal_indices(count, xp=xp, device=k.device)
        point = xp.take(x_t, rows, axis=-2)
        source = xp.take(x_tau, cols, axis=-2)
        source_dx = xp.take(dx_tau, cols, axis=-2)
        motion = xp.take(h_tau, cols, axis=-2) - xp.take(h_t, rows, axis=-2)
        motion_dx = xp.take(dh_tau, cols, axis=-2)
        sl_cross = slp_shape_derivative_kernel(
            point, shape_x=source, shape_dx=source_dx, h=motion, dh=motion_dx, k=k[..., None]
        )
        dl_cross = dlp_shape_derivative_kernel(
            point, shape_x=source, shape_dx=source_dx, h=motion, dh=motion_dx, k=k[..., None]
        )
        cross = off_diagonal_matrix(
            (alpha[..., None] * dl_cross - 1j * eta[..., None] * sl_cross) * w[0], count
        )
        residual = residual + xp.sum(cross * phi_t[None, ..., None, :], axis=(1, -1))
    residual = residual - dr_g
    return dr_j + xp.real(xp.sum(psi_t * xp.conj(residual), axis=(0, -1)) * w[0])
