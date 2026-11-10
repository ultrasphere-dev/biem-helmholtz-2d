from __future__ import annotations

from collections.abc import Callable, Sequence

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import (
    NystromInterpolant,
    QuadratureType,
    Shape,
    nystrom,
)

from ._acoustic import _coupled_kernels
from ._potential_inner_derivative import dlp_shape_derivative_kernel, slp_shape_derivative_kernel
from ._potential_shape_derivative import dlp_shape_derivative_split, slp_shape_derivative_split
from ._quadrature import log_cot_power_quadrature, trapezoidal_quadrature


def solve_adjoint(
    *,
    k: Array,
    shape: Sequence[Shape],
    alpha: Array,
    eta: Array,
    n: int,
    grad_phi_j: Callable[[Array], Array],
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
    shape : Sequence[Shape]
        Nonempty sequence of disjoint boundaries of (...) -> (..., 2).
    n : int
        Quadrature order, with $Q=2n-1$ nodes.
    grad_phi_j : Callable[[Array], Array]
        Density gradient under parameter measure of (...) -> (..., ...(B), C),
        where $C$ is the number of boundaries.

    Returns
    -------
    NystromInterpolant
        Adjoint density of (...) -> (..., ...(B), C).

    """
    if not shape:
        raise ValueError("At least one boundary is required")
    xp = array_namespace(k, alpha, eta)
    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    info = check_shapes("*B,*B,*B", k, alpha, eta, names="k,alpha,eta")
    batch_shape = info.unique["B"].shape_broadcasted
    forward = _coupled_kernels(shape=shape, k=k, alpha=alpha, eta=eta)

    def k_log(t: Array, tau: Array) -> Array:
        return xp.conj(xp.matrix_transpose(forward[QuadratureType.LOG_COT_POWER, 0](tau, t)))

    def k_cont(t: Array, tau: Array) -> Array:
        return xp.conj(xp.matrix_transpose(forward[QuadratureType.NO_SINGULARITY, 0](tau, t)))

    def coefficient(t: Array) -> Array:
        value = xp.broadcast_to(xp.conj(alpha) / 2, (*t.shape, *batch_shape))
        return value[..., None] * xp.ones(len(shape), device=t.device, dtype=t.dtype)

    def rhs(t: Array) -> Array:
        gradient = grad_phi_j(t)
        check_shapes(
            "...*BC,...,*B,C", gradient, t, alpha, (len(shape),), names="grad_phi_j,t,alpha,C"
        )
        if gradient.shape[: t.ndim] != t.shape or gradient.shape[-1] != len(shape):
            raise ValueError("grad_phi_j must have one value per parameter and scatterer")
        return -xp.broadcast_to(gradient, (*t.shape, *batch_shape, len(shape)))

    return nystrom(
        coefficient,
        {(QuadratureType.NO_SINGULARITY, 0): k_cont, (QuadratureType.LOG_COT_POWER, 0): k_log},
        rhs,
        n=n,
        xp=xp,
        device=k.device,
        dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta)),
    )


def _relative_motion(source: Shape, target: Shape, t: Array, /) -> Callable[[Array], Array]:
    def motion(parameter: Array, /) -> Array:
        return source.x(parameter) - target.x(t)

    return motion


def objective_derivative(
    *,
    k: Array,
    shape: Sequence[Shape],
    alpha: Array,
    eta: Array,
    n: int,
    phi: NystromInterpolant,
    adjoint: NystromInterpolant,
    dr_j: Array,
    dr_g: Array,
    h_shape: Sequence[Shape],
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
    shape : Sequence[Shape]
        Disjoint boundary parametrisations of (...,) -> (..., 2).
    alpha : Array
        Coupling parameter $\alpha$ of shape (...(B),).
    eta : Array
        Coupling parameter $\eta$ of shape (...(B),).
    n : int
        Integration order, with $2n-1$ nodes. It may differ from the
        Nyström order used to compute ``phi`` and ``adjoint``.
    phi : NystromInterpolant
        Forward density $\phi_r$ of (...) -> (..., ...(B), C).
    adjoint : NystromInterpolant
        Solution returned by :func:`solve_adjoint` of (...) -> (..., ...(B), C).
        Reuse it for all perturbations of the same objective and geometry.
    dr_j : Array
        Value $D_r j(r,\phi_r)[h]$ of shape (...(B),).
    dr_g : Array
        Shape derivative $D_r g[h]$ at quadrature nodes of shape ``(Q, ...(B), C)``.
    h_shape : Sequence[Shape]
        Shape perturbations $h$, each providing $h(t)$, $h'(t)$ and $h''(t)$
        of (...,) -> (..., 2), in boundary order.
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
    >>> phi = scattering_dirichlet(k=k, shape=[shape], incident_field=inc, alpha=a, eta=e, n=8)
    >>> class _ZeroShape:
    ...     def x(self, t, /): return xp.zeros((*t.shape, 2))
    ...     def dx(self, t, /): return xp.zeros((*t.shape, 2))
    ...     def ddx(self, t, /): return xp.zeros((*t.shape, 2))
    >>> from ie_circle import trapezoidal_quadrature
    >>> t, _ = trapezoidal_quadrature(8, xp=xp, device='cpu', dtype=xp.float64)
    >>> dr_g = xp.zeros((2 * 8 - 1, 1))
    >>> def gj(t): return xp.zeros_like(t)[..., None]
    >>> psi = solve_adjoint(k=k, shape=[shape], alpha=a, eta=e, n=8, grad_phi_j=gj)
    >>> objective_derivative(
    ...     k=k, shape=[shape], alpha=a, eta=e, n=8,
    ...     phi=phi, adjoint=psi, dr_j=xp.asarray(0),
    ...     dr_g=dr_g, h_shape=[_ZeroShape()],
    ... )
    np.float64(0.0)

    """
    if not shape or len(shape) != len(h_shape):
        raise ValueError("shape and h_shape must have the same nonzero length")
    xp = array_namespace(k, alpha, eta, dr_j, dr_g)
    t, w = trapezoidal_quadrature(
        n, xp=xp, device=k.device, dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    )
    n_nodes = 2 * n - 1
    phi_t, psi_t = phi(t), adjoint(t)
    values_and_names = ((phi_t, "phi"), (psi_t, "adjoint"), (dr_g, "dr_g"))
    infos = [
        check_shapes("Q*BC,QC", values, (n_nodes, len(shape)), names=f"{name},expected")
        for values, name in values_and_names
    ]
    for values, name in values_and_names:
        if values.shape[0] != n_nodes or values.shape[-1] != len(shape):
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
        broadcasted.append(xp.broadcast_to(values, (n_nodes, *batch_shape, len(shape))))
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
        (n_nodes, len(shape)),
        names="phi,adjoint,dr_g,k,alpha,eta,dr_j,expected",
    )
    _, weights = log_cot_power_quadrature(n, 0, xp=xp, device=k.device, dtype=t.dtype)
    indices = (
        xp.arange(n_nodes, device=k.device, dtype=xp.int64)[None, :]
        - xp.arange(n_nodes, device=k.device, dtype=xp.int64)[:, None]
    ) % n_nodes
    weights = weights[indices][(...,) + (None,) * batch_ndim]
    t_row = t[(slice(None), None) + (None,) * batch_ndim]
    tau = t[(None, slice(None)) + (None,) * batch_ndim]
    rows = []
    for i, target in enumerate(shape):
        contributions = []
        for j, source in enumerate(shape):
            h = h_shape[j]
            if i == j:
                sl = slp_shape_derivative_split(
                    t=t_row, tau=tau, k=k, x=source.x, dx=source.dx, h=h.x, dh=h.dx, eps=eps
                )
                dl = dlp_shape_derivative_split(
                    t=t_row,
                    tau=tau,
                    k=k,
                    x=source.x,
                    dx=source.dx,
                    ddx=source.ddx,
                    h=h.x,
                    dh=h.dx,
                    ddh=h.ddx,
                    eps=eps,
                )
                kernel = (alpha * dl[0] - 1j * eta * sl[0]) * weights
                kernel += (alpha * dl[1] - 1j * eta * sl[1]) * w[0]
            else:
                # Simultaneous translations leave a cross kernel unchanged.
                relative_motion = _relative_motion(h, h_shape[i], t_row)
                point = target.x(t_row)
                sl = slp_shape_derivative_kernel(
                    point,
                    shape_x=source.x,
                    shape_dx=source.dx,
                    h=relative_motion,
                    dh=h.dx,
                    k=k,
                    tau=tau,
                )
                dl = dlp_shape_derivative_kernel(
                    point,
                    shape_x=source.x,
                    shape_dx=source.dx,
                    h=relative_motion,
                    dh=h.dx,
                    k=k,
                    tau=tau,
                )
                kernel = (alpha * dl - 1j * eta * sl) * w[0]
            contributions.append(xp.sum(kernel * phi_t[None, ..., j], axis=1))
        rows.append(sum(contributions))
    residual = xp.stack(rows, axis=-1) - dr_g
    return dr_j + xp.real(xp.sum(psi_t * xp.conj(residual), axis=(0, -1)) * w[0])
