from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import (
    NystromInterpolant,
    QuadratureType,
    Shapes,
    log_cot_power_quadrature,
    nystrom,
    trapezoidal_quadrature,
)

from ._potential import dlp_kernel_split, slp_kernel_split
from ._potential_inner import dlp_kernel, slp_kernel
from ._potential_shape_derivative import (
    dlp_shape_derivative_split,
    slp_shape_derivative_split,
)


def _solve_adjoint(
    *,
    k: Array,
    shapes: Shapes,
    alpha: Array,
    eta: Array,
    n: int,
    rhs: Callable[[Array], Array],
    eps: float = 0,
) -> NystromInterpolant:
    r"""
    Solve the adjoint block system $A^* \Psi = \mathrm{rhs}$ for multiple scatterers.

    For $M$ scatterers, the adjoint operator is the block matrix:

    $$
    A^* = \begin{pmatrix}
    A_{11}^* & A_{21}^* & \cdots & A_{M1}^* \\
    A_{12}^* & A_{22}^* & \cdots & A_{M2}^* \\
    \vdots   & \vdots   & \ddots & \vdots   \\
    A_{1M}^* & A_{2M}^* & \cdots & A_{MM}^*
    \end{pmatrix}
    $$

    where $A_{jl}^*$ is the adjoint of the operator from scatterer $l$ to scatterer $j$.
    """
    M = shapes.n_shapes
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device

    def k_log(t: Array, tau: Array) -> Array:
        # Diagonal blocks only (log-singular part)
        # For adjoint: swap t and tau, then conjugate
        diag = []
        for j in range(M):
            shape_j = shapes[j]
            dlp_log, _ = dlp_kernel_split(
                t=tau,
                tau=t,
                k=k[..., None, None],
                x=shape_j.x,
                dx=shape_j.dx,
                ddx=shape_j.ddx,
                eps=eps,
            )
            slp_log, _ = slp_kernel_split(
                t=tau, tau=t, k=k[..., None, None], x=shape_j.x, dx=shape_j.dx, eps=eps
            )
            diag.append(alpha[j] * xp.conj(dlp_log) + 1j * eta[j] * xp.conj(slp_log))
        stacked = xp.stack(diag, axis=-1)
        eye = xp.eye(M, dtype=dtype, device=device)
        return stacked[..., :, None] * eye

    def k_cont(t: Array, tau: Array) -> Array:
        # All blocks (smooth/analytic part)
        # For adjoint: block (j, m) is A_{mj}^*(t, tau) = conj(A_{mj}(tau, t))
        blocks = []
        for j in range(M):
            row = []
            shape_j = shapes[j]
            for m in range(M):
                shape_m = shapes[m]
                if j == m:
                    # Diagonal: adjoint of A_{jj}, swap t and tau
                    _, dlp_cont = dlp_kernel_split(
                        t=tau,
                        tau=t,
                        k=k[..., None, None],
                        x=shape_j.x,
                        dx=shape_j.dx,
                        ddx=shape_j.ddx,
                        eps=eps,
                    )
                    _, slp_cont = slp_kernel_split(
                        t=tau, tau=t, k=k[..., None, None], x=shape_j.x, dx=shape_j.dx, eps=eps
                    )
                    row.append(alpha[j] * xp.conj(dlp_cont) + 1j * eta[j] * xp.conj(slp_cont))
                else:
                    # Off-diagonal: adjoint of A_{mj} is A_{jm}^*
                    # A_{mj}(tau, t) = kernel from scatterer j evaluated at points on scatterer m
                    # So A_{jm}^*(t, tau) = conj(A_{mj}(tau, t))
                    # We need: kernel from scatterer j, evaluated at x_m(tau), with source at x_j(t)
                    # But for adjoint, we swap: evaluate at x_j(t) with source parameter tau
                    xm_tau = shape_m.x(tau)  # (1, Q, 2)
                    dlp_val = dlp_kernel(xm_tau, shape_x=shape_j.x, shape_dx=shape_j.dx, k=k, tau=t)
                    slp_val = slp_kernel(xm_tau, shape_x=shape_j.x, shape_dx=shape_j.dx, k=k, tau=t)
                    # A_{mj} uses alpha_j, eta_j (coefficients of scatterer j)
                    row.append(alpha[j] * xp.conj(dlp_val) + 1j * eta[j] * xp.conj(slp_val))
            blocks.append(xp.stack(row, axis=-1))
        return xp.stack(blocks, axis=-2)

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
    shapes: Shapes,
    alpha: Array,
    eta: Array,
    n: int,
    phi: NystromInterpolant,
    grad_phi_j: Callable[[Array], Array],
    dr_j: Array,
    dr_g: Array,
    h_shapes: Shapes,
    eps: float = 0,
) -> Array:
    r"""
    Shape derivative $D_r J(r)[h]$ for multiple scatterers under the $L^2$ sesquilinear form.

    For $M$ scatterers with boundaries $\Gamma_1, \ldots, \Gamma_M$ and perturbations $h_1, \ldots, h_M$:

    $$
    \begin{aligned}
    A \Phi &= G, \\
    A^* \Psi &= -\operatorname{grad}_\Phi J, \\
    D_r J(r)[h] &= D_r j(r,\Phi_r)[h]
        + \operatorname{Re} \sum_{j=1}^M \bigl\langle
            \psi_j,\;
            \sum_{l=1}^M D_r A_{jl}[h_l] \phi_l
            - D_r g_j[h_j]
          \bigr\rangle,
    \end{aligned}
    $$

    where $A_{jl}$ is the block operator from scatterer $l$ to scatterer $j$.

    Parameters
    ----------
    k : Array
        Wave number of shape (...,).
    shapes : Shapes
        Boundary parametrisations of the $M$ scatterers.
    alpha : Array
        Coupling parameters $\alpha$ of shape (M,).
    eta : Array
        Coupling parameters $\eta$ of shape (M,).
    n : int
        Maximum order minus $1$, number of quadrature nodes $2n-1$.
    phi : NystromInterpolant
        Forward density $\Phi_r$ with $M$ components.
    grad_phi_j : Callable[[Array], Array]
        Riesz representation w.r.t. $\langle f,g\rangle=\int f\overline g$.
        Signature (...,) -> (..., M).
    dr_j : Array
        Value $D_r j(r,\Phi_r)[h]$ of shape (...,).
    dr_g : Array
        Shape derivative $D_r g[h]$ at quadrature nodes of shape ``(..., M, Q)``.
    h_shapes : Shapes
        Shape perturbations for all $M$ scatterers.
    eps : float
        Tolerance for switching to diagonal limit in singular kernel evaluations.

    Returns
    -------
    Array
        Shape derivative $D_r J(r)[h]$ of shape (...,).

    """
    from array_api_shape_check import check_shapes

    M = shapes.n_shapes
    xp = array_namespace(k, alpha, eta, dr_j, dr_g)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device
    t, w_trap = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    n_nodes = 2 * n - 1
    w_trap_val = w_trap[0]

    # dr_g has shape (..., M, Q)
    check_shapes("*BMQ", dr_g, names="dr_g")

    # Solve adjoint system
    # grad_phi_j returns (..., M), we need rhs to return (Q, M)
    def adjoint_rhs(t_in: Array) -> Array:
        grad_vals = grad_phi_j(t_in)  # (Q, M) or (..., M)
        return -grad_vals

    psi = _solve_adjoint(
        k=k,
        shapes=shapes,
        alpha=alpha,
        eta=eta,
        n=n,
        rhs=adjoint_rhs,
        eps=eps,
    )
    # psi(t) returns (..., M)
    psi_t = psi(t)
    # phi(t) returns (..., M)
    phi_t = phi(t)

    _, w_log_cot_raw = log_cot_power_quadrature(n, 0, xp=xp, device=device, dtype=dtype)
    roll_idx = (
        -xp.arange(n_nodes, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_nodes, device=device, dtype=xp.int64)[None, :]
    ) % n_nodes
    w_log_cot = w_log_cot_raw[roll_idx]

    t_row = t[:, None]
    tau_col = t[None, :]

    # Compute shape derivative contributions for each scatterer pair (j, l)
    # shape_deriv_phi[j] = sum_l D_r A_{jl}[h_l] phi_l
    shape_deriv_phi = xp.zeros(
        (M, n_nodes), dtype=xp.result_type(dtype, xp.complex64), device=device
    )

    for j in range(M):
        for l in range(M):
            # Compute shape derivative of kernel A_{jl} with respect to perturbation h_l
            shape_l = shapes[l]
            h_l = h_shapes[l]
            slp_log, slp_cont = slp_shape_derivative_split(
                t=t_row,
                tau=tau_col,
                k=k[..., None, None],
                x=shape_l.x,
                dx=shape_l.dx,
                h=h_l.x,
                dh=h_l.dx,
                eps=eps,
            )
            dlp_log, dlp_cont = dlp_shape_derivative_split(
                t=t_row,
                tau=tau_col,
                k=k[..., None, None],
                x=shape_l.x,
                dx=shape_l.dx,
                ddx=shape_l.ddx,
                h=h_l.x,
                dh=h_l.dx,
                ddh=h_l.ddx,
                eps=eps,
            )

            k_log_sd = alpha[l] * dlp_log - 1j * eta[l] * slp_log
            k_cont_sd = alpha[l] * dlp_cont - 1j * eta[l] * slp_cont

            # phi_l has shape (Q,)
            phi_l = phi_t[..., l]

            # Compute D_r A_{jl}[h_l] phi_l
            contrib = xp.sum(k_log_sd * phi_l * w_log_cot, axis=-1) + xp.sum(
                k_cont_sd * phi_l * w_trap_val, axis=-1
            )
            shape_deriv_phi[j] = shape_deriv_phi[j] + contrib

    # Subtract dr_g and compute inner product with psi
    # dr_g has shape (..., M, Q), shape_deriv_phi has shape (M, Q)
    # psi_t has shape (..., M, Q)
    inner_per_scatterer = xp.sum(
        psi_t * xp.conj(shape_deriv_phi - dr_g) * w_trap_val, axis=-1
    )  # (..., M)
    inner = xp.sum(inner_per_scatterer, axis=-1)  # (...)

    return dr_j + xp.real(inner)
