from __future__ import annotations

from biem_helmholtz_2d import _potential, _potential_shape_derivative


def test_slp_shape_derivative_numerical(xp):
    k = 1.5
    rho = 2.0

    def x(t):
        return xp.stack([rho * xp.cos(t), rho * xp.sin(t)], axis=-1)

    def dx(t):
        return xp.stack([-rho * xp.sin(t), rho * xp.cos(t)], axis=-1)

    # Perturbation
    def h(t):
        r = xp.cos(2 * t)
        return xp.stack([r * xp.cos(t), r * xp.sin(t)], axis=-1)

    def dh(t):
        r = xp.cos(2 * t)
        dr = -2 * xp.sin(2 * t)
        c = xp.cos(t)
        s = xp.sin(t)
        return xp.stack([dr * c - r * s, dr * s + r * c], axis=-1)

    t = xp.linspace(0.1, 2.0, 10, dtype=xp.float64)
    tau = 0.5

    # Filter out points too close to tau to avoid diagonal issues for now
    mask = xp.abs(t - tau) > 0.1
    t = t[mask]

    epsilon = 1e-4

    da_log, da_rem = _potential_shape_derivative.slp_shape_derivative(t, tau, k, x, h, dx, dh)

    def x_plus(t_val):
        return x(t_val) + epsilon * h(t_val)

    def dx_plus(t_val):
        return dx(t_val) + epsilon * dh(t_val)

    def x_minus(t_val):
        return x(t_val) - epsilon * h(t_val)

    def dx_minus(t_val):
        return dx(t_val) - epsilon * dh(t_val)

    p_log, p_rem = _potential.slp(t, tau, k, x_plus, dx_plus)
    m_log, m_rem = _potential.slp(t, tau, k, x_minus, dx_minus)

    num_log = (p_log - m_log) / (2 * epsilon)
    num_rem = (p_rem - m_rem) / (2 * epsilon)

    # Check relative error
    # Use higher tolerance for finite difference approximation
    assert xp.all(xp.abs(da_log - num_log) < 1e-3 * xp.abs(da_log) + 1e-4), "SLP Log part mismatch"
    assert xp.all(xp.abs(da_rem - num_rem) < 1e-3 * xp.abs(da_rem) + 1e-4), (
        "SLP Remainder part mismatch"
    )


def test_dlp_shape_derivative_numerical(xp):
    k = 1.5
    rho = 2.0

    def x(t):
        return xp.stack([rho * xp.cos(t), rho * xp.sin(t)], axis=-1)

    def dx(t):
        return xp.stack([-rho * xp.sin(t), rho * xp.cos(t)], axis=-1)

    def ddx(t):
        return xp.stack([-rho * xp.cos(t), -rho * xp.sin(t)], axis=-1)

    def h(t):
        r = xp.cos(2 * t)
        return xp.stack([r * xp.cos(t), r * xp.sin(t)], axis=-1)

    def dh(t):
        r = xp.cos(2 * t)
        dr = -2 * xp.sin(2 * t)
        c = xp.cos(t)
        s = xp.sin(t)
        return xp.stack([dr * c - r * s, dr * s + r * c], axis=-1)

    def ddh(t):
        # r = cos(2t), r' = -2sin(2t), r'' = -4cos(2t)
        r = xp.cos(2 * t)
        dr = -2 * xp.sin(2 * t)
        ddr = -4 * xp.cos(2 * t)
        c = xp.cos(t)
        s = xp.sin(t)
        # x' = r' c - r s
        # x'' = (r'' c - r' s) - (r' s + r c) = r'' c - 2 r' s - r c
        dx_val = ddr * c - 2 * dr * s - r * c
        dy_val = ddr * s + 2 * dr * c - r * s
        return xp.stack([dx_val, dy_val], axis=-1)

    t = xp.linspace(0.1, 2.0, 10, dtype=xp.float64)
    tau = 0.5

    mask = xp.abs(t - tau) > 0.1
    t = t[mask]

    epsilon = 1e-4

    da_log, da_rem = _potential_shape_derivative.dlp_shape_derivative(
        t, tau, k, x, h, dx, dh, ddx, ddh
    )

    def x_plus(t_val):
        return x(t_val) + epsilon * h(t_val)

    def dx_plus(t_val):
        return dx(t_val) + epsilon * dh(t_val)

    def ddx_plus(t_val):
        return ddx(t_val) + epsilon * ddh(t_val)

    def x_minus(t_val):
        return x(t_val) - epsilon * h(t_val)

    def dx_minus(t_val):
        return dx(t_val) - epsilon * dh(t_val)

    def ddx_minus(t_val):
        return ddx(t_val) - epsilon * ddh(t_val)

    p_log, p_rem = _potential.dlp(t, tau, k, x_plus, dx_plus, ddx_plus)
    m_log, m_rem = _potential.dlp(t, tau, k, x_minus, dx_minus, ddx_minus)

    num_log = (p_log - m_log) / (2 * epsilon)
    num_rem = (p_rem - m_rem) / (2 * epsilon)

    assert xp.all(xp.abs(da_log - num_log) < 1e-3 * xp.abs(da_log) + 1e-4), "DLP Log part mismatch"
    assert xp.all(xp.abs(da_rem - num_rem) < 1e-3 * xp.abs(da_rem) + 1e-4), (
        "DLP Remainder part mismatch"
    )
