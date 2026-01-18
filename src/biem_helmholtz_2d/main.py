from __future__ import annotations

import xp

from .nystrom import Nystrom
from .quadrature import PlaneWaveGFunc, UInf
from .shape import KressShape


def get_uinf(
    k: float,
    n: int,
    wave_direction: tuple[float, float] | Array,
    farfield_direction: list[tuple[float, float]] | tuple[float, float] | Array,
    eta: float,
    *,
    device: xp.device | None = None,
    dtype: xp.dtype | None = None,
) -> Array:
    shape = KressShape()
    t = xp.linspace(0, xp.pi * 2, 2 * n + 1, device=device, dtype=dtype)[:-1]
    direction_ = Array(wave_direction, dtype=t.dtype, device=t.device)
    phi = Nystrom(
        g=PlaneWaveGFunc(k=k, direction=direction_), shape=shape, eta=eta, k=k
    )(t)
    farfield_direction_ = Array(farfield_direction, dtype=t.dtype, device=t.device)
    t_ = t.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    phi_ = phi.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    uinf = UInf(k=k, eta=eta, shape=shape).forward(
        t=t_, phi=phi_, direction=farfield_direction_
    )
    return uinf
