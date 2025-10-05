from __future__ import annotations

import xp

from .quadrature import PlaneWaveGFunc, UInf
from .nystrom import Nystrom
from .shape import KressShape


def get_uinf(
    k: float,
    n: int,
    wave_direction: tuple[float, float] | xp.Tensor,
    farfield_direction: list[tuple[float, float]] | tuple[float, float] | xp.Tensor,
    eta: float,
    *,
    device: xp.device | None = None,
    dtype: xp.dtype | None = None,
) -> xp.Tensor:
    shape = KressShape()
    t = xp.linspace(0, xp.pi * 2, 2 * n + 1, device=device, dtype=dtype)[:-1]
    direction_ = xp.tensor(wave_direction, dtype=t.dtype, device=t.device)
    phi = Nystrom(
        g=PlaneWaveGFunc(k=k, direction=direction_), shape=shape, eta=eta, k=k
    )(t)
    farfield_direction_ = xp.tensor(
        farfield_direction, dtype=t.dtype, device=t.device
    )
    t_ = t.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    phi_ = phi.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    uinf = UInf(k=k, eta=eta, shape=shape).forward(
        t=t_, phi=phi_, direction=farfield_direction_
    )
    return uinf
