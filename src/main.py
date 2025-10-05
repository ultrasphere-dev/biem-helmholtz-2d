from __future__ import annotations

import torch

from .func import PlaneWaveGFunc, UInf
from .nystrom import Nystrom
from .shape import KressShape


def get_uinf(
    k: float,
    n: int,
    wave_direction: tuple[float, float] | torch.Tensor,
    farfield_direction: list[tuple[float, float]] | tuple[float, float] | torch.Tensor,
    eta: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    shape = KressShape()
    t = torch.linspace(0, torch.pi * 2, 2 * n + 1, device=device, dtype=dtype)[:-1]
    direction_ = torch.tensor(wave_direction, dtype=t.dtype, device=t.device)
    phi = Nystrom(
        g=PlaneWaveGFunc(k=k, direction=direction_), shape=shape, eta=eta, k=k
    )(t)
    farfield_direction_ = torch.tensor(
        farfield_direction, dtype=t.dtype, device=t.device
    )
    t_ = t.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    phi_ = phi.reshape([1] * int(farfield_direction_.ndim - 1) + [-1])
    uinf = UInf(k=k, eta=eta, shape=shape).forward(
        t=t_, phi=phi_, direction=farfield_direction_
    )
    return uinf
