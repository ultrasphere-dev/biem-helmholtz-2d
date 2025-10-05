from abc import ABCMeta, abstractmethod

import attrs
import torch
import torch.nn as nn
from torchsphharm.eigenfunction import shift_nth_row_n_steps

from .shape import VECTOR_AXIS, Shape


def Lk(t: torch.Tensor, tk: torch.Tensor, n: float) -> torch.Tensor:
    return torch.where(
        t == tk, 1, torch.sin(n * (t - tk)) / torch.tan((t - tk) / 2) / (2 * n)
    )


def RWeight(t: torch.Tensor, tk: torch.Tensor, n: int) -> torch.Tensor:
    m = (
        torch.arange(n - 1, dtype=t.dtype, device=t.device).reshape(
            [1] * int(t.ndim) + [-1]
        )
        + 1
    )
    res = -2 * torch.pi / n * torch.sum(
        1 / m * torch.cos(m * (t).unsqueeze(-1)), dim=-1
    ) - torch.pi / n**2 * torch.cos(n * (t))
    res = res.repeat(1, res.shape[0])
    res = shift_nth_row_n_steps(res, dim_row=1, dim_shift=0, cut_padding=False)
    # add one more row to the end of the matrix
    res = torch.cat((res, torch.zeros_like(res[0]).unsqueeze(0)), dim=0)
    res = res[: res.shape[0] // 2] + res[res.shape[0] // 2 :]
    return res


@attrs.frozen(kw_only=True)
class GFunc(nn.Module, metaclass=ABCMeta):
    k: float

    def __attrs_post_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


@attrs.frozen(kw_only=True)
class PlaneWaveGFunc(GFunc):
    direction: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_in = torch.exp(
            torch.tensor(1j)
            * self.k
            * torch.einsum("...i,i->...", x, self.direction.to(x))
        )
        return -u_in


@attrs.frozen(kw_only=True)
class UInf(nn.Module):
    k: float
    eta: float
    shape: Shape

    def forward(
        self, *, t: torch.Tensor, phi: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        gamma = torch.exp(torch.tensor(-1j * torch.pi / 4)) / torch.sqrt(
            torch.tensor(8 * torch.pi * self.k)
        )
        n = t.shape[-1] // 2
        xhat = direction / torch.linalg.vector_norm(
            direction, dim=VECTOR_AXIS, keepdim=True
        )
        y = self.shape.x(t)
        dy = self.shape.dx(t)
        ny = torch.stack([dy[..., 1], -dy[..., 0]], dim=VECTOR_AXIS)
        ny = ny / torch.linalg.vector_norm(ny, dim=VECTOR_AXIS, keepdim=True)
        inner = (
            (self.k * torch.einsum("...tv,...v->...t", ny, xhat) + self.eta)
            * torch.exp(-1j * self.k * torch.einsum("...v,...tv->...t", xhat, y))
            * self.shape.jacobian(t)
            * phi
        )
        return (
            gamma
            * torch.pi
            / n
            * torch.sum(
                inner,
                dim=-1,
            )
        )
