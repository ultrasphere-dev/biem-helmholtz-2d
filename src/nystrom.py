from logging import getLogger

import attrs
import attrs.setters
import torch
import torch.linalg
import torch.nn as nn
from attrs import field
from cm_time import timer

from .func import GFunc, RWeight
from .kernel import Kernel, get_final_kernels
from .shape import Shape

LOG = getLogger(__name__)


@attrs.define(kw_only=True)
class Nystrom(nn.Module):
    k: float = field(on_setattr=attrs.setters.frozen)
    g: GFunc = field(on_setattr=attrs.setters.frozen)
    shape: Shape = field(on_setattr=attrs.setters.frozen)
    eta: float = field(on_setattr=attrs.setters.frozen)
    m1: Kernel = field(init=False)
    m2: Kernel = field(init=False)
    l1: Kernel = field(init=False)
    l2: Kernel = field(init=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.m1, self.m2, self.l1, self.l2 = get_final_kernels(self.k, self.shape)

    def A(self, t: torch.Tensor) -> torch.Tensor:
        n = len(t) // 2
        tj = t.unsqueeze(0)
        ti = t.unsqueeze(1)
        with timer() as tr:
            r = RWeight(ti, tj, n)
        with timer() as tm1:
            m1 = self.m1(ti, tj)
        with timer() as tm2:
            m2 = self.m2(ti, tj)
        with timer() as tl1:
            l1 = self.l1(ti, tj)
        with timer() as tl2:
            l2 = self.l2(ti, tj)
        with timer() as tres:
            res = (
                -torch.pi / n * (l2 + 1j * self.eta * m2)
                - r * (l1 + 1j * self.eta * m1)
                + torch.eye(n * 2, device=t.device, dtype=t.dtype)
            )
        LOG.debug(
            f"{tr.elapsed=}, {tm1.elapsed=}, {tm2.elapsed=}, "
            f"{tl1.elapsed=}, {tl2.elapsed=}, {tres.elapsed=}"
        )
        return res

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return 2 * self.g(self.shape.x(t))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        with timer() as ta:
            A = self.A(t)
        with timer() as tb:
            b = self.b(t)
        with timer() as tphi:
            phi = torch.linalg.solve(A, b).squeeze()
        LOG.debug(f"{ta.elapsed=}, {tb.elapsed=}, {tphi.elapsed=}")
        return phi
