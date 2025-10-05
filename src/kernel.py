from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Literal

import attrs
import xp
import xp.nn as nn
from xp.special import bessel_j0, bessel_j1, bessel_y0, bessel_y1

from .shape import Shape


def _hankel(order: Literal[0, 1], x: xp.Tensor, kind: Literal[1, 2]) -> xp.Tensor:
    bessel_kind_1 = bessel_j0 if order == 0 else bessel_j1
    bessel_kind_2 = bessel_y0 if order == 0 else bessel_y1
    mp = 1 if kind == 1 else -1
    return bessel_kind_1(x) + mp * 1j * bessel_kind_2(x)


@attrs.frozen(kw_only=True)
class Kernel(nn.Module, metaclass=ABCMeta):
    k: float
    shape: Shape

    def __attrs_post_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        pass


def _is_diagonal(t: xp.Tensor, tau: xp.Tensor, eps: float = 1e-6) -> xp.Tensor:
    absFmod = xp.abs(xp.fmod(t - tau, 2 * xp.pi))
    # return absFmod < eps or absFmod > 2 * xp.pi - eps ambiguous
    return (absFmod < eps) | (absFmod > 2 * xp.pi - eps)


@attrs.frozen(kw_only=True)
class KernelDiagonal(Kernel):
    kernel_diagonal: Kernel
    kernel_nondiagonal: Kernel

    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        # return xp.where(
        #     # _is_diagonal(t, tau),
        #     xp.eye(len(t), device=t.device, dtype=t.dtype).bool(),
        #     self.kernel_diagonal(t, tau),
        #     self.kernel_nondiagonal(t, tau),
        # )
        eye = xp.eye(len(t), device=t.device, dtype=t.dtype)
        return eye * self.kernel_diagonal(t, tau) + (1 - eye) * xp.nan_to_num(
            self.kernel_nondiagonal(t, tau), nan=0.0, posinf=0.0, neginf=0.0
        )


class L(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return (
            1j
            * self.k
            / 2.0
            * (
                self.shape.dx(tau)[..., 1]
                * (self.shape.x(tau) - self.shape.x(t))[..., 0]
                - self.shape.dx(tau)[..., 0]
                * (self.shape.x(tau) - self.shape.x(t))[..., 1]
            )
            * _hankel(1, self.k * self.shape.r(t, tau), 1)
            / self.shape.r(t, tau)
        )


class M(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return (
            1j
            / 2.0
            * _hankel(0, self.k * self.shape.r(t, tau), 1)
            * self.shape.jacobian(tau)
        )


class L1(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return xp.where(
            _is_diagonal(t, tau),
            0,
            (
                self.k
                / (2.0 * xp.pi)
                * (
                    self.shape.dx(tau)[..., 1]
                    * (self.shape.x(t) - self.shape.x(tau))[..., 0]
                    - self.shape.dx(tau)[..., 0]
                    * (self.shape.x(t) - self.shape.x(tau))[..., 1]
                )
                * bessel_j1(self.k * self.shape.r(t, tau))
                / self.shape.r(t, tau)
            ),
        )


class M1(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return (
            -1
            / (2.0 * xp.pi)
            * bessel_j0(self.k * self.shape.r(t, tau))
            * self.shape.jacobian(tau)
        )


@attrs.frozen(kw_only=True)
class LM2(Kernel):
    kernel_original: Kernel
    kernel1: Kernel

    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return self.kernel_original(t, tau) - self.kernel1(t, tau) * xp.log(
            4 * xp.sin((t - tau) / 2.0) ** 2
        )


class L2Diagonal(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return (
            self.shape.dx(t)[..., 0] * self.shape.ddx(t)[..., 1]
            - self.shape.dx(t)[..., 1] * self.shape.ddx(t)[..., 0]
        ) / (2.0 * xp.pi * self.shape.jacobian(t) ** 2)


class M2Diagonal(Kernel):
    def forward(self, t: xp.Tensor, tau: xp.Tensor) -> xp.Tensor:
        return self.shape.jacobian(t) * (
            1j / 2.0
            - 0.5772156649015329 / xp.pi
            - 1.0
            / (2.0 * xp.pi)
            * xp.log((self.k * self.shape.jacobian(t)) ** 2 / 4.0)
        )


def get_final_kernels(k: float, shape: Shape) -> tuple[Kernel, Kernel, Kernel, Kernel]:
    l1 = L1(k=k, shape=shape)
    lo = L(k=k, shape=shape)
    l2nd = LM2(k=k, shape=shape, kernel_original=lo, kernel1=l1)
    l2d = L2Diagonal(k=k, shape=shape)
    l2 = KernelDiagonal(k=k, shape=shape, kernel_nondiagonal=l2nd, kernel_diagonal=l2d)

    m1 = M1(k=k, shape=shape)
    mo = M(k=k, shape=shape)
    m2nd = LM2(k=k, shape=shape, kernel_original=mo, kernel1=m1)
    m2d = M2Diagonal(k=k, shape=shape)
    m2 = KernelDiagonal(k=k, shape=shape, kernel_nondiagonal=m2nd, kernel_diagonal=m2d)
    return m1, m2, l1, l2
