from abc import ABCMeta, abstractmethod

import xp
from xp import Tensor

VECTOR_AXIS = -1


class Shape(metaclass=ABCMeta):
    @abstractmethod
    def x(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def dx(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def ddx(self, t: Tensor) -> Tensor:
        pass

    def jacobian(self, t: Tensor) -> Tensor:
        return xp.sqrt(xp.sum(self.dx(t) ** 2, dim=VECTOR_AXIS))


class KressShape(Shape):
    def x(self, t: Tensor) -> Tensor:
        return xp.stack(
            [xp.cos(t) + 0.65 * xp.cos(2 * t) - 0.65, 1.5 * xp.sin(t)],
            dim=VECTOR_AXIS,
        )

    def dx(self, t: Tensor) -> Tensor:
        return xp.stack(
            [-xp.sin(t) - 0.65 * 2 * xp.sin(2 * t), 1.5 * xp.cos(t)],
            dim=VECTOR_AXIS,
        )

    def ddx(self, t: Tensor) -> Tensor:
        return xp.stack(
            [-xp.cos(t) - 0.65 * 2 * 2 * xp.cos(2 * t), -1.5 * xp.sin(t)],
            dim=VECTOR_AXIS,
        )
