from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import attrs
from array_api._2024_12 import Array
from array_api_compat import array_namespace


class Shape(metaclass=ABCMeta):
    @abstractmethod
    def x(self, t: Array, /) -> Array:
        pass

    @abstractmethod
    def dx(self, t: Array, /) -> Array:
        pass

    @abstractmethod
    def ddx(self, t: Array, /) -> Array:
        pass

    @abstractmethod
    def __eq__(self, other: object, /) -> bool:
        pass

    def jacobian(self, t: Array) -> Array:
        xp = array_namespace(t)
        return xp.sqrt(xp.sum(self.dx(t) ** 2, axis=-1))


@attrs.define(frozen=True)
class Shapes:
    shapes: Sequence[Shape]

    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [shape.x(t) for shape in self.shapes],
            axis=-2,
        )

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [shape.dx(t) for shape in self.shapes],
            axis=-2,
        )

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [shape.ddx(t) for shape in self.shapes],
            axis=-2,
        )


@attrs.define(frozen=True)
class KressShape(Shape):
    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [xp.cos(t) + 0.65 * xp.cos(2 * t) - 0.65, 1.5 * xp.sin(t)],
            axis=-1,
        )

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [-xp.sin(t) - 0.65 * 2 * xp.sin(2 * t), 1.5 * xp.cos(t)],
            axis=-1,
        )

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [-xp.cos(t) - 0.65 * 2 * 2 * xp.cos(2 * t), -1.5 * xp.sin(t)],
            axis=-1,
        )
