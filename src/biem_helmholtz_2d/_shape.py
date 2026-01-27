from typing import Protocol

import attrs
from array_api._2024_12 import Array
from array_api_compat import array_namespace


class Shape(Protocol):
    def x(self, t: Array, /) -> Array: ...

    def dx(self, t: Array, /) -> Array: ...

    def ddx(self, t: Array, /) -> Array: ...


def jacobian(shape: Shape, t: Array, /) -> Array:
    xp = array_namespace(t)
    return xp.sqrt(xp.sum(shape.dx(t) ** 2, axis=-1))


@attrs.define(frozen=True)
class CircleShape(Shape):
    """Circle of radius ``rho`` centered at the origin."""

    rho: float

    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([self.rho * xp.cos(t), self.rho * xp.sin(t)], axis=-1)

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([-self.rho * xp.sin(t), self.rho * xp.cos(t)], axis=-1)

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([-self.rho * xp.cos(t), -self.rho * xp.sin(t)], axis=-1)


@attrs.define(frozen=True)
class KressShape(Shape):
    """
    Shape of x(t) = (cos(t) + 0.65 cos(2t) - 0.65, 1.5 sin(t)).

    References
    ----------
    Kress, R. (1991). Boundary integral equations in
    time-harmonic acoustic scattering.
    Mathematical and Computer Modelling, 15(3), 229--243.
    https://doi.org/10.1016/0895-7177(91)90068-I

    """

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
