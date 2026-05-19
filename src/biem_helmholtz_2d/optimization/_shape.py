from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import Shape


class RadiusShape(Shape):
    def r(self, t: Array, /) -> Array:
        raise NotImplementedError

    def dr(self, t: Array, /) -> Array:
        raise NotImplementedError

    def ddr(self, t: Array, /) -> Array:
        raise NotImplementedError

    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        r = self.r(t)
        return xp.stack([r * xp.cos(t), r * xp.sin(t)], axis=-1)

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        r = self.r(t)
        dr = self.dr(t)
        return xp.stack([dr * xp.cos(t) - r * xp.sin(t), dr * xp.sin(t) + r * xp.cos(t)], axis=-1)

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        r = self.r(t)
        dr = self.dr(t)
        ddr = self.ddr(t)
        return xp.stack(
            [
                ddr * xp.cos(t) - 2 * dr * xp.sin(t) - r * xp.cos(t),
                ddr * xp.sin(t) + 2 * dr * xp.cos(t) - r * xp.sin(t),
            ],
            axis=-1,
        )


class ParameterShape(RadiusShape):
    def __init__(self, parameters: Array, /) -> None:
        """
        Parameterized shape.

        The radius is
        (1, cos t, ..., cos (m - 1) t, sin t, ..., sin (m - 1) t) @ parameters.
        """
        super().__init__()
        if parameters.shape[0] % 2 != 1:
            raise ValueError("The number of parameters must be odd.")
        self.parameters = parameters
        self.m = parameters.shape[0] // 2 + 1

    def r(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.parameters)
        t = t[..., None]
        m_range = xp.arange(self.m, dtype=t.dtype, device=t.device)
        m_range_sin = xp.arange(1, self.m, dtype=t.dtype, device=t.device)
        basis = xp.concat(
            [xp.cos(t * m_range), xp.sin(t * m_range_sin)],
            axis=-1,
        )
        return xp.sum(self.parameters * basis, axis=-1)

    def dr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.parameters)
        t = t[..., None]
        m_range = xp.arange(self.m, dtype=t.dtype, device=t.device)
        m_range_sin = xp.arange(1, self.m, dtype=t.dtype, device=t.device)
        basis_derivative = xp.concat(
            [-xp.sin(t * m_range) * m_range, xp.cos(t * m_range_sin) * m_range_sin],
            axis=-1,
        )
        return xp.sum(self.parameters * basis_derivative, axis=-1)

    def ddr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.parameters)
        t = t[..., None]
        m_range = xp.arange(self.m, dtype=t.dtype, device=t.device)
        m_range_sin = xp.arange(1, self.m, dtype=t.dtype, device=t.device)
        basis_second_derivative = xp.concat(
            [
                -xp.cos(t * m_range) * (m_range**2),
                -xp.sin(t * m_range_sin) * (m_range_sin**2),
            ],
            axis=-1,
        )
        return xp.sum(self.parameters * basis_second_derivative, axis=-1)
