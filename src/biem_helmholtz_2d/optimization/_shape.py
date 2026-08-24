from array_api._2024_12 import Array
from array_api_compat import array_namespace
from attrs import frozen
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


@frozen
class ParameterShape(RadiusShape):
    sin_coefs: Array
    cos_coefs: Array

    def r(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        t_expanded = t[..., None]
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        cos_basis = xp.cos(t_expanded * m_cos)
        sin_basis = xp.sin(t_expanded * m_sin)
        cos_coefs = self.cos_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        sin_coefs = self.sin_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        return xp.sum(cos_coefs * cos_basis, axis=-1) + xp.sum(sin_coefs * sin_basis, axis=-1)

    def dr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        t_expanded = t[..., None]
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        cos_basis_derivative = -xp.sin(t_expanded * m_cos) * m_cos
        sin_basis_derivative = xp.cos(t_expanded * m_sin) * m_sin
        cos_coefs = self.cos_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        sin_coefs = self.sin_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        return xp.sum(cos_coefs * cos_basis_derivative, axis=-1) + xp.sum(
            sin_coefs * sin_basis_derivative, axis=-1
        )

    def ddr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        t_expanded = t[..., None]
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        cos_basis_second_derivative = -xp.cos(t_expanded * m_cos) * (m_cos**2)
        sin_basis_second_derivative = -xp.sin(t_expanded * m_sin) * (m_sin**2)
        cos_coefs = self.cos_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        sin_coefs = self.sin_coefs[(...,) + (None,) * t.ndim + (slice(None),)]
        return xp.sum(cos_coefs * cos_basis_second_derivative, axis=-1) + xp.sum(
            sin_coefs * sin_basis_second_derivative, axis=-1
        )
