from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
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
        return xp.stack([r * xp.cos(t)[..., None], r * xp.sin(t)[..., None]], axis=-1)

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        r, dr = self.r(t), self.dr(t)
        c, s = xp.cos(t)[..., None], xp.sin(t)[..., None]
        return xp.stack([dr * c - r * s, dr * s + r * c], axis=-1)

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        r, dr, ddr = self.r(t), self.dr(t), self.ddr(t)
        c, s = xp.cos(t)[..., None], xp.sin(t)[..., None]
        return xp.stack([ddr * c - 2 * dr * s - r * c, ddr * s + 2 * dr * c - r * s], axis=-1)


@frozen
class ParameterShape(RadiusShape):
    sin_coefs: Array
    cos_coefs: Array
    center: Array

    def __attrs_post_init__(self) -> None:
        array_namespace(self.cos_coefs, self.sin_coefs, self.center)
        if min(self.cos_coefs.ndim, self.sin_coefs.ndim, self.center.ndim) < 2:
            raise ValueError("coefficients and centers must have a boundary dimension")
        check_shapes("...CM,...CN,...CD,D", self.cos_coefs, self.sin_coefs, self.center, (2,))
        if (
            self.center.shape[-1] != 2
            or self.cos_coefs.shape[-2] != self.center.shape[-2]
            or self.sin_coefs.shape[-2] != self.center.shape[-2]
        ):
            raise ValueError("coefficients and centers must have one row per boundary")
        if self.cos_coefs.shape[-1] != self.sin_coefs.shape[-1] + 1:
            raise ValueError("cosine coefficients must include the constant term")

    def x(self, t: Array, /) -> Array:
        return super().x(t) + self.center

    def r(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        return xp.sum(self.cos_coefs * xp.cos(t[..., None, None] * m_cos), axis=-1) + xp.sum(
            self.sin_coefs * xp.sin(t[..., None, None] * m_sin), axis=-1
        )

    def dr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        return xp.sum(
            -self.cos_coefs * m_cos * xp.sin(t[..., None, None] * m_cos), axis=-1
        ) + xp.sum(self.sin_coefs * m_sin * xp.cos(t[..., None, None] * m_sin), axis=-1)

    def ddr(self, t: Array, /) -> Array:
        xp = array_namespace(t, self.cos_coefs, self.sin_coefs)
        m_cos = xp.arange(self.cos_coefs.shape[-1], dtype=t.dtype, device=t.device)
        m_sin = xp.arange(1, self.sin_coefs.shape[-1] + 1, dtype=t.dtype, device=t.device)
        return xp.sum(
            -self.cos_coefs * m_cos**2 * xp.cos(t[..., None, None] * m_cos), axis=-1
        ) + xp.sum(-self.sin_coefs * m_sin**2 * xp.sin(t[..., None, None] * m_sin), axis=-1)
