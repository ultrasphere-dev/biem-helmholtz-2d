from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull


_EULER_MASCHERONI: float = 0.57721566490153286060651209008240243104215933593992


def _scipy_jv_yv(
	order: int,
	x: Array,
	/,
	*,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> tuple[Array, Array]:
	from scipy.special import jv, yv

	x_cpu = xp.asarray(x, device="cpu")
	j = jv(order, x_cpu)
	y = yv(order, x_cpu)
	return xp.asarray(j, device=device, dtype=dtype), xp.asarray(y, device=device, dtype=dtype)


def neumann_y1_y2(
	x: Array,
	order: int,
	f: Callable[[Array], Array],
	fprime0: Array | None = None,
	eps: float = 0.0,
	/,
	*,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> tuple[Array, Array]:
	r"""
	Split Neumann functions into log-singular and analytic parts on nodes ``x``.

	For ``order == 0`` the split is

	$$
	Y_0(f(x)) = Y_0^{(1)}(x)\,\log\left(4\sin^2\frac{x}{2}\right) + Y_0^{(2)}(x).
	$$

	For ``order > 0`` the split is

	$$
	f(x)^{\mathrm{order}} Y_{\mathrm{order}}(f(x))
	= Y_{\mathrm{order}}^{(1)}(x)\,\log\left(4\sin^2\frac{x}{2}\right) + Y_{\mathrm{order}}^{(2)}(x).
	$$

	Parameters
	----------
	x : Array
		Quadrature nodes of shape (N',).
	order : int
		Order of the Neumann function.
	f : Callable[[Array], Array]
		Function evaluated at nodes. It must accept input of shape (N',)
		and return an array of shape (..., N').
	fprime0 : Array | None
		Value $f'(0)$ of shape (...,) required when ``order == 0``.
	eps : float
		If ``abs(x) <= eps``, replace $Y^{(2)}$ by its limit value.
	xp : ArrayNamespaceFull
		The array namespace.
	device : Any
		The device.
	dtype : Any
		The dtype.

	Returns
	-------
	Array
		$Y^{(1)}$ of shape (..., N').
	Array
		$Y^{(2)}$ of shape (..., N').

	"""
	if order == 0 and fprime0 is None:
		msg = "fprime0 (shape (...,)) is required when order == 0."
		raise ValueError(msg)

	fx = f(x)
	jv, yv = _scipy_jv_yv(order, fx, xp=xp, device=device, dtype=dtype)

	if order == 0:
		x_pow = 1
	else:
		x_pow = fx**order

	y1 = x_pow * jv / xp.pi
	log_kernel = xp.log(4 * xp.sin(x / 2) ** 2)
	y2 = x_pow * yv - y1 * log_kernel

	if eps < 0:
		msg = "eps must be non-negative."
		raise ValueError(msg)

	near0 = xp.abs(x) <= eps
	if order == 0:
		assert fprime0 is not None
		y2_lim = (2 / xp.pi) * (
			xp.log(xp.abs(fprime0) / 2) + _EULER_MASCHERONI
		)
	else:
		limit_scalar = -(
			(2**order) * math.factorial(order - 1)
		) / xp.pi
		y2_lim = xp.full_like(y2[..., 0], limit_scalar)

	y2 = xp.where(near0, y2_lim[..., None], y2)
	return y1, y2
