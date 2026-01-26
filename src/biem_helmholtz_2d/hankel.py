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
	t_singularity: float = 0,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> tuple[Array, Array]:
	r"""
	Split Neumann functions into log-singular and analytic parts on nodes ``x``.

	The split is

	$$
	f(x)^{\mathrm{order}} Y_{\mathrm{order}}(f(x))
	= Y_{\mathrm{order}}^{(1)}(x)\,\log\left(4\sin^2\frac{x - t_s}{2}\right) + Y_{\mathrm{order}}^{(2)}(x).
	$$

	Parameters
	----------
	x : Array
		Quadrature nodes of shape (N',).
	order : int
		Order of the Neumann function.
	f : Callable[[Array], Array]
		Function evaluated at nodes. It must accept input of shape (N',)
		and return an array of shape (..., N'). It is assumed to be smooth
		everywhere with $f(t_s) = 0$ and $f'(t_s) \ne 0$.
	fprime0 : Array | None
		Value $f'(t_s)$ of shape (...,) required when ``order == 0``.
	eps : float
		If ``abs(x - t_s) <= eps``, replace $Y^{(2)}$ by its limit value.
	t_singularity : float
		Singularity location $t_s$ in $[0, 2\pi)$.
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
	two_pi = 2 * xp.pi
	delta = xp.remainder(x - t_singularity + xp.pi, two_pi) - xp.pi
	log_kernel = xp.log(4 * xp.sin(delta / 2) ** 2)
	y2 = x_pow * yv - y1 * log_kernel

	if eps < 0:
		msg = "eps must be non-negative."
		raise ValueError(msg)

	near0 = xp.abs(delta) <= eps
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


def hankel_h1_h2(
	x: Array,
	order: int,
	f: Callable[[Array], Array],
	fprime0: Array | None = None,
	eps: float = 0.0,
	/,
	*,
	t_singularity: float = 0,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> tuple[Array, Array]:
	r"""
	Split Hankel functions of the first kind into log-singular and analytic parts.

	The split is

	$$
	f(x)^{\mathrm{order}} H_{\mathrm{order}}^{(1)}(f(x))
	= H_{\mathrm{order}}^{(1,1)}(x)\,\log\left(4\sin^2\frac{x - t_s}{2}\right)
	+ H_{\mathrm{order}}^{(1,2)}(x).
	$$

	Parameters
	----------
	x : Array
		Quadrature nodes of shape (N',).
	order : int
		Order of the Hankel function.
	f : Callable[[Array], Array]
		Function evaluated at nodes. It must accept input of shape (N',)
		and return an array of shape (..., N'). It is assumed to be smooth
		everywhere with $f(t_s) = 0$ and $f'(t_s) \ne 0$.
	fprime0 : Array | None
		Value $f'(t_s)$ of shape (...,) required when ``order == 0``.
	eps : float
		If ``abs(x - t_s) <= eps``, replace $H^{(1,2)}$ by its limit value.
	t_singularity : float
		Singularity location $t_s$ in $[0, 2\pi)$.
	xp : ArrayNamespaceFull
		The array namespace.
	device : Any
		The device.
	dtype : Any
		The dtype.

	Returns
	-------
	Array
		$H^{(1,1)}$ of shape (..., N').
	Array
		$H^{(1,2)}$ of shape (..., N').

	"""
	y1, y2 = neumann_y1_y2(
		x,
		order,
		f,
		fprime0,
		eps,
		t_singularity=t_singularity,
		xp=xp,
		device=device,
		dtype=dtype,
	)
	h1 = 1j * y1
	h2 = (xp.pi * y1) + (1j * y2)
	return h1, h2


