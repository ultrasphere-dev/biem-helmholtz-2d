from __future__ import annotations

from collections.abc import Callable
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull

from .quadrature import kussmaul_martensen_kress_quadrature, trapezoidal_quadrature


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


def integrate_neumann_kress(
	n: int,
	order: int,
	f: Callable[[Array], Array],
	g: Callable[[Array], Array],
	fprime0: Array | None = None,
	/,
	*,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> Array:
	r"""
	Integrate $\int_0^{2\pi} g(t) Y_{\mathrm{order}}(f(t))\,dt$ with Kress+trapezoidal.

	Let $N' := 2n - 1$ and $t_j := 2\pi j / N'$.

	Parameters
	----------
	n : int
		Quadrature parameter, using $N' := 2n - 1$ points.
	order : int
		Order of the Neumann function $Y_\mathrm{order}$.
	f, g : Callable[[Array], Array]
		Functions evaluated at nodes $t_j$.
		They must accept input of shape (N',) and return arrays of shape (..., N').
	fprime0 : Array | None
		Value $f'(0)$ of shape (...,) required when ``order == 0``.
	xp : ArrayNamespaceFull
		The array namespace.
	device : Any
		The device.
	dtype : Any
		The dtype.

	Returns
	-------
	Array
		Integrated value of shape (...,).

	"""
	if order == 0 and fprime0 is None:
		msg = "fprime0 (shape (...,)) is required when order == 0."
		raise ValueError(msg)

	t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
	_, r = kussmaul_martensen_kress_quadrature(n, xp=xp, device=device, dtype=dtype)

	ft = f(t)
	gt = g(t)

	jv, yv = _scipy_jv_yv(order, ft, xp=xp, device=device, dtype=dtype)
	y1 = jv / xp.pi

	log_kernel = xp.log(4 * xp.sin(t / 2) ** 2)
	y2 = yv - y1 * log_kernel

	if order == 0:
		y2_0 = (2 / xp.pi) * (xp.log(fprime0 / 2) + _EULER_MASCHERONI)
	else:
		y2_0 = xp.zeros_like(y2[..., 0])

	y2 = xp.concat(
		[xp.expand_dims(y2_0, axis=-1), y2[..., 1:]],
		axis=-1,
	)

	return xp.sum(gt * (r * y1 + w * y2), axis=-1)
