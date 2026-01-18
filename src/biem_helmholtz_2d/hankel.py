from __future__ import annotations

from typing import Any, Callable

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace

from .quadrature import kussmaul_martensen_kress_quadrature, trapezoidal_quadrature


_EULER_GAMMA = 0.5772156649015328606


def _j0_series(z: Array, /, *, n_terms: int = 40) -> Array:
	# J0(z) = sum_{m>=0} (-1)^m (z^2/4)^m / (m!)^2
	xp = array_namespace(z)
	t = (z * z) / 4
	term = xp.ones_like(z)
	acc = term
	for m in range(1, n_terms):
		term = term * (-t) / (m * m)
		acc = acc + term
	return acc


def _y0_series(z: Array, /, *, n_terms: int = 40) -> Array:
	# Y0(z) = (2/pi)(log(z/2)+gamma)J0(z) + (2/pi) * sum_{m>=1} (-1)^{m+1} H_m (z^2/4)^m/(m!)^2
	xp = array_namespace(z)

	j0z = _j0_series(z, n_terms=n_terms)
	t = (z * z) / 4

	term = xp.ones_like(z)  # term for J0 series recurrence
	harmonic = 0.0
	remainder = xp.zeros_like(z)
	for m in range(1, n_terms):
		harmonic = harmonic + 1.0 / m
		term = term * (-t) / (m * m)  # now term == (-1)^m t^m/(m!)^2
		remainder = remainder + (-harmonic) * term

	two_over_pi = 2.0 / xp.pi
	return two_over_pi * (xp.log(z / 2) + _EULER_GAMMA) * j0z + two_over_pi * remainder


def j0(z: Array, /) -> Array:
	"""Bessel function of the first kind $J_0$ of shape (...)."""

	xp = array_namespace(z)
	az = xp.abs(z)
	small = az <= 8
	series = _j0_series(z)

	# Asymptotic for large |z|
	large = xp.sqrt(2.0 / (xp.pi * az)) * xp.cos(az - xp.pi / 4)
	return xp.where(small, series, large)


def y0(z: Array, /) -> Array:
	"""Bessel function of the second kind $Y_0$ of shape (...)."""

	xp = array_namespace(z)
	az = xp.abs(z)
	small = az <= 8

	# Avoid log(0) / invalid at exactly 0; callers typically override the singular point.
	safe = xp.where(az == 0, xp.asarray(1.0, dtype=z.dtype, device=z.device), az)

	series = _y0_series(safe)
	large = xp.sqrt(2.0 / (xp.pi * az)) * xp.sin(az - xp.pi / 4)
	out = xp.where(small, series, large)
	return xp.where(az == 0, xp.nan * xp.ones_like(out), out)


def integrate_neumann_y0_kress(
	n: int,
	f: Callable[[Array], Array],
	g: Callable[[Array], Array],
	fprime0: Array,
	/,
	*,
	xp: ArrayNamespaceFull,
	device: Any,
	dtype: Any,
) -> Array:
	r"""Integrate $\int_0^{2\pi} g(t) Y_0(f(t))\,dt$ with Kress splitting.

	This uses the decomposition from `docs/neumann.typ`:

	$$
	Y_0(f(t)) = Y_0^{(1,f)}(t)\,\log\left(4\sin^2\frac{t}{2}\right) + Y_0^{(2,f)}(t),
	\quad Y_0^{(1,f)}(t)=\frac{J_0(f(t))}{\pi}.
	$$

	Parameters
	----------
	n : int
		Harmonics parameter, with $N' := 2n-1$ quadrature nodes.
	f : Callable[[Array], Array]
		Map from nodes $t$ of shape (N',) to values $f(t)$ of shape (..., N').
	g : Callable[[Array], Array]
		Map from nodes $t$ of shape (N',) to values $g(t)$ of shape (..., N').
	fprime0 : Array
		Value of $f'(0)$ of shape (...).
	xp : ArrayNamespaceFull
		The array namespace.
	device : Any
		The device.
	dtype : Any
		The dtype.

	Returns
	-------
	Array
		Approximated integral of shape (...).

	"""

	t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
	_, r = kussmaul_martensen_kress_quadrature(n, xp=xp, device=device, dtype=dtype)

	ft = f(t)
	gt = g(t)

	y1 = j0(ft) / xp.pi
	log_sing = xp.log(4.0 * xp.square(xp.sin(t / 2.0)))

	y2 = y0(ft) - y1 * log_sing
	y2_0 = (2.0 / xp.pi) * (xp.log(fprime0 / 2.0) + _EULER_GAMMA)

	if t.shape[0] == 1:
		y2 = xp.expand_dims(y2_0, axis=-1)
	else:
		y2 = xp.concat([xp.expand_dims(y2_0, axis=-1), y2[..., 1:]], axis=-1)

	return xp.sum(gt * (r * y1 + w * y2), axis=-1)

