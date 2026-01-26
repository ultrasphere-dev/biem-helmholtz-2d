from __future__ import annotations

from typing import Any

import pytest  # type: ignore[import-not-found]

from biem_helmholtz_2d.quadrature import (
	trapezoidal_quadrature,
	garrick_wittich_quadrature,
	kussmaul_martensen_kress_quadrature,
)


@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_kussmaul_martensen_kress_quadrature_exactness(
	xp: Any, device: Any, dtype: Any, f_case: str
) -> None:
	n = 6
	t, w = kussmaul_martensen_kress_quadrature(n, xp=xp, device=device, dtype=dtype)
	two_pi = xp.pi * 2

	if f_case == "one":
		f = xp.ones_like(t)
		expected = xp.zeros_like(t)
	elif f_case == "exp1":
		f = xp.exp(1j * t)
		expected = -two_pi * xp.exp(1j * t)
		# From docs/quadrature.typ:
		# ∫ log(4 sin^2((τ-t)/2)) e^{i m τ} dτ = e^{i m t} (-2π/|m|).
	else:
		f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
		expected = -(two_pi / 3) * xp.exp(1j * 3 * t) - (two_pi / 4) * xp.exp(-1j * 4 * t)

	approx = xp.sum(w * f[None, :], axis=1)
	assert xp.max(xp.abs(approx - expected)) < 1e-10


@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_garrick_wittich_quadrature_exactness(xp: Any, device: Any, dtype: Any, f_case: str) -> None:
	n = 6
	t, w = garrick_wittich_quadrature(n, xp=xp, device=device, dtype=dtype)
	two_pi = xp.pi * 2

	if f_case == "one":
		f = xp.ones_like(t)
		expected = xp.zeros_like(t)
	elif f_case == "exp1":
		f = xp.exp(1j * t)
		expected = (two_pi * 1j) * xp.exp(1j * t)
		# From docs/quadrature.typ:
		# p.v.∫ cot((τ-t)/2) e^{i m τ} dτ = e^{i m t} (2π i sgn(m)).
	else:
		f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
		expected = (
			(two_pi * 1j) * xp.exp(1j * 3 * t)
			+ (-two_pi * 1j) * xp.exp(-1j * 4 * t)
		)

	approx = xp.sum(w * f[None, :], axis=1)
	assert xp.max(xp.abs(approx - expected)) < 1e-10


@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_trapezoidal_quadrature_exactness(xp: Any, device: Any, dtype: Any, f_case: str) -> None:
	n = 6
	t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
	two_pi = xp.pi * 2

	if f_case == "one":
		f = xp.ones_like(t)
		expected = two_pi  # Trapezoidal is exact on Fourier modes with |m|<n (here m=0).
	elif f_case == "exp1":
		f = xp.exp(1j * t)
		expected = 0
	else:
		f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
		expected = two_pi  # Nonzero modes integrate to 0; only the constant term remains.

	approx = w * xp.sum(f)
	assert xp.abs(approx - expected) < 1e-10
