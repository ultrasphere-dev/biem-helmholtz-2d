"""Tests for multi-scatterer acoustic scattering."""

from typing import Any

import pytest
from array_api.latest import ArrayNamespace
from ie_circle import CircleShape, ShapeList

from biem_helmholtz_2d._acoustic import far_field, near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_two_circle_scattering(
    xp: ArrayNamespace,
    device: Any,
    dtype: Any,
    seed: int,
) -> None:
    """Test scattering from two circles with randomized parameters."""
    import numpy as np

    rng = np.random.default_rng(seed)

    # Random parameters
    k_val = rng.uniform(0.5, 2.0)
    radius1 = rng.uniform(0.3, 0.8)
    radius2 = rng.uniform(0.3, 0.8)
    center1 = rng.uniform(-2.0, 2.0, size=2)
    center2 = center1 + rng.uniform(2.0, 4.0, size=2)  # Ensure separation

    # Create shapes
    shape1 = CircleShape(radius1)
    # For translated circle, we need a custom shape

    class TranslatedCircle:
        def __init__(self, rho: float, center: Any) -> None:
            self._rho = rho
            self._center = xp.asarray(center, device=device, dtype=dtype)

        def x(self, t: Any) -> Any:
            xp_local = xp
            return xp_local.stack(
                [
                    self._rho * xp_local.cos(t) + self._center[0],
                    self._rho * xp_local.sin(t) + self._center[1],
                ],
                axis=-1,
            )

        def dx(self, t: Any) -> Any:
            xp_local = xp
            return xp_local.stack(
                [
                    -self._rho * xp_local.sin(t),
                    self._rho * xp_local.cos(t),
                ],
                axis=-1,
            )

        def ddx(self, t: Any) -> Any:
            xp_local = xp
            return xp_local.stack(
                [
                    -self._rho * xp_local.cos(t),
                    -self._rho * xp_local.sin(t),
                ],
                axis=-1,
            )

    shape2 = TranslatedCircle(radius2, center2)

    # Create Shapes object
    shapes = ShapeList([shape1, shape2])

    # Setup scattering problem
    k = xp.asarray(k_val, device=device, dtype=dtype)
    alpha = xp.asarray([1.0, 1.0], device=device, dtype=dtype)
    eta = xp.asarray([1.0, 1.0], device=device, dtype=dtype)
    direction = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
    incident_field = plane_wave(k, direction)

    n = 24

    # Solve for density
    density = scattering_dirichlet(
        k=k,
        shapes=shapes,
        incident_field=incident_field,
        alpha=alpha,
        eta=eta,
        n=n,
    )

    # Check density shape
    t_test = xp.linspace(0, 2 * np.pi, 5, device=device, dtype=dtype)
    density_vals = density(t_test)
    assert density_vals.shape == (5, 2), f"Expected shape (5, 2), got {density_vals.shape}"

    # Compute near field at test points
    x_test = xp.asarray(
        [[0.0, 3.0], [3.0, 0.0], [-2.0, -2.0]],
        device=device,
        dtype=dtype,
    )
    uscat = near_field(density, x_test, n=n, shapes=shapes, k=k, alpha=alpha, eta=eta)
    assert uscat.shape == (3,), f"Expected shape (3,), got {uscat.shape}"

    # Compute far field
    direction_ff = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
    ff = far_field(density, direction_ff, n=n, shapes=shapes, k=k, alpha=alpha, eta=eta)
    assert ff.shape == (), f"Expected scalar shape, got {ff.shape}"

    # Verify results are finite
    assert xp.all(xp.isfinite(uscat)), "Near field contains non-finite values"
    assert xp.isfinite(ff), "Far field is non-finite"
