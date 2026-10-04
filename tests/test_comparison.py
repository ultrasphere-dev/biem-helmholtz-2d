"""Comparison tests between biem_helmholtz_2d and biem_helmholtz_sphere."""

import numpy as np
import pytest
import ultrasphere as us
from array_api.latest import ArrayNamespace
from biem_helmholtz_sphere import biem as sphere_biem
from biem_helmholtz_sphere import plane_wave as sphere_plane_wave
from ie_circle import ShapeList

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave


class TranslatedCircle:
    """Circle with arbitrary center."""

    def __init__(self, radius: float, center: tuple[float, float]) -> None:
        self._radius = radius
        self._center = center

    def x(self, t):
        xp = np
        return xp.stack(
            [
                self._center[0] + self._radius * xp.cos(t),
                self._center[1] + self._radius * xp.sin(t),
            ],
            axis=-1,
        )

    def dx(self, t):
        xp = np
        return xp.stack(
            [
                -self._radius * xp.sin(t),
                self._radius * xp.cos(t),
            ],
            axis=-1,
        )

    def ddx(self, t):
        xp = np
        return xp.stack(
            [
                -self._radius * xp.cos(t),
                -self._radius * xp.sin(t),
            ],
            axis=-1,
        )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_two_circle_comparison(
    xp: ArrayNamespace,
    device: object,
    dtype: object,
    seed: int,
) -> None:
    """Compare 2D scattered field from biem_helmholtz_2d and biem_helmholtz_sphere."""
    rng = np.random.default_rng(seed)

    # Random parameters
    k_val = rng.uniform(0.5, 2.0)
    radius1 = rng.uniform(0.3, 0.8)
    radius2 = rng.uniform(0.3, 0.8)
    center1 = tuple(rng.uniform(-1.0, 1.0, size=2))
    center2 = tuple(center1 + rng.uniform(2.5, 4.0, size=2))

    # Setup for biem_helmholtz_2d
    k_2d = xp.asarray(k_val, device=device, dtype=dtype)
    alpha_2d = xp.asarray([1.0, 1.0], device=device, dtype=dtype)
    eta_2d = xp.asarray([1.0, 1.0], device=device, dtype=dtype)
    direction_2d = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
    uin_2d = plane_wave(k_2d, direction_2d)

    shapes_2d = ShapeList([
        TranslatedCircle(radius1, center1),
        TranslatedCircle(radius2, center2),
    ])

    # Solve with biem_helmholtz_2d
    n_quad = 64
    phi_2d = scattering_dirichlet(
        shapes=shapes_2d,
        k=k_2d,
        incident_field=uin_2d,
        alpha=alpha_2d,
        eta=eta_2d,
        n=n_quad,
    )

    # Evaluation points
    n_eval = 5
    eval_points_2d = rng.uniform(-3.0, 3.0, size=(n_eval, 2))
    # Ensure points are outside both circles
    for i in range(n_eval):
        while (
            np.linalg.norm(eval_points_2d[i] - np.array(center1)) < radius1 + 0.1
            or np.linalg.norm(eval_points_2d[i] - np.array(center2)) < radius2 + 0.1
        ):
            eval_points_2d[i] = rng.uniform(-3.0, 3.0, size=2)

    # Evaluate scattered field with biem_helmholtz_2d
    x_eval_2d = xp.asarray(eval_points_2d, device=device, dtype=dtype)
    uscat_2d = near_field(
        phi_2d,
        x_eval_2d,
        shapes=shapes_2d,
        k=k_2d,
        alpha=alpha_2d,
        eta=eta_2d,
        n=n_quad,
    )

    # Setup for biem_helmholtz_sphere
    c_polar = us.create_polar()
    k_sphere = np.asarray(k_val)
    direction_sphere = np.asarray([1.0, 0.0])
    uin_sphere, _ = sphere_plane_wave(k=k_sphere, direction=direction_sphere)

    centers_sphere = np.asarray([center1, center2])
    radii_sphere = np.asarray([radius1, radius2])
    eta_sphere = np.asarray(1.0)

    # Solve with biem_helmholtz_sphere
    n_end = 30
    calc_sphere = sphere_biem(
        c_polar,
        uin=uin_sphere,
        centers=centers_sphere,
        radii=radii_sphere,
        k=k_sphere,
        n_end=n_end,
        eta=eta_sphere,
        kind="outer",
    )

    # Evaluate scattered field with biem_helmholtz_sphere
    # biem_helmholtz_sphere expects shape (c_ndim, n_points)
    x_eval_sphere = eval_points_2d.T
    uscat_sphere = calc_sphere.uscat(x_eval_sphere, expand_x=False)

    # Compare results
    uscat_2d_np = np.asarray(uscat_2d)
    uscat_sphere_np = np.asarray(uscat_sphere)

    # Check shapes match
    assert uscat_2d_np.shape == uscat_sphere_np.shape, (
        f"Shape mismatch: {uscat_2d_np.shape} vs {uscat_sphere_np.shape}"
    )

    # Check values are close
    max_diff = np.max(np.abs(uscat_2d_np - uscat_sphere_np))
    rel_error = max_diff / np.max(np.abs(uscat_sphere_np))

    print(f"\nSeed {seed}:")
    print(f"  k={k_val:.3f}, r1={radius1:.3f}, r2={radius2:.3f}")
    print(f"  center1={center1}, center2={center2}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative error: {rel_error:.2e}")

    # Should be close (within numerical tolerance)
    assert rel_error < 1e-2, f"Relative error too large: {rel_error:.2e}"
