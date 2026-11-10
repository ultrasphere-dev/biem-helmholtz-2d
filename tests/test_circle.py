from typing import Any

from array_api.latest import ArrayNamespace
from biem_helmholtz_sphere import biem
from biem_helmholtz_sphere import plane_wave as sphere_plane_wave
from ultrasphere import create_from_branching_types

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave
from biem_helmholtz_2d.optimization._shape import ParameterShape


def test_three_circles(xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    """Compare the scattered field for three sound-soft circles with the sphere solver."""
    centers = xp.asarray([[0, 0], [4, 0], [1.5, 3.5]], dtype=dtype, device=device)
    radii = xp.asarray([0.8, 1, 0.6], dtype=dtype, device=device)
    points = xp.asarray(
        [[-2, 2], [2, 2], [6, 1], [0, -3], [3, 4], [2, -1.5]], dtype=dtype, device=device
    )
    direction = xp.asarray([1, 0.4], dtype=dtype, device=device)
    k = xp.asarray(1.2, dtype=dtype, device=device)
    alpha = xp.asarray(1, dtype=dtype, device=device)
    eta = xp.asarray(1, dtype=dtype, device=device)
    shapes = [
        ParameterShape(
            center=centers[i],
            cos_coefs=radii[i : i + 1],
            sin_coefs=xp.zeros(0, dtype=dtype, device=device),
        )
        for i in range(3)
    ]
    density = scattering_dirichlet(
        k=k, shape=shapes, incident_field=plane_wave(k, direction), alpha=alpha, eta=eta, n=16
    )
    actual = near_field(density, points, n=16, shape=shapes, k=k, alpha=alpha, eta=eta)

    incident, incident_grad = sphere_plane_wave(k=k, direction=direction)
    reference = biem(
        create_from_branching_types("a"),
        centers=centers,
        radii=radii,
        k=k,
        n_end=12,
        eta=eta,
        uin=incident,
        uin_grad=incident_grad,
        alpha=1,
        beta=0,
        kind="outer",
    )
    expected = reference.uscat(xp.moveaxis(points, -1, 0))
    assert actual.shape == expected.shape == (6,)
    tolerance = 2e-8 + 200 * xp.finfo(dtype).eps
    assert bool(xp.all(xp.abs(actual - expected) <= tolerance * (1 + xp.abs(expected))))
