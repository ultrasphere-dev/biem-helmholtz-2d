from typing import Any

from array_api.latest import Array, ArrayNamespace
from array_api_compat import array_namespace
from ie_circle import KressShape

from ._acoustic import far_field, scattering_dirichlet


def example_3_1(n: int, /, *, xp: ArrayNamespace, dtype: Any, device: Any) -> Array:
    k = xp.asarray(1.0, device=device, dtype=dtype)
    eta = xp.asarray(0.0, device=device, dtype=dtype)
    alpha = xp.asarray(1.0)

    def incident_field(x: Array) -> Array:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    density = scattering_dirichlet(
        k=xp.asarray(1.0),
        shape=KressShape(),
        incident_field=incident_field,
        eta=eta,
        alpha=alpha,
        n=n,
    )
    return far_field(
        density=density,
        direction=xp.asarray((1.0, 0)),
        k=xp.asarray(1.0),
        shape=KressShape(),
        n=n,
        alpha=alpha,
        eta=eta,
    )


def example_3_1_answer() -> complex:
    return -1.62745750 + 0.60222591j
