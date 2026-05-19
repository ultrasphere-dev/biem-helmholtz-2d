from typing import Any

from array_api.latest import Array, ArrayNamespace
from array_api_compat import array_namespace
from ie_circle import KressShape
from matplotlib import pyplot as plt

from ._acoustic import far_field, plot_ner_field, scattering_dirichlet


def example_3_1(n: int, /, *, xp: ArrayNamespace, dtype: Any, device: Any) -> Array:
    k = xp.asarray(1.0, device=device, dtype=dtype)
    eta = xp.asarray(0.0, device=device, dtype=dtype)
    alpha = xp.asarray(1.0, device=device, dtype=dtype)
    shape = KressShape()

    def incident_field(x: Array) -> Array:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    density = scattering_dirichlet(
        k=k,
        shape=shape,
        incident_field=incident_field,
        eta=eta,
        alpha=alpha,
        n=n,
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_ner_field(
        density,
        incident_field,
        xlim=(-6.0, 6.0),
        ylim=(-6.0, 6.0),
        k=k,
        shape=shape,
        n=n,
        alpha=alpha,
        eta=eta,
        ax_re=ax[0],
        ax_im=ax[1],
        ax_abs=ax[2],
    )
    fig.savefig("example_3_1.png")
    direction = xp.asarray((1.0, 0), device=device, dtype=dtype)
    return far_field(
        density,
        direction,
        k=k,
        shape=shape,
        n=n,
        alpha=alpha,
        eta=eta,
    )


def example_3_1_answer() -> complex:
    return -1.62745750 + 0.60222591j
