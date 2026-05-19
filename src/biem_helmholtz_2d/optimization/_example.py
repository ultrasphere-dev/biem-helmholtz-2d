import pathlib
from datetime import datetime
from typing import Any

from array_api.latest import Array, ArrayNamespace
from array_api_compat import array_namespace
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import trange

from biem_helmholtz_2d.optimization._shape import ParameterShape

from .._acoustic import near_field, plot_ner_field, scattering_dirichlet


def example_optimization(*, xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    # start from circle
    n = 10
    parameters = xp.zeros(n * 2 - 1, dtype=dtype, device=device)
    parameters[0] = 1
    k = xp.asarray(1.0, device=device, dtype=dtype)
    eta = xp.asarray(0.0, device=device, dtype=dtype)
    alpha_ = xp.asarray(1.0, device=device, dtype=dtype)
    point_to_minimize = xp.asarray([[3, 3]], dtype=dtype, device=device)

    def incident_field(x: Array) -> Array:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    def objective(
        parameters: Array, /, *, ax_re: Axes | None = None, ax_abs: Axes | None = None
    ) -> Array:
        shape = ParameterShape(parameters)
        density = scattering_dirichlet(
            k=k,
            shape=shape,
            incident_field=incident_field,
            eta=eta,
            alpha=alpha_,
            n=n,
        )
        u_in = incident_field(point_to_minimize)
        u_scat = near_field(
            density,
            point_to_minimize,
            k=k,
            shape=shape,
            n=n + 20,
            alpha=alpha_,
            eta=eta,
        )
        u = u_in + u_scat
        if ax_re is not None or ax_abs is not None:
            plot_ner_field(
                density,
                incident_field,
                xlim=(-6.0, 6.0),
                ylim=(-6.0, 6.0),
                k=k,
                shape=shape,
                n=n,
                alpha=alpha_,
                eta=eta,
                ax_re=ax_re,
                ax_im=None,
                ax_abs=ax_abs,
            )
        return xp.sum(xp.abs(u) ** 2)

    def objective_num_diff(parameters: Array, /, *, eps: float = 1e-4) -> Array:
        grad = []
        for i in range(parameters.shape[0]):
            parameters[i] += eps
            u_plus = objective(parameters)
            parameters[i] -= 2 * eps
            u_minus = objective(parameters)
            parameters[i] += eps
            u_grad_i = (u_plus - u_minus) / (2 * eps)
            grad.append(u_grad_i)
        return xp.stack(grad)

    # simple optimization
    alpha = 0.1
    beta = 0.1
    val_hist = []

    path = pathlib.Path(
        f"optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}_k{k}_a{alpha}_b{beta}_n{n}"
    )
    path.mkdir(parents=True, exist_ok=True)
    pbar = trange(25)
    for i in pbar:
        if i % 5 == 0:
            fig_opt, (ax_re, ax_abs) = plt.subplots(1, 2, figsize=(10, 5))
            val = objective(parameters, ax_re=ax_re, ax_abs=ax_abs)
            fig_opt.suptitle(f"Iteration {i}")
            fig_opt.savefig(path / f"{i:03d}.png")
            plt.close(fig_opt)
        else:
            val = objective(parameters)
        pbar.set_postfix({"objective": val})

        grad = objective_num_diff(parameters)
        # reguralization
        grad /= (
            1
            + alpha
            * xp.concat(
                [
                    xp.arange(n, dtype=dtype, device=device),
                    xp.arange(1, n, dtype=dtype, device=device),
                ]
            )
            ** 3
        )
        # descent step
        parameters -= beta * grad
        # parameters[0] = 1
        # parameters = torch.clamp(parameters, -1, 1)
        # parameters[0] = 1
        # parameters[1:] = parameters[1:] / (2 * torch.sum(torch.abs(parameters[1:])))
        val_hist.append(float(val))

    fig, ax = plt.subplots()
    ax.plot(val_hist)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optimization history\n(k={k}, alpha={alpha}, beta={beta}, n={n})")
    fig.savefig(path / "optimization_history.png")
