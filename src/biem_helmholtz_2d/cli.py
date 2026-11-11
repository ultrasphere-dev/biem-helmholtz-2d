import pathlib
import warnings
from datetime import datetime
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Annotated, Any

import array_api_compat.numpy as xp
import typer
from rich.logging import RichHandler

from biem_helmholtz_2d.optimization._accuracy import example_accuracy, example_accuracy_plot
from biem_helmholtz_2d.optimization._example import example_optimization, example_optimization_plot

try:
    import array_api_compat.torch as torch_xp
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch_xp = None

app = typer.Typer(pretty_exceptions_enable=False)

LOG = getLogger(__name__)

_CENTERS = [[0, 0], [5, 0]]
_K = 5
_ALPHA = 1
_ETA = 1
_TARGET_POINT = (-3, 3)
_DESIRED_TOTAL_FIELD = 0
_N_SOLVE = 20
_N_MODES = 10


def _namespace(device: str, /) -> Any:
    if device == "cpu":
        return xp
    if device == "cuda" or device.startswith("cuda:"):
        if torch_xp is None:
            raise typer.BadParameter("CUDA requires the 'cuda' extra: uv sync --extra cuda")
        if not torch_xp.cuda.is_available():
            raise typer.BadParameter("PyTorch cannot access CUDA on this system")
        return torch_xp
    raise typer.BadParameter("device must be 'cpu', 'cuda', or 'cuda:N'")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = INFO
    if verbose:
        level = DEBUG
    basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def optimize(
    device: str = typer.Option("cpu", help="Array device: cpu, cuda, or cuda:N."),
    n_quadrature: int | None = typer.Option(
        None, help="Field and sensitivity integration order (defaults to the Nyström order)."
    ),
) -> None:
    """Compare optimization results with/without Hilbertian regularization."""
    warnings.filterwarnings("ignore")
    path = pathlib.Path(f"optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    namespace = _namespace(device)
    dtype = namespace.float64
    for alpha_reg in [
        0,
        1,
        1e-2,
        1e-4,
    ]:
        example_optimization(
            k=_K,
            centers=namespace.asarray(_CENTERS, dtype=dtype, device=device),
            n=_N_SOLVE,
            n_quadrature=n_quadrature,
            alpha=_ALPHA,
            eta=_ETA,
            n_modes=_N_MODES,
            n_steps=20,
            target_point=_TARGET_POINT,
            desired_total_field=_DESIRED_TOTAL_FIELD,
            alpha_reg=alpha_reg,
            path=path / f"alpha_reg_{alpha_reg}",
        )


@app.command()
def accuracy(
    n_reference: Annotated[int, typer.Option(help="Reference order for both stages.")],
    device: str = typer.Option("cpu", help="Array device: cpu, cuda, or cuda:N."),
    n_solve: Annotated[
        list[int] | None, typer.Option(help="Nyström orders; repeat for each value.")
    ] = None,
    n_quadrature: Annotated[
        list[int] | None, typer.Option(help="Integration orders; repeat for each value.")
    ] = None,
    step: float = typer.Option(1e-5, help="Central-difference step."),
    seed: int = typer.Option(0, help="Random seed for reproducible stress cases."),
) -> None:
    """Export field and sensitivity errors for independent quadrature orders."""
    warnings.filterwarnings("ignore")
    namespace = _namespace(device)
    solve_orders = n_solve or [_N_SOLVE // 2, _N_SOLVE]
    quadrature_orders = n_quadrature or [_N_SOLVE // 2, _N_SOLVE]
    path = pathlib.Path(f"accuracy/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    example_accuracy(
        path=path,
        centers=namespace.asarray(_CENTERS, dtype=namespace.float64, device=device),
        k=_K,
        alpha=_ALPHA,
        eta=_ETA,
        target_point=_TARGET_POINT,
        desired_total_field=_DESIRED_TOTAL_FIELD,
        solve_orders=solve_orders,
        quadrature_orders=quadrature_orders,
        reference_order=n_reference,
        step=step,
        n_modes=_N_MODES,
        seed=seed,
    )
    LOG.info("Saved accuracy studies to %s", path)


@app.command("plot-accuracy")
def plot_accuracy(path: pathlib.Path | None = None) -> None:
    """Plot studies under a directory, or all studies in accuracy/."""
    root = path if path is not None else pathlib.Path("accuracy")
    paths = [file.parent for file in root.rglob("accuracy.json")]
    if not paths:
        LOG.warning("No accuracy studies found.")
        return
    for directory in paths:
        example_accuracy_plot(directory)
        LOG.info("Saved %s", directory / "accuracy.svg")


_JSON_FILES = ("optimization_history.json", "optimized_shape.json", "optimized_near_field.json")


@app.command()
def plot() -> None:
    """Plot all optimization runs that have JSON data but missing PNG files."""
    optimization_dir = pathlib.Path("optimization")
    if not optimization_dir.is_dir():
        LOG.warning("No optimization/ directory found.")
        return

    unplotted = []
    for subdir in sorted(optimization_dir.rglob("*")):
        if not subdir.is_dir():
            continue
        has_all_json = all((subdir / j).exists() for j in _JSON_FILES)
        has_any_svg = any(subdir.rglob("*.svg"))
        if has_all_json and not has_any_svg:
            unplotted.append(subdir)

    if not unplotted:
        LOG.info("No unplotted optimization runs found.")
        return

    LOG.info("Plotting %d run(s)...", len(unplotted))
    for d in unplotted:
        LOG.info("Plotting %s", d.name)
        example_optimization_plot(d)
