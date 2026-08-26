import pathlib
import warnings
from datetime import datetime
from logging import DEBUG, INFO, basicConfig, getLogger

import typer
from rich.logging import RichHandler

from biem_helmholtz_2d.optimization._example import example_optimization, example_optimization_plot

app = typer.Typer(pretty_exceptions_enable=False)

LOG = getLogger(__name__)


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
def optimize() -> None:
    """Compare optimization results with/without Hilbertian regularization."""
    warnings.filterwarnings("ignore")
    path = pathlib.Path(f"optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    import numpy as np

    xp = np
    dtype = np.float64
    device = None
    for alpha_reg in [
        0,
        1,
        1e-2,
        1e-4,
    ]:
        example_optimization(
            k=5,
            n=100,
            alpha=1,
            eta=1,
            n_modes=10,
            n_steps=20,
            target_point=(-3, 3),
            desired_total_field=0,
            xp=xp,
            dtype=dtype,
            device=device,
            alpha_reg=alpha_reg,
            path=path / f"alpha_reg_{alpha_reg}",
        )


_JSON_FILES = ("optimization_history.json", "optimized_shape.json", "optimized_near_field.json")
_PNG_FILES = (
    "optimization_history.png",
    "optimized_shape.png",
    "optimized_near_field.png",
    "optimized_coefficients.png",
)


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
        has_all_png = all((subdir / p).exists() for p in _PNG_FILES)
        if has_all_json and not has_all_png:
            unplotted.append(subdir)

    if not unplotted:
        LOG.info("No unplotted optimization runs found.")
        return

    LOG.info("Plotting %d run(s)...", len(unplotted))
    for d in unplotted:
        LOG.info("Plotting %s", d.name)
        example_optimization_plot(d)
