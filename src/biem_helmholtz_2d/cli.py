from logging import DEBUG, INFO, basicConfig, getLogger

import typer
from rich.logging import RichHandler

from biem_helmholtz_2d.optimization._example import example_optimization

app = typer.Typer()

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
    import numpy as np

    xp = np
    dtype = np.float64
    device = None
    example_optimization(xp=xp, dtype=dtype, device=device, alpha_reg=0)
    example_optimization(xp=xp, dtype=dtype, device=device)
