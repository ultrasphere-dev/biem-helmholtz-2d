from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path

import typer

# import tracemalloc


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
def farfield_showcase() -> None:
    N = 32
    t = xp.linspace(0, xp.pi * 2, 2 * N + 1)[:-1]
    directions = xp.stack([xp.cos(t), xp.sin(t)], dim=-1)
    result = get_uinf(1, N, wave_direction=(1, 0), farfield_direction=directions, eta=1)
    result_abs = xp.abs(result)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(t, result_abs, label="Amplitude")
    ax.plot(t, result.real, label="Real")
    # set title
    ax.set_title("Farfield pattern for a plane wave")
    # set x and y labels
    ax.set_xlabel("Theta (Radians)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    Path(".cache").mkdir(exist_ok=True)
    fig.savefig(".cache/farfield_showcase_2d.svg")
    fig.savefig(".cache/farfield_showcase_2d.png")
