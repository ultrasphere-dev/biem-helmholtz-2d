import pathlib
from datetime import datetime
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Any, Self

import array_api_compat.numpy as xp
from cyclopts import App
from rich.logging import RichHandler

from biem_helmholtz_2d.optimization._accuracy import example_accuracy, example_accuracy_plot
from biem_helmholtz_2d.optimization._example import example_optimization, example_optimization_plot

try:
    import array_api_compat.torch as torch_xp
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch_xp = None

app = App(name="biem-helmholtz-2d")

LOG = getLogger(__name__)

_CENTERS = [[0, 0], [5, 0]]
_K = 5
_ALPHA = 1
_ETA = 1
_TARGET_POINT = (-3, 3)
_DESIRED_TOTAL_FIELD = 0
_N_SOLVE = 200
_N_MODES = 10


def _namespace(device: str, /) -> Any:
    if device == "cpu":
        return xp
    if device == "cuda" or device.startswith("cuda:"):
        if torch_xp is None:
            raise ValueError("CUDA requires the 'cuda' extra: uv sync --extra cuda")
        if not torch_xp.cuda.is_available():
            raise ValueError("PyTorch cannot access CUDA on this system")
        return torch_xp
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")


class Device(str):
    """A validated array device name."""

    def __new__(cls, value: str, /) -> Self:
        """Validate the device name."""
        _namespace(value)
        return super().__new__(cls, value)


class Values(str):
    """A single comma-separated list of positive quadrature parameters."""

    values: list[int]

    def __new__(cls, value: str, /) -> Self:
        """Parse one comma-separated list of positive integers."""
        values = [int(part) for part in value.split(",")]
        if min(values) < 1:
            raise ValueError("quadrature parameters must be positive integers")
        result = super().__new__(cls, value)
        result.values = values
        return result


_CPU = Device("cpu")


@app.command()
def optimize(
    *,
    device: Device = _CPU,
    n_quad: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Compare optimization results with/without Hilbertian regularization.

    Parameters
    ----------
    device : Device
        Array device: cpu, cuda, or cuda:N.
    n_quad : int or None
        Integration parameter; defaults to the solve parameter.
    verbose : bool
        Enable debug logging.

    """
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
            n_quad=n_quad,
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
    *,
    n_quad_ref: int | None = None,
    device: Device = _CPU,
    n_solve: Values | None = None,
    n_quad: Values | None = None,
    step: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> None:
    """
    Export field and sensitivity errors for independent quadrature parameters.

    Parameters
    ----------
    n_quad_ref : int or None
        Reference parameter; defaults to 1.5 times the largest study parameter, rounded up.
    device : Device
        Array device: cpu, cuda, or cuda:N.
    n_solve : Values or None
        Comma-separated solve parameters; defaults to powers of two and the optimization value.
    n_quad : Values or None
        Comma-separated integration parameters; defaults to the same values.
    step : float
        Central-difference step.
    seed : int
        Random seed for reproducible stress cases.
    verbose : bool
        Enable debug logging.

    """
    namespace = _namespace(device)
    default_values = [2**power for power in range(1, _N_SOLVE.bit_length())]
    if not default_values or default_values[-1] != _N_SOLVE:
        default_values.append(_N_SOLVE)
    path = pathlib.Path(f"accuracy/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    example_accuracy(
        path=path,
        centers=namespace.asarray(_CENTERS, dtype=namespace.float64, device=device),
        k=_K,
        alpha=_ALPHA,
        eta=_ETA,
        target_point=_TARGET_POINT,
        desired_total_field=_DESIRED_TOTAL_FIELD,
        n_solve=n_solve.values if n_solve is not None else default_values,
        n_quad=n_quad.values if n_quad is not None else default_values,
        n_quad_ref=n_quad_ref,
        step=step,
        n_modes=_N_MODES,
        seed=seed,
    )
    LOG.info("Saved accuracy studies to %s", path)


@app.command()
def plot_accuracy(*, path: pathlib.Path | None = None, verbose: bool = False) -> None:
    """
    Plot studies under a directory, or all studies in accuracy/.

    Parameters
    ----------
    path : pathlib.Path or None
        Directory to search; defaults to accuracy/.
    verbose : bool
        Enable debug logging.

    """
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
def plot(*, verbose: bool = False) -> None:
    """
    Plot optimization runs that have JSON data but missing SVG files.

    Parameters
    ----------
    verbose : bool
        Enable debug logging.

    """
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


def main() -> None:
    """Configure logging and run the selected command."""
    command, bound, _ = app.parse_args()
    basicConfig(
        level=DEBUG if bound.arguments.get("verbose", False) else INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    command(*bound.args, **bound.kwargs)
