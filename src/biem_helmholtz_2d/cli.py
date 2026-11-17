import pathlib
from datetime import datetime
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Annotated, Any, Literal, Self

import array_api_compat.numpy as xp
from cyclopts import App, Parameter
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


class Points(str):
    """A comma-separated list of x,y target coordinates."""

    values: list[list[float]]

    def __new__(cls, value: str, /) -> Self:
        """Parse coordinates as x1,y1,x2,y2,... ."""
        coordinates = [float(part) for part in value.split(",")]
        if len(coordinates) < 2 or len(coordinates) % 2:
            raise ValueError("target must contain x,y coordinate pairs")
        result = super().__new__(cls, value)
        result.values = [coordinates[i : i + 2] for i in range(0, len(coordinates), 2)]
        return result


def _configuration(
    objective: Literal["point", "cloaking"], /, *, device: Device, target: Points | None
) -> dict[str, Any]:
    namespace = _namespace(device)
    if objective == "point":
        config = {
            "centers": [[-2.5, 0], [2.5, 0]],
            "k": 4,
            "alpha": 1,
            "eta": 1,
            "target": [[0.5, 2.5]],
            "desired_total_field": 0,
            "n_solve": 256,
            "mode_counts": (3, 6, 9, 12),
            "include_cross_section": False,
        }
    elif objective == "cloaking":
        # Equal-area samples on a Fermat spiral in the unit disk.
        n_targets = 16
        i = namespace.arange(n_targets, dtype=namespace.float64, device=device)
        radius = namespace.sqrt((i + 0.5) / n_targets)
        angle = i * namespace.pi * (3 - 5**0.5)
        config = {
            "centers": [[-3.2, 0], [0, -3.2], [3.2, 0], [0, 3.2]],
            "k": 1,
            "alpha": 1,
            "eta": 1,
            "target": namespace.stack(
                [radius * namespace.cos(angle), radius * namespace.sin(angle)], axis=-1
            ),
            "desired_total_field": 0,
            "n_solve": 200,
            "mode_counts": (5, 10, 20),
            "include_cross_section": True,
        }
    else:
        raise ValueError(f"unknown objective: {objective}")
    config["centers"] = namespace.asarray(config["centers"], dtype=namespace.float64, device=device)
    config["target"] = namespace.asarray(
        target.values if target is not None else config["target"],
        dtype=namespace.float64,
        device=device,
    )
    return config


_DEFAULT_DEVICE = Device("cuda" if torch_xp is not None and torch_xp.cuda.is_available() else "cpu")


@app.command()
def optimize(
    *,
    objective: Literal["point", "cloaking"] = "point",
    target: Points | None = None,
    weight_target: float = 1.0,
    device: Device = _DEFAULT_DEVICE,
    n_quad: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Compare optimization results with/without Hilbertian regularization.

    Parameters
    ----------
    objective : Literal["point", "cloaking"]
        Minimize the total field at the target(s), optionally with the scattering cross section.
    target : Points or None
        Comma-separated coordinates x1,y1,x2,y2,...; defaults depend on the objective.
    weight_target : float
        Weight of the mean squared total field at the target points.
    device : Device
        Array device: cpu, cuda, or cuda:N.
    n_quad : int or None
        Integration parameter; defaults to the solve parameter.
    verbose : bool
        Enable debug logging.

    """
    path = pathlib.Path(f"optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config = _configuration(objective, device=device, target=target)
    n_solve = config.pop("n_solve")
    mode_counts = config.pop("mode_counts")
    for alpha_reg in (0, 1, 1e-2, 1e-4):
        for n_modes in mode_counts:
            example_optimization(
                **config,
                n=n_solve,
                n_quad=n_quad,
                n_modes=n_modes,
                n_steps=25,
                weight_target=weight_target,
                alpha_reg=alpha_reg,
                path=path / f"alpha_reg_{alpha_reg}" / f"n_modes_{n_modes}",
            )


@app.command()
def accuracy(
    *,
    objective: Literal["point", "cloaking"] = "point",
    target: Points | None = None,
    weight_target: float = 1.0,
    n_quad_ref: Annotated[int | None, Parameter(name=("--n-reference", "--n-quad-ref"))] = None,
    device: Device = _DEFAULT_DEVICE,
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
    objective : Literal["point", "cloaking"]
        Minimize the total field at the target(s), optionally with the scattering cross section.
    target : Points or None
        Comma-separated coordinates x1,y1,x2,y2,...; defaults depend on the objective.
    weight_target : float
        Weight of the mean squared total field at the target points.
    n_quad_ref : int or None
        Reference parameter; defaults to 1.5 times the largest study parameter, rounded up.
    device : Device
        Array device: cpu, cuda, or cuda:N.
    n_solve : Values or None
        Comma-separated solve parameters; defaults to powers of two and the optimization value.
    n_quad : Values or None
        Comma-separated integration parameters; defaults to the same values.
    step : float
        Central-difference step at the center of the logarithmic sweep.
    seed : int
        Random seed for reproducible stress cases.
    verbose : bool
        Enable debug logging.

    """
    config = _configuration(objective, device=device, target=target)
    default_solve = config.pop("n_solve")
    n_modes = max(config.pop("mode_counts"))
    default_values = [2**power for power in range(1, default_solve.bit_length())]
    if not default_values or default_values[-1] != default_solve:
        default_values.append(default_solve)
    path = pathlib.Path(f"accuracy/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    example_accuracy(
        **config,
        path=path,
        weight_target=weight_target,
        n_solve=n_solve.values if n_solve is not None else default_values,
        n_quad=n_quad.values if n_quad is not None else default_values,
        n_reference=n_quad_ref,
        n_modes=n_modes,
        step=step,
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
        LOG.info("Plotted %s", directory)


_JSON_FILES = ("optimization_history.json", "optimized_shape.json", "optimized_near_field.json")


@app.command()
def plot_optimize(*, path: pathlib.Path | None = None, verbose: bool = False) -> None:
    """
    Plot unplotted optimization comparisons under a directory.

    Parameters
    ----------
    path : pathlib.Path or None
        Directory to search; defaults to optimization/.
    verbose : bool
        Enable debug logging.

    """
    root = path if path is not None else pathlib.Path("optimization")
    if not root.is_dir():
        LOG.warning("No optimization directory found: %s", root)
        return
    groups = set()
    for file in root.rglob("optimization_history.json"):
        group = file.parent
        if group.name.startswith("n_modes_"):
            group = group.parent
        if group.name.startswith("alpha_reg_"):
            group = group.parent
        groups.add(group)
    for group in sorted(groups):
        if all(
            (group / name).is_file()
            for name in (
                "optimized_shape.svg",
                "optimized_coefficients.svg",
                "optimization_history.svg",
            )
        ):
            continue
        cases = [
            file.parent
            for file in group.rglob("optimization_history.json")
            if file.parent == group
            or file.parent.parent == group
            or file.parent.parent.parent == group
        ]
        if any(not all((case / name).is_file() for name in _JSON_FILES) for case in cases):
            LOG.warning("Incomplete optimization data in %s", group)
            continue
        example_optimization_plot(group)
        LOG.info("Plotted %s", group)


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
