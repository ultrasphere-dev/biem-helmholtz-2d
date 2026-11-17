import json
from itertools import combinations
from math import dist
from pathlib import Path
from typing import Literal

import pytest
from cyclopts.exceptions import CycloptsError

from biem_helmholtz_2d.cli import Device, _configuration, app, torch_xp


def invoke(args: list[str]) -> None:
    app(args, exit_on_error=False, print_error=False, result_action="return_value")


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The help message includes the CLI name and commands."""
    invoke(["--help"])
    output = capsys.readouterr().out
    assert "biem-helmholtz-2d" in output
    assert "accuracy" in output
    assert "plot-accuracy" in output
    assert "plot-optimize" in output
    expected_device = "cuda" if torch_xp is not None and torch_xp.cuda.is_available() else "cpu"
    for name in ("accuracy", "optimize"):
        _, bound, _ = app.parse_args([name], exit_on_error=False, print_error=False)
        bound.apply_defaults()
        assert bound.arguments["device"] == expected_device


@pytest.mark.parametrize(
    "args",
    [
        ["--n-solve", "3", "4"],
        ["--n-quad", "3", "--n-quad", "4"],
        ["--n-quad", "3,,4"],
        ["--n-solve", "0,2"],
    ],
)
def test_invalid_quadrature_lists(args: list[str]) -> None:
    with pytest.raises(CycloptsError):
        invoke(["accuracy", *args])


@pytest.mark.parametrize(
    ("objective", "custom_target"), [("point", False), ("cloaking", False), ("cloaking", True)]
)
def test_accuracy_and_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    objective: Literal["point", "cloaking"],
    custom_target: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    args = [
        "accuracy",
        "--objective",
        objective,
        "--n-solve",
        "3,4",
        "--n-quad",
        "3,5",
        "--n-reference",
        "8",
    ]
    if custom_target:
        args += ["--target", "0,0.1,0,0.6,0.3,0.5", "--weight-target", "0.7"]
    invoke(args)
    files = list((tmp_path / "accuracy").rglob("accuracy.json"))
    assert len(files) == 1
    directory = files[0].parent
    data = json.loads(files[0].read_text())
    assert data["settings"]["n_reference"] == 8
    assert data["settings"]["n_solve"] == [3, 4]
    assert data["settings"]["n_quad"] == [3, 5]
    defaults = _configuration(objective, device=Device("cpu"), target=None)
    assert data["settings"]["centers"] == defaults["centers"].tolist()
    expected_target = (
        [[0, 0.1], [0, 0.6], [0.3, 0.5]] if custom_target else defaults["target"].tolist()
    )
    assert len(data["settings"]["target"]) == len(expected_target)
    for actual, expected in zip(data["settings"]["target"], expected_target, strict=True):
        assert actual == pytest.approx(expected, abs=1e-14)
    assert data["settings"]["weight_target"] == pytest.approx(0.7 if custom_target else 1)
    assert data["settings"]["include_cross_section"] is (objective == "cloaking")
    assert data["total_field"]["extent"] == pytest.approx(
        [-5, 5, -3, 3.5] if objective == "point" else [-5.7, 5.7, -6.2, 6.2]
    )
    assert len(data["central_difference"]["steps"]) == 7
    assert len(data["central_difference"]["values"]) == 7
    assert all(len(values) == 2 for values in data["central_difference"]["values"])
    figures = ["scattered_field.svg", "total_field.svg"] + [
        f"{case}/{name}.svg"
        for case in ("highest_mode", "random")
        for name in ("central_difference", "shape_derivative")
    ]
    assert all((directory / name).is_file() for name in figures)
    invoke(["plot-accuracy", "--path", str(directory)])
    assert all((directory / name).is_file() for name in figures)
    scattered = (directory / "scattered_field.svg").read_text()
    title = (
        "Maximum scattered-field difference over targets"
        if len(expected_target) > 1
        else "Scattered-field difference at the target"
    )
    assert title in scattered
    assert "reference: solve and integration $n=8$" in scattered
    assert "Integration parameter $n$ ($2n-1$ nodes)" in scattered
    assert "Real part of the total field" in (directory / "total_field.svg").read_text()
    central = (directory / "highest_mode" / "central_difference.svg").read_text()
    assert "Central-difference validation: highest Fourier mode" in central
    assert "Finite-difference step size $h$" in central


def test_configurations() -> None:
    point = _configuration("point", device=Device("cpu"), target=None)
    cloak = _configuration("cloaking", device=Device("cpu"), target=None)
    assert point["target"].tolist() == [[0.5, 2.5]]
    centers, targets = cloak["centers"], cloak["target"]
    assert len(centers) == 4 and len(targets) == 16
    assert min(dist(a, b) for a, b in combinations(centers, 2)) > 4
    assert min(dist(center, (0, 0)) for center in centers) > 3
    squared_radii = [float(target @ target) for target in targets]
    assert squared_radii == pytest.approx([(i + 0.5) / 16 for i in range(16)])


@pytest.mark.parametrize("objective", ["point", "cloaking"])
def test_optimize_objective_options(objective: str) -> None:
    _, bound, _ = app.parse_args(
        [
            "optimize",
            "--device",
            "cpu",
            "--objective",
            objective,
            "--target",
            "0,0.1,0,0.6",
            "--weight-target",
            "0.7",
            "--n-quad",
            "17",
        ],
        exit_on_error=False,
        print_error=False,
    )
    assert bound.arguments["objective"] == objective
    assert bound.arguments["target"].values == [[0, 0.1], [0, 0.6]]
    assert bound.arguments["weight_target"] == pytest.approx(0.7)
    assert bound.arguments["n_quad"] == 17


def test_invalid_target() -> None:
    with pytest.raises(CycloptsError):
        invoke(["accuracy", "--target", "0,1,2"])
