import json
from pathlib import Path

import pytest
from cyclopts.exceptions import CycloptsError

from biem_helmholtz_2d.cli import app, torch_xp


def invoke(args: list[str]) -> None:
    app(args, exit_on_error=False, print_error=False, result_action="return_value")


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The help message includes the CLI name and commands."""
    invoke(["--help"])
    output = capsys.readouterr().out
    assert "biem-helmholtz-2d" in output
    assert "accuracy" in output
    assert "plot-accuracy" in output
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


def test_accuracy_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    invoke(["accuracy", "--n-solve", "3,4", "--n-quad", "3,5", "--n-reference", "8"])
    files = list((tmp_path / "accuracy").rglob("accuracy.json"))
    assert len(files) == 1
    directory = files[0].parent
    data = json.loads(files[0].read_text())
    assert data["settings"]["n_reference"] == 8
    assert data["settings"]["n_solve"] == [3, 4]
    assert data["settings"]["n_quad"] == [3, 5]
    assert "total_field" in data
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
