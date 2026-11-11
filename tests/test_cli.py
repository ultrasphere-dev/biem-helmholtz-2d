import json
import math
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure
from typer.testing import CliRunner

from biem_helmholtz_2d.cli import app

runner = CliRunner()


def test_help():
    """The help message includes the CLI name."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "accuracy" in result.output
    assert "plot-accuracy" in result.output
    assert runner.invoke(app, ["accuracy"]).exit_code != 0  # A reference order is required.


def test_accuracy_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "accuracy",
            "--n-solve",
            "3",
            "--n-solve",
            "4",
            "--n-quadrature",
            "3",
            "--n-quadrature",
            "5",
            "--n-reference",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    files = list((tmp_path / "accuracy").rglob("accuracy.json"))
    assert {file.parent.name for file in files} == {"highest_mode", "random"}
    studies = {file.parent.name: json.loads(file.read_text()) for file in files}
    assert (
        studies["highest_mode"]["settings"]["coefficients"]
        == studies["random"]["settings"]["coefficients"]
    )
    for case, data in studies.items():
        settings = data["settings"]
        assert settings["case"] == case
        assert settings["reference_order"] == 6
        assert settings["seed"] == 0
        for body in settings["coefficients"] + settings["perturbation_coefficients"]:
            assert sum(math.hypot(*pair) for pair in body) == pytest.approx(1)
        assert all(len(body) == 10 for body in settings["coefficients"])
        if case == "highest_mode":
            assert all(
                body[:-1] == [[0, 0]] * 9 and body[-1] == [1, 0]
                for body in settings["perturbation_coefficients"]
            )
        else:
            assert all(
                any(pair != [0, 0] for pair in body[:-1])
                for body in settings["perturbation_coefficients"]
            )
        assert {(item["n_solve"], item["n_quadrature"]) for item in data["records"]} == {
            (3, 3),
            (3, 5),
            (4, 3),
            (4, 5),
        }
        for item in data["records"]:
            assert all(
                item[key] >= 0
                for key in ("field_error", "field_derivative_error", "objective_derivative_error")
            )
    savefig = Figure.savefig

    def check_ticks(fig: Figure, *args: Any, **kwargs: Any) -> None:
        for axis in fig.axes:
            assert list(axis.get_xticks()) == [3, 5]
            assert [label.get_text() for label in axis.get_xticklabels()] == ["3", "5"]
            assert len(axis.xaxis.get_minorticklocs()) == 0
        savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_ticks)
    result = runner.invoke(app, ["plot-accuracy", "--path", str(files[0].parent.parent)])
    assert result.exit_code == 0, result.output
    assert all((file.parent / "accuracy.svg").is_file() for file in files)
