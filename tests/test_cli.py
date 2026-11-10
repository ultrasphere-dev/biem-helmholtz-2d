import json
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
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert len(data["records"]) == 4
    assert data["settings"]["reference_order"] == 6
    assert {(item["n_solve"], item["n_quadrature"]) for item in data["records"]} == {
        (3, 3),
        (3, 5),
        (4, 3),
        (4, 5),
    }
    assert all(item["field_error"] >= 0 for item in data["records"])
    assert all(item["field_derivative_error"] >= 0 for item in data["records"])
    assert all(item["objective_derivative_error"] >= 0 for item in data["records"])
    savefig = Figure.savefig

    def check_ticks(fig: Figure, *args: Any, **kwargs: Any) -> None:
        for axis in fig.axes:
            assert list(axis.get_xticks()) == [3, 5]
            assert [label.get_text() for label in axis.get_xticklabels()] == ["3", "5"]
            assert len(axis.xaxis.get_minorticklocs()) == 0
        savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_ticks)
    result = runner.invoke(app, ["plot-accuracy"])
    assert result.exit_code == 0, result.output
    assert (files[0].parent / "accuracy.svg").is_file()
