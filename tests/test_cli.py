import json
from pathlib import Path

import pytest
from cyclopts.exceptions import CycloptsError

from biem_helmholtz_2d.cli import app


def invoke(args: list[str]) -> None:
    app(args, exit_on_error=False, print_error=False, result_action="return_value")


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The help message includes the CLI name and commands."""
    invoke(["--help"])
    output = capsys.readouterr().out
    assert "biem-helmholtz-2d" in output
    assert "accuracy" in output
    assert "plot-accuracy" in output


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
    invoke(["accuracy", "--n-solve", "3,4", "--n-quad", "3,5"])
    files = list((tmp_path / "accuracy").rglob("accuracy.json"))
    assert {file.parent.name for file in files} == {"highest_mode", "random"}
    for file in files:
        data = json.loads(file.read_text())
        assert data["settings"]["n_quad_ref"] == 8
        assert {(row["n_solve"], row["n_quad"]) for row in data["records"]} == {
            (3, 3),
            (3, 5),
            (4, 3),
            (4, 5),
        }
    invoke(["plot-accuracy", "--path", str(files[0].parent.parent)])
    assert all((file.parent / "accuracy.svg").is_file() for file in files)
