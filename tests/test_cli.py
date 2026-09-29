from typer.testing import CliRunner

from nebius_physical_ai.cli import app


def test_info_runs() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Physical AI" in result.output
