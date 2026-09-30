from pathlib import Path

from typer.testing import CliRunner

from nebius_physical_ai.cli import app


def test_markers_writes_pdf(tmp_path: Path) -> None:
    out = tmp_path / "m.pdf"
    result = CliRunner().invoke(app, ["markers", "--out", str(out)])
    assert result.exit_code == 0 and out.read_bytes().startswith(b"%PDF")
