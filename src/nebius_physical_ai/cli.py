from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich import print

from nebius_physical_ai.capture.markers import DPI, render_marker_sheet
from nebius_physical_ai.capture.mat import MatSpec, ScenePerceiver
from nebius_physical_ai.capture.server import create_app

app = typer.Typer(no_args_is_help=True)

WidthOpt = Annotated[float, typer.Option(help="Mat width between marker centres (mm)")]
HeightOpt = Annotated[float, typer.Option(help="Mat height between marker centres (mm)")]
MarkerOpt = Annotated[float, typer.Option(help="Printed marker side (mm)")]


@app.command()
def info() -> None:
    print("[bold]Preflight[/bold] — Physical AI track entry")


@app.command()
def markers(
    out: Path = Path("markers.pdf"),
    width_mm: WidthOpt = 300.0,
    height_mm: HeightOpt = 300.0,
    marker_mm: MarkerOpt = 50.0,
) -> None:
    """Write the printable corner-marker sheet."""
    render_marker_sheet(MatSpec(width_mm, height_mm, marker_mm)).save(out, resolution=DPI)
    print(f"wrote {out}")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    width_mm: WidthOpt = 300.0,
    height_mm: HeightOpt = 300.0,
    marker_mm: MarkerOpt = 50.0,
    ssl_certfile: Path | None = None,
    ssl_keyfile: Path | None = None,
) -> None:
    """Serve the phone capture page. Phones need HTTPS for camera access."""
    perceiver = ScenePerceiver(MatSpec(width_mm, height_mm, marker_mm))
    uvicorn.run(
        create_app(perceiver),
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
