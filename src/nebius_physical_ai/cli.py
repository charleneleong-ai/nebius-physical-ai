import json
import time
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich import print
from rich.progress import track

from nebius_physical_ai.capture.markers import DPI, render_marker_sheet
from nebius_physical_ai.capture.mat import DetectedObject, MatSpec, ScenePerceiver
from nebius_physical_ai.capture.server import create_app
from nebius_physical_ai.policy.gr00t import DEFAULT_PORT, Gr00tClient
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler
from nebius_physical_ai.twin.rehearsal import Rehearsal
from nebius_physical_ai.twin.rollout import STACK_TASK, StackCheck
from nebius_physical_ai.twin.scene import A3_MAT, Twin, TwinLayout

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


@app.command()
def demos(
    out: Path,
    episodes: int = 100,
    seed: int = 0,
    task: str = STACK_TASK,
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
) -> None:
    """Record scripted pick-and-place demos in the twin as a LeRobot v2.1 dataset (successes only)."""
    sampler = SceneSampler(TwinLayout(MatSpec(width_mm, height_mm)), seed=seed)
    writer, attempts = LeRobotWriter(out), 0
    for _ in track(range(episodes), description="recording demos"):
        while True:
            attempts += 1
            with Twin.from_objects(sampler.sample(), sampler.layout, start_pose=sampler.start_pose()) as twin:
                frames = list(PickPlaceExpert(twin, task).run())
                twin.settle(0.5)
                stacked = StackCheck()(twin)
            if stacked:
                writer.add_episode(frames, task)
                break
    writer.finalize()
    print(f"wrote {episodes} episodes ({writer.total_frames} frames) to {out}; expert success {episodes}/{attempts}")


@app.command()
def rehearse(
    objects: Annotated[
        str, typer.Option(help='JSON list, e.g. [{"label":"blue","x_mm":130,"y_mm":150,"area_mm2":900}]')
    ] = "",
    n: int = 64,
    batch_size: int = 16,
    seed: int = 0,
    port: int = DEFAULT_PORT,
    task: str = STACK_TASK,
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
    out: Path | None = None,
) -> None:
    """Rehearse one scene N times across randomised variants against a running GR00T server."""
    layout = TwinLayout(MatSpec(width_mm, height_mm))
    scene = [DetectedObject(**o) for o in json.loads(objects)] if objects else SceneSampler(layout, seed).sample()
    with Gr00tClient(port=port) as client:
        start = time.time()
        result = Rehearsal(client, task, StackCheck(), layout, batch_size=batch_size).run(scene, n, seed)
    summary = result.summary() | {"seconds": round(time.time() - start, 1)}
    print(summary)
    if out:
        out.write_text(
            json.dumps(summary | {"scene": [o.__dict__ for o in scene], "variants": result.variants}, indent=2)
        )
