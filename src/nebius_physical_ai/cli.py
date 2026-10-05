import json
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer
import uvicorn
from rich import print
from rich.progress import track

from nebius_physical_ai.capture.camera import DEFAULT_CAMERA_MIX, CameraPose, MarkerCalibrator, parse_camera_mix
from nebius_physical_ai.capture.lift import HeightLifter, MatView
from nebius_physical_ai.capture.markers import DPI, render_marker_sheet
from nebius_physical_ai.capture.mat import ARUCO_DICT, DetectedObject, MatSpec, ScenePerceiver
from nebius_physical_ai.capture.server import create_app
from nebius_physical_ai.pipeline import add_pipeline_cli
from nebius_physical_ai.policy.gr00t import DEFAULT_PORT, Gr00tClient
from nebius_physical_ai.tracking import Tracker
from nebius_physical_ai.trust.plots import StressPlots
from nebius_physical_ai.trust.stress import StressCase, StressReport
from nebius_physical_ai.twin.dataset import LeRobotWriter
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler, UnreachableTarget
from nebius_physical_ai.twin.randomize import Episode, SceneRandomizer
from nebius_physical_ai.twin.rehearsal import Rehearsal, SceneRun
from nebius_physical_ai.twin.rollout import STACK_TASK, BatchRollout, RolloutResult, StackCheck
from nebius_physical_ai.twin.scene import A3_MAT, RENDER_ASPECT, Twin, TwinLayout

app = typer.Typer(no_args_is_help=True)
MAX_ATTEMPTS_PER_EPISODE = 10  # demos give up instead of looping forever on ranges the expert can't solve

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


def build_randomizer(
    sampler: SceneSampler,
    seed: int,
    real_objects: Path | None,
    camera: Path | None,
    camera_mix: str,
    role_agnostic: bool = False,
) -> SceneRandomizer:
    """The demo randomiser: real objects as distractors, and cameras centred on a calibrated pose when one is given.

    Role-agnostic episodes name the objects by their looks and never use the held-out colours or real objects."""
    real = [DetectedObject(**o) for o in json.loads(real_objects.read_text())] if real_objects else None
    try:
        mix = parse_camera_mix(camera_mix)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    pose = CameraPose.load(camera) if camera else None
    return SceneRandomizer(
        sampler.layout, seed=seed, real_objects=real, camera_pose=pose, camera_mix=mix, role_agnostic=role_agnostic
    )


@app.command()
def demos(
    out: Path,
    episodes: int = 100,
    seed: int = 0,
    task: str = STACK_TASK,
    randomize: Annotated[
        bool, typer.Option(help="Random cameras, lights, shapes, sizes, colours and distractors per episode")
    ] = False,
    real_objects: Annotated[
        Path | None, typer.Option(help="Objects JSON (npai lift) placed as distractors when randomizing")
    ] = None,
    camera: Annotated[
        Path | None,
        typer.Option(help="Calibrated camera (npai calibrate-camera) that randomized episodes are centred on"),
    ] = None,
    camera_mix: Annotated[
        str,
        typer.Option(help="With --camera, shares of pose-centred, wide-arc and straight-down cameras (sum 1)"),
    ] = ",".join(str(share) for share in DEFAULT_CAMERA_MIX),
    role_agnostic: Annotated[
        bool,
        typer.Option(
            help="Name the pick and place objects by their looks, with a prompt per episode (needs --randomize)"
        ),
    ] = False,
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
) -> None:
    """Record scripted pick-and-place demos in the twin as a LeRobot v2.1 dataset (successes only).

    With --role-agnostic each episode is told what to pick and where to put it in its own words (`Pick green can and
    place on red box`), and a quarter keep the original blue and red cubes and prompt."""
    if (camera or role_agnostic) and not randomize:
        raise typer.BadParameter("--camera and --role-agnostic need --randomize: only randomized episodes vary")
    sampler = SceneSampler(TwinLayout(MatSpec(width_mm, height_mm)), seed=seed)
    randomizer = build_randomizer(sampler, seed, real_objects, camera, camera_mix, role_agnostic) if randomize else None
    writer, attempts = LeRobotWriter(out), 0
    for _ in track(range(episodes), description="recording demos"):
        while True:
            attempts += 1
            if attempts > MAX_ATTEMPTS_PER_EPISODE * episodes:
                raise RuntimeError(f"expert solved only {len(writer.episodes)} of {attempts} scenes; check the ranges")
            episode = (
                randomizer.episode(sampler, task) if randomizer else Episode(sampler.sample(), "blue", "red", task)
            )
            look = randomizer.look() if randomizer else None
            start = sampler.start_pose()
            with Twin.from_objects(episode.objects, sampler.layout, start_pose=start, look=look) as twin:
                try:
                    frames = list(PickPlaceExpert(twin, episode.prompt, episode.pick, episode.place).run())
                except UnreachableTarget:  # draw another scene
                    continue
                twin.settle(0.5)
                stacked = StackCheck(episode.pick, episode.place)(twin)
            if stacked:
                writer.add_episode(frames, episode.prompt)
                break
    writer.finalize()
    print(f"wrote {episodes} episodes ({writer.total_frames} frames) to {out}; expert success {episodes}/{attempts}")
    if role_agnostic:
        print(f"{len(writer.tasks)} distinct prompts, e.g. {writer.tasks[:3]}")
    if randomizer and randomizer.camera_pose:
        print(f"cameras (pose, arc, top_down) over every attempt: {dict(randomizer.camera_counts)}")


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
    pick: Annotated[str, typer.Option(help="Label of the object to pick, as in --objects")] = "blue",
    place: Annotated[str, typer.Option(help="Label of the object to place it on")] = "red",
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
    out: Path | None = None,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
) -> None:
    """Rehearse one scene N times across randomised variants against a running GR00T server."""
    layout = TwinLayout(MatSpec(width_mm, height_mm))
    scene = [DetectedObject(**o) for o in json.loads(objects)] if objects else SceneSampler(layout, seed).sample()
    config = {"n": n, "batch_size": batch_size, "seed": seed, "task": task}
    with Tracker("rehearsal", config=config, enabled=wandb_log) as tracker, Gr00tClient(port=port) as client:
        rehearsal = Rehearsal(client, task, StackCheck(pick, place), layout, batch_size=batch_size)
        run = SceneRun.rehearse(rehearsal, 0, scene, n, seed)
        tracker.log_scene(run)
    print({k: v for k, v in run.record().items() if k not in ("scene", "variants")})
    if out:
        out.write_text(json.dumps(run.record(), indent=2))


@app.command()
def sweep(
    out: Path,
    scenes: int = 20,
    n: int = 32,
    batch_size: int = 32,
    port: int = DEFAULT_PORT,
    task: str = STACK_TASK,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    """Rehearse many sampled scenes, alternating ordinary and hard layouts (cubes close, edge of reach)."""
    layout = TwinLayout()
    samplers = [SceneSampler(layout, seed=2024), SceneSampler.hard(layout, seed=2025)]
    records: list[dict[str, Any]] = []
    config = {"scenes": scenes, "n": n, "batch_size": batch_size, "task": task}
    with Tracker("sweep", config=config, enabled=wandb_log) as tracker, Gr00tClient(port=port) as client:
        rehearsal = Rehearsal(client, task, StackCheck(), layout, batch_size=batch_size)
        for i in range(scenes):
            run = SceneRun.rehearse(rehearsal, i, samplers[i % 2].sample(), n, seed=i, hard=bool(i % 2))
            tracker.log_scene(run)
            records.append(run.record())
            out.write_text(json.dumps(records))
            print(f"{i} hard={run.tags['hard']} p={run.result.p:.3f} ci={run.result.summary()['ci95']} {run.seconds}s")


@app.command()
def stress(
    out: Path,
    scenes: int = 20,
    n: int = 32,
    k: int = 8,
    batch_size: int = 16,
    port: int = DEFAULT_PORT,
    task: str = STACK_TASK,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    """Sim-to-sim calibration test: rehearse the perceived scene, then score it on k attempts in a hidden world."""
    layout = TwinLayout()
    samplers = [SceneSampler(layout, seed=3024), SceneSampler.hard(layout, seed=3025)]
    cases: list[StressCase] = []
    config = {"scenes": scenes, "n": n, "k": k, "batch_size": batch_size, "task": task}
    with Tracker("stress", config=config, enabled=wandb_log) as tracker, Gr00tClient(port=port) as client:
        rehearsal = Rehearsal(client, task, StackCheck(), layout, batch_size=batch_size)
        for i in range(scenes):
            case = StressCase.run(rehearsal, i, samplers[i % 2].sample(), n, k, seed=i, hard=bool(i % 2))
            cases.append(case)
            tracker.log_scene(case.prediction, true_rate=case.true_rate)
            out.write_text(json.dumps([c.record() for c in cases]))
            print(f"{i} p={case.prediction.result.p:.2f} true={case.true_rate:.2f} {case.prediction.seconds}s")
        report = StressReport(cases)
        summary = report.summary()
        tracker.log_summary(summary, report.tables())
    print(summary)


@app.command()
def stress_report(results: Path, out_dir: Path = Path("docs/assets/stress"), sweep: Path | None = None) -> None:
    """Rebuild a stress-test report offline from its JSON: summary plus PNG figures."""
    report = StressReport([StressCase.from_record(r) for r in json.loads(results.read_text())])
    plots = StressPlots(report, out_dir)
    paths = [plots.predicted_vs_true(), plots.coverage_curve()]
    if sweep:
        paths.append(StressPlots.sweep_distribution(json.loads(sweep.read_text()), out_dir))
    print(report.summary())
    print("wrote", *map(str, paths))


@app.command()
def calibrate_camera(
    video: Path,
    out: Path = Path("configs/front_camera.json"),
    hfov_deg: Annotated[float | None, typer.Option(help="Known horizontal FOV; omit to fit it from the clip")] = None,
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
) -> None:
    """Measure the fixed top camera's pose (and optionally its FOV) from a clip of the printed mat."""
    calibrator = MarkerCalibrator(MatSpec(width_mm, height_mm))
    correspondences = calibrator.correspondences(video)
    if hfov_deg is None:
        hfov_deg, rms = calibrator.fit_hfov_from(*correspondences)
        print(f"fitted HFOV {hfov_deg:.1f}° (RMS {rms:.2f} px); move the phone around the mat for a reliable fit")
    pose = calibrator.pose_from(*correspondences, hfov_deg)
    out.parent.mkdir(parents=True, exist_ok=True)
    pose.save(out)
    x, y, z = pose.position_mm
    print(
        f"camera at ({x:.0f}, {y:.0f}) mm on the mat, {z:.0f} mm up, tilted {pose.tilt_deg:.1f}°, "
        f"reprojection {pose.reprojection_px:.2f} px -> {out}"
    )


@app.command()
def evaluate(
    name: str,
    scenes: int = 20,
    seed: int = 1234,
    max_steps: int = 480,
    batch_size: int = 10,
    port: int = DEFAULT_PORT,
    task: str = STACK_TASK,
    camera: Annotated[Path | None, typer.Option(help="Calibrated top camera (npai calibrate-camera)")] = None,
    wandb_log: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    """One rollout per unseen scene (seed 1234 = the v1 evaluation set), full-res cameras, success rate to W&B."""
    layout = TwinLayout(front_camera=CameraPose.load(camera) if camera else None)
    sampler = SceneSampler(layout, seed=seed)
    episodes = [(sampler.sample(), sampler.start_pose()) for _ in range(scenes)]
    results: list[RolloutResult] = []
    successes = 0
    config = {"scenes": scenes, "seed": seed, "max_steps": max_steps, "port": port, "task": task, "camera": str(camera)}
    with Tracker("eval", name=name, config=config, enabled=wandb_log) as tracker, Gr00tClient(port=port) as client:
        for start in range(0, scenes, batch_size):
            with ExitStack() as stack:
                batch = episodes[start : start + batch_size]
                twins = [stack.enter_context(Twin.from_objects(o, layout, start_pose=p)) for o, p in batch]
                results += BatchRollout(twins, client, task).run(StackCheck(), max_steps)
            successes = sum(r.success for r in results)
            print(f"{len(results)}/{scenes}: {successes} successes")
        rows = [[i, r.success, r.steps] for i, r in enumerate(results)]
        tracker.log_summary(
            {"success_rate": successes / scenes, "successes": successes, "n": scenes},
            {"episodes": (["scene", "success", "steps"], rows)},
        )
    print(f"{name}: {successes}/{scenes} successes")


# Preflight pipeline CLI
add_pipeline_cli(app)


def marker_alignment(twin: Twin, layout: TwinLayout, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Printed marker corners: (detected in `image`, projected through the twin's front camera), both (N, 2) px."""
    corners, ids, _ = cv2.aruco.ArucoDetector(ARUCO_DICT).detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return np.empty((0, 2)), np.empty((0, 2))
    mat_points, image_points = layout.mat.board.matchImagePoints(corners, ids)
    world = np.array([[*layout.mat_to_world(x, y), 0.0] for x, y in mat_points.reshape(-1, 3)[:, :2]])
    return image_points.reshape(-1, 2), twin.project(world, "front_cam", image.shape[:2])


@app.command()
def compare_twin(
    frame: Path,
    camera: Path,
    out: Path = Path("docs/assets/capture/real_vs_twin.jpg"),
    objects: Annotated[
        Path | None, typer.Option(help="Objects JSON (npai lift) instead of perceiving the frame")
    ] = None,
    hide_arm: Annotated[bool, typer.Option(help="Hide the twin's arm when the real one isn't in shot")] = True,
) -> None:
    """Rebuild a real frame's objects in the twin, render from the calibrated camera: real | twin | overlay.

    Also checks the camera: printed marker corners detected in the frame (green) vs projected through the twin (red).
    """
    pose = CameraPose.load(camera)
    if (image := cv2.imread(str(frame))) is None:
        raise typer.BadParameter(f"can't read {frame}")
    if objects:
        found = [DetectedObject(**o) for o in json.loads(objects.read_text())]
    elif (obs := ScenePerceiver(A3_MAT).observe(image)).ok:
        found = obs.objects
    else:
        raise typer.BadParameter(obs.error or "perception failed")
    layout = TwinLayout(A3_MAT, front_camera=pose)
    with Twin.from_objects(found, layout) as twin:
        twin.settle(0.5)
        if hide_arm:
            twin.hide_arm()
        sim = cv2.cvtColor(twin.render("front_cam"), cv2.COLOR_RGB2BGR)
        real = cv2.resize(pose.crop_frame(image, RENDER_ASPECT), sim.shape[1::-1])
        seen, projected = marker_alignment(twin, layout, real)
    panels = {"real": real, "twin": sim, "overlay": cv2.addWeighted(real, 0.55, sim, 0.45, 0)}
    for p, q in zip(np.int32(seen), np.int32(projected)):
        cv2.circle(panels["overlay"], tuple(p), 6, (0, 200, 0), 2)
        cv2.drawMarker(panels["overlay"], tuple(q), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    for text, img in panels.items():
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(list(panels.values())))
    for o in found:
        rgb = "#{:02x}{:02x}{:02x}".format(*(round(255 * c) for c in o.rgb)) if o.rgb else "palette"
        print(
            f"{o.label} {o.shape} ({o.x_mm:.0f}, {o.y_mm:.0f}) mm {o.length_mm:.0f}x{o.width_mm:.0f} mm yaw {o.yaw_deg:.0f} {rgb}"
        )
    if len(seen):
        err = np.linalg.norm(seen - projected, axis=1)
        print(f"camera check: {len(err)} marker corners, median {np.median(err):.1f} px, max {err.max():.1f} px")
    print(f"wrote {out}")


def read_frames(video: Path, indices: Iterable[int]) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(video)), []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise typer.BadParameter(f"can't read frame {i} of {video}")
        frames.append(frame)
    return frames


@app.command()
def lift(
    video: Path,
    points: Annotated[Path, typer.Argument(help="Point maps from scripts/moge_points.py for the same clip")],
    out: Path = Path("objects.json"),
    width_mm: WidthOpt = A3_MAT.width_mm,
    height_mm: HeightOpt = A3_MAT.height_mm,
) -> None:
    """Object heights and parallax-free footprints from a clip's MoGe-3 point maps, written as the twin's objects."""
    spec, maps = MatSpec(width_mm, height_mm), np.load(points)
    views = []
    for frame, pts, valid in zip(read_frames(video, maps["indices"]), maps["points"], maps["valid"]):
        if (view := MatView.solve(frame, spec, float(maps["hfov"]))) is not None:
            views.append((view, frame, pts.astype(np.float32), valid))
    if not views:
        raise typer.BadParameter(f"no frame of {video} shows two corner markers")
    reference = min(views, key=lambda v: v[0].tilt_deg)[1]  # most top-down view: least warp and parallax
    if not (obs := ScenePerceiver(spec).observe(reference)).objects:
        raise typer.BadParameter(f"no objects perceived in the reference frame: {obs.error or 'none found'}")
    objects = obs.objects
    lifted = HeightLifter().lift(objects, [(v, p, m) for v, _, p, m in views], spec)
    out.write_text(json.dumps([asdict(o) for o in lifted], indent=2))
    for before, after in zip(objects, lifted):
        print(
            f"{after.label} {after.shape}: height {after.height_mm:.0f} mm, "
            f"footprint {before.length_mm:.0f}x{before.width_mm:.0f} -> {after.length_mm:.0f}x{after.width_mm:.0f} mm"
        )
    print(f"{len(views)} views -> {out}")
