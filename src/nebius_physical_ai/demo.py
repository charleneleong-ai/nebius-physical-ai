"""The public demo: upload a photo of the mat for a live check, and browse fully rehearsed scenes.

A free CPU host can't run GR00T, so an upload runs the pipeline up to the trust gate (`rehearse=False`): ArUco markers
place the twin on the mat, a Token Factory vision model reads the scene, and the gate either accepts the twin or answers
ASK_HUMAN with its reason. The gallery shows the whole pipeline: scenes rehearsed on a GPU with the same job image
(`npai preflight-job`), each with its verdict, success rate, rollout video and Nemotron's explanation. Uploads spend
Token Factory credits, so they are capped per hour across all visitors.

    GALLERY_DIR=gallery npai demo --port 7860
"""

import asyncio
import base64
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nebius_physical_ai.job import result_record
from nebius_physical_ai.perception import decode_frame
from nebius_physical_ai.pipeline import PreflightPipeline
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec

STATIC = Path(__file__).parent / "capture/static"
MAX_UPLOAD_BYTES = 8_000_000
PER_VISITOR_PER_HOUR = 5  # one visitor can't spend the whole hourly cap
CHECK_TIMEOUT_S = 60.0


class HourlyCap:
    """At most `limit` calls in any rolling hour, shared by every visitor."""

    def __init__(self, limit: int) -> None:
        self.limit, self.calls = limit, deque[float]()

    def take(self) -> float | None:
        """None when a call may go ahead (and counts it); otherwise seconds until the next slot frees."""
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        if len(self.calls) >= self.limit:
            return 3600 - (now - self.calls[0]) if self.calls else 3600.0
        self.calls.append(now)
        return None

    def refund(self) -> None:
        """Give back the latest call, when it ended before spending anything."""
        if self.calls:
            self.calls.pop()


class Admission:
    """Who may spend a vision call now: the hourly caps (everyone's and each visitor's) and one call at a time."""

    def __init__(self, cap_per_hour: int) -> None:
        self.cap, self.lock = HourlyCap(cap_per_hour), asyncio.Lock()
        self.visitors: defaultdict[str, HourlyCap] = defaultdict(lambda: HourlyCap(PER_VISITOR_PER_HOUR))

    def take(self, visitor: str) -> None:
        """Count a call against both caps, or raise 429 without counting it against either."""
        if self.lock.locked():  # no queue behind the call in flight
            raise HTTPException(429, "another photo is being checked; try again in half a minute")
        if (wait := self.visitors[visitor].take()) is not None:
            raise HTTPException(429, f"your hourly limit is reached; try again in {wait / 60:.0f} minutes")
        if (wait := self.cap.take()) is not None:
            self.visitors[visitor].refund()
            raise HTTPException(429, f"the demo's hourly limit is reached; try again in {wait / 60:.0f} minutes")

    def refund(self, visitor: str) -> None:
        self.visitors[visitor].refund()
        self.cap.refund()


def gallery_index(root: Path) -> list[dict[str, Any]]:
    """Every rehearsed scene under `root`: <scene>/result.json plus <scene>.jpg and its videos."""
    scenes = []
    for result in sorted(root.glob("*/result.json")):
        name = result.parent.name
        scenes.append({"name": name, "frame": f"{name}.jpg", **json.loads(result.read_text())})
    return scenes


async def read_upload(request: Request) -> bytes:
    """The request body, refused as soon as it passes MAX_UPLOAD_BYTES rather than after buffering all of it."""
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"send one JPEG or PNG under {MAX_UPLOAD_BYTES // 1_000_000} MB")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"send one JPEG or PNG under {MAX_UPLOAD_BYTES // 1_000_000} MB")
    if not body:
        raise HTTPException(400, "the upload is empty")
    return bytes(body)


def create_demo(gallery: Path, cap_per_hour: int = 30) -> FastAPI:
    app = FastAPI(title="Preflight demo")
    app.state.admission = admission = Admission(cap_per_hour)
    app.mount("/gallery", StaticFiles(directory=gallery), name="gallery")

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC / "demo.html")

    @app.get("/api/gallery")
    def scenes() -> list[dict[str, Any]]:
        return gallery_index(gallery)

    @app.get("/api/task")
    def trained_task() -> dict[str, str]:
        """The one instruction the checkpoint was fine-tuned on, so the page never has to guess what counts as
        in-distribution: `pick`/`place` are selectors (`blue`, not `blue cube`) and only they make `standard_names`."""
        return {"pick": DEFAULT_TASK.pick, "place": DEFAULT_TASK.place, "instruction": DEFAULT_TASK.instruction}

    @app.get("/api/suites")
    def suites() -> list[dict[str, Any]]:
        """Benchmark suites with their scores and rollouts (deploy/suite_cards.py), empty when not rendered."""
        path = gallery / "suites.json"
        return json.loads(path.read_text()) if path.exists() else []

    @app.post("/api/check")
    async def check(request: Request, pick: str = DEFAULT_TASK.pick, place: str = DEFAULT_TASK.place) -> dict[str, Any]:
        try:  # naming an object the checkpoint never trained on is allowed: the gate is what reports that
            task = TaskSpec(pick=pick, place=place)
        except ValueError as err:
            raise HTTPException(400, str(err)) from err
        body = await read_upload(request)
        frame_b64 = base64.b64encode(body).decode()
        try:
            decode_frame(frame_b64)  # a bad file is refused before it costs anyone a slot
        except ValueError as err:
            raise HTTPException(400, str(err)) from err
        visitor = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
        admission.take(visitor)
        async with admission.lock:  # one vision call at a time keeps the spend predictable
            started, pipeline = time.time(), PreflightPipeline(rehearse=False, task=task)
            try:
                result = await asyncio.wait_for(pipeline.run(frame_b64), CHECK_TIMEOUT_S)
            except Exception as err:
                admission.refund(visitor)
                status = 504 if isinstance(err, TimeoutError) else 502
                raise HTTPException(status, f"the vision model did not answer usably: {type(err).__name__}") from err
            finally:
                await pipeline.close()
        if result.scene_graph is None:  # stopped at the markers, before the vision call: nothing was spent
            admission.refund(visitor)
        return result_record(result, round(time.time() - started, 1), task)

    return app


def demo(port: int = 7860, cap_per_hour: int = 30) -> None:
    """Serve the public demo page (GALLERY_DIR holds the rehearsed scenes)."""
    uvicorn.run(create_demo(Path(os.environ.get("GALLERY_DIR", "gallery")), cap_per_hour), host="0.0.0.0", port=port)
