from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from nebius_physical_ai.capture.mat import SceneObservation, ScenePerceiver

STATIC_DIR = Path(__file__).parent / "static"


def perceive_jpeg(perceiver: ScenePerceiver, body: bytes) -> SceneObservation | None:
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else perceiver.observe(image)


def create_app(perceiver: ScenePerceiver | None = None) -> FastAPI:
    app = FastAPI(title="Preflight capture")
    perceiver = perceiver or ScenePerceiver()
    app.state.latest = None

    @app.get("/")
    def phone_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "phone.html")

    @app.post("/frame")
    async def frame(request: Request) -> dict[str, Any]:
        obs = await run_in_threadpool(perceive_jpeg, perceiver, await request.body())
        if obs is None:
            raise HTTPException(400, "body is not a decodable image")
        app.state.latest = obs
        return obs.to_dict()

    @app.get("/latest")
    def latest() -> dict[str, Any] | None:
        return app.state.latest and app.state.latest.to_dict()

    return app
