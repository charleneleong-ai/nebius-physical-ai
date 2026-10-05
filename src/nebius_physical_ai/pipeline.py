"""Preflight pipeline: ArUco-corrected perception (+ Cosmos 3 semantics) → Twin → GR00T rehearsal → Nemotron verdict.
Updated for Oct 2026 Token Factory SOTA models.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import ScenePerceiver
from nebius_physical_ai.cosmos import SERVED_MODEL_NAME
from nebius_physical_ai.perception import (
    SCENE_PROMPT,
    CosmosSceneGraph,
    MarkerScene,
    SceneError,
    decode_frame,
    fuse,
    precheck,
    vl_detections,
    vl_jpeg_b64,
)
from nebius_physical_ai.policy.gr00t import DEFAULT_PORT, Gr00tClient
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec
from nebius_physical_ai.trust.verdict import Assessment, TrustPolicy, Verdict
from nebius_physical_ai.twin.rehearsal import Perturbation, Rehearsal, RehearsalResult
from nebius_physical_ai.twin.rollout import StackCheck
from nebius_physical_ai.twin.scene import TwinLayout

TOKEN_FACTORY_URL = "https://api.tokenfactory.nebius.com/v1"
TOKEN_FACTORY_KEY_ENV = "NEBIUS_API_KEY"  # the Token Factory key, as documented in .env.example
SEED = 42
# Both Token Factory models reason before answering and the thinking counts against max_tokens: Qwen3.8 used ~2050
# thinking tokens on one scene photo and returned nothing at a 2048 cap.
VL_MAX_TOKENS = 8192
EXPLAIN_MAX_TOKENS = 4096
LLM_TIMEOUT_S = 120.0


def reply_text(data: dict[str, Any], what: str) -> str:
    """The assistant text of a chat completion; raises on a truncated or empty reply instead of returning garbage."""
    choice = data["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    if not content and choice.get("finish_reason") == "stop":
        # Cosmos3-Super-Reasoner on Token Factory answers in `reasoning` and leaves `content` null. Only a finished
        # reply counts: a truncated one is thinking that never reached its answer.
        content = (choice["message"].get("reasoning") or "").strip()
    if choice.get("finish_reason") == "length" or not content:
        used = (data.get("usage") or {}).get("completion_tokens")
        raise RuntimeError(
            f"{what} ran out of tokens before answering (finish_reason={choice.get('finish_reason')!r}, "
            f"completion_tokens={used}); a reasoning model spends its budget thinking, so raise max_tokens"
        )
    if "</think>" in content:  # a server without a reasoning parser leaves the thinking in the content
        content = content.split("</think>", 1)[1].strip()
    return content


def strip_fence(text: str) -> str:
    """The text without a surrounding ```json fence."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].removesuffix("```").strip()
    return body


JSON_OPENER = re.compile(r"[{\[]")


def top_level_values(body: str) -> list[Any]:
    """Every JSON object or array in the text, in order, skipping ones nested inside another (and fragments that
    merely look like JSON, such as `{x1, y1, x2, y2}` quoted while thinking)."""
    decoder, found, pos = json.JSONDecoder(), [], 0
    while (match := JSON_OPENER.search(body, pos)) is not None:
        try:
            value, pos = decoder.raw_decode(body, match.start())
        except json.JSONDecodeError:
            pos = match.start() + 1
            continue
        found.append(value)
    return found


def parse_json_object(text: str, what: str) -> dict[str, Any]:
    """JSON object from model text, tolerating a ```json fence or prose around the braces."""
    body = strip_fence(text)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    else:
        if not isinstance(parsed, dict):
            raise TypeError(f"{what} returned {type(parsed).__name__}, expected a JSON object")
        return parsed
    # Reasoning models think aloud first, so take the last top-level JSON object in the text (the answer),
    # preferring one that has the scene graph's `objects` key over a fragment quoted while thinking.
    found = [v for v in top_level_values(body) if isinstance(v, dict)]
    if found:
        return ([o for o in found if "objects" in o] or found)[-1]
    raise ValueError(f"{what} did not return JSON: {text[:200]!r}")


def is_detection_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(o, dict) and "bbox_2d" in o for o in value)


def parse_scene_json(text: str, what: str) -> dict[str, Any]:
    """A scene graph from model text: an object with `objects`, or a bare list of detections (what Cosmos 3 sends
    when it has nothing else to say). The last such answer wins, after any chain of thought."""
    for value in reversed(top_level_values(strip_fence(text))):
        if isinstance(value, dict) and "objects" in value:
            return value
        if is_detection_list(value):
            return {"objects": value}
    return parse_json_object(text, what)


# ---------------------------------------------------------------------------
# Token Factory model registry (Oct 2026 SOTA)
# ---------------------------------------------------------------------------

TOKEN_FACTORY_MODELS = {
    # Vision-Language (fallback for Cosmos)
    "qwen3.8-27b-vl": {
        "model_id": "Qwen/Qwen3.8-27B",
        "type": "image2text",
        "context_window_k": 262,
        "tokens_per_sec": 218,
        "quantization": "fp4",
        "regions": ["us-central1"],
        "description": "Qwen3.8-27B dense VL model, 27B params, 262K context, 218 tok/s, fp4",
    },
    "qwen3-30b-a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "type": "text2text",
        "context_window_k": 262,
        "tokens_per_sec": 70,
        "quantization": "fp8",
        "regions": ["eu-north1"],
    },
    "qwen3-235b": {
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "type": "text2text",
        "context_window_k": 262,
        "tokens_per_sec": 27,
        "quantization": "fp8",
        "regions": ["eu-north1"],
    },
    "qwen3.5-397b": {
        "model_id": "Qwen/Qwen3.5-397B-A17B",
        "type": "text2text",
        "context_window_k": 262,
        "tokens_per_sec": 80,
        "quantization": "fp4",
        "regions": ["us-central1"],
    },
    # Cosmos3 Reasoner: in the console price list ($0.10 in / $0.30 out per 1M tokens, eu-north1) but, on 2026-10-05,
    # not in /v1/models for our key in any region and a direct call returned 404. Use it with `--tf-vl-model` if it appears.
    "cosmos3-super-reasoner": {
        "model_id": "nvidia/Cosmos3-Super-Reasoner",
        "type": "image2text",
        "regions": ["eu-north1"],
        "description": "NVIDIA Cosmos3 Super Reasoner (64B); replies arrive in the `reasoning` field",
    },
    # Nemotron (NVIDIA - orchestration agent)
    "nemotron-3-ultra": {
        "model_id": "nvidia/Nemotron-3-Ultra-550b-a55b",
        "type": "text2text",
        "context_window_k": 1024,
        "tokens_per_sec": 523,
        "quantization": "fp4",
        "regions": ["us-central1"],
    },
    "nemotron-3-super": {
        "model_id": "nvidia/nemotron-3-super-120b-a12b",
        "type": "text2text",
        "context_window_k": 256,
        "tokens_per_sec": 127,
        "quantization": "fp4",
        "regions": ["us-central1"],
    },
    "nemotron-3.5-lightning": {
        "model_id": "nvidia/Nemotron-3_5-Lightning",
        "type": "text2text",
        "context_window_k": 1024,
        "tokens_per_sec": 314,
        "quantization": "bf16",
        "regions": ["eu-north1"],
    },
    # Vision-Language fallbacks
    "kimi-k3": {
        "model_id": "moonshotai/Kimi-K3",
        "type": "image2text",
        "context_window_k": 1024,
        "tokens_per_sec": 265,
        "quantization": "fp4",
        "regions": ["eu-west2"],
    },
    "glm-5.3-flash": {
        "model_id": "zai-org/GLM-5.3-Flash",
        "type": "image2text",
        "context_window_k": 1024,
        "tokens_per_sec": 349,
        "quantization": "fp4",
        "regions": ["us-north1"],
    },
    "minicpm-v-4.5": {
        "model_id": "openbmb/MiniCPM-V-4_5",
        "type": "image2text",
        "context_window_k": 32,
        "tokens_per_sec": 49,
        "quantization": "fp16",
        "regions": ["eu-north1"],
    },
    # DeepSeek
    "deepseek-v4.1-flash": {
        "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
        "type": "image2text",
        "context_window_k": 1048,
        "tokens_per_sec": 139,
        "quantization": "fp8",
        "regions": ["us-north1"],
    },
}


# ---------------------------------------------------------------------------
# Vision-language clients: semantics for the marker-corrected scene
# ---------------------------------------------------------------------------


class ChatVLClient:
    """OpenAI-compatible chat client that turns a frame into a scene graph.

    `json_mode` asks the server for a JSON object. A vLLM server for a reasoning model (Cosmos 3) should
    leave it off: constrained decoding would apply to the thinking as well.
    """

    what = "VL perception"
    json_mode = True
    requires_key = True
    missing_key = f"{TOKEN_FACTORY_KEY_ENV} not set"

    def __init__(
        self, base_url: str, model: str, api_key: str | None, timeout: float, client: httpx.AsyncClient | None
    ) -> None:
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def perceive(self, frame_b64: str, prompt: str | None = None) -> CosmosSceneGraph:
        """Send a JPEG frame (base64) to the vision model, return its scene graph."""
        if self.requires_key and not self.api_key:
            raise RuntimeError(self.missing_key)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a robot perception system. Output only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or SCENE_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    ],
                },
            ],
            "max_tokens": VL_MAX_TOKENS,
            "temperature": 0.1,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await self._client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        content = parse_scene_json(reply_text(resp.json(), f"{self.what} reply"), self.what)
        return CosmosSceneGraph.from_json(content)

    async def close(self):
        await self._client.aclose()


def openai_base_url(url: str) -> str:
    """An endpoint's managed URL with the /v1 prefix vLLM's OpenAI-compatible server serves under."""
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


class CosmosClient(ChatVLClient):
    """Cosmos 3 behind vLLM's OpenAI-compatible server: hosted on your own GPU (scripts/cosmos_local.sh)."""

    what = "Cosmos perception"
    json_mode = False
    requires_key = False  # an endpoint prototyped without --auth token takes none

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = LLM_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ):
        url = endpoint_url or os.environ.get("COSMOS_ENDPOINT_URL")
        super().__init__(
            openai_base_url(url) if url else "",
            model or os.environ.get("COSMOS_MODEL", SERVED_MODEL_NAME),
            api_key or os.environ.get("COSMOS_API_KEY"),
            timeout,
            client,
        )

    async def perceive(self, frame_b64: str, prompt: str | None = None) -> CosmosSceneGraph:
        if not self.base_url:
            raise RuntimeError(
                "COSMOS_ENDPOINT_URL not set (`mise run cosmos-up` hosts Cosmos on the A100); use --tf-vl for Token Factory"
            )
        return await super().perceive(frame_b64, prompt)


class TokenFactoryVLClient(ChatVLClient):
    """Token Factory client for vision-language models (fallback for Cosmos)."""

    what = "Token Factory VL perception"
    DEFAULT_VL_MODEL = "qwen3.8-27b-vl"  # Best VL fallback Oct 2026

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_VL_MODEL,
        base_url: str = TOKEN_FACTORY_URL,
        timeout: float = LLM_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(
            base_url,
            TOKEN_FACTORY_MODELS[model]["model_id"],
            api_key or os.environ.get(TOKEN_FACTORY_KEY_ENV),
            timeout,
            client,
        )


# ---------------------------------------------------------------------------
# Nemotron agent (Token Factory) - orchestration & explanation
# ---------------------------------------------------------------------------


class NemotronAgent:
    """Nemotron agent for verdict explanation via Token Factory.

    Without an API key it raises, unless `allow_fallback` asks for a canned explanation (offline runs only), so a
    missing key can't pass for a working agent.
    """

    DEFAULT_MODEL = "nemotron-3-ultra"  # Best reasoning model Oct 2026

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = TOKEN_FACTORY_URL,
        timeout: float = LLM_TIMEOUT_S,
        allow_fallback: bool = False,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key or os.environ.get(TOKEN_FACTORY_KEY_ENV)
        self.model = TOKEN_FACTORY_MODELS[model]["model_id"]
        self.base_url = base_url
        self.timeout = timeout
        self.allow_fallback = allow_fallback
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def explain_verdict(
        self,
        assessment: Assessment,
        rehearsal_summary: dict[str, Any],
        scene: MarkerScene,
    ) -> str:
        if not self.api_key:
            if self.allow_fallback:
                return self._fallback_explanation(assessment)
            raise RuntimeError(f"{TOKEN_FACTORY_KEY_ENV} not set; pass --offline-explain for a canned explanation")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Preflight, a robot safety verifier. Explain verdicts clearly in 2-3 sentences.",
                    },
                    {"role": "user", "content": self._build_prompt(assessment, rehearsal_summary, scene)},
                ],
                "max_tokens": EXPLAIN_MAX_TOKENS,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return reply_text(resp.json(), "Nemotron explanation")

    def _build_prompt(
        self,
        assessment: Assessment,
        rehearsal_summary: dict[str, Any],
        scene: MarkerScene,
    ) -> str:
        lo, hi = assessment.interval
        objects = "; ".join(
            f"{o.label} ({o.shape}, {o.length_mm:.0f}x{o.width_mm:.0f} mm at ({o.x_mm:.0f}, {o.y_mm:.0f}) mm)"
            for o in scene.objects
        )
        f = assessment.fidelity
        notes = "; ".join(scene.warnings) or "none"

        return f"""Preflight verdict: {assessment.verdict.value}

Scene ({scene.objects[0].label} is picked and placed on {scene.objects[1].label}; the rest are distractors): {objects}
Perception: {len(scene.markers_found)} of 4 corner markers visible, marker fit {scene.residual_mm:.1f} mm; notes: {notes}
Rehearsal: {rehearsal_summary["successes"]}/{rehearsal_summary["n"]} successes in the twin
Calibrated success probability: {assessment.p:.1%} (95% CI: {lo:.1%}-{hi:.1%})
Twin fidelity: {f.score:.2f} (perception {f.perception:.2f}, robustness to perception error {f.robustness:.2f})

Explain this verdict in 2-3 sentences for a human operator. Include:
1. Why this verdict (reference success rate, fidelity, specific failure modes)
2. What to do next (rearrange, retry, human help)
3. One specific adjustment if RETRY"""

    def _fallback_explanation(self, assessment: Assessment) -> str:
        verdict, p = assessment.verdict.value, assessment.p
        if verdict == "GO":
            return f"GO: {p:.0%} calibrated success, twin fidelity OK. Proceed with execution."
        if verdict == "RETRY":
            return f"RETRY: {p:.0%} calibrated success is too low. Rearrange objects and re-rehearse."
        return f"ASK_HUMAN: twin fidelity {assessment.fidelity.score:.2f} is too low. Manual intervention needed."

    async def close(self):
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    scene: MarkerScene | None  # None when the mat could not be located or lacks a task block
    scene_graph: CosmosSceneGraph | None  # None when the vision model was not consulted
    rehearsal_summary: dict[str, Any]  # empty when the rehearsal was skipped
    assessment: Assessment | None  # None when the rehearsal was skipped
    explanation: str
    verdict: str
    rehearsal: RehearsalResult | None = None  # the rollouts behind `rehearsal_summary`, with example videos

    @classmethod
    def ask_human(
        cls, reason: str, scene: MarkerScene | None = None, scene_graph: CosmosSceneGraph | None = None
    ) -> PipelineResult:
        """The twin can't be trusted for this frame, so there is nothing to rehearse: don't guess."""
        return cls(scene, scene_graph, {}, None, f"ASK_HUMAN: {reason}", Verdict.ASK_HUMAN.value)


class PreflightPipeline:
    """End-to-end Preflight pipeline.

    The ArUco markers place every object on the mat in millimetres; Cosmos 3 (or the Token Factory fallback)
    adds labels, plausibility notes and a cross-check; the twin is rehearsed against GR00T; Nemotron explains.
    """

    def __init__(
        self,
        layout: TwinLayout | None = None,
        gr00t_port: int = DEFAULT_PORT,
        perturbation: Perturbation | None = None,
        n_rollouts: int = 200,
        batch_size: int = 32,
        use_cosmos: bool | None = None,
        tf_vl_model: str = "qwen3.8-27b-vl",
        nemotron_model: str = "nemotron-3-ultra",
        allow_fallback: bool = False,
        seed: int = SEED,
        markers_only: bool = False,
        task: TaskSpec | None = None,
    ):
        self.task = task or DEFAULT_TASK
        self.layout = layout or TwinLayout()
        self.perceiver = ScenePerceiver(self.layout.mat)
        self.gr00t = Gr00tClient(port=gr00t_port)
        self.perception: ChatVLClient | None = None
        if use_cosmos is None:  # Cosmos costs GPU hours; Token Factory runs on credits, so Cosmos only when deployed
            use_cosmos = bool(os.environ.get("COSMOS_ENDPOINT_URL"))
        if not markers_only:
            self.perception = CosmosClient() if use_cosmos else TokenFactoryVLClient(model=tf_vl_model)
        self.nemotron = NemotronAgent(model=nemotron_model, allow_fallback=allow_fallback)
        self.rehearsal = Rehearsal(
            self.gr00t,
            self.task.instruction,
            StackCheck(),  # replaced per run with the resolved names of the task's two blocks
            self.layout,
            perturbation=perturbation,
            batch_size=batch_size,
        )
        self.n_rollouts, self.seed = n_rollouts, seed
        self.trust_policy = TrustPolicy()

    async def run(self, frame_b64: str) -> PipelineResult:
        """Run the pipeline on a phone frame (base64 JPEG or PNG)."""
        image = decode_frame(frame_b64)
        observation, location = await asyncio.to_thread(self.perceiver.perceive, image)
        if location is None:
            return PipelineResult.ask_human(f"{observation.error}, so the twin can't be placed on the mat")
        try:  # geometry alone first: no vision-model spend on a frame that can't support the task
            precheck(observation.objects, self.task, vision=self.perception is not None)
            scene = fuse(observation, location, task=self.task) if self.perception is None else None
        except SceneError as err:
            return PipelineResult.ask_human(str(err))

        # Fail fast on a dead GR00T server instead of after paying for perception.
        if not await asyncio.to_thread(self.gr00t.ping):
            raise RuntimeError(f"GR00T server at {self.gr00t.address} did not answer a ping")

        scene_graph = None
        if self.perception is not None:
            scene_graph = await self.perception.perceive(vl_jpeg_b64(image))
            height, width = image.shape[:2]
            detections, notes = vl_detections(scene_graph, location, (width, height))
            try:
                scene = fuse(observation, location, detections, notes, self.task)
            except SceneError as err:  # a name only the vision model could resolve, and it did not
                return PipelineResult.ask_human(str(err), None, scene_graph)
            scene.warnings.extend(f"vision model: {issue}" for issue in scene_graph.issues)
        if not self.task.in_distribution:
            scene.warnings.append(self.task.out_of_distribution_note())

        if scene.perception < self.trust_policy.min_fidelity:
            reason = (
                f"twin fidelity {scene.perception:.2f} is below {self.trust_policy.min_fidelity:.2f} "
                f"({len(scene.markers_found)} of 4 markers, fit {scene.residual_mm:.1f} mm"
                + (f"; {'; '.join(scene.warnings)}" if scene.warnings else "")
                + ")"
            )
            return PipelineResult.ask_human(reason, scene, scene_graph)

        # Rehearsal is blocking MuJoCo + ZeroMQ work. The twin names blocks by label, so check the resolved ones.
        self.rehearsal.success = StackCheck(top=scene.objects[0].label, bottom=scene.objects[1].label)
        result = await asyncio.to_thread(self.rehearsal.run, scene.objects, self.n_rollouts, self.seed)

        assessment = Assessment.of(
            result,
            perceived_pick=scene.objects[0],
            perception=scene.perception,
            policy=self.trust_policy,
        )
        summary = result.summary()
        explanation = await self.nemotron.explain_verdict(assessment, summary, scene)

        return PipelineResult(scene, scene_graph, summary, assessment, explanation, assessment.verdict.value, result)

    async def close(self):
        self.gr00t.close()
        if self.perception is not None:
            await self.perception.close()
        await self.nemotron.close()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def log_to_wandb(
    frame: Path, frame_b64: str, pipeline: PreflightPipeline, result: PipelineResult, seconds: float, name: str | None
) -> str | None:
    """One W&B run for this answer (project and entity from WANDB_PROJECT / WANDB_ENTITY); returns its URL."""
    from nebius_physical_ai.tracking import Tracker  # imported late: wandb is slow to load and optional for plain runs

    vision = pipeline.perception.model.split("/")[-1] if pipeline.perception else "markers-only"
    config = {
        "frame": frame.name,
        "vision_model": vision,
        "nemotron_model": pipeline.nemotron.model,
        "n": pipeline.n_rollouts,
        "seed": pipeline.seed,
        "gr00t": pipeline.gr00t.address,
        "pick": pipeline.task.pick,
        "place": pipeline.task.place,
        "instruction": pipeline.task.instruction,
        "in_distribution": pipeline.task.in_distribution,
    }
    with Tracker("preflight", name=name or f"{frame.stem}-{vision}", config=config) as tracker:
        tracker.log_preflight(decode_frame(frame_b64), result, seconds)
        return tracker.run.url


def add_pipeline_cli(app):
    # Annotated and typer are module-level imports: this module postpones annotations and Typer resolves them from globals.
    from rich import print
    from rich.markup import escape

    @app.command()
    def preflight(
        frame: Annotated[Path, typer.Argument(help="Phone frame image with the mat's corner markers in view")],
        n: int = typer.Option(200, help="Rollouts per scene"),
        port: int = DEFAULT_PORT,
        use_cosmos: bool | None = typer.Option(
            None,
            "--cosmos/--tf-vl",
            help="Vision model: Cosmos 3, or Token Factory VL. Default: Cosmos if COSMOS_ENDPOINT_URL is set",
        ),
        markers_only: bool = typer.Option(
            False, "--markers-only", help="Skip the vision model: geometry from the ArUco markers alone"
        ),
        camera: Annotated[
            Path | None, typer.Option(help="Calibrated top camera for the twin (npai calibrate-camera)")
        ] = None,
        tf_vl_model: str = typer.Option("qwen3.8-27b-vl", help="Token Factory VL model"),
        nemotron_model: str = typer.Option("nemotron-3-ultra", help="Nemotron model for explanation"),
        offline_explain: bool = typer.Option(
            False, "--offline-explain", help="Canned explanation when no Token Factory key is set"
        ),
        seed: int = typer.Option(SEED, help="Rehearsal seed"),
        pick: str = typer.Option(
            DEFAULT_TASK.pick,
            help="Object to pick: a colour the markers see, or words the vision model uses (pomade tin)",
        ),
        place: str = typer.Option(DEFAULT_TASK.place, help="Object to place it on, named the same way"),
        prompt: Annotated[
            str | None,
            typer.Option(help="Instruction for the policy (default: the checkpoint's own, or one from the names)"),
        ] = None,
        wandb_log: Annotated[
            bool, typer.Option("--wandb/--no-wandb", help="Log this run to W&B (WANDB_PROJECT, WANDB_ENTITY)")
        ] = False,
        run_name: Annotated[str | None, typer.Option(help="W&B run name (default: frame and vision model)")] = None,
    ) -> None:
        """Run full Preflight pipeline on a frame."""
        frame_b64 = base64.b64encode(frame.read_bytes()).decode()
        layout = TwinLayout(front_camera=CameraPose.load(camera)) if camera else None
        try:
            task = TaskSpec(pick, place, prompt)
        except ValueError as err:
            raise typer.BadParameter(str(err)) from err

        async def _run() -> None:
            pipeline = PreflightPipeline(
                layout=layout,
                n_rollouts=n,
                gr00t_port=port,
                use_cosmos=use_cosmos,
                tf_vl_model=tf_vl_model,
                nemotron_model=nemotron_model,
                allow_fallback=offline_explain,
                seed=seed,
                markers_only=markers_only,
                task=task,
            )
            started = time.time()
            try:
                result = await pipeline.run(frame_b64)
            except (ValueError, TypeError, RuntimeError, httpx.HTTPError) as err:
                print(f"[red]preflight failed:[/red] {escape(str(err))}")
                raise typer.Exit(1) from err
            finally:
                await pipeline.close()
            if result.scene:
                scene = result.scene
                print(
                    f"markers {scene.markers_found}, fit {scene.residual_mm:.1f} mm, perception {scene.perception:.2f}"
                )
                for o in scene.objects:
                    dims = f"{o.length_mm:.0f}x{o.width_mm:.0f} mm"
                    print(f"perceived {o.label}: ({o.x_mm:.0f}, {o.y_mm:.0f}) mm {dims} confidence {o.confidence:.2f}")
                for warning in scene.warnings:
                    print(f"[yellow]note:[/yellow] {escape(warning)}")
            print(f"[bold]{result.verdict}[/bold]")
            print(escape(result.explanation))
            if result.assessment:
                print(result.rehearsal_summary)
                print(result.assessment.record())
            if wandb_log:
                url = log_to_wandb(frame, frame_b64, pipeline, result, round(time.time() - started, 1), run_name)
                print(f"W&B run: {url}")

        asyncio.run(_run())
