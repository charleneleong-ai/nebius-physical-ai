import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from synthetic_scene import CLEAN, CLUTTERED, Block, SyntheticScene, scene
from typer.testing import CliRunner

from nebius_physical_ai.capture.camera import CameraPose
from nebius_physical_ai.capture.mat import DetectedObject, MatSpec
from nebius_physical_ai.cli import app
from nebius_physical_ai.cosmos import SERVED_MODEL_NAME
from nebius_physical_ai.perception import SCENE_PROMPT, CosmosSceneGraph, MarkerScene, decode_frame
from nebius_physical_ai.pipeline import (
    TOKEN_FACTORY_MODELS,
    CosmosClient,
    NemotronAgent,
    PipelineResult,
    PreflightPipeline,
    TokenFactoryVLClient,
    openai_base_url,
)
from nebius_physical_ai.task import DEFAULT_TASK, TaskSpec
from nebius_physical_ai.trust.verdict import Assessment, Fidelity, Verdict
from nebius_physical_ai.twin.rehearsal import RehearsalResult
from nebius_physical_ai.twin.rollout import STACK_TASK
from nebius_physical_ai.twin.scene import TwinLayout


def rehearsal_result(outcomes: list[bool]) -> RehearsalResult:
    variants = [
        {
            "index": i,
            "success": ok,
            "steps": 160,
            "physics": {"friction": 1.0, "mass_kg": 0.05},
            "objects": [{"x_mm": 120.0, "y_mm": 80.0}],
        }
        for i, ok in enumerate(outcomes)
    ]
    return RehearsalResult(sum(outcomes), len(outcomes), [160] * len(outcomes), variants)


def mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


SCENE_JSON = {
    "objects": [{"label": "blue", "bbox_2d": [100, 200, 160, 260], "confidence": 0.9}],
    "relations": [{"type": "near"}],
    "issues": ["blue is already on red"],
}


def marker_scene(warnings: list[str] | None = None) -> MarkerScene:
    objects = [
        DetectedObject("blue", 130, 150, 900, length_mm=30, width_mm=30),
        DetectedObject("red", 220, 150, 1600, length_mm=40, width_mm=40),
        DetectedObject("yellow", 90, 60, 1600, length_mm=40, width_mm=40, shape="cylinder"),
    ]
    return MarkerScene(objects, [0, 1, 2, 3], 0.4, 0.9, warnings or [])


class TestTokenFactoryClient:
    def test_sends_the_frame_with_the_marker_prompt_and_parses_the_scene_graph(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"], seen["body"] = request.headers["authorization"], json.loads(request.content)
            return chat_reply(json.dumps(SCENE_JSON))

        client = TokenFactoryVLClient(api_key="k", client=mock_client(handler))
        result = asyncio.run(client.perceive("QUJD"))
        body = seen["body"]
        assert seen["auth"] == "Bearer k" and body["model"] == TOKEN_FACTORY_MODELS["qwen3.8-27b-vl"]["model_id"]
        text, image = body["messages"][1]["content"]
        assert image["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
        assert text["text"] == SCENE_PROMPT and "bbox_2d" in SCENE_PROMPT and "0-1000" in SCENE_PROMPT
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] >= 8192  # 2048 produced an empty reply on a real photo
        assert result.objects[0]["label"] == "blue" and result.issues == ["blue is already on red"]

    def test_without_a_key_it_fails_before_any_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
            asyncio.run(TokenFactoryVLClient(client=mock_client(handler)).perceive("QUJD"))

    def test_reads_json_inside_a_code_fence(self) -> None:
        fenced = "```json\n" + json.dumps(SCENE_JSON) + "\n```"
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply(fenced)))
        assert asyncio.run(client.perceive("QUJD")).objects[0]["label"] == "blue"

    def test_reads_json_surrounded_by_prose(self) -> None:
        wordy = "Sure! Here is the scene graph:\n" + json.dumps(SCENE_JSON) + "\nLet me know if you need more."
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply(wordy)))
        assert asyncio.run(client.perceive("QUJD")).relations == [{"type": "near"}]

    def test_takes_the_final_answer_after_chain_of_thought_with_stray_braces(self) -> None:
        thinking = 'Let me think. The box format is {x1, y1, x2, y2} and a draft was {"objects": []}. Final answer:\n'
        client = TokenFactoryVLClient(
            api_key="k", client=mock_client(lambda r: chat_reply(thinking + json.dumps(SCENE_JSON)))
        )
        result = asyncio.run(client.perceive("QUJD"))
        assert result.objects[0]["label"] == "blue" and result.issues == ["blue is already on red"]

    def test_nested_objects_are_not_mistaken_for_the_answer(self) -> None:
        trailing = json.dumps(SCENE_JSON) + "\nNote: the {blue} cube is closest."
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply(trailing)))
        result = asyncio.run(client.perceive("QUJD"))
        assert result.objects == SCENE_JSON["objects"] and result.relations == [{"type": "near"}]

    def test_a_bare_list_of_detections_is_the_objects_of_the_scene(self) -> None:
        # Cosmos 3 on a real photo: no wrapper object, no thinking, just a fenced list (reply captured 2026-10-05)
        reply = (
            '```json\n[\n    {"label": "yellow cylindrical container", "bbox_2d": [630, 453, 774, 693], "shape": "cylinder",'
            ' "confidence": 0.95},\n    {"label": "multicolored ball cluster", "bbox_2d": [465, 228, 586, 453],'
            ' "shape": "sphere cluster", "confidence": 0.97}\n]\n```'
        )
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply(reply)))
        graph = asyncio.run(client.perceive("QUJD"))
        assert [o["label"] for o in graph.objects] == ["yellow cylindrical container", "multicolored ball cluster"]
        assert graph.objects[1]["bbox_2d"] == [465, 228, 586, 453] and graph.issues == []

    def test_a_detection_list_after_thinking_is_not_cut_down_to_its_last_element(self) -> None:
        thinking = (
            'Boxes look like [x1, y1, x2, y2] and the first is {"label": "draft", "bbox_2d": [1, 2, 3, 4]}. Answer:\n'
        )
        answer = json.dumps(
            [{"label": "blue", "bbox_2d": [10, 20, 30, 40]}, {"label": "red", "bbox_2d": [50, 60, 70, 80]}]
        )
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply(thinking + answer)))
        assert [o["label"] for o in asyncio.run(client.perceive("QUJD")).objects] == ["blue", "red"]

    def test_a_list_that_is_not_detections_is_still_rejected(self) -> None:
        client = TokenFactoryVLClient(
            api_key="k", client=mock_client(lambda r: chat_reply('[{"label": "blue"}, {"label": "red"}]'))
        )
        with pytest.raises(TypeError, match="expected a JSON object"):
            asyncio.run(client.perceive("QUJD"))

    def test_reports_a_reasoning_model_that_ran_out_of_tokens(self) -> None:
        def thinking_only(request: httpx.Request) -> httpx.Response:
            choice = {"message": {"content": "", "reasoning_content": "hmm..."}, "finish_reason": "length"}
            return httpx.Response(200, json={"choices": [choice], "usage": {"completion_tokens": 2048}})

        client = TokenFactoryVLClient(api_key="k", client=mock_client(thinking_only))
        with pytest.raises(RuntimeError, match=r"ran out of tokens.*completion_tokens=2048"):
            asyncio.run(client.perceive("QUJD"))

    def test_rejects_a_reply_cut_off_mid_json(self) -> None:
        cut = {"choices": [{"message": {"content": '{"objects": [{"label": "bl'}, "finish_reason": "length"}]}
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: httpx.Response(200, json=cut)))
        with pytest.raises(RuntimeError, match="ran out of tokens"):
            asyncio.run(client.perceive("QUJD"))

    def test_a_finished_reply_that_arrives_in_the_reasoning_field_is_used(self) -> None:
        message = {"content": None, "reasoning": json.dumps(SCENE_JSON)}  # how Cosmos3-Super-Reasoner answers
        body = {"choices": [{"message": message, "finish_reason": "stop"}]}
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: httpx.Response(200, json=body)))
        assert asyncio.run(client.perceive("QUJD")).issues == ["blue is already on red"]

    def test_a_truncated_reasoning_field_is_not_mistaken_for_an_answer(self) -> None:
        message = {"content": None, "reasoning": '{"objects": [{"label": "bl'}
        body = {"choices": [{"message": message, "finish_reason": "length"}]}
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: httpx.Response(200, json=body)))
        with pytest.raises(RuntimeError, match="ran out of tokens"):
            asyncio.run(client.perceive("QUJD"))

    def test_the_cosmos3_reasoner_is_selectable_by_name(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["model"] = json.loads(request.content)["model"]
            return chat_reply(json.dumps(SCENE_JSON))

        client = TokenFactoryVLClient(api_key="k", model="cosmos3-super-reasoner", client=mock_client(handler))
        asyncio.run(client.perceive("QUJD"))
        assert seen["model"] == "nvidia/Cosmos3-Super-Reasoner"

    def test_rejects_prose_instead_of_json(self) -> None:
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply("I see two cubes.")))
        with pytest.raises(ValueError, match="did not return JSON.*I see two cubes"):
            asyncio.run(client.perceive("QUJD"))

    def test_rejects_a_json_array(self) -> None:
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: chat_reply("[1, 2]")))
        with pytest.raises(TypeError, match="expected a JSON object"):
            asyncio.run(client.perceive("QUJD"))

    def test_surfaces_http_errors(self) -> None:
        client = TokenFactoryVLClient(api_key="k", client=mock_client(lambda r: httpx.Response(401)))
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.perceive("QUJD"))


class TestCosmosClient:
    def test_posts_to_the_openai_route_with_the_endpoint_key_and_served_name(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
            return chat_reply(json.dumps(SCENE_JSON))

        client = CosmosClient("https://abc.example", "endpoint-key", client=mock_client(handler))
        result = asyncio.run(client.perceive("QUJD"))
        assert seen["url"] == "https://abc.example/v1/chat/completions" and seen["auth"] == "Bearer endpoint-key"
        assert seen["body"]["model"] == SERVED_MODEL_NAME
        assert "response_format" not in seen["body"]  # constrained decoding would also constrain the thinking
        assert seen["body"]["messages"][1]["content"][1]["image_url"]["url"].endswith("QUJD")
        assert result.issues == ["blue is already on red"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://a.example",
            "https://a.example/",
            "https://a.example/v1",
            "https://a.example/v1/",
            " https://a.example ",
        ],
    )
    def test_the_v1_prefix_is_added_exactly_once(self, url: str) -> None:
        assert openai_base_url(url) == "https://a.example/v1"

    def test_settings_come_from_the_environment_the_deploy_command_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COSMOS_ENDPOINT_URL", "https://env.example/")
        monkeypatch.setenv("COSMOS_API_KEY", "env-key")
        monkeypatch.setenv("COSMOS_MODEL", "my-model")
        client = CosmosClient()
        assert (client.base_url, client.api_key, client.model) == ("https://env.example/v1", "env-key", "my-model")

    def test_the_model_name_defaults_to_what_the_endpoint_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COSMOS_MODEL", raising=False)
        assert CosmosClient("https://a.example").model == SERVED_MODEL_NAME

    def test_a_missing_url_fails_before_any_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COSMOS_ENDPOINT_URL", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        with pytest.raises(RuntimeError, match="COSMOS_ENDPOINT_URL"):
            asyncio.run(CosmosClient(client=mock_client(handler)).perceive("QUJD"))

    def test_an_endpoint_without_auth_gets_no_authorization_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COSMOS_API_KEY", raising=False)
        headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            headers.update(request.headers)
            return chat_reply(json.dumps(SCENE_JSON))

        asyncio.run(CosmosClient("https://a.example", client=mock_client(handler)).perceive("QUJD"))
        assert "authorization" not in headers

    def test_thinking_left_in_the_content_is_dropped(self) -> None:
        reply = "<think>two cubes, one blue...</think>\n```json\n" + json.dumps(SCENE_JSON) + "\n```"
        client = CosmosClient("https://a.example", client=mock_client(lambda r: chat_reply(reply)))
        assert asyncio.run(client.perceive("QUJD")).objects[0]["label"] == "blue"


class TestNemotronAgent:
    assessment: ClassVar[Assessment] = Assessment(0.42, (0.31, 0.55), Fidelity(0.9, 0.8), Verdict.RETRY)
    summary: ClassVar[dict[str, Any]] = {"successes": 13, "n": 32, "p": 0.406, "ci95": [0.25, 0.59]}

    def test_prompt_carries_the_verdict_counts_scene_and_fidelity(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return chat_reply("Rearrange the cubes.")

        agent = NemotronAgent(api_key="k", client=mock_client(handler))
        scene_ = marker_scene(["vision model: blue is already on red"])
        text = asyncio.run(agent.explain_verdict(self.assessment, self.summary, scene_))
        prompt = captured["body"]["messages"][1]["content"]
        assert text == "Rearrange the cubes."
        assert captured["body"]["model"] == TOKEN_FACTORY_MODELS["nemotron-3-ultra"]["model_id"]
        for expected in (
            "RETRY",
            "13/32",
            "42.0%",
            "31.0%-55.0%",
            "0.80",
            "blue (box, 30x30 mm at (130, 150) mm)",
            "yellow (cylinder, 40x40 mm at (90, 60) mm)",
            "4 of 4 corner markers",
            "marker fit 0.4 mm",
            "vision model: blue is already on red",
        ):
            assert expected in prompt

    def test_prompt_names_the_task_objects_not_always_blue_and_red(self) -> None:
        objects = [
            DetectedObject("pomade_tin", 90, 60, 1600, length_mm=40, width_mm=40, shape="cylinder"),
            DetectedObject("red", 220, 150, 1600, length_mm=40, width_mm=40),
        ]
        scene_ = MarkerScene(objects, [0, 1, 2, 3], 0.4, 0.9, [])
        prompt = NemotronAgent(api_key="k")._build_prompt(self.assessment, self.summary, scene_)
        assert "pomade_tin is picked and placed on red; the rest are distractors" in prompt

    def test_no_notes_reads_as_none(self) -> None:
        agent = NemotronAgent(api_key="k")
        assert "notes: none" in agent._build_prompt(self.assessment, self.summary, marker_scene())

    def test_truncated_explanation_raises_instead_of_returning_empty_text(self) -> None:
        empty = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        agent = NemotronAgent(api_key="k", client=mock_client(lambda r: httpx.Response(200, json=empty)))
        with pytest.raises(RuntimeError, match="Nemotron explanation ran out of tokens"):
            asyncio.run(agent.explain_verdict(self.assessment, self.summary, marker_scene()))

    def test_missing_key_raises_instead_of_pretending_to_explain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="--offline-explain"):
            asyncio.run(NemotronAgent().explain_verdict(self.assessment, self.summary, marker_scene()))

    @pytest.mark.parametrize(
        ("assessment", "expected"),
        [
            (Assessment(0.9, (0.8, 0.95), Fidelity(1.0, 1.0), Verdict.GO), "GO"),
            (Assessment(0.2, (0.1, 0.3), Fidelity(1.0, 1.0), Verdict.RETRY), "RETRY"),
            (Assessment(0.9, (0.8, 0.95), Fidelity(0.3, 1.0), Verdict.ASK_HUMAN), "ASK_HUMAN"),
        ],
    )
    def test_offline_explanation_names_the_verdict(
        self, assessment: Assessment, expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        agent = NemotronAgent(allow_fallback=True)
        assert asyncio.run(agent.explain_verdict(assessment, self.summary, marker_scene())).startswith(expected)


class FakeRehearsal:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes, self.calls = outcomes, []

    def run(self, objects: list[DetectedObject], n: int, seed: int) -> RehearsalResult:
        self.calls.append((objects, n, seed))
        return rehearsal_result(self.outcomes)


class FakeVision:
    def __init__(self, graph: CosmosSceneGraph) -> None:
        self.graph, self.frames, self.closed = graph, [], False

    async def perceive(self, frame_b64: str, prompt: str | None = None) -> CosmosSceneGraph:
        self.frames.append(frame_b64)
        return self.graph

    async def close(self) -> None:
        self.closed = True


def confirming(
    s: SyntheticScene, blocks: tuple[Block, ...] = CLEAN, confidence: float = 0.9, issues=()
) -> CosmosSceneGraph:
    return CosmosSceneGraph([s.vl_object(b, confidence) for b in blocks], [], [], list(issues))


def pipeline(
    outcomes: list[bool],
    monkeypatch: pytest.MonkeyPatch,
    graph: CosmosSceneGraph | None = None,
    alive: bool = True,
    task: TaskSpec | None = None,
) -> tuple[PreflightPipeline, FakeVision | None, FakeRehearsal]:
    """A pipeline whose GR00T and vision model are fakes; `graph=None` means markers only."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    pipe = PreflightPipeline(
        n_rollouts=len(outcomes), use_cosmos=False, allow_fallback=True, markers_only=graph is None, task=task
    )
    vision = None
    if graph is not None:
        asyncio.run(pipe.perception.close())  # drop the real HTTP client we are replacing
        vision = pipe.perception = FakeVision(graph)
    rehearsal = pipe.rehearsal = FakeRehearsal(outcomes)
    pipe.gr00t.ping = lambda: alive
    return pipe, vision, rehearsal


def run(pipe: PreflightPipeline, s: SyntheticScene) -> PipelineResult:
    try:
        return asyncio.run(pipe.run(s.b64()))
    finally:
        asyncio.run(pipe.close())


class TestPipelineRun:
    def test_a_clean_marker_scene_with_high_success_is_go(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, rehearsal = pipeline([True] * 32, monkeypatch)
        result = run(pipe, scene(CLEAN))
        assert result.verdict == "GO" and result.explanation.startswith("GO")
        objects, n, seed = rehearsal.calls[0]
        assert ([o.label for o in objects], n, seed) == (["blue", "red"], 32, 42)  # pick first; seed 42 by default
        assert (objects[0].x_mm, objects[0].y_mm) == pytest.approx((130, 150), abs=4.0)  # from the markers
        assert result.scene.perception > 0.9 and result.scene_graph is None

    def test_distractors_are_rehearsed_so_clutter_cannot_hide_behind_a_clean_twin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipe, _, rehearsal = pipeline([True] * 8, monkeypatch)
        run(pipe, scene(CLUTTERED))
        labels = [o.label for o in rehearsal.calls[0][0]]
        assert labels[:2] == ["blue", "red"] and sorted(labels[2:]) == ["dark", "yellow"]

    def test_low_success_with_a_trusted_twin_is_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, _ = pipeline([True] * 2 + [False] * 30, monkeypatch)
        result = run(pipe, scene(CLEAN))
        assert result.verdict == "RETRY" and result.assessment.interval[1] < 0.3

    def test_a_confirming_vision_model_adds_semantics_and_keeps_marker_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        graph = confirming(s, issues=["blue block is already resting on red"])
        pipe, vision, _ = pipeline([True] * 32, monkeypatch, graph)
        result = run(pipe, s)
        assert result.verdict == "GO" and result.scene_graph is graph
        assert [o.confidence for o in result.scene.objects] == pytest.approx([0.9, 0.9])
        assert "vision model: blue block is already resting on red" in result.scene.warnings
        sent = decode_frame(vision.frames[0])
        assert max(sent.shape[:2]) <= 1280  # the vision model gets a downscaled frame

    def test_a_vision_model_that_cannot_find_a_block_where_the_markers_do_stops_the_rehearsal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        graph = CosmosSceneGraph([s.vl_object(CLEAN[0], 0.95, at=(60, 40)), s.vl_object(CLEAN[1])], [], [])
        pipe, _, rehearsal = pipeline([True] * 32, monkeypatch, graph)
        result = run(pipe, s)
        assert result.verdict == "ASK_HUMAN" and rehearsal.calls == [] and result.assessment is None
        assert "did not confirm the blue block" in result.explanation and result.rehearsal_summary == {}

    def test_a_low_confidence_vision_model_means_ask_human_not_a_rehearsal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        pipe, _, rehearsal = pipeline([True] * 32, monkeypatch, confirming(s, confidence=0.4))
        result = run(pipe, s)
        assert result.verdict == "ASK_HUMAN" and rehearsal.calls == []
        assert "twin fidelity 0.3" in result.explanation and "below 0.60" in result.explanation

    def test_two_markers_are_not_enough_to_trust_the_twin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, rehearsal = pipeline([True] * 32, monkeypatch)
        result = run(pipe, scene(CLEAN, markers=(0, 2)))
        assert result.verdict == "ASK_HUMAN" and rehearsal.calls == []
        assert "2 of 4 markers" in result.explanation

    def test_a_frame_without_the_mat_is_ask_human_and_needs_no_servers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = scene(CLEAN, markers=(1,))
        pipe, vision, rehearsal = pipeline([True], monkeypatch, confirming(s), alive=False)  # GR00T is down
        result = run(pipe, s)
        assert result.verdict == "ASK_HUMAN" and "saw 1" in result.explanation and result.scene is None
        assert vision.frames == [] and rehearsal.calls == []

    def test_a_missing_task_block_is_ask_human_before_any_vision_spend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = scene((Block("blue", 130, 150), Block("yellow", 90, 60, 20)))
        pipe, vision, rehearsal = pipeline([True], monkeypatch, confirming(s), alive=False)
        result = run(pipe, s)
        assert result.verdict == "ASK_HUMAN" and "no red block" in result.explanation
        assert vision.frames == [] and rehearsal.calls == []

    def test_a_dead_gr00t_server_fails_before_the_vision_model_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = scene(CLEAN)
        pipe, vision, rehearsal = pipeline([True], monkeypatch, confirming(s), alive=False)
        with pytest.raises(RuntimeError, match="did not answer a ping"):
            run(pipe, s)
        assert vision.frames == [] and rehearsal.calls == []

    def test_an_undecodable_frame_is_a_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, _ = pipeline([True], monkeypatch)
        with pytest.raises(ValueError, match="not a decodable image"):
            asyncio.run(pipe.run("aGVsbG8="))
        asyncio.run(pipe.close())

    @pytest.mark.parametrize(
        ("endpoint", "use_cosmos", "expected"),
        [
            (None, None, TokenFactoryVLClient),  # no endpoint deployed: spend credits, not GPU hours
            ("https://abc.example", None, CosmosClient),
            (None, True, CosmosClient),  # explicit wins, and fails at the first call without a URL
            ("https://abc.example", False, TokenFactoryVLClient),
        ],
    )
    def test_vision_backend_follows_the_endpoint_unless_told_otherwise(
        self, endpoint: str | None, use_cosmos: bool | None, expected: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if endpoint:
            monkeypatch.setenv("COSMOS_ENDPOINT_URL", endpoint)
        else:
            monkeypatch.delenv("COSMOS_ENDPOINT_URL", raising=False)
        pipe = PreflightPipeline(use_cosmos=use_cosmos)
        try:
            assert type(pipe.perception) is expected
        finally:
            asyncio.run(pipe.close())

    def test_the_marker_spec_follows_the_twin_layout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert PreflightPipeline(use_cosmos=False, markers_only=True).perceiver.spec == TwinLayout().mat
        custom = TwinLayout(mat=MatSpec(300.0, 300.0))
        pipe = PreflightPipeline(layout=custom, markers_only=True)
        assert pipe.perceiver.spec == custom.mat
        asyncio.run(pipe.close())

    def test_close_releases_every_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, vision, _ = pipeline([True], monkeypatch, confirming(scene(CLEAN)))
        asyncio.run(pipe.close())
        assert vision.closed and pipe.gr00t._socket.closed and pipe.nemotron._client.is_closed

    def test_markers_only_has_no_vision_client_to_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, _ = pipeline([True], monkeypatch)
        assert pipe.perception is None
        asyncio.run(pipe.close())

    def test_rehearses_the_checkpoint_prompt(self) -> None:
        pipe = PreflightPipeline(use_cosmos=False, markers_only=True, allow_fallback=True)
        try:
            assert pipe.rehearsal.task == STACK_TASK
        finally:
            asyncio.run(pipe.close())

    def test_the_policy_is_told_the_tasks_instruction(self) -> None:
        derived = PreflightPipeline(
            use_cosmos=False, markers_only=True, allow_fallback=True, task=TaskSpec("yellow", "dark")
        )
        explicit = PreflightPipeline(
            use_cosmos=False, markers_only=True, allow_fallback=True, task=TaskSpec("yellow", "dark", "stack it")
        )
        try:
            assert derived.rehearsal.task == "Pick yellow and place on dark"
            assert explicit.rehearsal.task == "stack it"
        finally:
            asyncio.run(derived.close())
            asyncio.run(explicit.close())

    def test_the_default_task_checks_blue_on_red_and_adds_no_distribution_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipe, _, rehearsal = pipeline([True] * 8, monkeypatch)
        result = run(pipe, scene(CLEAN))
        assert (rehearsal.success.top, rehearsal.success.bottom) == ("blue", "red")
        assert result.scene.warnings == []

    def test_colour_names_pick_the_blocks_and_the_check_follows_them(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, rehearsal = pipeline([True] * 8, monkeypatch, task=TaskSpec("yellow", "dark"))
        result = run(pipe, scene(CLUTTERED))
        assert [o.label for o in rehearsal.calls[0][0][:2]] == ["yellow", "dark"]
        assert (rehearsal.success.top, rehearsal.success.bottom) == ("yellow", "dark")
        assert any("outside what it was trained on" in w for w in result.scene.warnings)

    def test_a_task_can_name_objects_only_the_vision_model_knows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = scene(CLUTTERED)
        graph = CosmosSceneGraph([s.vl_object(Block("pomade tin", 90, 60, 20)), s.vl_object(CLUTTERED[1])], [], [])
        pipe, _, rehearsal = pipeline([True] * 32, monkeypatch, graph, task=TaskSpec("pomade tin", "red"))
        result = run(pipe, s)
        objects, _, _ = rehearsal.calls[0]
        assert [o.label for o in objects[:2]] == ["pomade_tin", "red"]  # the twin and the check use the resolved names
        assert (rehearsal.success.top, rehearsal.success.bottom) == ("pomade_tin", "red")
        assert result.verdict == "GO" and result.scene_graph is graph
        assert any("outside what it was trained on" in w for w in result.scene.warnings)

    def test_a_name_the_vision_model_does_not_find_is_ask_human_with_its_scene_graph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = scene(CLEAN)
        graph = confirming(s)
        pipe, vision, rehearsal = pipeline([True], monkeypatch, graph, task=TaskSpec("pomade tin", "red"))
        result = run(pipe, s)
        assert result.verdict == "ASK_HUMAN" and rehearsal.calls == [] and result.scene_graph is graph
        assert (
            "no pomade tin block" in result.explanation and "vision model names ['blue', 'red']" in result.explanation
        )
        assert len(vision.frames) == 1  # the model was asked: only it could have named the object

    def test_a_name_only_vision_can_supply_stops_a_markers_only_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe, _, rehearsal = pipeline([True], monkeypatch, task=TaskSpec("pomade tin", "red"))
        result = run(pipe, scene(CLUTTERED))
        assert result.verdict == "ASK_HUMAN" and rehearsal.calls == []
        assert "no pomade tin block" in result.explanation


def frame_file(tmp_path: Path) -> Path:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"ABC")
    return frame


class TestPreflightCommand:
    def test_command_prints_markers_objects_notes_and_the_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        seen: dict[str, Any] = {}

        async def fake_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            seen.update(
                frame=frame_b64, n=self.n_rollouts, seed=self.seed, port=self.gr00t.address, vision=self.perception
            )
            result = rehearsal_result([True] * 32)
            scene_ = marker_scene(["vision model: stack looks fine"])
            assessment = Assessment.of(result, scene_.objects[0])
            return PipelineResult(scene_, None, result.summary(), assessment, "Stack looks safe.", "GO")

        monkeypatch.setattr(PreflightPipeline, "run", fake_run)
        out = CliRunner().invoke(
            app, ["preflight", str(frame_file(tmp_path)), "--n", "8", "--port", "5557", "--tf-vl", "--offline-explain"]
        )
        assert out.exit_code == 0, out.output
        text = " ".join(out.output.split())
        assert "markers [0, 1, 2, 3], fit 0.4 mm, perception 0.90" in text
        assert "perceived blue: (130, 150) mm 30x30 mm confidence 1.00" in text
        assert "perceived yellow: (90, 60) mm" in text and "note: vision model: stack looks fine" in text
        assert "GO" in text and "Stack looks safe." in text
        assert seen.pop("port").endswith(":5557") and isinstance(seen.pop("vision"), TokenFactoryVLClient)
        assert seen == {"frame": "QUJD", "n": 8, "seed": 42}

    @pytest.mark.parametrize(
        ("endpoint", "flags", "expected"),
        [
            (None, [], TokenFactoryVLClient),
            ("https://abc.example", [], CosmosClient),
            ("https://abc.example", ["--tf-vl"], TokenFactoryVLClient),
            (None, ["--cosmos"], CosmosClient),
        ],
    )
    def test_the_default_vision_model_needs_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str | None, flags: list[str], expected: type
    ) -> None:
        if endpoint:
            monkeypatch.setenv("COSMOS_ENDPOINT_URL", endpoint)
        else:
            monkeypatch.delenv("COSMOS_ENDPOINT_URL", raising=False)
        seen: list[type] = []

        async def fake_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            seen.append(type(self.perception))
            return PipelineResult.ask_human("no red block on the mat")

        monkeypatch.setattr(PreflightPipeline, "run", fake_run)
        out = CliRunner().invoke(app, ["preflight", str(frame_file(tmp_path)), *flags])
        assert out.exit_code == 0, out.output
        assert seen == [expected]

    def test_markers_only_and_camera_options_reach_the_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pose = CameraPose((1.0, 2.0, 3.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), 69.0, (480, 640), 0.5)
        camera = pose.save(tmp_path / "front_camera.json")
        seen: dict[str, Any] = {}

        async def fake_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            seen.update(vision=self.perception, camera=self.layout.front_camera)
            return PipelineResult.ask_human("no red block on the mat")

        monkeypatch.setattr(PreflightPipeline, "run", fake_run)
        out = CliRunner().invoke(
            app, ["preflight", str(frame_file(tmp_path)), "--markers-only", "--camera", str(camera)]
        )
        assert out.exit_code == 0, out.output
        assert seen == {"vision": None, "camera": pose}

    def test_an_ask_human_result_prints_the_reason_and_no_rehearsal_numbers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            return PipelineResult.ask_human("need 2 corner markers, saw 1, so the twin can't be placed on the mat")

        monkeypatch.setattr(PreflightPipeline, "run", fake_run)
        out = CliRunner().invoke(app, ["preflight", str(frame_file(tmp_path)), "--markers-only"])
        text = " ".join(out.output.split())
        assert out.exit_code == 0 and "ASK_HUMAN: need 2 corner markers, saw 1" in text
        assert "p_cal" not in text and "successes" not in text and "perceived" not in text

    def test_pipeline_failure_is_a_one_line_error_not_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            raise RuntimeError("GR00T server at tcp://x:5557 did not answer a ping")

        monkeypatch.setattr(PreflightPipeline, "run", failing_run)
        out = CliRunner().invoke(app, ["preflight", str(frame_file(tmp_path)), "--markers-only"])
        assert out.exit_code == 1
        assert "preflight failed: GR00T server at tcp://x:5557 did not answer a ping" in " ".join(out.output.split())
        assert "Traceback" not in out.output

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ([], DEFAULT_TASK),
            (["--pick", "pomade tin", "--place", "glasses case"], TaskSpec("pomade tin", "glasses case")),
            (["--prompt", "stack it"], TaskSpec(prompt="stack it")),
        ],
    )
    def test_the_task_options_reach_the_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: TaskSpec
    ) -> None:
        seen: list[TaskSpec] = []

        async def fake_run(self: PreflightPipeline, frame_b64: str) -> PipelineResult:
            seen.append(self.task)
            return PipelineResult.ask_human("no red block on the mat")

        monkeypatch.setattr(PreflightPipeline, "run", fake_run)
        out = CliRunner().invoke(app, ["preflight", str(frame_file(tmp_path)), "--markers-only", *flags])
        assert out.exit_code == 0, out.output
        assert seen == [expected]

    def test_picking_an_object_onto_itself_is_a_usage_error(self, tmp_path: Path) -> None:
        out = CliRunner().invoke(app, ["preflight", str(frame_file(tmp_path)), "--pick", "red", "--place", "red"])
        assert out.exit_code == 2 and "same object" in " ".join(out.output.split())

    def test_command_is_registered_once_with_the_new_switches(self) -> None:
        out = CliRunner().invoke(app, ["preflight", "--help"])
        assert out.exit_code == 0
        for flag in (
            "--tf-vl",
            "--offline-explain",
            "--seed",
            "--markers-only",
            "--camera",
            "--pick",
            "--place",
            "--prompt",
        ):
            assert flag in out.output
