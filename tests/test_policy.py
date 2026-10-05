import json

import msgpack
import numpy as np
import pytest

from nebius_physical_ai.capture.mat import DetectedObject
from nebius_physical_ai.policy.gr00t import (
    SPLIT,
    STOCK,
    Gr00tClient,
    JointLayout,
    MsgCodec,
    Observation,
    action_keys,
)
from nebius_physical_ai.tracking import Tracker
from nebius_physical_ai.twin.rehearsal import (
    VARIANT_COLUMNS,
    Perturbation,
    Rehearsal,
    RehearsalResult,
    SceneRun,
    Variant,
)
from nebius_physical_ai.twin.rollout import BatchRollout, StackCheck
from nebius_physical_ai.twin.scene import SO101_XML, Twin, TwinLayout

A3 = TwinLayout()
BLOCKS = [DetectedObject("blue", 130.0, 150.0, 900.0), DetectedObject("red", 220.0, 150.0, 900.0)]
needs_model = pytest.mark.skipif(not SO101_XML.exists(), reason="SO-101 menagerie model not fetched")


class HoldPolicy:
    """Stub policy: commands the same LeRobot-unit target for every chunk."""

    def __init__(self, target: np.ndarray, horizon: int = 16) -> None:
        self.chunk = np.tile(target, (horizon, 1))
        self.batch_sizes: list[int] = []

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        self.batch_sizes.append(len(observations))
        return np.stack([self.chunk] * len(observations))


class SlipperyFails:
    """Success check that passes exactly the variants whose blocks are grippy (friction > 0.8)."""

    def __call__(self, twin: Twin) -> bool:
        return bool(twin.model.geom_friction[-1, 0] > 0.8)


def test_codec_round_trips_nested_arrays() -> None:
    payload = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [["pick"]], "c": {"d": np.zeros(2)}}
    out = MsgCodec.unpack(MsgCodec.pack(payload))
    np.testing.assert_array_equal(out["a"], payload["a"])
    assert out["a"].dtype == np.float32 and out["b"] == [["pick"]]


def test_stack_batches_observations() -> None:
    img = np.zeros((4, 5, 3), np.uint8)
    inp = Observation.stack([Observation(img, img, np.arange(6.0), "a"), Observation(img, img, np.ones(6), "b")])
    assert inp["video"]["front"].shape == (2, 1, 4, 5, 3)
    assert inp["state"]["single_arm"].shape == (2, 1, 5) and inp["state"]["gripper"].shape == (2, 1, 1)
    assert inp["language"]["annotation.human.task_description"] == [["a"], ["b"]]


class TestJointLayouts:
    JOINTS = np.arange(6.0)  # pan, lift, elbow, wrist flex, wrist roll, gripper

    def test_stock_groups_five_arm_joints_and_the_gripper(self) -> None:
        parts = STOCK.split(self.JOINTS)
        assert list(parts) == ["single_arm", "gripper"]
        assert parts["single_arm"].tolist() == [0, 1, 2, 3, 4] and parts["gripper"].tolist() == [5]

    def test_split_gives_the_wrist_roll_a_block_of_its_own(self) -> None:
        parts = SPLIT.split(self.JOINTS)
        assert list(parts) == ["single_arm", "wrist_roll", "gripper"]
        assert parts["single_arm"].tolist() == [0, 1, 2, 3] and parts["wrist_roll"].tolist() == [4]
        assert parts["gripper"].tolist() == [5]

    @pytest.mark.parametrize("layout", [STOCK, SPLIT])
    def test_the_blocks_put_back_together_are_the_six_joints_in_order(self, layout: JointLayout) -> None:
        batch = np.stack([self.JOINTS, self.JOINTS + 10])[:, None]  # (batch, time, 6)
        rebuilt = np.concatenate([layout.split(batch)[k] for k in layout.keys], axis=-1)
        np.testing.assert_array_equal(rebuilt, batch)

    def test_a_split_observation_carries_the_roll_under_its_own_key(self) -> None:
        img = np.zeros((4, 5, 3), np.uint8)
        state = Observation.stack([Observation(img, img, np.arange(6.0), "a")], SPLIT)["state"]
        assert {k: v.shape for k, v in state.items()} == {
            "single_arm": (1, 1, 4),
            "wrist_roll": (1, 1, 1),
            "gripper": (1, 1, 1),
        }
        assert state["wrist_roll"].item() == 4.0

    def test_layouts_are_found_by_their_action_keys(self) -> None:
        assert JointLayout.from_keys(["single_arm", "gripper"]) is STOCK
        assert JointLayout.from_keys(("single_arm", "wrist_roll", "gripper")) is SPLIT
        with pytest.raises(ValueError, match="unknown action layout"):
            JointLayout.from_keys(["left_arm", "right_arm"])


class TestActionKeys:
    KEYS = ("single_arm", "wrist_roll", "gripper")

    def test_servers_wrap_each_modality_as_marked_json(self) -> None:
        reply = {
            "action": {"__ModalityConfig__": True, "as_json": {"delta_indices": [0, 1], "modality_keys": self.KEYS}}
        }
        assert action_keys(reply) == tuple(self.KEYS)

    def test_the_older_marker_and_a_json_string_payload_are_read_too(self) -> None:
        payload = json.dumps({"modality_keys": self.KEYS})
        assert action_keys({"action": {"__ModalityConfig_class__": True, "as_json": payload}}) == tuple(self.KEYS)
        assert action_keys({"action": {"as_json": payload.encode()}}) == tuple(self.KEYS)

    @pytest.mark.parametrize("reply", [{}, {"video": {}}, {"action": {"as_json": {}}}, "nonsense", None])
    def test_a_reply_that_names_no_action_blocks_gives_none(self, reply: object) -> None:
        assert action_keys(reply) is None


class FakeServer:
    """Stands in for the policy server behind Gr00tClient.call: answers by endpoint and remembers the requests."""

    def __init__(self, keys: list[str] | None, fail: bool = False) -> None:
        self.keys, self.fail, self.requests = keys, fail, []

    def __call__(self, endpoint: str, data: dict | None = None):
        self.requests.append((endpoint, data))
        if endpoint == "get_modality_config":
            if self.fail:
                raise RuntimeError("GR00T server error: unknown endpoint")
            return {"action": {"__ModalityConfig__": True, "as_json": {"modality_keys": self.keys}}}
        batch = len(data["observation"]["language"]["annotation.human.task_description"])
        arm_width = 4 if "wrist_roll" in self.keys else 5
        # Each block answers with its own index as the value, so the test can see where it lands.
        chunk = {k: np.full((batch, 16, arm_width if k == "single_arm" else 1), i) for i, k in enumerate(self.keys)}
        return chunk, {}


class TestClientLayout:
    def client(self, monkeypatch: pytest.MonkeyPatch, server: FakeServer, **kwargs: object) -> Gr00tClient:
        monkeypatch.setattr(Gr00tClient, "connect", lambda self: None)
        client = Gr00tClient(**kwargs)
        monkeypatch.setattr(client, "call", server)
        return client

    @pytest.mark.parametrize(
        ("keys", "layout"), [(["single_arm", "gripper"], STOCK), (["single_arm", "wrist_roll", "gripper"], SPLIT)]
    )
    def test_the_layout_is_whatever_the_served_checkpoint_names(
        self, monkeypatch: pytest.MonkeyPatch, keys: list[str], layout: JointLayout
    ) -> None:
        assert self.client(monkeypatch, FakeServer(keys)).layout is layout

    def test_a_server_that_cannot_say_is_taken_to_be_stock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.client(monkeypatch, FakeServer(None, fail=True)).layout is STOCK

    def test_the_server_is_asked_once_and_an_explicit_layout_is_never_asked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = FakeServer(["single_arm", "wrist_roll", "gripper"])
        client = self.client(monkeypatch, server)
        _ = client.layout, client.layout
        assert [e for e, _ in server.requests] == ["get_modality_config"]
        explicit = FakeServer(["single_arm", "gripper"])
        assert self.client(monkeypatch, explicit, layout=SPLIT).layout is SPLIT and explicit.requests == []

    def test_a_split_checkpoint_is_sent_split_state_and_its_actions_come_back_in_joint_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = FakeServer(["single_arm", "wrist_roll", "gripper"])
        img = np.zeros((4, 5, 3), np.uint8)
        out = self.client(monkeypatch, server).act_batch([Observation(img, img, np.arange(6.0), "t")] * 2)
        sent = server.requests[-1][1]["observation"]["state"]
        assert set(sent) == {"single_arm", "wrist_roll", "gripper"} and sent["single_arm"].shape[-1] == 4
        assert out.shape == (2, 16, 6)  # four arm joints, the roll, the gripper
        assert out[0, 0].tolist() == [0, 0, 0, 0, 1, 2]  # each block's marker lands in its own joint columns


class TestMsgpackNumpyReplies:
    """N1.7's server answers in msgpack_numpy's format, with bytes keys; these payloads are built by hand."""

    @staticmethod
    def packed(arr: np.ndarray, **extra: object) -> dict:
        return {
            b"nd": True,
            b"type": arr.dtype.str,
            b"kind": b"",
            b"shape": list(arr.shape),
            b"data": arr.tobytes(),
            **extra,
        }

    def test_an_array_with_bytes_keys_decodes_to_the_same_array(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.5
        out = MsgCodec.unpack(msgpack.packb({"action": {"single_arm": self.packed(arr)}}))
        np.testing.assert_array_equal(out["action"]["single_arm"], arr)
        assert out["action"]["single_arm"].dtype == np.float32 and out["action"]["single_arm"].flags.writeable

    def test_text_keys_and_other_dtypes_decode_too(self) -> None:
        arr = np.arange(6, dtype=np.int64).reshape(3, 2)
        payload = {"nd": True, "type": arr.dtype.str, "kind": "", "shape": [3, 2], "data": arr.tobytes()}
        out = MsgCodec.unpack(msgpack.packb(payload))
        np.testing.assert_array_equal(out, arr)

    def test_a_numpy_scalar_decodes_to_a_scalar(self) -> None:
        payload = {b"nd": False, b"type": np.dtype(np.float32).str, b"data": np.float32(2.5).tobytes()}
        assert MsgCodec.unpack(msgpack.packb(payload)) == pytest.approx(2.5)

    def test_object_and_structured_arrays_are_never_decoded(self) -> None:
        arr = np.zeros(2)
        forged = self.packed(arr) | {b"kind": b"O"}
        assert MsgCodec.unpack(msgpack.packb(forged)) == forged  # left as the dict it arrived as

    def test_ordinary_dicts_and_the_modality_marker_pass_through(self) -> None:
        reply = {"action": {"__ModalityConfig__": True, "as_json": {"modality_keys": ["single_arm"]}}, "type": "x"}
        assert MsgCodec.unpack(msgpack.packb(reply)) == reply


@pytest.mark.parametrize(
    ("successes", "n", "lo", "hi"), [(0, 20, 0.0, 0.161), (10, 20, 0.299, 0.701), (16, 20, 0.584, 0.919)]
)
def test_wilson_interval(successes: int, n: int, lo: float, hi: float) -> None:
    assert RehearsalResult(successes, n, [], []).wilson() == pytest.approx((lo, hi), abs=0.002)


def test_variant_rows_are_relative_to_perceived_pick() -> None:
    variant = {
        "index": 0,
        "success": True,
        "steps": 160,
        "physics": {"friction": 0.61234, "mass_kg": 0.04567},
        "objects": [{"x_mm": BLOCKS[0].x_mm + 3.0, "y_mm": BLOCKS[0].y_mm - 1.5}],
    }
    rows = RehearsalResult(1, 1, [160], [variant]).variant_rows(7, BLOCKS[0])
    assert dict(zip(VARIANT_COLUMNS, rows[0])) == {
        "scene_id": 7,
        "index": 0,
        "success": True,
        "steps": 160,
        "friction": 0.612,
        "mass_kg": 0.0457,
        "pick_dx_mm": 3.0,
        "pick_dy_mm": -1.5,
    }


def test_variant_sampling() -> None:
    rng = np.random.default_rng(0)
    exact = Variant.sample(BLOCKS, Perturbation.none(), rng)
    assert exact.objects == BLOCKS and exact.physics.friction == 1.0
    p = Perturbation()
    variants = [Variant.sample(BLOCKS, p, rng) for _ in range(200)]
    dx = np.array([v.objects[0].x_mm - BLOCKS[0].x_mm for v in variants])
    assert abs(dx.mean()) < 1.5 and 3.5 < dx.std() < 6.5
    frictions = [v.physics.friction for v in variants]
    assert p.friction[0] <= min(frictions) and max(frictions) <= p.friction[1]


@needs_model
class TestBatchRollout:
    def test_lerobot_units_round_trip(self) -> None:
        twin = Twin.from_objects(BLOCKS, A3)
        joints = np.array([-50.0, 0.0, 25.0, 90.0, -10.0, 40.0])
        assert twin.to_lerobot(twin.from_lerobot(joints)) == pytest.approx(joints)

    def test_twins_track_target_in_one_batch_per_chunk(self) -> None:
        twins = [Twin.from_objects(BLOCKS, A3) for _ in range(3)]
        target = np.array([30.0, -20.0, 20.0, 40.0, 0.0, 50.0])
        policy = HoldPolicy(target)
        results = BatchRollout(twins, policy, "hold", image_size=(64, 80), record=True).run(StackCheck(), 48)
        assert [r.steps for r in results] == [48] * 3 and not any(r.success for r in results)
        assert [len(r.frames) for r in results] == [4] * 3  # one per chunk plus the final state
        assert results[0].frames[-1].shape == (64, 80, 3)
        assert policy.batch_sizes == [3, 3, 3]
        for twin in twins:
            assert twin.to_lerobot(twin.arm_qpos()) == pytest.approx(target, abs=3)

    def test_stack_check_detects_blue_on_red(self) -> None:
        twin = Twin.from_objects(BLOCKS, A3)
        check = StackCheck()
        twin.settle(0.2)
        assert not check(twin)
        twin.data.joint("blue").qpos[:3] = twin.block_pos("red") + [0, 0, 0.031]
        twin.settle(0.3)
        assert check(twin)

    def test_rehearsal_aggregates_per_variant_outcomes(self) -> None:
        policy = HoldPolicy(np.zeros(6))
        rehearsal = Rehearsal(policy, "t", SlipperyFails(), A3, batch_size=4, max_steps=16, image_size=(32, 40))
        result = rehearsal.run(BLOCKS, n=10, seed=3)
        assert result.n == 10 and [v["index"] for v in result.variants] == list(range(10))
        first = {ok: next(v["index"] for v in result.variants if v["success"] == ok) for ok in (True, False)}
        assert {k: i for k, (i, _) in result.examples.items()} == {"success": first[True], "failure": first[False]}
        media = Tracker.media(SceneRun(4, BLOCKS, result, 1.0))
        assert set(media) == {"video/scene_4_success", "video/scene_4_failure", "media/scene_4_start"}
        assert [v["success"] for v in result.variants] == [v["physics"]["friction"] > 0.8 for v in result.variants]
        assert policy.batch_sizes == [4, 4, 2]
