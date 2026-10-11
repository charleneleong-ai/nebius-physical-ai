import io
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, Self

import msgpack
import numpy as np
import zmq

CAMERA_KEYS = ("front", "wrist")
DEFAULT_PORT = int(os.environ.get("PREFLIGHT_GR00T_PORT", "5556"))


@dataclass(frozen=True)
class JointLayout:
    """How the arm's six values (pan, lift, elbow, wrist flex, wrist roll, gripper) are grouped into the named blocks
    a GR00T checkpoint reads as state and writes as actions. Each block is (key, start, end) over the six."""

    blocks: tuple[tuple[str, int, int], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _, _ in self.blocks)

    def split(self, joints: np.ndarray) -> dict[str, np.ndarray]:
        """Joint values (..., 6) as one array per block."""
        return {key: joints[..., start:end] for key, start, end in self.blocks}

    @classmethod
    def from_keys(cls, keys: Iterable[str]) -> "JointLayout":
        wanted = tuple(keys)
        for layout in LAYOUTS.values():
            if layout.keys == wanted:
                return layout
        raise ValueError(f"unknown action layout {list(wanted)}; known: {[list(v.keys) for v in LAYOUTS.values()]}")


# The five arm joints as one relative block and the gripper, as in NVIDIA's examples/SO100 config, versus the wrist roll
# in a block of its own (absolute), which is how the community SO-101 checkpoints were trained.
STOCK = JointLayout((("single_arm", 0, 5), ("gripper", 5, 6)))
SPLIT = JointLayout((("single_arm", 0, 4), ("wrist_roll", 4, 5), ("gripper", 5, 6)))
LAYOUTS = {"stock": STOCK, "split": SPLIT}


def action_keys(config: Any) -> tuple[str, ...] | None:
    """The action blocks in a server's `get_modality_config` reply, or None when it names none.

    Servers wrap each modality as {"__ModalityConfig__": True, "as_json": {...}}; the payload may be a JSON string."""
    entry = config.get("action") if isinstance(config, dict) else None
    if isinstance(entry, dict) and "as_json" in entry:
        entry = entry["as_json"]
    if isinstance(entry, bytes | str):
        entry = json.loads(entry)
    keys = entry.get("modality_keys") if isinstance(entry, dict) else None
    return tuple(keys) if keys else None


class MsgCodec:
    """Wire format of Isaac-GR00T's PolicyServer: msgpack, ndarrays as embedded .npy bytes.

    Requests go out in that format, which every server version reads. N1.7's server answers in msgpack_numpy's format
    instead ({nd, type, kind, shape, data}), so both are decoded."""

    @staticmethod
    def encode_ndarray(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def field(obj: dict, name: str) -> Any:
        """A key of a decoded dict whether msgpack gave it as text or as bytes (msgpack_numpy writes bytes keys)."""
        return obj.get(name, obj.get(name.encode()))

    @classmethod
    def decode_ndarray(cls, obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        nd, dtype, data = (cls.field(obj, k) for k in ("nd", "type", "data"))
        if nd is None or dtype is None or data is None or cls.field(obj, "kind") not in (None, "", b""):
            return obj  # not an array, or one of object or structured dtype, which are never decoded
        values = np.frombuffer(data, dtype=np.dtype(dtype.decode() if isinstance(dtype, bytes) else dtype))
        return values.reshape(cls.field(obj, "shape")).copy() if nd is True else values[0]

    @classmethod
    def pack(cls, data: Any) -> bytes:
        return msgpack.packb(data, default=cls.encode_ndarray)

    @classmethod
    def unpack(cls, data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=cls.decode_ndarray)


@dataclass
class Observation:
    front: np.ndarray  # (H, W, 3) uint8
    wrist: np.ndarray  # (H, W, 3) uint8
    joints: np.ndarray  # (6,) LeRobot units: 5 arm joints in [-100, 100], gripper in [0, 100]
    task: str

    @staticmethod
    def stack(observations: list["Observation"], layout: JointLayout = STOCK) -> dict[str, Any]:
        """Batch observations into the server's (batch, time=1, ...) layout, state grouped as the checkpoint expects."""
        joints = np.stack([o.joints for o in observations]).astype(np.float32)[:, None]
        return {
            "video": {cam: np.stack([getattr(o, cam) for o in observations])[:, None] for cam in CAMERA_KEYS},
            "state": layout.split(joints),
            "language": {"annotation.human.task_description": [[o.task] for o in observations]},
        }


class Policy(Protocol):
    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        """Returns action chunks (batch, horizon, 6) of absolute joint targets in LeRobot units."""
        ...


class Gr00tClient:
    """ZeroMQ client for `gr00t/eval/run_gr00t_server.py` (Isaac-GR00T n1d6 and N1.7).

    The joint layout (`STOCK` or `SPLIT`) is whatever the served checkpoint's action modality names, asked of the
    server on first use; pass `layout` to skip the question."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        timeout_ms: int = 120_000,
        layout: JointLayout | None = None,
    ) -> None:
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self._layout = layout
        self._context = zmq.Context.instance()
        self._socket = self.connect()

    def connect(self) -> zmq.Socket:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)
        return socket

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self._socket.send(MsgCodec.pack(request))
            reply = self._socket.recv()
        except zmq.Again:
            self._socket.close()
            self._socket = self.connect()  # a REQ socket is stuck after a timeout
            raise TimeoutError(f"GR00T server at {self.address} did not reply") from None
        if reply == b"ERROR":
            raise RuntimeError("GR00T server error")
        response = MsgCodec.unpack(reply)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ping(self) -> bool:
        try:
            self.call("ping")
        except (TimeoutError, RuntimeError):
            return False
        return True

    @property
    def layout(self) -> JointLayout:
        if self._layout is None:
            self._layout = self.detect_layout()
        return self._layout

    def detect_layout(self) -> JointLayout:
        """The layout named by the server's action modality; STOCK when the server can't say."""
        try:
            keys = action_keys(self.call("get_modality_config"))
        except (RuntimeError, TimeoutError):
            return STOCK
        return STOCK if keys is None else JointLayout.from_keys(keys)

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        layout = self.layout
        observation = Observation.stack(observations, layout)
        action, _info = self.call("get_action", {"observation": observation, "options": None})
        return np.concatenate([action[k] for k in layout.keys], axis=-1)
