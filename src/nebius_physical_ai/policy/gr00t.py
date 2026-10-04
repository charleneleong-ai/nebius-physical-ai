import io
import os
from dataclasses import dataclass
from typing import Any, Protocol, Self

import msgpack
import numpy as np
import zmq

ARM_KEYS = ("single_arm", "gripper")
CAMERA_KEYS = ("front", "wrist")
DEFAULT_PORT = int(os.environ.get("PREFLIGHT_GR00T_PORT", "5556"))


class MsgCodec:
    """Wire format of Isaac-GR00T's PolicyServer: msgpack, ndarrays as embedded .npy bytes."""

    @staticmethod
    def encode_ndarray(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def decode_ndarray(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

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
    def stack(observations: list["Observation"]) -> dict[str, Any]:
        """Batch observations into the server's (batch, time=1, ...) layout."""
        joints = np.stack([o.joints for o in observations]).astype(np.float32)[:, None]
        return {
            "video": {cam: np.stack([getattr(o, cam) for o in observations])[:, None] for cam in CAMERA_KEYS},
            "state": {"single_arm": joints[..., :5], "gripper": joints[..., 5:]},
            "language": {"annotation.human.task_description": [[o.task] for o in observations]},
        }


class Policy(Protocol):
    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        """Returns action chunks (batch, horizon, 6) of absolute joint targets in LeRobot units."""
        ...


class Gr00tClient:
    """ZeroMQ client for `gr00t/eval/run_gr00t_server.py` (Isaac-GR00T n1d6)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout_ms: int = 120_000) -> None:
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
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

    def act_batch(self, observations: list[Observation]) -> np.ndarray:
        action, _info = self.call("get_action", {"observation": Observation.stack(observations), "options": None})
        return np.concatenate([action[k] for k in ARM_KEYS], axis=-1)
