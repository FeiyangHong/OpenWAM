from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from openwam.deploy.server import (
    ACTION_CHUNK,
    OBS_CHUNK,
    PolicyServer,
    _decode_wire_message,
    _encode_wire_message,
)


def test_msgpack_wire_preserves_compressed_image_bytes():
    message = {"type": OBS_CHUNK, "images": {"head_camera": b"jpeg"}, "count": 4}
    encoded = _encode_wire_message(message, True)
    decoded, binary = _decode_wire_message(encoded)
    assert binary is True
    assert decoded == message


def test_predict_chunk_preprocesses_once_and_returns_requested_actions():
    server = PolicyServer(engine=SimpleNamespace(), cfg=OmegaConf.create({}))
    calls = {"preprocess": 0, "predict": 0}

    class Preprocessor:
        def preprocess(self, obs):
            calls["preprocess"] += 1
            return obs

    class Policy:
        def predict_action(self, obs):
            calls["predict"] += 1
            return np.array([calls["predict"], 2.0], dtype=np.float32)

    server._policy = Policy()
    server._obs_preprocessor = Preprocessor()
    result = server.predict_chunk({"images": {}}, 3)
    result["type"] = ACTION_CHUNK

    assert calls == {"preprocess": 1, "predict": 3}
    assert result["actions"] == [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]
    assert result["count"] == 3
