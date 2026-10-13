import asyncio
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from benchmarks.robotwin.r2_policy_client import PairedClient
from benchmarks.robotwin.r2_policy_protocol import chunk_seeds, reset_identity, validate_reset_ack
from benchmarks.robotwin.r2_policy_service import PairedSession, PolicyOutputError, serve
from benchmarks.utils.client import ServerError, build_payload, encode_numpy_b64
from scripts.statewam.r2.common import GOAL_ID, stable_hash


class Engine:
    def __init__(self):
        self.architecture = SimpleNamespace(binary_command_dims=(), uses_proprioception=True)
        self._dit_cache = None
        self._vace_cache = {"old": object()}
        self._prompt_embed_cache = {"old": object()}
        self.calls = []
        self.actions = np.arange(640, dtype=np.float32).reshape(32, 20)

    def generate(self, conditions):
        self.calls.append(conditions)
        self._vace_cache["used"] = 1
        self._prompt_embed_cache[conditions["prompt"]] = 1
        return {"actions": self.actions.copy()}


def setup():
    instruction = "Adjust the bottle."
    identity = {"goal_id": GOAL_ID, "group": "E0", "task": "adjust_bottle", "condition": "demo_clean",
                "split": "development", "scene_id": "development/adjust_bottle/demo_clean/200001",
                "policy_seed": 123, "replicate": 0, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                "scene_manifest_sha256": "c" * 64, "instruction_sha256": stable_hash(instruction)}
    cfg = OmegaConf.create({"dataloader": {"multiview": True, "camera_layout": ["head_camera", "left_camera", "right_camera"],
                                         "action_mode": "eef", "unify_action": True, "height": 384, "width": 320},
                            "inference": {"denoise_steps": 10, "denoise_mode": "sync", "num_frames": 33, "video_num_frames": 9},
                            "optimization": {"compile": {"enabled": False}, "dit_cache": {"enabled": False}}})
    engine = Engine()
    session = PairedSession(engine, cfg, {key: identity[key] for key in ("group", "checkpoint_sha256", "protocol_sha256")})
    return session, engine, identity, instruction


def reset_payload(identity, instruction, attempt=1):
    return {"type": "reset", "identity": identity, "attempt": attempt,
            "reset": reset_identity(identity, attempt=attempt), "instruction": instruction}


def observation(session, index):
    image = encode_numpy_b64(np.full((16, 16, 3), 128, dtype=np.uint8))
    return {**build_payload(image, image, image, format_prompt_for_inference(session.instruction), [0.] * 20),
            "episode_key": session.expected["episode_key"], "attempt": session.expected["attempt"], "action_index": index}


def test_real_sync_executor_pairs_noise_and_clears_all_engine_caches():
    session, engine, identity, instruction = setup()
    ack = session.reset(reset_payload(identity, instruction))
    validate_reset_ack(ack, reset_identity(identity, attempt=1))
    assert not engine._vace_cache and not engine._prompt_embed_cache
    for index in range(33):
        result = session.predict(observation(session, index))
        assert result["action"] == engine.actions[index % 32].tolist()
        assert result["chunk_index"] == index // 32
        assert (result["generation"] is not None) == (index % 32 == 0)
    assert len(engine.calls) == 2
    for index, call in enumerate(engine.calls):
        seeds = chunk_seeds(identity, index)
        assert (call["seed"], call["action_seed"]) == (seeds["world"], seeds["action"])
        assert call["first_frame_image"][0].size == (320, 384)
    session.reset(reset_payload(identity, instruction, attempt=2))
    assert not engine._vace_cache and not engine._prompt_embed_cache
    assert not session.policy._executor._action_buffer and not session.chunk_timings
    assert session.predict(observation(session, 0))["action"] == engine.actions[0].tolist()
    assert engine.calls[-1]["seed"] == engine.calls[0]["seed"]


@pytest.mark.parametrize("change", [{"group": "E3"}, {"checkpoint_sha256": "d" * 64}, {"protocol_sha256": "d" * 64}])
def test_reset_rejects_different_model_or_protocol(change):
    session, _, identity, instruction = setup()
    with pytest.raises(ValueError, match="loaded checkpoint"):
        session.reset(reset_payload({**identity, **change}, instruction))


def test_request_validation_does_not_silently_reset_or_consume_buffer():
    session, engine, identity, instruction = setup()
    session.reset(reset_payload(identity, instruction))
    session.predict(observation(session, 0))
    for changes in ({"action_index": 0}, {"attempt": 2}, {"prompt": "changed"}, {"images": {}}, {"state": [0.] * 80}):
        with pytest.raises(ValueError):
            session.predict({**observation(session, 1), **changes})
    assert session.next_action == 1 and len(engine.calls) == 1
    session.predict(observation(session, 1))


@pytest.mark.parametrize("actions", [np.zeros((31, 20)), np.zeros((32, 80)), np.full((32, 20), np.nan)])
def test_invalid_model_chunks_fail_without_retries(actions):
    session, engine, identity, instruction = setup()
    engine.actions = actions
    session.reset(reset_payload(identity, instruction))
    with pytest.raises(PolicyOutputError):
        session.predict(observation(session, 0))
    with pytest.raises(ValueError, match="reset required"):
        session.predict(observation(session, 0))
    assert len(engine.calls) == 1


def test_actual_websocket_reset_order_disconnect_and_stop(tmp_path):
    session, engine, identity, instruction = setup()
    output, stop = tmp_path / "service", tmp_path / "STOP"
    failures = []

    def run():
        try:
            asyncio.run(serve(session, port=0, output=output, stop_file=stop, max_messages=100,
                              max_seconds=30, idle_seconds=15))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client = None
    try:
        deadline = time.monotonic() + 10
        while not (output / "ready.json").exists() and time.monotonic() < deadline and thread.is_alive():
            time.sleep(0.05)
        ready = json.loads((output / "ready.json").read_text())
        client = PairedClient(f"ws://127.0.0.1:{ready['port']}", timeout=3)
        client.reset(identity, attempt=1, instruction=instruction)
        cameras = {name: np.full((16, 16, 3), 128, dtype=np.uint8) for name in ("head_camera", "left_camera", "right_camera")}
        for index in range(33):
            action, result = client.predict(cameras=cameras, state=np.zeros(20, dtype=np.float32), instruction=instruction)
            assert result["action_index"] == index and action.tolist() == engine.actions[index % 32].tolist()
        assert len(engine.calls) == 2
        with pytest.raises(ServerError, match="out-of-order"):
            client.transport.predict_once(observation(session, 0))
        client.close()
        deadline = time.monotonic() + 3
        while session.expected is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert session.expected is None and not engine._vace_cache and not engine._prompt_embed_cache
        client.reset(identity, attempt=2, instruction=instruction)
        action, result = client.predict(cameras=cameras, state=[0.] * 20, instruction=instruction)
        assert result["action_index"] == 0 and result["attempt"] == 2
        assert engine.calls[-1]["seed"] == engine.calls[0]["seed"]
    finally:
        if client:
            client.close()
        stop.touch()
        thread.join(timeout=10)
    assert not thread.is_alive() and not failures
    assert json.loads((output / "state.json").read_text())["state"] == "stopped"


def test_client_never_retries_an_ambiguous_request(monkeypatch):
    _, _, identity, instruction = setup()
    client = PairedClient("ws://127.0.0.1:1")
    client.expected, client.instruction = reset_identity(identity, attempt=1), instruction
    calls = []

    def dropped(payload):
        calls.append(payload)
        raise OSError("connection lost after send")

    monkeypatch.setattr(client.transport, "predict_once", dropped)
    cameras = {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in ("head_camera", "left_camera", "right_camera")}
    with pytest.raises(OSError):
        client.predict(cameras=cameras, state=[0.] * 20, instruction=instruction)
    with pytest.raises(ValueError, match="Explicit reset"):
        client.predict(cameras=cameras, state=[0.] * 20, instruction=instruction)
    assert len(calls) == 1


def test_service_exits_at_idle_budget_without_assistant_monitoring(tmp_path):
    session, _, _, _ = setup()
    output, stop = tmp_path / "service", tmp_path / "STOP"
    start = time.monotonic()
    asyncio.run(serve(session, port=0, output=output, stop_file=stop, max_messages=10,
                      max_seconds=5, idle_seconds=0.2))
    assert time.monotonic() - start < 5
    state = json.loads((output / "state.json").read_text())
    assert state["stop_reason"] == "idle_budget" and state["messages"] == 0
    assert json.loads(stop.read_text())["reason"] == "idle_budget"


def test_watchdog_remains_live_during_blocking_generation(tmp_path, monkeypatch):
    session, engine, identity, instruction = setup()
    output, stop = tmp_path / "service", tmp_path / "STOP"
    generating, release = threading.Event(), threading.Event()
    original = engine.generate

    def slow(conditions):
        generating.set()
        assert release.wait(timeout=5)
        return original(conditions)

    monkeypatch.setattr(engine, "generate", slow)
    errors = []

    def run():
        try:
            asyncio.run(serve(session, port=0, output=output, stop_file=stop, max_messages=10,
                              max_seconds=10, idle_seconds=1))
        except BaseException as error:
            errors.append(error)

    server = threading.Thread(target=run, daemon=True)
    server.start()
    client = None
    request = None
    try:
        deadline = time.monotonic() + 5
        while not (output / "ready.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        ready = json.loads((output / "ready.json").read_text())
        client = PairedClient(f"ws://127.0.0.1:{ready['port']}", timeout=5)
        client.reset(identity, attempt=1, instruction=instruction)

        def predict():
            try:
                client.predict(cameras={name: np.zeros((16, 16, 3), dtype=np.uint8)
                                        for name in ("head_camera", "left_camera", "right_camera")},
                               state=[0.] * 20, instruction=instruction)
            except Exception:
                pass  # The watchdog is expected to close the socket.

        request = threading.Thread(target=predict, daemon=True)
        request.start()
        assert generating.wait(timeout=3)
        deadline = time.monotonic() + 3
        while not stop.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert stop.exists() and not release.is_set()
    finally:
        release.set()
        stop.touch(exist_ok=True)
        if request:
            request.join(timeout=5)
        if client:
            client.close()
        server.join(timeout=5)
    assert not server.is_alive() and not errors
