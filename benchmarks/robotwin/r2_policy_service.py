"""R2 identity-bound synchronous policy service, separate from the public WS API."""

import asyncio
import json
import math
from pathlib import Path
import time

import numpy as np

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from benchmarks.robotwin.r2_policy_protocol import chunk_seeds, reset_identity, validate_reset_ack
from scripts.statewam.r2.common import atomic_json, sha256, stable_hash

BINDING_FIELDS = {"group", "checkpoint_sha256", "protocol_sha256"}
CAMERAS = {"head_camera", "left_wrist_camera", "right_wrist_camera"}


def build_session(checkpoint_dir, *, checkpoint_name, group, protocol_sha256):
    """Load one explicitly named R2 checkpoint with a shared eager recipe."""
    import inspect
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import build_server_from_config

    directory = Path(checkpoint_dir).resolve()
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Checkpoint name must be a basename")
    saved = OmegaConf.load(directory / "config.yaml")
    if OmegaConf.select(saved, "r2.group") != group or group not in ("E0", "E1", "E2", "E3"):
        raise ValueError("Checkpoint configuration belongs to another group")
    stats_hash = sha256(directory / "normalization_stats.npy")
    if stats_hash != OmegaConf.select(saved, "r2.statistics_sha256"):
        raise ValueError("Checkpoint statistics do not match the train-only identity")
    digest = sha256(directory / checkpoint_name)
    config_hash = sha256(directory / "config.yaml")
    deploy = OmegaConf.create({
        "inference": {"denoise_steps": 10, "denoise_mode": "sync", "inference_mode": "sync",
                      "inference_horizon": 32, "num_frames": 33, "video_num_frames": 9,
                      "height": 384, "width": 320, "cfg_scale": 1.0, "cfg_merge": False},
        "optimization": {"compile": {"enabled": False}, "dit_cache": {"enabled": False},
                         "decode_video": False, "prompt_embed_cache": {"enabled": True, "maxsize": 32}}})
    server = build_server_from_config(deploy, str(directory), device="cuda:0", ckpt_name=checkpoint_name)
    architecture = server.engine.architecture
    if "action_seed" not in inspect.signature(architecture.generate).parameters:
        raise ValueError("Architecture cannot accept the independent action-noise stream")
    if any(parameter.is_floating_point() and parameter.dtype != torch.bfloat16 for parameter in architecture.parameters()):
        raise ValueError("R2 policy weights must be bf16")
    architecture.eval()
    session = PairedSession(server.engine, server.cfg,
                            {"group": group, "checkpoint_sha256": digest, "protocol_sha256": protocol_sha256})
    session.provenance = {"checkpoint": str(directory / checkpoint_name), "checkpoint_sha256": digest,
                          "config_sha256": config_hash, "statistics_sha256": stats_hash,
                          "deploy": OmegaConf.to_container(deploy, resolve=True),
                          "representation": architecture.repr_contract}
    return session


class PolicyOutputError(RuntimeError):
    """Invalid model output is a policy failure, never an infrastructure retry."""


class PairedSession:
    """One episode stream over the real WAMPolicy and SyncInferenceExecutor."""

    def __init__(self, engine, cfg, binding):
        from omegaconf import OmegaConf
        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.policy import WAMPolicy

        if set(binding) != BINDING_FIELDS:
            raise ValueError("Require explicit checkpoint, group and protocol binding")
        self.engine, self.architecture, self.binding = engine, engine.architecture, dict(binding)
        required = {"inference.denoise_steps": 10, "inference.denoise_mode": "sync",
                    "inference.num_frames": 33, "inference.video_num_frames": 9,
                    "optimization.compile.enabled": False, "optimization.dit_cache.enabled": False,
                    "dataloader.action_mode": "eef", "dataloader.unify_action": True}
        if any(OmegaConf.select(cfg, key) != value for key, value in required.items()):
            raise ValueError("R2 requires the explicit eager, cache-off, ten-step EEF recipe")
        if engine._dit_cache is not None:
            raise ValueError("R2 main comparison forbids velocity cache")
        self.preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        if not self.preprocessor.multiview or (self.preprocessor.img_height, self.preprocessor.img_width) != (384, 320):
            raise ValueError("R2 requires the frozen three-view 384x320 canvas")
        # The facade supplies paired noise only when the existing executor
        # requests a new chunk; buffered actions never consume a noise stream.
        self.policy = WAMPolicy(self, cfg, {"inference_mode": "sync", "inference_horizon": 32})
        self.expected = None
        self.instruction = None
        self.next_action = self.next_chunk = 0
        self.chunk_timings = []
        self.failed = False

    def clear(self):
        self.policy.reset()
        if self.engine._prompt_embed_cache is not None:
            self.engine._prompt_embed_cache.clear()
        self.engine._vace_cache.clear()
        if self.engine._dit_cache is not None:
            self.engine._dit_cache.reset()
        self.expected = self.instruction = None
        self.next_action = self.next_chunk = 0
        self.chunk_timings.clear()
        self.failed = False

    def reset(self, payload):
        expected = reset_identity(payload["identity"], attempt=payload["attempt"])
        if payload.get("reset") != expected:
            raise ValueError("Reset envelope identity differs")
        if any(expected["identity"][key] != value for key, value in self.binding.items()):
            raise ValueError("Episode does not match loaded checkpoint/protocol")
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or stable_hash(instruction) != expected["identity"]["instruction_sha256"]:
            raise ValueError("Instruction text differs from scene identity")
        self.clear()
        self.expected, self.instruction = expected, instruction
        ack = {"type": "reset_ack", **expected, "next_action": 0, "next_chunk": 0,
               "cleared": {"executor": not self.policy._executor._action_buffer,
                           "prompt": not self.engine._prompt_embed_cache, "vace": not self.engine._vace_cache,
                           "velocity": self.engine._dit_cache is None,
                           "temporary_state": not self.chunk_timings and self.next_chunk == 0}}
        validate_reset_ack(ack, expected)
        return ack

    def generate(self, conditions):
        if self.expected is None or self.failed or self.next_action != self.next_chunk * 32:
            raise ValueError("Generation outside an active ordered chunk boundary")
        seeds = chunk_seeds(self.expected["identity"], self.next_chunk)
        start = time.monotonic()
        result = self.engine.generate({**conditions, "seed": seeds["world"], "action_seed": seeds["action"]})
        actions = result["actions"]
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (32, 20) or not np.isfinite(actions).all():
            raise PolicyOutputError("Require a finite 32x20 raw EEF action chunk")
        self.chunk_timings.append({"chunk_index": self.next_chunk, "seeds": seeds,
                                   "seconds": time.monotonic() - start,
                                   "actions_sha256": stable_hash(actions.tolist())})
        self.next_chunk += 1
        return {**result, "actions": actions}

    def predict(self, payload):
        if self.expected is None or self.failed:
            raise ValueError("Explicit reset required before predictions")
        if (payload.get("episode_key") != self.expected["episode_key"] or payload.get("attempt") != self.expected["attempt"]
                or type(payload.get("action_index")) is not int or payload["action_index"] != self.next_action):
            raise ValueError("Stale, duplicate or out-of-order action request")
        if payload.get("prompt") != format_prompt_for_inference(self.instruction):
            raise ValueError("Prompt changed within the fixed episode")
        images = payload.get("images", {})
        if set(images) != CAMERAS or any(images[name] is None for name in CAMERAS):
            raise ValueError("R2 does not permit missing-camera black-fill")
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("Require finite raw 20D EEF proprio")
        start = time.monotonic()
        try:
            obs = self.preprocessor.preprocess(dict(payload))
            action = np.asarray(self.policy.predict_action(obs))
            if action.shape != (20,) or not np.isfinite(action).all():
                raise PolicyOutputError("Invalid final policy action")
        except BaseException:
            self.failed = True
            raise
        index = self.next_action
        self.next_action += 1
        return {"type": "action", "episode_key": self.expected["episode_key"], "attempt": self.expected["attempt"],
                "action_index": index, "chunk_index": index // 32, "seeds": chunk_seeds(self.expected["identity"], index // 32),
                "action": action.tolist(), "latency_ms": (time.monotonic() - start) * 1000,
                "generation": self.chunk_timings[-1] if index % 32 == 0 else None}


async def serve(session, *, port, output, stop_file, max_messages, max_seconds, idle_seconds):
    """Bounded single-owner service; watchdog remains live during inference.

    A supervised GPU job must enforce termination if an in-flight model call
    ignores the STOP deadline. Disconnects invalidate the in-memory episode.
    """
    import websockets

    if (type(max_messages) is not int or max_messages <= 0
            or any(not math.isfinite(value) or value <= 0 for value in (max_seconds, idle_seconds))):
        raise ValueError("Service requires finite positive message, wall and idle budgets")
    output, stop_file = Path(output), Path(stop_file)
    output.mkdir(parents=True, exist_ok=False)
    owner = None
    started = last_activity = time.monotonic()
    count = 0
    stop_event = asyncio.Event()
    state = {"binding": session.binding, "state": "starting", "started_at": time.time(),
             "max_messages": max_messages, "max_seconds": max_seconds, "idle_seconds": idle_seconds}

    async def handle(socket):
        nonlocal owner, count, last_activity
        if owner is not None:
            await socket.close(code=1013, reason="One R2 episode owner per service")
            return
        owner = socket
        try:
            async for raw in socket:
                if stop_file.exists() or stop_event.is_set() or count >= max_messages:
                    break
                count += 1
                last_activity = time.monotonic()
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("Expected an object message")
                    kind = payload.get("type")
                    if kind == "ping":
                        result = {"type": "pong", "binding": session.binding, "protocol": "r2-paired-ws-v1"}
                    elif kind == "reset":
                        result = session.reset(payload)
                    elif kind == "obs":
                        result = await asyncio.to_thread(session.predict, payload)
                    else:
                        raise ValueError("Unsupported R2 message")
                except Exception as error:
                    session.failed = True
                    result = {"type": "error", "code": "policy_output_error" if isinstance(error, PolicyOutputError) else
                              ("protocol_error" if isinstance(error, ValueError) else "internal_error"),
                              "message": f"{type(error).__name__}: {error}"}
                # Persist the result before send: an ambiguous delivery cannot
                # cause an unrecorded duplicate model call after reconnection.
                with (output / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time": time.time(), "message_index": count,
                                             "request_sha256": stable_hash(raw), "result": result}, allow_nan=False) + "\n")
                    stream.flush()
                last_activity = time.monotonic()
                await socket.send(json.dumps(result, allow_nan=False))
                if count >= max_messages:
                    stop_event.set()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            session.clear()
            owner = None

    async def watchdog():
        while not stop_event.is_set():
            elapsed = time.monotonic() - started
            idle = time.monotonic() - last_activity
            if stop_file.exists() or elapsed >= max_seconds or idle >= idle_seconds:
                state["stop_reason"] = "external_stop" if stop_file.exists() else ("wall_budget" if elapsed >= max_seconds else "idle_budget")
                if not stop_file.exists():
                    atomic_json(stop_file, {"requested_at": time.time(), "reason": state["stop_reason"]})
                stop_event.set()
                break
            state.update(messages=count, heartbeat=time.time())
            atomic_json(output / "state.json", state)
            await asyncio.sleep(0.5)

    task = None
    try:
        async with websockets.serve(handle, "127.0.0.1", port, max_size=32 * 1024**2, ping_interval=None,
                                    compression=None, close_timeout=2) as server:
            state.update(state="running", port=server.sockets[0].getsockname()[1])
            atomic_json(output / "ready.json", state)
            task = asyncio.create_task(watchdog())
            await stop_event.wait()
    finally:
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        session.policy.shutdown()
        state.update(state="stopped", messages=count, ended_at=time.time(), wall_seconds=time.monotonic() - started)
        atomic_json(output / "state.json", state)
