"""Dedicated TrainingRTC chunk service. The robot client owns the action clock."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from omegaconf import OmegaConf

from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import (
    MAX_MESSAGE_BYTES, _decode_wire_message, _encode_wire_message,
    _build_argparser, _load_deploy_yaml, _apply_inference_overrides,
    _apply_compile_enabled_override, _log_attention_backends, merge_deploy_cfg,
)

logger = logging.getLogger(__name__)
PROTOCOL = "openwam-training-rtc-v1"


def _integer(data, name, minimum=0):
    value = data.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class RTCChunkServer:
    """One active client, one model worker, no server-side action buffer."""
    def __init__(self, engine, cfg):
        self.engine, self.cfg = engine, cfg
        arch = engine.architecture
        rtc = getattr(arch, "training_rtc", None)
        if rtc is None or not rtc.enabled:
            raise ValueError("RTC server requires an RTC-trained checkpoint")
        self.max_prefix = rtc.max_prefix_steps
        self.horizon = int(OmegaConf.select(cfg, "inference.num_frames")) - 1
        self.contract = dict(getattr(arch, "repr_contract", {}))
        if (self.contract.get("representation") != "eef" or
                self.contract.get("raw_action_dim") != 58 or
                self.contract.get("rot6d_convention") != "column" or
                self.contract.get("action_layout") != "left_eef9,left_hand20,right_eef9,right_hand20"):
            raise ValueError("RTC Wuji server requires the raw58 column-rot6d checkpoint contract")
        self.preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        self.session_id = uuid.uuid4().hex
        self.episode_id = 0
        self.last_request = -1
        self.last_start = -1
        self.last_prefix = None
        self.last_execute = None
        self._connected = False

    def reset(self):
        self.episode_id += 1
        self.last_request = self.last_start = -1
        self.last_prefix = self.last_execute = None
        self.engine.reset_video_saving(self.session_id)
        return {"type": "reset_ack", "session_id": self.session_id, "episode_id": self.episode_id}

    def _conditions(self, request):
        obs = self.preprocessor.preprocess(dict(request))
        return {
            "first_frame_image": [obs["image"]], "prompt": obs.get("prompt", ""),
            "proprio": obs.get("state"), "video_save_session_id": self.session_id,
            "video_save_episode_id": self.episode_id,
        }

    def dispatch(self, request):
        kind = request.get("type")
        if kind == "ping":
            return {"type": "pong", **self.contract, "rtc_protocol": PROTOCOL,
                    "action_horizon": self.horizon, "max_prefix_steps": self.max_prefix,
                    "session_id": self.session_id, "control_hz": 30}
        if kind == "reset":
            return self.reset()
        if kind not in ("rtc_chunk", "rtc_warmup"):
            raise ValueError("RTC server accepts rtc_chunk/rtc_warmup, not legacy obs/obs_chunk")
        if request.get("session_id") != self.session_id or request.get("episode_id") != self.episode_id:
            raise ValueError("Stale RTC session/episode; reset and start a new episode")
        if self.episode_id == 0:
            raise ValueError("Reset is required before RTC generation")
        request_id = _integer(request, "request_id")
        if request_id <= self.last_request:
            raise ValueError("RTC request IDs must increase; do not retry generation")
        start = _integer(request, "start_step")
        execute = _integer(request, "execute_steps", 1)
        prefix = np.asarray(request.get("rtc_prefix", []), dtype=np.float32)
        if prefix.shape == (0,):
            prefix = np.empty((0, 58), dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 58 or not np.isfinite(prefix).all():
            raise ValueError("rtc_prefix must be finite raw actions [d,58]")
        d = len(prefix)
        if d > self.max_prefix or execute + d > self.horizon or (d and d >= execute):
            raise ValueError("RTC requires d <= trained maximum, 0 < d < execute_steps and d+s <= H")
        if kind == "rtc_chunk":
            if self.last_start < 0:
                if start != 0 or d != 0:
                    raise ValueError("First chunk must start at step 0 without a prefix")
            elif start != self.last_start + self.last_prefix + self.last_execute - d:
                raise ValueError("RTC action timeline does not match the previous committed chunk")
        # Consume ID even on generation failure: retries are ambiguous.
        self.last_request = request_id
        t0 = time.monotonic()
        conditions = self._conditions(request)
        if kind == "rtc_warmup":
            if self.last_start >= 0 or not d:
                raise ValueError("Warmup needs a nonempty prefix and must precede executed chunks")
            # Same observation: warm up both scalar and token timestep paths.
            self.engine.generate(conditions)
            self.engine.generate(dict(conditions, rtc_prefix=prefix))
            return {"type": "rtc_warmup_ack", "session_id": self.session_id,
                    "episode_id": self.episode_id, "request_id": request_id,
                    "latency_ms": (time.monotonic() - t0) * 1000}
        if d:
            conditions["rtc_prefix"] = prefix
        result = self.engine.generate(conditions)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 58) or not np.isfinite(actions).all():
            raise ValueError("RTC model returned malformed actions")
        self.last_start, self.last_prefix, self.last_execute = start, d, execute
        return {"type": "rtc_action_chunk", "session_id": self.session_id,
                "episode_id": self.episode_id, "request_id": request_id,
                "start_step": start, "prefix_steps": d, "execute_steps": execute,
                "actions": actions.tolist(), "latency_ms": (time.monotonic() - t0) * 1000}

    def run(self, host, port):
        import websockets

        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-model")

        async def handler(socket):
            if self._connected:
                await socket.close(code=1013, reason="RTC server already has an active client")
                return
            self._connected = True
            self.session_id = uuid.uuid4().hex
            self.episode_id = 0
            self.last_request = self.last_start = -1
            self.last_prefix = self.last_execute = None
            try:
                async for message in socket:
                    binary = isinstance(message, bytes)
                    try:
                        request, binary = _decode_wire_message(message)
                        response = await asyncio.get_running_loop().run_in_executor(worker, self.dispatch, request)
                    except Exception as exc:
                        logger.exception("RTC request failed")
                        response = {"type": "error", "code": "rtc_request_error", "message": str(exc)}
                    await socket.send(_encode_wire_message(response, binary))
            finally:
                self._connected = False

        async def serve():
            async with websockets.serve(handler, host, port, max_size=MAX_MESSAGE_BYTES,
                                        compression=None, ping_interval=None):
                logger.info("TrainingRTC server started on ws://%s:%d", host, port)
                await asyncio.Future()

        try:
            asyncio.run(serve())
        finally:
            worker.shutdown(wait=True)
            self.engine.shutdown()


def main(argv=None):
    parser = _build_argparser()
    parser.description = "Dedicated Wuji TrainingRTC server (client owns action execution)"
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if (args.inference_mode not in (None, "sync") or args.inference_delay_steps is not None
            or args.inference_horizon is not None):
        parser.error("RTC chunk server has no async action executor; set prefix/execution length on the client")
    cfg = _load_deploy_yaml(args.config)
    # Explicit defaults for this entry only. Existing synchronous deploy config is untouched.
    OmegaConf.update(cfg, "optimization.dit_cache.enabled", False)
    OmegaConf.update(cfg, "optimization.compile.enabled", False)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.update(cfg, "inference.rtc.enabled", True)
    OmegaConf.update(cfg, "inference.inference_mode", "sync")
    OmegaConf.update(cfg, "inference.inference_delay_steps", None)
    cfg = _apply_inference_overrides(cfg, args)
    _apply_compile_enabled_override(cfg, args.compile_enabled)
    ckpt_dir = args.ckpt_dir or OmegaConf.select(cfg, "checkpoint_path")
    if not ckpt_dir:
        parser.error("--ckpt-dir is required")
    _log_attention_backends(logger)
    training, arch = load_from_checkpoint_dir(ckpt_dir, device=args.device or cfg.get("device", "cuda"),
                                               ckpt_name=args.ckpt_name)
    merged = merge_deploy_cfg(training, cfg)
    engine = JointInferenceEngine(merged, architecture=arch)
    server = RTCChunkServer(engine, merged)
    server.run(args.host or OmegaConf.select(merged, "server.host", default="0.0.0.0"),
               args.port or OmegaConf.select(merged, "server.port", default=8848))


if __name__ == "__main__":
    main()
