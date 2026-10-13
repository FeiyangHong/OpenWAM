"""One paired RoboTwin episode with original control limits and durable evidence.

The caller owns the frozen experiment/scene manifest and service lifecycle.
This module neither selects checkpoints nor expands seeds or retry budgets.
"""

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger, validate_identity
from benchmarks.robotwin.r2_instructions import replay_instruction
from benchmarks.robotwin.r2_policy_client import PairedClient
from benchmarks.robotwin.r2_scene_manifest import inspect_probe
from benchmarks.robotwin.r2_scene_probe import configuration, initial_observation, numeric_difference
from benchmarks.utils.action_conversion import eef20d_to_ee16d, robotwin_endpose_to_eef20d
from benchmarks.utils.client import ServerError
from scripts.statewam.r2.common import atomic_json, sha256, stable_hash


@contextmanager
def count_physics_steps(scene):
    """Count the pinned Python Scene.step without altering its implementation."""
    cls, counter = type(scene), {"steps": 0}
    original = cls.step

    def counted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is scene:
            counter["steps"] += 1
        return result

    cls.step = counted
    try:
        yield counter
    finally:
        cls.step = original


class EpisodeVideo:
    """Lossless input frames, official H264/CRF23 output, verified on close."""

    def __init__(self, path, *, shape):
        self.path, self.shape, self.frames = Path(path), tuple(shape), 0
        height, width, channels = self.shape
        if channels != 3 or height % 2 or width % 2 or self.path.exists():
            raise ValueError("Require an unused path and even-sized RGB frames")
        self.log = self.path.with_suffix(".ffmpeg.log").open("x")
        self.process = subprocess.Popen([
            "ffmpeg", "-nostdin", "-n", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", "10", "-i", "pipe:0",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-threads", "2", "-crf", "23", str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def write(self, frame):
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.shape != self.shape:
            raise ValueError("Video frame shape/dtype changed")
        self.process.stdin.write(np.ascontiguousarray(value).tobytes())
        self.frames += 1

    def close(self):
        try:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise
            if code != 0 or self.frames < 1:
                raise RuntimeError("Video encoding failed or no frames recorded")
            result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                     "-show_entries", "stream=width,height,nb_read_frames,codec_name", "-of", "json", str(self.path)],
                                    capture_output=True, text=True, check=True, timeout=30)
            streams = json.loads(result.stdout)["streams"]
            if len(streams) != 1 or streams[0] != {"codec_name": "h264", "width": self.shape[1],
                                                   "height": self.shape[0], "nb_read_frames": str(self.frames)}:
                raise ValueError("Closed video does not match recorded frame count/geometry")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.path), "-f", "null", "-"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=60)
            return {"frames": self.frames, "fps": 10, "stream": streams[0], "sha256": sha256(self.path),
                    "frame_rule": "initial observation and every subsequent observation; video FPS is not control frequency"}
        finally:
            self.log.close()


def verify_initial(expected, actual):
    if numeric_difference(expected["metadata"], actual["metadata"]) or expected["rgb"] != actual["rgb"]:
        raise ValueError("Initial scene differs from paired manifest; do not consume a policy episode")


def observation_payload(observation):
    endpose = observation["endpose"]
    state = robotwin_endpose_to_eef20d(endpose["left_endpose"], endpose["right_endpose"],
                                     endpose["left_gripper"], endpose["right_gripper"])
    cameras = {name: observation["observation"][name]["rgb"] for name in ("head_camera", "left_camera", "right_camera")}
    return cameras, state


def run_episode(module, *, robotwin_root, scene_row, identity, ledger_root, service_url, stop_file):
    """Run one complete original-limit episode or preserve an explicit invalid.

    Initial-state mismatches and protocol errors remain unclassified partials
    requiring diagnosis. Only evidenced transport/STOP events become invalid.
    """
    from websockets.exceptions import ConnectionClosed

    identity = validate_identity(identity)
    root, stop_file = Path(robotwin_root), Path(stop_file)
    probe = Path(scene_row["probe"])
    verified = inspect_probe(probe, task=identity["task"], condition=identity["condition"], seed=scene_row["seed"])
    verified["scene_id"] = scene_row["scene_id"]
    if (verified != scene_row or not verified["expert_eligible"] or identity["scene_id"] != scene_row["scene_id"]
            or identity["policy_seed"] != scene_row["policy_seed"]
            or identity["instruction_sha256"] != scene_row["instruction"]["text_sha256"]):
        raise ValueError("Episode scene differs from independently verified manifest row")
    cfg, limit = configuration(module, root, identity["task"], identity["condition"])
    if stable_hash(cfg) != scene_row["config_sha256"]:
        raise ValueError("Live environment configuration differs from scene preparation")
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
        raise ValueError("Planner fallback forbidden")
    instruction = replay_instruction(scene_row["instruction"], robotwin_root=root, task=identity["task"],
                                     episode_parameters=scene_row["instruction"]["episode_parameters"])
    with EpisodeLedger(ledger_root, identity) as ledger:
        if stop_file.exists():
            return "stopped"
        attempt = ledger.begin()
        if attempt is None:
            return "already_complete"
        attempt_index = int(attempt.name.split("_")[-1])
        env, client, video = module.class_decorator(identity["task"]), PairedClient(service_url), None
        start = time.monotonic()
        rows, outcome, reason, evidence = [], None, None, None
        counters = {"steps": 0}
        try:
            random.seed(scene_row["seed"])
            env.setup_demo(now_ep_num=0, seed=scene_row["seed"], is_test=True, **copy.deepcopy(cfg))
            if env.step_lim != limit or env.scene.render_system.device.cuda_id != 0:
                raise ValueError("Original action limit or render device differs")
            env.set_instruction(instruction)
            with count_physics_steps(env.scene) as counters:
                obs, _, _ = initial_observation(env, attempt / "initial")
                verify_initial(json.loads((probe / "reset_0/initial.json").read_text()),
                               json.loads((attempt / "initial/initial.json").read_text()))
                ack = client.reset(identity, attempt=attempt_index, instruction=instruction)
                atomic_json(attempt / "reset_ack.json", ack)
                video = EpisodeVideo(attempt / "video.mp4", shape=obs["observation"]["head_camera"]["rgb"].shape)
                video.write(obs["observation"]["head_camera"]["rgb"])
                while env.take_action_cnt < limit:
                    if stop_file.exists():
                        outcome, reason, evidence = "invalid", "external_interrupt", json.loads(stop_file.read_text())
                        break
                    if env.get_instruction() != instruction or env.step_lim != limit:
                        raise ValueError("Instruction or upstream action limit changed")
                    cameras, proprio = observation_payload(obs)
                    before = counters["steps"]
                    request_start = time.monotonic()
                    try:
                        action, response = client.predict(cameras=cameras, state=proprio, instruction=instruction)
                    except ServerError as error:
                        if error.code not in ("policy_output_error", "internal_error"):
                            raise
                        outcome, reason = "failure", "nonfinite_action" if error.code == "policy_output_error" else "model_error"
                        evidence = {"code": error.code, "message": error.message}
                        break
                    except (OSError, TimeoutError, ConnectionClosed) as error:
                        outcome, reason, evidence = "invalid", "transport_disconnect", {"error": f"{type(error).__name__}: {error}"}
                        if stop_file.exists():
                            reason = "external_interrupt"
                            evidence["stop"] = json.loads(stop_file.read_text())
                        break
                    request_seconds = time.monotonic() - request_start
                    converted = eef20d_to_ee16d(action)
                    if converted.shape != (16,) or not np.isfinite(converted).all():
                        outcome, reason, evidence = "failure", "nonfinite_action", {"action": action.tolist()}
                        break
                    control_start, previous_count = time.monotonic(), env.take_action_cnt
                    env.take_action(converted, action_type="ee")
                    if env.take_action_cnt != previous_count + 1:
                        raise ValueError("Upstream policy-action counter did not advance exactly once")
                    row = {"action_index": previous_count, "eef20": action.tolist(), "ee16": converted.tolist(),
                           "proprio": proprio.tolist(), "rgb_sha256": {name: hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
                                                                     for name, rgb in cameras.items()},
                           "response": response, "request_seconds": request_seconds,
                           "control_seconds": time.monotonic() - control_start, "physics_steps": counters["steps"] - before}
                    rows.append(row)
                    with (attempt / "actions.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                    atomic_json(attempt / "progress.json", {"actions": env.take_action_cnt, "physics_steps": counters["steps"],
                                                              "updated_at": time.time(), "attempt": attempt_index})
                    if env.eval_success:
                        outcome, reason = "success", "goal_reached"
                        obs = env.now_obs
                        video.write(obs["observation"]["head_camera"]["rgb"])
                        break
                    if env.take_action_cnt < limit:
                        obs = env.get_obs()
                        video.write(obs["observation"]["head_camera"]["rgb"])
                if outcome is None:
                    outcome, reason = "failure", "action_limit"
            closing_video, video = video, None
            video_report = closing_video.close()
            stats = {"policy_actions": env.take_action_cnt, "physics_steps": counters["steps"],
                     "wall_seconds": time.monotonic() - start, "step_limit": limit, "timestep": env.scene.get_timestep(),
                     "physics_scope": "post-setup initial observation through final control; setup steps excluded", "video": video_report}
            atomic_json(attempt / "rollout.json", {"identity": identity, "stats": stats, "actions": rows,
                                                   "outcome": outcome, "reason": reason, "evidence": evidence})
            artifacts = {"video": "video.mp4", "rollout": "rollout.json"}
            if outcome == "invalid":
                atomic_json(attempt / "interruption.json", {"reason": reason, "evidence": evidence, "time": time.time()})
                artifacts["evidence"] = "interruption.json"
            ledger.finish(attempt, outcome=outcome, reason=reason, stats=stats, artifacts=artifacts)
            return outcome
        except BaseException as error:
            atomic_json(attempt / "error.json", {"error": f"{type(error).__name__}: {error}",
                                                 "time": time.time(), "policy_actions_recorded": len(rows),
                                                 "classification": "unclassified; diagnosis required, no automatic retry"})
            raise
        finally:
            client.close()
            try:
                if video is not None:
                    video.close()
            finally:
                env.close_env()
