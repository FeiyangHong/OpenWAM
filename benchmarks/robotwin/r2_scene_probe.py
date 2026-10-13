"""Bounded expert/reset evidence for R2 scene preparation, without a policy.

Collects two fresh initial observations after one expert eligibility check.
It measures differences; it does not invent or freeze reset tolerances.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_instructions import freeze_instruction, replay_instruction
from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cuda_identity import verify_cuda_devices


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite initial state")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported scene metadata value {type(value)}")


def pose_data(pose):
    return {"p": json_value(pose.p), "q": json_value(pose.q)}


def bind_renderer():
    """Process-local device binding; preserve the pinned CPU physics config."""
    import sapien
    import sapien.core as core
    from sapien.wrapper.scene import Scene

    device = sapien.Device("cuda:0")
    if device.cuda_id != 0 or not device.can_render():
        raise RuntimeError("Renderer did not resolve the leased CUDA device")
    engine_class = core.Engine

    class BoundRenderer(core.SapienRenderer):
        def __init__(self):
            # The 3.0.0b1 compatibility wrapper drops **args and constructs the
            # default GPU. Invoke its device-aware native parent explicitly.
            sapien.pysapien.render.SapienRenderer.__init__(self, device=device)

    class BoundEngine(engine_class):
        def create_scene(self, config=None):
            sapien.physx.set_scene_config(config if config is not None else core.SceneConfig())
            return Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])

    def bound_renderer(*args, **kwargs):
        if args or kwargs:
            raise ValueError("Unexpected renderer configuration override")
        return BoundRenderer()

    core.Engine, core.SapienRenderer = BoundEngine, bound_renderer
    return {"cuda_id": device.cuda_id, "pci_string": device.pci_string, "name": device.name,
            "binding": "Process-local Engine scene and native SapienRenderer device; unchanged CPU physics config"}


def configuration(module, root, task, condition):
    directory = root / "task_config"
    cfg = yaml.safe_load((directory / f"{condition}.yml").read_text())
    embodiments = yaml.safe_load((directory / "_embodiment_config.yml").read_text())
    cameras = yaml.safe_load((directory / "_camera_config.yml").read_text())
    limits = yaml.safe_load((directory / "_eval_step_limit.yml").read_text())
    if len(cfg["embodiment"]) != 1 or task not in limits:
        raise ValueError("Require explicit single embodiment and upstream task step limit")
    robot = embodiments[cfg["embodiment"][0]]["file_path"]
    robot_config = module.get_embodiment_config(robot)
    camera = cameras[cfg["camera"]["head_camera_type"]]
    cfg.update(task_name=task, task_config=condition, eval_mode=True, render_freq=0,
               left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
               left_embodiment_config=copy.deepcopy(robot_config), right_embodiment_config=copy.deepcopy(robot_config),
               head_camera_h=camera["h"], head_camera_w=camera["w"])
    return cfg, int(limits[task])


def initial_observation(env, directory):
    """Exactly one get_obs call; reuse this observation for the first policy action."""
    observation = env.get_obs()
    images, image_info = {}, {}
    for name in ("head_camera", "left_camera", "right_camera"):
        rgb = np.asarray(observation["observation"][name]["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or float(rgb.std()) <= 1:
            raise ValueError(f"Invalid policy RGB view {name}")
        images[name] = rgb.copy()
        image_info[name] = {"shape": list(rgb.shape), "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
    actors = [{"name": a.get_name(), "pose": pose_data(a.get_pose())}
              for a in env.scene.get_all_actors()]
    articulations = [{"name": a.get_name(), "pose": pose_data(a.get_pose()),
                      "qpos": json_value(a.get_qpos()), "qvel": json_value(a.get_qvel())}
                     for a in env.scene.get_all_articulations()]
    lights = [{"kind": kind, "pose": pose_data(light.get_global_pose()), "color": json_value(light.get_color())}
              for kind, collection in (("directional", env.direction_light_lst), ("point", env.point_light_lst))
              for light in collection]
    metadata = json_value({"actors": actors, "articulations": articulations, "lights": lights,
                          "ambient_light": env.scene.ambient_light, "table_z_bias": env.table_z_bias,
                          "crazy_random_light": env.crazy_random_light, "scene_info": env.info,
                          "step_lim": env.step_lim, "timestep": env.scene.get_timestep(),
                          "camera_config": {name: {k: v for k, v in values.items() if k != "rgb"}
                                            for name, values in observation["observation"].items()},
                          "endpose": observation["endpose"], "joint_action": observation["joint_action"]})
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "rgb.npz", **images)
    atomic_json(directory / "initial.json", {"metadata": metadata, "rgb": image_info})
    return observation, metadata, images


def numeric_difference(a, b, path="metadata"):
    differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f"Different scene fields at {path}")
        for key in a:
            differences.extend(numeric_difference(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(f"Different scene element count at {path}")
        for i, (left, right) in enumerate(zip(a, b)):
            differences.extend(numeric_difference(left, right, f"{path}[{i}]"))
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Nonfinite reset comparison")
        if a != b:
            differences.append({"field": path, "absolute_difference": abs(a - b)})
    elif type(a) is not type(b) or a != b:
        raise ValueError(f"Different scene identity at {path}")
    return differences


def no_grasp_pose_evidence(error, robotwin_root):
    if type(error) is not AssertionError or str(error) != "target_pose cannot be None for move action.":
        return None
    root = Path(robotwin_root).resolve()
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    if len(frames) < 2:
        return None
    expected = (("envs/_base_task.py", "grasp_actor", ("pre_grasp_pose", "grasp_pose")),
                ("envs/utils/action.py", "__init__", ("target_pose",)))
    evidence = []
    for (frame, line), (name, function, fields) in zip(frames[-2:], expected):
        path = root / name
        if (Path(frame.f_code.co_filename).resolve() != path or frame.f_code.co_name != function
                or any(key not in frame.f_locals or frame.f_locals[key] is not None for key in fields)):
            return None
        evidence.append({"source": name, "source_sha256": sha256(path), "function": function,
                         "line": line, "none_fields": list(fields)})
    return evidence


def record_probe_error(record, error, *, unstable_error, phase, robotwin_root=None):
    if phase == "expert" and isinstance(error, unstable_error):
        record.update(completed=True, expert_eligible=False,
                      rejection={"kind": "unstable_initial_scene", "detail": str(error)})
        return True
    evidence = no_grasp_pose_evidence(error, robotwin_root) if phase == "expert" and robotwin_root is not None else None
    if evidence is not None:
        record.update(completed=True, expert_eligible=False,
                      rejection={"kind": "expert_unsuccessful", "detail": str(error),
                                 "classifier": "r2-expert-no-grasp-pose-v1", "evidence": evidence})
        return True
    record["error"] = f"{type(error).__name__}: {error}"
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", choices=("demo_clean", "demo_randomized"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uuids = verify_cuda_devices()
    if len(uuids) != 1:
        raise ValueError("Single leased GPU required for this scene probe")
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
        raise ValueError("Planner fallback is forbidden")
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    from benchmarks.robotwin.eval_policy_wrapper import bootstrap_robotwin_module, _robotwin_git, VERIFIED_ROBOTWIN_COMMIT
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    if _robotwin_git(str(root), "rev-parse", "HEAD") != VERIFIED_ROBOTWIN_COMMIT:
        raise ValueError("Unverified RoboTwin source version")
    module = bootstrap_robotwin_module()
    renderer = bind_renderer()
    cfg, step_limit = configuration(module, root, args.task, args.condition)
    record = {"goal_id": GOAL_ID, "task": args.task, "condition": args.condition, "seed": args.seed,
              "cuda_uuids": uuids, "renderer": renderer, "config_sha256": stable_hash(cfg),
              "tool_sha256": sha256(__file__), "started_at": time.time(), "completed": False,
              "limits": "One expert and two resets. No policy, final scene manifest, frozen tolerance or full physics snapshot."}
    atomic_json(output / "config.json", cfg)
    env = module.class_decorator(args.task)
    phase = "expert"
    try:
        random.seed(args.seed)
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **copy.deepcopy(cfg))
        if env.scene.render_system.device.cuda_id != 0 or env.step_lim != step_limit:
            raise ValueError("Render device or task limit mismatch")
        episode = env.play_once()
        record["expert_eligible"] = bool(env.plan_success and env.check_success())
        record["expert_info"] = json_value(episode)
        env.close_env()
        if not record["expert_eligible"]:
            record["rejection"] = {"kind": "expert_unsuccessful", "detail": "Upstream expert plan or success check failed"}
            record["completed"] = True
            return
        phase = "replay"
        instruction = freeze_instruction(root, args.task, episode["info"], seed=args.seed)
        atomic_json(output / "instruction.json", instruction)
        metadata, images = [], []
        for index in range(2):
            if Path(os.environ["R2_STOP_FILE"]).exists():
                record["stopped"] = True
                return
            random.seed(args.seed)
            env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **copy.deepcopy(cfg))
            env.set_instruction(replay_instruction(instruction, robotwin_root=root, task=args.task,
                                                   episode_parameters=episode["info"]))
            _, state, rgb = initial_observation(env, output / f"reset_{index}")
            metadata.append(state)
            images.append(rgb)
            env.close_env()
        record["numeric_differences"] = numeric_difference(*metadata)
        record["rgb_differences"] = {}
        for name in images[0]:
            delta = np.abs(images[0][name].astype(np.int16) - images[1][name].astype(np.int16))
            record["rgb_differences"][name] = {"max": int(delta.max()), "mean": float(delta.mean()),
                                                "fraction_changed": float((delta != 0).mean())}
        record["completed"] = True
    except BaseException as error:
        if not record_probe_error(record, error, unstable_error=module.UnStableError, phase=phase, robotwin_root=root):
            raise
    finally:
        env.close_env()
        record["ended_at"] = time.time()
        atomic_json(output / "scene_probe.json", record)


if __name__ == "__main__":
    main()
