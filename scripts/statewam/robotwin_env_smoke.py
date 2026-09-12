"""Check real RoboTwin reset, RGB rendering and expert planning without a policy."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/robotwin"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--condition", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
        raise RuntimeError("Environment smoke requires cuRobo, not the planner fallback")
    from eval_policy_wrapper import bootstrap_robotwin_module

    module = bootstrap_robotwin_module()
    config_root = Path(os.environ["ROBOTWIN_PATH"]) / "task_config"
    cfg = yaml.safe_load((config_root / f"{args.condition}.yml").read_text(encoding="utf-8"))
    embodiments = yaml.safe_load((config_root / "_embodiment_config.yml").read_text(encoding="utf-8"))
    cameras = yaml.safe_load((config_root / "_camera_config.yml").read_text(encoding="utf-8"))
    if len(cfg["embodiment"]) != 1:
        raise ValueError("This smoke uses the published single-embodiment dual-arm protocol")
    robot_path = embodiments[cfg["embodiment"][0]]["file_path"]
    robot_config = module.get_embodiment_config(robot_path)
    camera = cameras[cfg["camera"]["head_camera_type"]]
    cfg.update(
        task_name=args.task, task_config=args.condition, eval_mode=True, render_freq=0,
        left_robot_file=robot_path, right_robot_file=robot_path, dual_arm_embodied=True,
        left_embodiment_config=robot_config, right_embodiment_config=robot_config,
        head_camera_h=camera["h"], head_camera_w=camera["w"],
    )
    env = module.class_decorator(args.task)
    record = {"goal_id": "r1/robotwin-smoke", "task": args.task, "condition": args.condition,
              "seed": args.seed, "pass": False}
    try:
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **cfg)
        observation = env.get_obs()
        record["cameras"] = {}
        for name, payload in observation["observation"].items():
            if "rgb" not in payload:
                continue
            rgb = np.asarray(payload["rgb"])
            if not np.isfinite(rgb).all() or float(rgb.std()) <= 1:
                raise RuntimeError(f"Nonfinite or blank camera: {name}")
            Image.fromarray(rgb.astype(np.uint8)).save(output / f"{name}.png")
            record["cameras"][name] = {"shape": list(rgb.shape), "std": float(rgb.std())}
        if not {"head_camera", "left_camera", "right_camera"}.issubset(record["cameras"]):
            raise RuntimeError("Missing a required policy RGB view")
        env.play_once()
        record["expert_plan_success"] = bool(env.plan_success)
        record["expert_task_success"] = bool(env.check_success())
        record["pass"] = record["expert_plan_success"] and record["expert_task_success"]
        if not record["pass"]:
            raise RuntimeError("This seed did not pass the expert plan/task check")
    finally:
        (output / "environment_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
        env.close_env()
    print(f"Environment smoke passed: {output}")


if __name__ == "__main__":
    main()
