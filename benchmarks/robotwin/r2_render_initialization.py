"""Bounded initialization trace for the simulator's default graphics context."""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.statewam.r2.cuda_identity import verify_cuda_devices


def mark(stage):
    print(json.dumps({"stage": stage, "time": time.time()}), flush=True)
    time.sleep(3)
    if Path(os.environ["R2_STOP_FILE"]).exists():
        raise RuntimeError("Diagnostic stop requested")


def main():
    uuids = verify_cuda_devices()
    if len(uuids) != 2:
        raise ValueError("Diagnostic requires two explicitly leased GPUs")
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(Path(os.environ["R2_RUN_DIR"]) / "runtime")
    mark("before_imports")
    from benchmarks.robotwin.eval_policy_wrapper import _prewarm_cuda_for_curobo, bootstrap_robotwin_module
    _prewarm_cuda_for_curobo()
    mark("cuda_curobo_warm")
    import sapien
    mark("sapien_import")
    from benchmarks.robotwin.r2_scene_probe import bind_renderer, configuration
    print(bind_renderer(), flush=True)
    mark("renderer_bound")
    module = bootstrap_robotwin_module()
    mark("robotwin_import")
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    cfg, _ = configuration(module, root, "adjust_bottle", "demo_clean")
    env = module.class_decorator("adjust_bottle")
    try:
        env.setup_demo(now_ep_num=0, seed=200001, is_test=True, **cfg)
        mark("scene_setup")
        env.get_obs()
        mark("first_observation")
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
