"""Run bounded upstream RoboTwin evaluation with per-episode smoke evidence."""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/robotwin"))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args, output):
    from eval_policy_wrapper import bootstrap_robotwin_module

    module = bootstrap_robotwin_module()
    config = yaml.safe_load((ROOT / "benchmarks/robotwin/policy_config.yml").read_text(encoding="utf-8"))
    config.update(task_name=args.task, task_config=args.condition, ckpt_setting=args.label,
                  seed=0, host="127.0.0.1", port=args.port)
    (output / "policy_config.yml").write_text(yaml.safe_dump(config))
    episodes, resets = [], []
    original_eval = module.eval_policy

    def instrumented_eval(task_name, env, env_config, model, seed, **kwargs):
        kwargs["test_num"] = args.episodes
        state = {"active": False, "seed": None}
        setup, close = env.setup_demo, env.close_env
        (output / "task_config.yml").write_text(yaml.safe_dump(json.loads(json.dumps(env_config, default=str))))

        def setup_recorded(*a, **kw):
            if len(resets) >= 100:
                raise SystemExit("Exceeded 100 reset attempts; inspect environment failures")
            state.update(active=False, seed=kw.get("seed"))
            entry = {"seed": state["seed"]}
            resets.append(entry)
            try:
                return setup(*a, **kw)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                (output / "resets.json").write_text(json.dumps(resets, indent=2) + "\n")

        def close_recorded(*a, **kw):
            if state["active"]:
                episodes.append({"episode": len(episodes), "seed": state["seed"],
                                 "success": bool(env.eval_success), "action_steps": env.take_action_cnt,
                                 "step_limit": env.step_lim, "instruction": env.get_instruction(),
                                 "seconds": time.monotonic() - state["start"]})
                (output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
                state["active"] = False
            return close(*a, **kw)

        env.setup_demo, env.close_env = setup_recorded, close_recorded
        original_decorator = module.eval_function_decorator

        def decorated(policy, name):
            fn = original_decorator(policy, name)
            if name != "eval":
                return fn

            def recorded_policy(*a, **kw):
                if not state["active"]:
                    state.update(active=True, start=time.monotonic())
                return fn(*a, **kw)
            return recorded_policy

        module.eval_function_decorator = decorated
        try:
            return original_eval(task_name, env, env_config, model, seed, **kwargs)
        finally:
            module.eval_function_decorator = original_decorator

    module.eval_policy = instrumented_eval
    module.main(config)
    if len(episodes) != args.episodes or any(e["action_steps"] <= 0 for e in episodes):
        raise RuntimeError("Not all requested episodes completed action execution")
    (output / "smoke_result.json").write_text(json.dumps({
        "goal_id": "r1/robotwin-smoke", "task": args.task, "condition": args.condition,
        "label": args.label, "episodes": episodes, "pass": True,
        "successes": sum(e["success"] for e in episodes), "count": len(episodes),
        "note": "Engineering smoke only. This JSON uses the actual episode count; upstream _result.txt hardcodes 100.",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--condition", choices=["demo_clean", "demo_randomized"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--episodes", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
        raise ValueError("A fallback-planner run cannot pass this smoke")
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    with (output / "console.log").open("w") as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            try:
                run(args, output)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
