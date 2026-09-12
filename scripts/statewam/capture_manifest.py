"""Record code, environment and optional dataset provenance without credentials."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--archives", type=Path, nargs="*", default=[])
    parser.add_argument("--robotwin", type=Path)
    args = parser.parse_args()
    sources = []
    for directory in ("scripts/statewam", "openwam/model/architectures/statewam", "tests/statewam"):
        sources.extend((ROOT / directory).glob("*.py"))
    sources.extend((ROOT / "docs/statewam").rglob("*.md"))
    for filename in subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines():
        sources.append(ROOT / filename)
    for filename in ("configs/model/statewam.yaml", "configs/deploy.yaml", "benchmarks/robotwin/policy_config.yml",
                     "benchmarks/robotwin/eval_policy_wrapper.py", "benchmarks/robotwin/openwam2robotwin_interface.py"):
        sources.append(ROOT / filename)
    packages = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()})
    record = {
        "goal_id": "r1/robotwin-smoke", "python": sys.executable, "python_version": platform.python_version(),
        "packages": dict(packages), "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "code_files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(sources)) if p.is_file()},
    }
    record["archives"] = {str(p.resolve()): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in args.archives}
    if args.robotwin:
        checkout = args.robotwin.resolve()
        files = [checkout / p for p in (
            "script/eval_policy.py", "envs/_base_task.py", "envs/adjust_bottle.py", "envs/robot/planner.py",
            "task_config/demo_clean.yml", "task_config/demo_randomized.yml",
            "assets/embodiments/aloha-agilex/curobo_left.yml", "assets/embodiments/aloha-agilex/curobo_right.yml",
        )]
        record["robotwin"] = {
            "path": str(checkout), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout, text=True),
            "files": {str(p.relative_to(checkout)): sha256(p) for p in files},
        }
    if args.dataset:
        record["dataset"] = {
            "root": str(args.dataset.resolve()), "split": "train", "task": "adjust_bottle",
            "hf_repo": "TianxingChen/RoboTwin2.0", "hf_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
            "variant": "aloha-agilex_clean_50", "normalization": "Published Clean2Random normalization_stats.npy reused, not refitted",
            "note": "Training subset for engineering profiling, not held-out efficacy evaluation",
            "files": {str(p.relative_to(args.dataset)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in sorted(args.dataset.rglob("*")) if p.is_file()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
