"""Four fixed, restartable scene-preparation lanes; no policy evaluation."""

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin import r2_scene_manifest as shards
from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256


def verify_provenance(asset_report, environment_report):
    assets = json.loads(asset_report.read_text())
    environment = json.loads(environment_report.read_text())
    if any(record.get("goal_id") != GOAL_ID or record.get("passed") is not True for record in (assets, environment)):
        raise ValueError("Scene queues require passed R2 asset and environment audits")
    if Path(environment["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Scene queue uses another Python environment")
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}
    if packages != environment["packages"]:
        raise ValueError("Simulator package versions changed since audit")
    if any(sha256(row["path"]) != row["sha256"] for row in environment["dependency_patches"]):
        raise ValueError("Simulator dependency patches changed")
    from benchmarks.robotwin.r2_scene_provenance import ASSET_NAMES, extracted_inventory
    if set(assets["groups"]) != set(ASSET_NAMES):
        raise ValueError("Incomplete scene asset audit")
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    if str(root) != assets["robotwin_root"]:
        raise ValueError("Generated asset paths belong to another RoboTwin checkout")
    for name in ASSET_NAMES:
        print(f"Verifying scene asset binding: {name}", flush=True)
        if ((root / "assets" / name).resolve() != Path(assets["groups"][name]["extracted_directory"])
                or extracted_inventory(root / "assets", name) != assets["groups"][name]["files"]):
            raise ValueError("Expanded scene assets changed since audit")
    return {"asset_report": str(asset_report.resolve()), "asset_report_sha256": sha256(asset_report),
            "environment_report": str(environment_report.resolve()), "environment_report_sha256": sha256(environment_report)}


def lane_entries(lane):
    if type(lane) is not int or not 0 <= lane < 4:
        raise ValueError("Scene preparation has exactly four lanes numbered 0..3")
    entries = [{"split": split, "task": task, "condition": condition}
               for split in ("development", "final") for task in TASKS for condition in shards.CONDITIONS]
    return entries[lane::4]


def verify_shard(directory, entry):
    marker = json.loads((directory / "COMPLETE.json").read_text())
    if marker != {"manifest_sha256": sha256(directory / "manifest.json")}:
        raise ValueError("Scene shard manifest hash differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    if (manifest["identity"] != identity or identity["goal_id"] != GOAL_ID
            or any(identity[key] != value for key, value in entry.items())
            or identity["first_seed"] != shards.seed_start(entry["task"], entry["split"])
            or identity["target"] != shards.COUNTS[entry["split"]]
            or identity["max_candidates"] != 1000
            or identity["max_infrastructure_attempts_per_seed"] != 2):
        raise ValueError("Scene shard identity differs from fixed queue")
    if any(sha256(ROOT / name) != digest for name, digest in identity["sources"].items()):
        raise ValueError("Scene shard source drift")
    accepted, rejected = manifest["accepted"], manifest["rejected"]
    if manifest["count"] != identity["target"] or len(accepted) != identity["target"]:
        raise ValueError("Scene shard accepted count is incomplete")
    rows = sorted(accepted + rejected, key=lambda row: row["seed"])
    first = identity["first_seed"]
    if (not rows or len(rows) > identity["max_candidates"]
            or [row["seed"] for row in rows] != list(range(first, first + len(rows)))
            or rows[-1] not in accepted):
        raise ValueError("Scene candidate ledger has a gap, duplicate or excess candidate")
    for row in rows:
        candidate = directory / "candidates" / str(row["seed"])
        attempts = sorted(candidate.glob("attempt_*"))
        complete = [path for path in attempts if shards.completed_attempt(path)]
        if not 1 <= len(attempts) <= 2 or len(complete) != 1:
            raise ValueError("Invalid candidate attempt count or completion")
        probe = complete[0] / "probe"
        if Path(row["probe"]).resolve() != probe.resolve():
            raise ValueError("Scene evidence is outside its candidate attempt")
        verified = shards.inspect_probe(probe, task=entry["task"], condition=entry["condition"], seed=row["seed"])
        if verified["expert_eligible"]:
            verified["scene_id"] = f"{entry['split']}/{entry['task']}/{entry['condition']}/{row['seed']}"
        if row != verified or (row in accepted) != verified["expert_eligible"]:
            raise ValueError("Scene manifest differs from independently verified evidence")
    return {"entry": entry, "path": str(directory.resolve()), "count": len(accepted),
            "rejected_count": len(rejected), "manifest_sha256": marker["manifest_sha256"]}


def run_lane(output, *, lane, stop_file, resume=False, provenance=None):
    entries = lane_entries(lane)
    identity = {"goal_id": GOAL_ID, "protocol_id": "r2-scene-queue-v2", "lane": lane,
                "lanes": 4, "entries": entries, "automatic_infrastructure_retries": 0,
                "provenance": provenance,
                "sources": {str(path.relative_to(ROOT)): sha256(path) for path in (
                    Path(__file__).resolve(), ROOT / "benchmarks/robotwin/r2_scene_manifest.py",
                    ROOT / "benchmarks/robotwin/r2_scene_probe.py",
                    ROOT / "benchmarks/robotwin/r2_scene_provenance.py",
                    ROOT / "benchmarks/robotwin/r2_instructions.py",
                    ROOT / "benchmarks/robotwin/eval_policy_wrapper.py")}}
    output = output.resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "identity.json", identity)
    elif json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("Scene queue identity changed on resume")
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "COMPLETE.json").exists():
            raise ValueError("Scene queue already complete; do not rerun")
        status = {"identity": identity, "state": "running", "completed_shards": [],
                  "run_directory": os.environ.get("R2_RUN_DIR"), "updated_at": time.time()}
        try:
            for index, entry in enumerate(entries):
                if stop_file.exists():
                    status["state"] = "stopped"
                    break
                if any(sha256(ROOT / name) != digest for name, digest in identity["sources"].items()):
                    raise ValueError("Scene queue source drift")
                directory = output / "shards" / entry["split"] / entry["task"] / entry["condition"]
                status.update(current_entry=entry, updated_at=time.time())
                atomic_json(output / "progress.json", status)
                if not (directory / "COMPLETE.json").exists():
                    result = shards.run_shard(directory, **entry, stop_file=stop_file, resume=directory.exists())
                    if result == "stopped":
                        status["state"] = "stopped"
                        break
                    if result != "completed":
                        raise RuntimeError(f"Scene shard did not complete: {result}")
                status["completed_shards"].append(verify_shard(directory, entry))
                status.update(next_entry=index + 1, updated_at=time.time())
                atomic_json(output / "progress.json", status)
            else:
                if provenance is not None:
                    verified = verify_provenance(Path(provenance["asset_report"]), Path(provenance["environment_report"]))
                    if verified != provenance:
                        raise ValueError("Scene queue provenance changed during execution")
                status["state"] = "completed"
                atomic_json(output / "manifest.json", {
                    "identity": identity, "shards": status["completed_shards"],
                    "reset_tolerance_frozen": False,
                    "limits": "Preparation evidence only; not a frozen evaluation protocol or policy result."})
                atomic_json(output / "COMPLETE.json", {"manifest_sha256": sha256(output / "manifest.json")})
            return status["state"]
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            status["updated_at"] = time.time()
            atomic_json(output / "progress.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=range(4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--asset-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    args = parser.parse_args()
    from scripts.statewam.r2.cuda_identity import verify_cuda_devices
    if len(verify_cuda_devices()) != 1:
        raise ValueError("Each scene queue requires exactly one leased GPU")
    provenance = verify_provenance(args.asset_report, args.environment_report)
    result = run_lane(args.output, lane=args.lane, stop_file=Path(os.environ["R2_STOP_FILE"]), resume=args.resume,
                      provenance=provenance)
    return 0 if result in ("completed", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
