"""Collect and verify the complete fixed R2 scene matrix without a policy."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.robotwin.r2_scene_pool import verify_lane
from benchmarks.robotwin.r2_scene_queue import lane_entries
from scripts.statewam.r2.common import GOAL_ID, ROOT, TASKS, atomic_json, sha256, stable_hash

PROTOCOL = "r2-evaluation-scenes-v1"


def validate_rows(rows):
    if set(rows) != {"development", "final"}:
        raise ValueError("Require separate development and final scene lists")
    ids, seeds_by_split = set(), {}
    for split, count in (("development", 5), ("final", 20)):
        expected = {(task, condition): count for task in TASKS for condition in ("demo_clean", "demo_randomized")}
        observed = dict.fromkeys(expected, 0)
        seeds_by_split[split] = set()
        if len(rows[split]) != 20 * count:
            raise ValueError("Incomplete fixed scene matrix")
        for row in rows[split]:
            key = row["task"], row["condition"]
            seed = row["seed"]
            scene_id = f"{split}/{row['task']}/{row['condition']}/{seed}"
            start = (200000 if split == "development" else 400000) + TASKS.index(row["task"]) * 10000
            if (key not in observed or row["scene_id"] != scene_id or row["scene_id"] in ids
                    or type(seed) is not int or not start <= seed < start + 1000
                    or row["expert_eligible"] is not True):
                raise ValueError("Duplicate, out-of-range or unqualified scene")
            ids.add(scene_id)
            observed[key] += 1
            seeds_by_split[split].add((key, seed))
        if observed != expected:
            raise ValueError("Per-task/condition scene count differs")
    if seeds_by_split["development"] & seeds_by_split["final"]:
        raise ValueError("Development and final seed overlap")


def collect(pool):
    pool = Path(pool).resolve()
    marker = json.loads((pool / "COMPLETE.json").read_text())
    if marker != {"manifest_sha256": sha256(pool / "manifest.json")}:
        raise ValueError("Scene pool has no intact completion marker")
    manifest = json.loads((pool / "manifest.json").read_text())
    identity = json.loads((pool / "identity.json").read_text())
    if manifest["identity"] != identity or identity["goal_id"] != GOAL_ID or identity["lanes"] != 4:
        raise ValueError("Pool identity differs")
    if any(sha256(ROOT / name) != digest for name, digest in identity["sources"].items()):
        raise ValueError("Scene preparation sources differ")
    lanes = [verify_lane(pool / f"lane_{lane}", lane) for lane in range(4)]
    if manifest["lanes"] != lanes:
        raise ValueError("Pool lane evidence differs")
    rows = {"development": [], "final": []}
    shard_hashes = {}
    for lane in range(4):
        for entry in lane_entries(lane):
            path = pool / f"lane_{lane}/shards" / entry["split"] / entry["task"] / entry["condition"] / "manifest.json"
            shard = json.loads(path.read_text())
            shard_hashes[str(path)] = sha256(path)
            rows[entry["split"]].extend({"task": entry["task"], "condition": entry["condition"], **row}
                                        for row in shard["accepted"])
    for values in rows.values():
        values.sort(key=lambda row: (row["task"], row["condition"], row["seed"]))
    validate_rows(rows)
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "preparation_complete": True,
            "source_pool": str(pool), "pool_manifest_sha256": marker["manifest_sha256"],
            "shard_sha256": shard_hashes, "rows": rows, "rows_sha256": stable_hash(rows),
            "cross_process_reset_accepted": False,
            "limits": "Complete expert/reset/instruction preparation evidence; protocol freeze still requires independent cross-process reset acceptance"}


def verify(path):
    record = json.loads(Path(path).read_text())
    if record != collect(record["source_pool"]):
        raise ValueError("Collected scenes differ from current independently verified pool evidence")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing scene collection")
    atomic_json(args.output, collect(args.pool))


if __name__ == "__main__":
    main()
