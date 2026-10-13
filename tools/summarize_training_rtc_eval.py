"""Summarize robot-free absolute-EEF evaluation results without model inference."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def stats(values):
    return dict(zip(("min", "p50", "p90", "max"), map(float, np.percentile(values, [0, 50, 90, 100])))) if values else {}


def summarize(rows):
    m = [r["metrics"]["right"] for r in rows]
    out = dict(n=len(m), reverse=sum(x["cosine"] is not None and x["cosine"] < 0 for x in m))
    out["reversed_step_mm"] = stats([x["after_mm"] for x in m if x["cosine"] is not None and x["cosine"] < 0])
    for threshold in (0.1, 1, 3):
        moving = [x for x in m if min(x["before_mm"], x["after_mm"]) > threshold]
        out[f"moving_{threshold}mm"] = dict(n=len(moving), reverse=sum(x["cosine"] < 0 for x in moving))
        forward = [x for x in moving if x["gt_cosine"] is not None and x["gt_cosine"] > 0]
        out[f"gt_forward_moving_{threshold}mm"] = dict(n=len(forward), reverse=sum(x["cosine"] < 0 for x in forward))
    for name in ("before_mm", "after_mm", "first_error_mm", "suffix_error_mm", "prefix_error_mm",
                 "state_to_first_mm", "state_to_end_mm", "seam_velocity_change_mm_per_tick",
                 "internal_step_median_mm", "internal_velocity_change_median_mm_per_tick",
                 "returned_prefix_xyz_error_mm"):
        out[name] = stats([x[name] for x in m])
    out["seam_internal_velocity_change_ratio"] = stats([
        x["seam_velocity_change_mm_per_tick"] / max(x["internal_velocity_change_median_mm_per_tick"], 0.1) for x in m])
    if rows and "projection" in rows[0]:
        out["projection"] = {key: stats([r["projection"][key] for r in rows]) for key in rows[0]["projection"]}
        out["projection"]["chunks_right_xyz_changed_gt_0_01mm"] = sum(r["projection"]["right_xyz_max_mm"] > 0.01 for r in rows)
        out["applied_reverse"] = sum(r["applied_metrics"]["right"]["cosine"] is not None and
                                     r["applied_metrics"]["right"]["cosine"] < 0 for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--previous", type=Path)
    args = ap.parse_args()
    rows = []
    for line in (args.run / "results.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # Permit progress summaries while the writer is still active.
    grouped = collections.defaultdict(list)
    request_groups = collections.defaultdict(list)
    sequence_groups = collections.defaultdict(list)
    for r in rows:
        grouped[f"{r['checkpoint']}/{r['mode']}"].append(r)
        if "request" in r:
            request_groups[f"{r['checkpoint']}/{r['mode']}/{r['request']}"].append(r)
            sequence_groups[f"{r['checkpoint']}/{r['mode']}/{r['case']}"].append(r)
    result = dict(records=len(rows), completed=(args.run / "complete.json").exists(),
                  groups={k: summarize(v) for k, v in grouped.items()},
                  request_groups={k: summarize(v) for k, v in request_groups.items()},
                  sequence_groups={k: summarize(v) for k, v in sequence_groups.items()})
    if args.previous:
        previous = [json.loads(line) for line in args.previous.read_text().splitlines()]
        original = {(r["case"], r["seed"]): r for r in previous if r["mode"] == "gt_prefix_10steps"}
        comparison = []
        for r in rows:
            if r["checkpoint"] != 9000 or r["mode"] != "clean":
                continue
            old = original[(r["case"], r["seed"])]
            delta = np.array(r["actions"]) - np.array(old["actions"])
            comparison.append(dict(case=r["case"], seed=r["seed"],
                                   actions_max_abs=float(np.abs(delta).max()),
                                   right_xyz_max_mm=float(np.linalg.norm(delta[:, 29:32], axis=1).max() * 1000)))
        result["9000_repeat"] = comparison
    (args.run / "summary.json").write_text(json.dumps(result, indent=2))
    for key, group in result["groups"].items():
        print(key, "n", group["n"], "reverse", group["reverse"], "moving3", group["moving_3mm"],
              "seam_mm", round(group["after_mm"]["p50"], 3),
              "suffix_error_mm", round(group["suffix_error_mm"]["p50"], 3),
              "prefix_error_mm", round(group["prefix_error_mm"]["p50"], 3))
    if result.get("9000_repeat"):
        print("9000 repeat maximum right XYZ difference mm", max(x["right_xyz_max_mm"] for x in result["9000_repeat"]))


if __name__ == "__main__":
    main()
