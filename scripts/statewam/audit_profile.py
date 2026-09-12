"""Verify four-rank update counts, finite gradients and exact Adam-state resume."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = []
    for rank in range(4):
        summary = json.loads((args.profile / f"summary_rank{rank}.json").read_text())
        rows = [json.loads(line) for line in (args.profile / f"steps_rank{rank}.jsonl").read_text().splitlines()]
        updates = summary["optimizer_update_final"]
        assert summary["world_size"] == 4 and 200 <= updates <= 500
        assert [r["micro_step"] for r in rows] == list(range(1, len(rows) + 1))
        assert len(rows) == updates * summary["gradient_accumulation"]
        assert summary["sample_exposures"] == len(rows) * summary["batch_per_gpu"] * 4
        for row in rows:
            assert all(math.isfinite(v) for v in row["metrics"].values())
            assert len(row["state_gradient_norms"]) == 18
            assert all(math.isfinite(v) for v in row["state_gradient_norms"].values())
        for key in rows[0]["state_gradient_norms"]:
            assert any(row["state_gradient_norms"][key] > 0 for row in rows[1:]), key
        restored = json.loads((args.resume / f"resume_rank{rank}.json").read_text())
        assert restored["restored_position"][:2] == [len(rows), updates]
        assert summary["optimizer_state"] and summary["optimizer_state"] == restored["optimizer_state"]
        resumed_summary = json.loads((args.resume / f"summary_rank{rank}.json").read_text())
        assert resumed_summary["optimizer_update_final"] == updates + 1
        ranks.append({"rank": rank, "updates": updates, "resume_seconds": restored["seconds"],
                      "adam_state_hashes_equal": True, "resume_final_update": updates + 1,
                      "peak_allocated_bytes": summary["peak_allocated_bytes"],
                      "steady_end_to_end_samples_per_second": summary["steady_end_to_end_samples_per_second"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"goal_id": "r1/robotwin-smoke", "pass": True, "ranks": ranks}, indent=2) + "\n")
    print(f"Four-rank profiling/resume audit passed: {args.output}")


if __name__ == "__main__":
    main()
