"""Collect the P0 acceptance snapshot without claiming later stages complete."""

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, ROOT, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    for path in sorted((ARTIFACTS / "runs").glob("*/job.json")):
        state = json.loads((path.parent / "state.json").read_text())
        performance_path = path.parent / "performance_summary.json"
        performance = json.loads(performance_path.read_text()) if performance_path.exists() else None
        jobs.append({"run_id": path.parent.name, "job_sha256": sha256(path),
                     "state": state, "supervisor_live_now": alive(state.get("supervisor")),
                     "child_live_now": alive(state.get("child")), "performance": performance})
    by_id = {row["run_id"]: row for row in jobs}
    required = ("p0-data-001", "p0-source-audit-002", "p0-reader-audit-002", "p1-regression-001")
    for run_id in required:
        job = by_id[run_id]
        if job["state"]["state"] != "completed" or job["child_live_now"]:
            raise ValueError(f"Required run not completed: {run_id}")
    sources = json.loads((ARTIFACTS / "runs/p0-source-audit-002/source_audit.json").read_text())
    reader = json.loads((ARTIFACTS / "runs/p0-reader-audit-002/reader_audit.json").read_text())
    summary = json.loads((args.data / "data_summary.json").read_text())
    if sources["status"] != "source_identity_passed" or reader["status"] != "passed":
        raise ValueError("Source or reader audit not passed")
    for key, name in (("episode_manifest_sha256", "episode_manifest.json"), ("statistics_sha256", "normalization_stats.npy")):
        if summary[key] != reader[key] or summary[key] != sha256(args.data / name):
            raise ValueError("Reader and prepared data identities differ")
    for split in ("train", "val"):
        for key in ("episodes", "windows"):
            if sum(row[split][key] for row in summary["counts"].values()) != reader["splits"][split][key]:
                raise ValueError("Data and reader counts differ")
    suites = ET.parse(ARTIFACTS / "runs/p1-regression-001/results.xml").getroot().findall("testsuite")
    tests = {name: sum(int(suite.get(name, 0)) for suite in suites) for name in ("tests", "failures", "errors", "skipped")}
    if tests["failures"] or tests["errors"] or not tests["tests"]:
        raise ValueError("Regression failures")
    data_files = [p for p in args.data.rglob("*") if p.is_file()]
    disk = {"logical_bytes_local_files": sum(p.stat().st_size for p in data_files if not p.is_symlink()),
            "logical_bytes_referenced_symlink_files": sum(p.stat().st_size for p in data_files if p.is_symlink()),
            "note": "Logical bytes, not unique physical storage blocks; reused symlink targets excluded from local total"}
    code = {}
    for base in ("scripts/statewam/r2", "openwam/model/architectures/statewam", "tests/statewam"):
        code.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / base).glob("*.py"))})
    for name in ("openwam/dataloader/robotwin.py", "openwam/dataloader/utils/episode_manifest.py"):
        code[name] = sha256(ROOT / name)
    now = time.time()
    atomic_json(ARTIFACTS / "manifests/jobs.json", {"goal_id": GOAL_ID, "observed_at": now, "jobs": jobs})
    record = {"goal_id": GOAL_ID, "protocol_id": "r2-data-v1", "stage": "P0", "status": "passed",
              "observed_at": now, "data_root": str(args.data.resolve()), "counts": summary["counts"],
              "required_runs": list(required), "regression": tests, "data_storage": disk,
              "regression_limit": "One CUDA-only R1 test deferred to GPU engineering checks; no full R1 pass claimed",
              "source_scope": sources["asset_scope"], "code_files": code,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "later_stages": "P1 incomplete; P2 GPU validation, profiling, P3 training and P4 evaluation not started",
              "data_files": {name: sha256(args.data / name) for name in (
                  "source_manifest.json", "episode_manifest.json", "normalization_stats.npy",
                  "statistics_provenance.json", "data_summary.json")}}
    atomic_json(ARTIFACTS / "reports/p0_acceptance.json", record)
    print(json.dumps({"status": record["status"], "regression": tests, "storage": disk}, indent=2))


if __name__ == "__main__":
    main()
