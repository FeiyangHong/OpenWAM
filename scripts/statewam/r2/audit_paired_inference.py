"""Independently verify saved R2 real-model WS identities, chunks and resets."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_policy_protocol import reset_identity, validate_action_response, validate_reset_ack
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash


def audit(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (plan["goal_id"] != GOAL_ID or plan["groups"] != ["E0", "E3"] or plan["actions_per_repeat"] != [33, 32]
            or plan["generated_chunks_per_group"] != 3 or set(report["groups"]) != {"E0", "E3"}
            or report["plan_sha256"] != sha256(directory / "plan.json") or report["passed"] is not True):
        raise ValueError("Incomplete or changed paired inference plan/result")
    sources = json.loads((directory / "source_snapshot/manifest.json").read_text())
    if any(sha256(directory / "source_snapshot" / name) != digest for name, digest in sources.items()):
        raise ValueError("Saved source snapshot changed")
    arrays, chunks = {}, {}
    for group in plan["groups"]:
        root = directory / group
        row = report["groups"][group]
        provenance = json.loads((root / "provenance.json").read_text())
        responses = json.loads((root / "responses.json").read_text())
        if (provenance != row["provenance"] or sha256(provenance["checkpoint"]) != provenance["checkpoint_sha256"]
                or sha256(root / "responses.json") != row["responses_sha256"] or sha256(root / "actions.npz") != row["actions_sha256"]):
            raise ValueError("Inference evidence/checkpoint identity changed")
        with np.load(root / "actions.npz", allow_pickle=False) as values:
            if set(values.files) != {"first", "repeated"}:
                raise ValueError("Missing action arrays")
            arrays[group] = [values[name].copy() for name in ("first", "repeated")]
        if [list(value.shape) for value in arrays[group]] != [[33, 20], [32, 20]] or not all(np.isfinite(value).all() for value in arrays[group]):
            raise ValueError("Incomplete/nonfinite action arrays")
        if not np.array_equal(arrays[group][0][:32], arrays[group][1]):
            raise ValueError("Reset did not reproduce the same first chunk")
        if np.array_equal(arrays[group][0][0], arrays[group][0][32]):
            raise ValueError("Next chunk did not change actions")
        chunks[group] = []
        for replicate, length in enumerate(plan["actions_per_repeat"]):
            ack = json.loads((root / f"reset_{replicate}.json").read_text())
            expected = reset_identity(ack["identity"], attempt=1)
            validate_reset_ack(ack, expected)
            identity = expected["identity"]
            if (identity["group"] != group or identity["replicate"] != replicate or identity["split"] != "repeatability"
                    or identity["checkpoint_sha256"] != provenance["checkpoint_sha256"]
                    or identity["protocol_sha256"] != report["plan_sha256"] or len(responses[replicate]) != length):
                raise ValueError("Reset/model/repeat identity mismatch")
            for index, response in enumerate(responses[replicate]):
                validate_action_response(response, expected, index)
                if not np.array_equal(np.asarray(response["action"]), arrays[group][replicate][index]):
                    raise ValueError("Response differs from saved raw actions")
                generation = response["generation"]
                if index % 32 == 0:
                    if generation["chunk_index"] != index // 32 or generation["seeds"] != response["seeds"] or generation["seconds"] <= 0:
                        raise ValueError("Generation identity/timing mismatch")
                    if index == 0 and generation["actions_sha256"] != stable_hash(arrays[group][replicate][:32].tolist()):
                        raise ValueError("Complete chunk hash differs")
                    chunks[group].append(generation)
                elif generation is not None:
                    raise ValueError("Buffered action improperly regenerated")
        if chunks[group] != row["generated_chunks"]:
            raise ValueError("Summary differs from raw chunk evidence")
        requests = [json.loads(line)["result"] for line in (root / "service/requests.jsonl").read_text().splitlines()]
        expected_requests = []
        for replicate in range(2):
            expected_requests.append(json.loads((root / f"reset_{replicate}.json").read_text()))
            expected_requests.extend(responses[replicate])
        if requests != expected_requests:
            raise ValueError("Server ledger has missing, duplicate or different responses")
        state = json.loads((root / "service/state.json").read_text())
        if state["state"] != "stopped" or state["messages"] != 67:
            raise ValueError("Service did not terminate at the expected bounded count")
    if not all(np.array_equal(a, b) for a, b in zip(arrays["E0"], arrays["E3"])):
        raise ValueError("Step-zero E0 and zero-gate E3 actions differ")
    for left, right in zip(chunks["E0"], chunks["E3"]):
        if left["seeds"] != right["seeds"] or left["actions_sha256"] != right["actions_sha256"]:
            raise ValueError("Groups do not share chunk noise/actions")
    return {"goal_id": GOAL_ID, "report_sha256": sha256(directory / "report.json"),
            "plan_sha256": report["plan_sha256"], "passed": True, "group_reset_exact": True,
            "zero_gate_group_actions_exact": True, "messages_per_group": 67,
            "limits": "Two step-zero models, saved non-final scene, real WS; no physical rollout or trained policy efficacy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite an audit")
    atomic_json(args.output, audit(args.input))


if __name__ == "__main__":
    main()
