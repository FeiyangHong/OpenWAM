"""Bounded E0/E3 real-model WS diagnostic using a saved, non-final scene.

Three generated chunks per group: 33 actions then 32 after a fresh reset.
No physical rollout, training, checkpoint selection or policy success score.
"""

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_policy_client import PairedClient
from benchmarks.robotwin.r2_policy_service import build_session, serve
from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.source_snapshot import capture_sources, assert_sources_unchanged


def probe_group(session, *, directory, identity, instruction, cameras, proprio, outer_stop):
    stop = directory / "SERVICE_STOP"
    service = directory / "service"
    errors = []

    def run():
        try:
            asyncio.run(serve(session, port=0, output=service, stop_file=stop, max_messages=80,
                              max_seconds=1200, idle_seconds=300))
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client, actions, responses = None, [], []
    try:
        deadline = time.monotonic() + 30
        while not (service / "ready.json").exists() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        ready = json.loads((service / "ready.json").read_text())
        client = PairedClient(f"ws://127.0.0.1:{ready['port']}", timeout=300)
        for replicate, length in ((0, 33), (1, 32)):
            ack = client.reset({**identity, "replicate": replicate}, attempt=1, instruction=instruction)
            atomic_json(directory / f"reset_{replicate}.json", ack)
            rows, results = [], []
            for index in range(length):
                if outer_stop.exists():
                    raise InterruptedError("External supervised STOP")
                action, response = client.predict(cameras=cameras, state=proprio, instruction=instruction)
                rows.append(action)
                results.append(response)
                atomic_json(directory / "progress.json", {"replicate": replicate, "actions": index + 1, "time": time.time()})
            actions.append(np.stack(rows))
            responses.append(results)
        np.savez_compressed(directory / "actions.npz", first=actions[0], repeated=actions[1])
        atomic_json(directory / "responses.json", responses)
        difference = np.abs(actions[0][:32] - actions[1])
        generations = [response["generation"] for results in responses for response in results if response["generation"]]
        if len(generations) != 3 or generations[0]["seeds"] != generations[2]["seeds"] or generations[0]["seeds"] == generations[1]["seeds"]:
            raise ValueError("Real service chunk count/noise pairing differs")
        return {"provenance": session.provenance, "generated_chunks": generations,
                "finite": bool(all(np.isfinite(value).all() for value in actions)),
                "same_input_repeat_max_abs": float(difference.max()), "same_input_repeat_mean_abs": float(difference.mean()),
                "same_input_repeat_exact": bool(np.array_equal(actions[0][:32], actions[1])),
                "changed_chunk_differs": bool(not np.array_equal(actions[0][0], actions[0][32])),
                "actions_sha256": sha256(directory / "actions.npz"), "responses_sha256": sha256(directory / "responses.json")}
    finally:
        if client:
            client.close()
        atomic_json(stop, {"requested_at": time.time(), "reason": "bounded_probe_finished"})
        thread.join(timeout=15)
        if thread.is_alive() or errors:
            raise RuntimeError(f"Paired service failed to close: thread_alive={thread.is_alive()}, errors={errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--scene-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(verify_cuda_devices()) != 1:
        raise ValueError("One supervised GPU required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = capture_sources(output / "source_snapshot")
    for directory in (ROOT / "benchmarks/robotwin", ROOT / "benchmarks/utils"):
        for path in directory.glob("*.py"):
            name = str(path.relative_to(ROOT))
            sources[name] = sha256(path)
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if sha256(target) != sources[name]:
                raise ValueError("Source changed during snapshot")
    atomic_json(output / "source_snapshot/manifest.json", sources)
    scene = args.scene_probe.resolve()
    record = json.loads((scene / "scene_probe.json").read_text())
    instruction = json.loads((scene / "instruction.json").read_text())
    initial = json.loads((scene / "reset_0/initial.json").read_text())
    if not record["completed"] or not record["expert_eligible"] or record["seed"] >= 400000:
        raise ValueError("Require a completed non-final expert/reset probe")
    with np.load(scene / "reset_0/rgb.npz", allow_pickle=False) as data:
        cameras = {key: data[key].copy() for key in data.files}
    endpose = initial["metadata"]["endpose"]
    proprio = robotwin_endpose_to_eef20d(endpose["left_endpose"], endpose["right_endpose"], endpose["left_gripper"], endpose["right_gripper"])
    plan = {"goal_id": GOAL_ID, "protocol_id": "r2-paired-inference-engineering-v1", "groups": ["E0", "E3"],
            "matrix": str(args.matrix.resolve()), "scene_probe_sha256": sha256(scene / "scene_probe.json"),
            "scene_inputs": {name: sha256(scene / name) for name in ("instruction.json", "reset_0/initial.json", "reset_0/rgb.npz")},
            "actions_per_repeat": [33, 32], "generated_chunks_per_group": 3,
            "limits": "Real model and real WS with saved simulator observations, not physical rollout or policy success evaluation"}
    atomic_json(output / "plan.json", plan)
    report = {"plan_sha256": sha256(output / "plan.json"), "groups": {}, "started_at": time.time(), "passed": False}
    try:
        for group in plan["groups"]:
            if Path(os.environ["R2_STOP_FILE"]).exists():
                raise InterruptedError("External supervised STOP")
            assert_sources_unchanged(sources)
            directory = output / group
            directory.mkdir()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            session = build_session(args.matrix / group, checkpoint_name="checkpoint_step_0.safetensors", group=group,
                                    protocol_sha256=report["plan_sha256"])
            load_seconds = time.monotonic() - start
            atomic_json(directory / "provenance.json", session.provenance)
            identity = {"goal_id": GOAL_ID, **session.binding, "scene_manifest_sha256": plan["scene_probe_sha256"],
                        "instruction_sha256": instruction["text_sha256"], "task": record["task"], "condition": record["condition"],
                        "split": "repeatability", "replicate": 0, "scene_id": f"engineering/{record['task']}/{record['condition']}/{record['seed']}",
                        "policy_seed": 20260914}
            result = probe_group(session, directory=directory, identity=identity, instruction=instruction["text"],
                                 cameras=cameras, proprio=proprio, outer_stop=Path(os.environ["R2_STOP_FILE"]))
            result.update(load_seconds=load_seconds, total_seconds=time.monotonic() - start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report["groups"][group] = result
            atomic_json(output / "report.json", report)
            del session
            gc.collect()
            torch.cuda.empty_cache()
        assert_sources_unchanged(sources)
        if any(not row["finite"] or not row["changed_chunk_differs"] for row in report["groups"].values()):
            raise ValueError("Invalid or noise-insensitive model output")
        report["passed"] = True
    finally:
        report["ended_at"] = time.time()
        atomic_json(output / "report.json", report)


if __name__ == "__main__":
    main()
