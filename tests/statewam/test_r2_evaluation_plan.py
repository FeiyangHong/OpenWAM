import copy
from collections import Counter
import json

import pytest

from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256, stable_hash
from scripts.statewam.r2.evaluation_plan import build_schedule, scheduled_episodes, verify_export, GROUPS, CONDITIONS
from scripts.statewam.r2.formal_training import TRAINING
from benchmarks.robotwin.r2_policy_protocol import chunk_seeds


def scene_fixture():
    rows = {split: [{"task": task, "condition": condition, "seed": base + 10000 * index + seed,
                    "scene_id": f"{split}/{task}/{condition}/{base + 10000 * index + seed}",
                    "expert_eligible": True, "instruction": {"text_sha256": "c" * 64}, "policy_seed": seed}
                   for index, task in enumerate(TASKS) for condition in CONDITIONS for seed in range(count)]
            for split, base, count in (("development", 200000, 5), ("final", 400000, 20))}
    return {"rows": rows, "rows_sha256": stable_hash(rows)}


def schedule_fixture():
    return build_schedule(scene_fixture(), {group: {"weight_sha256": str(index) * 64}
                                            for index, group in enumerate(GROUPS)}, "a" * 64)


def test_finite_reproducible_matrix_and_rotating_devices():
    schedule = schedule_fixture()
    assert schedule == schedule_fixture()
    assert len(schedule["waves"]) == 90
    assert schedule["counts"] == {"repeatability": 40, "development": 400, "final": 1600}
    for split in ("development", "final"):
        blocks = [block for wave in schedule["waves"] for block in wave["blocks"] if block["split"] == split]
        assert Counter((block["group"], block["lane"]) for block in blocks) == {
            (group, lane): 10 for group in GROUPS for lane in (0, 1)}
    assert len({row["episode_key"] for row in scheduled_episodes(schedule)}) == 2040
    assert all(len(wave["blocks"]) == 2 for wave in schedule["waves"])


def test_group_replicate_and_attempt_do_not_change_paired_noise():
    episodes = scheduled_episodes(schedule_fixture())
    repeated = episodes[:2]
    assert repeated[0]["episode_key"] != repeated[1]["episode_key"]
    assert chunk_seeds(repeated[0]["identity"], 0) == chunk_seeds(repeated[1]["identity"], 0)
    first = next(row for row in episodes if row["identity"]["split"] == "final")
    paired = [row for row in episodes if row["identity"]["scene_id"] == first["identity"]["scene_id"]]
    assert len(paired) == 4
    assert len({stable_hash(chunk_seeds(row["identity"], 5)) for row in paired}) == 1


def test_reject_incomplete_scene_or_checkpoint_matrix():
    scenes = scene_fixture()
    checkpoints = {group: {"weight_sha256": "b" * 64} for group in GROUPS}
    with pytest.raises(ValueError, match="four final"):
        build_schedule(scenes, {"E0": checkpoints["E0"]}, "a" * 64)
    scenes["rows"]["final"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        build_schedule(scenes, checkpoints, "a" * 64)


def test_preselected_first_development_scene_only_for_diagnostics():
    rows = scheduled_episodes(schedule_fixture())
    repeat = [row for row in rows if row["identity"]["split"] == "repeatability"]
    assert all(row["identity"]["group"] == "E0" and row["scene"]["seed"] % 10000 == 0 for row in repeat)
    altered = copy.deepcopy(scene_fixture())
    altered["rows"]["development"].reverse()
    checkpoint = {group: {"weight_sha256": str(index) * 64} for index, group in enumerate(GROUPS)}
    assert build_schedule(altered, checkpoint, "a" * 64) == schedule_fixture()


def export_fixture(root):
    root.mkdir()
    protocol = root.parent / "protocol.yaml"
    protocol.write_text("fixture protocol")
    plan = {"training": copy.deepcopy(TRAINING), "protocol_id": "r2-pilot-v1", "matrix": {"sha256": "matrix"}}
    recipe = plan["training"]
    files = (f"checkpoint_step_{recipe['updates']}.safetensors", "config.yaml", "normalization_stats.npy")
    for name in files:
        (root / name).write_bytes(b"test-only export payload")
    identity = {key: recipe[key] for key in ("updates", "seed", "micro_batch", "accumulation", "world_size",
                                            "global_batch", "workers", "zero_stage")}
    identity.update(group="E0", purpose="training", protocol_id=plan["protocol_id"], matrix_sha256="matrix",
                    statistics_sha256=sha256(root / "normalization_stats.npy"),
                    plan_identity={"plan": plan, "protocol_path": str(protocol), "protocol_sha256": sha256(protocol)})
    report = {"goal_id": GOAL_ID, "passed": True, "all_tensor_values_exact": True, "identity": identity,
              "position": {"optimizer_update": recipe["updates"], "next_occurrence": recipe["exposures"]},
              "weight_name": files[0], "files": {name: sha256(root / name) for name in files}}
    atomic_json(root / "export.json", report)
    atomic_json(root / "COMPLETE.json", {"goal_id": GOAL_ID, "export_sha256": sha256(root / "export.json")})
    return plan, report


@pytest.mark.parametrize("case", [None, "engineering", "profiling", "group", "plan", "batch", "workers",
                                 "position", "missing", "weights", "statistics", "marker", "protocol", "outside"])
def test_formal_export_requires_exact_final_training_identity_and_files(tmp_path, case):
    directory = tmp_path / "export"
    plan, report = export_fixture(directory)
    if case is None:
        assert verify_export(directory, plan, "E0")["weight_sha256"] == report["files"][report["weight_name"]]
        return
    if case in ("engineering", "profiling"):
        report["identity"]["purpose"] = case
    elif case == "group":
        report["identity"]["group"] = "E3"
    elif case == "plan":
        report["identity"]["plan_identity"]["plan"] = {"other": True}
    elif case == "batch":
        report["identity"]["global_batch"] = 16
    elif case == "workers":
        report["identity"]["workers"] = 1
    elif case == "position":
        report["position"]["optimizer_update"] -= 1
    elif case == "missing":
        del report["files"]["config.yaml"]
    elif case == "weights":
        (directory / report["weight_name"]).write_bytes(b"changed")
    elif case == "statistics":
        report["identity"]["statistics_sha256"] = "different"
    elif case == "protocol":
        (tmp_path / "protocol.yaml").write_text("changed")
    elif case == "outside":
        outside = tmp_path / "outside"
        outside.write_text("not inside export")
        report["files"]["../outside"] = sha256(outside)
    atomic_json(directory / "export.json", report)
    atomic_json(directory / "COMPLETE.json", {"goal_id": GOAL_ID,
        "export_sha256": "changed" if case == "marker" else sha256(directory / "export.json")})
    with pytest.raises(ValueError):
        verify_export(directory, plan, "E0")
