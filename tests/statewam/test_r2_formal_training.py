import copy
import json

import pytest
import yaml

from scripts.statewam.r2 import formal_training as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.evaluation_plan import EXECUTION


def recipe():
    return {"goal_id": GOAL_ID, "protocol_id": module.PROTOCOL, "status": "frozen",
            "groups": list(module.GROUPS), "tasks": list(module.TASKS),
            "training": copy.deepcopy(module.TRAINING), "inference": copy.deepcopy(module.INFERENCE),
            "evaluation": copy.deepcopy(module.EVALUATION), "evaluation_execution": copy.deepcopy(EXECUTION)}


def test_exact_finite_four_group_recipe():
    assert module.validate_recipe(recipe())["training"]["exposures"] == 8048 * 64


@pytest.mark.parametrize("section,key,value", [
    ("training", "updates", 200), ("training", "seed", 0), ("training", "micro_batch", 8),
    ("training", "global_batch", 128), ("training", "world_size", 5),
    ("training", "checkpoint_selection", "best"), ("training", "world_freeze", "no_grad"),
    ("training", "learning_rate", 1e-3), ("inference", "velocity_cache", True),
    ("inference", "denoise_steps", 5), ("inference", "planner_fallback", True),
    ("evaluation", "final_episodes", 40), ("evaluation", "max_invalid_retries", 2),
    ("evaluation", "development_nodes", [500, 8048]),
])
def test_recipe_rejects_budget_and_fairness_drift(section, key, value):
    plan = recipe()
    plan[section][key] = value
    with pytest.raises(ValueError):
        module.validate_recipe(plan)


def test_formal_source_inventory_includes_evaluation(tmp_path):
    names = ["openwam/model.py", "scripts/train.py", "configs/train.yaml",
             "benchmarks/robotwin/policy.py", "benchmarks/utils/nested/helper.py"]
    for name in names + ["docs/notes.md"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    assert {str(path.relative_to(tmp_path)) for path in module.executable_paths(tmp_path)} == set(names)


def test_evidence_rejects_changed_bytes_and_other_goal(tmp_path):
    path = tmp_path / "evidence.json"
    atomic_json(path, {"goal_id": GOAL_ID, "passed": True})
    ref = {"path": str(path), "sha256": sha256(path)}
    assert module.read_evidence(ref)["passed"]
    atomic_json(path, {"goal_id": "other", "passed": True})
    with pytest.raises(ValueError, match="changed"):
        module.read_evidence(ref)
    ref["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="another Goal"):
        module.read_evidence(ref)


def test_freeze_marker_and_complete_gate_set_required(tmp_path):
    plan = recipe()
    plan["evidence"] = {}
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(FileNotFoundError):
        module.load_protocol(path)
    atomic_json(tmp_path / "protocol.COMPLETE.json", {"goal_id": GOAL_ID, "protocol_sha256": sha256(path)})
    with pytest.raises(ValueError, match="acceptance gate"):
        module.load_protocol(path)


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    names = ("episode_manifest.json", "normalization_stats.npy")
    for name in names:
        (root / name).write_text("test identity only\n")
    hashes = {name: sha256(root / name) for name in names}
    identity = {"matrix_sha256": "matrix", "episode_manifest_sha256": hashes[names[0]],
                "statistics_sha256": hashes[names[1]]}
    rows = {split: [{"task": task, "condition": condition, "seed": base + 10000 * index + seed,
                     "scene_id": f"{split}/{task}/{condition}/{base + 10000 * index + seed}",
                     "expert_eligible": True}
                    for index, task in enumerate(module.TASKS)
                    for condition in ("demo_clean", "demo_randomized") for seed in range(count)]
            for split, base, count in (("development", 200000, 5), ("final", 400000, 20))}
    data = {
        "p0": {"stage": "P0", "status": "passed", "data_root": str(root), "data_files": hashes},
        "capacity": {"workload_audit_passed": True, "state_effective_parameter_counts_equal": True,
                     "groups": {group: {"identity": dict(identity)} for group in module.GROUPS}},
        "profiling": {"paired_profiling_passed": True, "completed_groups": ["E0", "E3"],
                      "groups": {group: {"identity": dict(identity)} for group in ("E0", "E3")}},
        "resume": {"passed": True, "identity": {**identity, "group": "E3", "world_size": 4,
                    "micro_batch": 16, "accumulation": 1, "global_batch": 64},
                   "saved_states": [{"contents_exact": True}]},
        "training_faults": {"passed": True, "final_checkpoint_bytes_identical_to_reference": True},
        "physical_recovery": {"passed": True, "environment_sha256": "env",
                              "checks": dict.fromkeys(module.PHYSICAL_CHECKS, True)},
        "pilot_lifecycle": {"passed": True, "service_reuse_and_fresh_simulators": True, "completed_waves": 2,
                            "records": [{"audited": True, "attempt": 1} for _ in range(8)],
                            "matrix_sha256": "matrix", "environment_sha256": "env",
                            "assets_sha256": "assets", "source_sha256": {}},
        "scenes": {"preparation_complete": True, "rows_sha256": "rows", "rows": rows},
        "cross_process_reset": {"passed": True, "scene_rows_sha256": "rows", "environment_sha256": "env",
                                "checks": [{"task": task, "condition": condition, "initial_exact": True}
                                           for task in module.TASKS for condition in ("demo_clean", "demo_randomized")]}}
    plan = {**recipe(), "matrix": {"sha256": "matrix"}, "environment": {"sha256": "env"},
            "assets": {"sha256": "assets"}, "source_sha256": {}}
    return plan, data


def test_all_acceptance_gates_required(evidence):
    plan, data = evidence
    module.validate_evidence(plan, data)
    for name in data:
        partial = {key: value for key, value in data.items() if key != name}
        with pytest.raises(ValueError, match="acceptance gate"):
            module.validate_evidence(plan, partial)


@pytest.mark.parametrize("gate,key,value", [
    ("resume", "passed", False), ("resume", "saved_states", []),
    ("physical_recovery", "checks", {}), ("physical_recovery", "environment_sha256", "other"),
    ("cross_process_reset", "checks", []), ("cross_process_reset", "scene_rows_sha256", "other"),
    ("scenes", "preparation_complete", False),
    ("pilot_lifecycle", "passed", False), ("pilot_lifecycle", "records", []),
    ("pilot_lifecycle", "source_sha256", {"changed": "hash"}),
    ("training_faults", "final_checkpoint_bytes_identical_to_reference", False),
])
def test_incomplete_or_mismatched_evidence_rejected(evidence, gate, key, value):
    plan, data = evidence
    data[gate][key] = value
    with pytest.raises(ValueError):
        module.validate_evidence(plan, data)


def test_different_data_and_changed_p0_files_rejected(evidence):
    plan, data = evidence
    data["capacity"]["groups"]["E1"]["identity"]["statistics_sha256"] = "different"
    with pytest.raises(ValueError, match="different P0"):
        module.validate_evidence(plan, data)
    root = module.Path(data["p0"]["data_root"])
    (root / "episode_manifest.json").write_text("changed\n")
    with pytest.raises(ValueError, match="P0 data evidence changed"):
        module.validate_evidence(plan, data)


def test_timing_summary_does_not_retain_gradient_or_sample_evidence():
    from scripts.statewam.r2.profile_metrics import timing_sample, summarize_steps

    row = {"microstep_wall_seconds": 1.0, "phase_timings": {}, "wait_seconds": 0.1,
           "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
           "gradients": {"huge": [1, 1]}, "samples": [{"occurrence": 1}], "noise_hashes": ["hash"]}
    compact = timing_sample(row)
    assert not {"gradients", "samples", "noise_hashes"} & compact.keys()
    assert summarize_steps([row] * 12, accumulation=1, micro_batch=16, world_size=4) == summarize_steps(
        [compact] * 12, accumulation=1, micro_batch=16, world_size=4)
