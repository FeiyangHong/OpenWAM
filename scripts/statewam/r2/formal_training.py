"""Four-group R2 training from an immutable, fully gated pilot protocol.

There is no CLI override for the budget, seed, batch, geometry or optimizer.
Engineering and profiling entry points retain their independent limits.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import GOAL_ID, ROOT, TASKS, sha256

GROUPS = ["E0", "E1", "E2", "E3"]
PROTOCOL = "r2-pilot-v1"
PHYSICAL_CHECKS = {"original_limit_episode", "cooperative_stop", "forced_interrupt", "same_scene_retry",
                   "completed_failure_dedup", "cache_reset", "video_decode", "resource_release"}
INFERENCE = {"denoise_steps": 10, "mode": "sync", "precision": "bf16", "compile": False,
             "velocity_cache": False, "num_frames": 33, "video_frames": 9, "height": 384, "width": 320,
             "action_chunk": 32, "action_representation": "eef20_unified80_ee16",
             "planner_fallback": False, "task_limits": "pinned_upstream", "reset_tolerance": "exact"}
TRAINING = {"seed": 42, "world_size": 4, "micro_batch": 16, "accumulation": 1, "global_batch": 64,
            "workers": 4, "updates": 8048, "exposures": 515072, "save_every": 500,
            "checkpoint_selection": "final_update", "sampling": "paired_task_uniform_replacement",
            "precision": "bf16", "zero_stage": 2, "world_freeze": "parameters_only_input_differentiable",
            "learning_rate": 1e-4, "action_lr": 1e-5, "adam_betas": [0.9, 0.95], "adam_eps": 1e-8,
            "weight_decay": 0.01, "max_grad_norm": 1.0, "lr_scheduler": None,
            "lambda_video": 1.0, "lambda_action": 1.0, "gradient_checkpointing": True}
EVALUATION = {"development_nodes": [8048], "development_episodes": 400, "repeatability_episodes": 40,
              "final_episodes": 1600, "max_invalid_retries": 1, "bootstrap_samples": 10000,
              "bootstrap_seed": 20260912, "comparisons": ["E3-E0", "E3-E1", "E2-E1", "E3-E2"]}


def executable_paths(root=ROOT):
    from scripts.statewam.r2.source_snapshot import source_paths

    root = Path(root)
    return sorted(set(source_paths(root)) | {path for directory in ("benchmarks/robotwin", "benchmarks/utils")
                                            for path in (root / directory).rglob("*.py") if path.is_file()})


def validate_recipe(plan):
    from scripts.statewam.r2.evaluation_plan import EXECUTION

    if (plan["goal_id"] != GOAL_ID or plan["protocol_id"] != PROTOCOL or plan["status"] != "frozen"
            or plan["groups"] != GROUPS or plan["tasks"] != list(TASKS)):
        raise ValueError("Require the frozen four-group, ten-task R2 pilot protocol")
    recipe = plan["training"]
    if recipe != TRAINING:
        raise ValueError("Formal recipe differs from the selected finite common budget")
    if plan["inference"] != INFERENCE:
        raise ValueError("Inference or physical-control protocol differs")
    evaluation = plan["evaluation"]
    if evaluation != EVALUATION:
        raise ValueError("Evaluation count, checkpoint selection or analysis rules differ")
    if plan.get("evaluation_execution") != EXECUTION:
        raise ValueError("Evaluation execution order, devices or deadlines were not frozen")
    return plan


def read_evidence(reference):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Evidence requires an explicit path and SHA256")
    path = Path(reference["path"])
    if not path.is_absolute() or sha256(path) != reference["sha256"]:
        raise ValueError("Protocol evidence is missing or changed")
    value = json.loads(path.read_text())
    if value["goal_id"] != GOAL_ID:
        raise ValueError("Evidence belongs to another Goal")
    return value


def configure_formal(cfg, plan, group):
    from omegaconf import OmegaConf

    recipe = plan["training"]
    training_fields = ("learning_rate", "action_lr", "adam_betas", "adam_eps", "weight_decay", "max_grad_norm",
                       "lr_scheduler", "lambda_video", "lambda_action", "zero_stage")
    if any(OmegaConf.to_container(cfg.training, resolve=True)[key] != recipe[key] for key in training_fields):
        raise ValueError("Matrix optimizer/loss recipe differs from formal protocol")
    if (cfg.r2.group != group or cfg.r2.world_freeze != recipe["world_freeze"]
            or cfg.training.mixed_precision != recipe["precision"]
            or cfg.training.use_gradient_checkpointing is not True
            or cfg.training.use_gradient_checkpointing_offload is not False
            or cfg.training.batch_size != recipe["micro_batch"]
            or cfg.training.gradient_accumulation_steps != recipe["accumulation"]):
        raise ValueError("Matrix precision, freezing or batch differs")
    data = cfg.dataloader
    if (data.variant != "clean_50" or data.action_mode != "eef" or data.unify_action is not True
            or data.num_frames != 33 or data.video_stride != 4 or data.height != 384 or data.width != 320
            or data.multiview is not True or list(data.camera_layout) != ["head_camera", "left_camera", "right_camera"]):
        raise ValueError("Matrix training data geometry or clean-only contract differs")
    if (sha256(data.episode_manifest) != cfg.r2.episode_manifest_sha256
            or sha256(data.normalization_stats_path) != cfg.r2.statistics_sha256):
        raise ValueError("Formal data/statistics content differs from matrix")
    p0 = read_evidence(plan["evidence"]["p0"])
    root = Path(p0["data_root"])
    if (Path(data.dataset_dir).resolve() != (root / "dataset").resolve()
            or Path(data.episode_manifest).resolve() != (root / "episode_manifest.json").resolve()
            or Path(data.normalization_stats_path).resolve() != (root / "normalization_stats.npy").resolve()
            or cfg.r2.episode_manifest_sha256 != p0["data_files"]["episode_manifest.json"]
            or cfg.r2.statistics_sha256 != p0["data_files"]["normalization_stats.npy"]):
        raise ValueError("Formal data/statistics identity differs from P0 acceptance")
    cfg.training.max_steps = recipe["updates"]
    cfg.training.save_steps = recipe["save_every"]
    cfg.training.dataset_num_workers = recipe["workers"]
    cfg.r2.protocol_id = plan["protocol_id"]
    cfg.r2.budget_frozen = True
    return cfg


def validate_evidence(plan, evidence):
    if set(evidence) != {"p0", "capacity", "profiling", "resume", "training_faults", "physical_recovery", "cross_process_reset", "scenes", "pilot_lifecycle"}:
        raise ValueError("Missing or extra formal acceptance gate")
    p0, capacity, profiling = (evidence[key] for key in ("p0", "capacity", "profiling"))
    if p0["stage"] != "P0" or p0["status"] != "passed":
        raise ValueError("Data/source gate not passed")
    for name, digest in p0["data_files"].items():
        path = (Path(p0["data_root"]) / name).resolve()
        if not path.is_relative_to(Path(p0["data_root"]).resolve()) or sha256(path) != digest:
            raise ValueError("P0 data evidence changed")
    if (not capacity["workload_audit_passed"] or not capacity["state_effective_parameter_counts_equal"]
            or set(capacity["groups"]) != set(GROUPS)):
        raise ValueError("Four-group workload/capacity gate not passed")
    if not profiling["paired_profiling_passed"] or set(profiling["completed_groups"]) != {"E0", "E3"}:
        raise ValueError("Paired real profiling gate not passed")
    for group in GROUPS:
        if capacity["groups"][group]["identity"]["matrix_sha256"] != plan["matrix"]["sha256"]:
            raise ValueError("Capacity evidence has another initialization")
    identities = [capacity["groups"][group]["identity"] for group in GROUPS]
    identities += [profiling["groups"][group]["identity"] for group in ("E0", "E3")]
    identities.append(evidence["resume"]["identity"])
    for identity in identities:
        if (identity["episode_manifest_sha256"] != p0["data_files"]["episode_manifest.json"]
                or identity["statistics_sha256"] != p0["data_files"]["normalization_stats.npy"]):
            raise ValueError("Engineering evidence has different P0 data/statistics")
    for group in ("E0", "E3"):
        if profiling["groups"][group]["identity"]["matrix_sha256"] != plan["matrix"]["sha256"]:
            raise ValueError("Profiling evidence has another initialization")
    resume = evidence["resume"]
    if not resume["passed"] or resume["identity"]["group"] != "E3":
        raise ValueError("Real-model resume gate not passed")
    for key in ("world_size", "micro_batch", "accumulation", "global_batch"):
        if resume["identity"][key] != plan["training"][key]:
            raise ValueError("Resume audit does not cover selected batch configuration")
    if (resume["identity"]["matrix_sha256"] != plan["matrix"]["sha256"]
            or not resume["saved_states"] or not all(row["contents_exact"] for row in resume["saved_states"])):
        raise ValueError("Resume full-state identity or exactness missing")
    if not evidence["training_faults"]["passed"] or not evidence["training_faults"]["final_checkpoint_bytes_identical_to_reference"]:
        raise ValueError("Training write-interruption/STOP gate not passed")
    physical = evidence["physical_recovery"]
    if (not physical["passed"] or physical["checks"] != {key: True for key in PHYSICAL_CHECKS}
            or physical["environment_sha256"] != plan["environment"]["sha256"]):
        raise ValueError("Real physical episode recovery gates not passed")
    scenes, reset = evidence["scenes"], evidence["cross_process_reset"]
    if (scenes["preparation_complete"] is not True or reset["passed"] is not True
            or reset["scene_rows_sha256"] != scenes["rows_sha256"]
            or reset["environment_sha256"] != plan["environment"]["sha256"]):
        raise ValueError("Scene matrix/reset identity or completion differs")
    expected_pairs = {(task, condition) for task in TASKS for condition in ("demo_clean", "demo_randomized")}
    checked = reset["checks"]
    if (len(checked) != len(expected_pairs) or {(row["task"], row["condition"]) for row in checked} != expected_pairs
            or any(row["initial_exact"] is not True for row in checked)):
        raise ValueError("Cross-process initial-state audit lacks task/condition coverage")
    from scripts.statewam.r2.evaluation_scenes import validate_rows
    validate_rows(scenes["rows"])
    lifecycle = evidence["pilot_lifecycle"]
    if (lifecycle["passed"] is not True or lifecycle["service_reuse_and_fresh_simulators"] is not True
            or lifecycle["completed_waves"] != 2 or len(lifecycle["records"]) != 8
            or any(row["audited"] is not True or row["attempt"] != 1 for row in lifecycle["records"])
            or lifecycle["matrix_sha256"] != plan["matrix"]["sha256"]
            or lifecycle["environment_sha256"] != plan["environment"]["sha256"]
            or lifecycle["assets_sha256"] != plan["assets"]["sha256"]
            or lifecycle["source_sha256"] != plan["source_sha256"]):
        raise ValueError("Real pilot execution lifecycle differs or has not passed")


def load_protocol(path):
    path = Path(path).resolve()
    plan = validate_recipe(yaml.safe_load(path.read_text()))
    marker = json.loads(path.with_name("protocol.COMPLETE.json").read_text())
    if marker != {"goal_id": GOAL_ID, "protocol_sha256": sha256(path)}:
        raise ValueError("Formal protocol lacks an intact freeze marker")
    evidence = {key: read_evidence(value) for key, value in plan["evidence"].items()}
    validate_evidence(plan, evidence)
    matrix = read_evidence(plan["matrix"])
    environment = read_evidence(plan["environment"])
    assets = read_evidence(plan["assets"])
    if (environment["passed"] is not True or set(matrix["groups"]) != set(GROUPS)
            or matrix["all_original_values_identical"] is not True):
        raise ValueError("Matrix/environment identity incomplete")
    if assets.get("passed") is not True:
        raise ValueError("Asset identity gate not passed")
    pool = Path(evidence["scenes"]["source_pool"])
    pool_identity = json.loads((pool / "identity.json").read_text())
    if (pool_identity["asset_report_sha256"] != plan["assets"]["sha256"]
            or pool_identity["environment_report_sha256"] != plan["environment"]["sha256"]):
        raise ValueError("Formal assets/environment differ from prepared scenes")
    from scripts.statewam.r2.evaluation_scenes import verify
    verify(plan["evidence"]["scenes"]["path"])
    if not plan["source_sha256"]:
        raise ValueError("Require frozen executable source identities")
    for name, digest in plan["source_sha256"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != digest:
            raise ValueError("Formal executable source changed after protocol freeze")
    expected_paths = {str(source.relative_to(ROOT)) for source in executable_paths(ROOT)}
    if set(plan["source_sha256"]) != expected_paths:
        raise ValueError("Formal source inventory is incomplete or has new files")
    return plan


def validate_call(args, plan_identity, *, seed, protocol_id):
    path = Path(plan_identity["protocol_path"])
    plan = load_protocol(path)
    if plan != plan_identity["plan"] or sha256(path) != plan_identity["protocol_sha256"]:
        raise ValueError("Training invocation protocol identity changed")
    recipe = plan["training"]
    if seed != recipe["seed"] or protocol_id != plan["protocol_id"] or args.group not in GROUPS:
        raise ValueError("Formal training group/seed differs")
    for key in ("updates", "micro_batch", "accumulation", "workers", "save_every"):
        if getattr(args, key) != recipe[key]:
            raise ValueError("Formal training settings differ from frozen recipe")
    if (args.matrix.resolve() != Path(plan["matrix"]["path"]).parent
            or args.end_update is not None or args.audit_workload):
        raise ValueError("Formal matrix, early stop or workload override is forbidden")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    plan = load_protocol(args.protocol)
    recipe = plan["training"]
    settings = SimpleNamespace(matrix=Path(plan["matrix"]["path"]).parent, group=args.group, output=args.output,
        updates=recipe["updates"], micro_batch=recipe["micro_batch"], accumulation=recipe["accumulation"],
        workers=recipe["workers"], save_every=recipe["save_every"], end_update=None, resume=args.resume, audit_workload=False)
    from scripts.statewam.r2.train import run_experiment
    run_experiment(settings, protocol_id=plan["protocol_id"], seed=recipe["seed"], purpose="training",
                   plan_identity={"protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol), "plan": plan})


if __name__ == "__main__":
    main()
