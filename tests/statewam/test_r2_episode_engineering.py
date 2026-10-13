import json

import pytest

from benchmarks.robotwin import r2_episode_engineering as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash


@pytest.fixture
def plan_path(tmp_path, monkeypatch):
    probe, checkpoint = tmp_path / "probe", tmp_path / "checkpoint"
    checkpoint.mkdir()
    files = [checkpoint / name for name in ("checkpoint_step_0.safetensors", "config.yaml", "normalization_stats.npy")]
    files += [tmp_path / name for name in ("assets.json", "environment.json")]
    for path in files:
        path.write_text("fixture\n")
    atomic_json(probe / "scene_probe.json", {"task": "adjust_bottle", "condition": "demo_clean", "seed": 200001})
    atomic_json(probe / "reset_0/initial.json", {"metadata": {"step_lim": 400}})
    row = {"expert_eligible": True, "instruction": {"text_sha256": "a" * 64}, "policy_seed": 123}
    monkeypatch.setattr(module, "inspect_probe", lambda *args, **kwargs: dict(row))
    plan = {"goal_id": GOAL_ID, "protocol_id": module.PROTOCOL, "purpose": "engineering", "group": "E0",
            "checkpoint_name": "checkpoint_step_0.safetensors", "checkpoint_dir": str(checkpoint),
            "automatic_retries": 0, "max_episodes": 1, "startup_seconds": 600, "episode_seconds": 1800,
            "preflight_seconds": 1200, "provenance_seconds": 900,
            "fault": {"mode": "none", "after_actions": 0, "latest_action": 0},
            "service_seconds": 2400, "idle_seconds": 300, "max_messages": 401, "scene_probe": str(probe),
            "scene_row_sha256": stable_hash(row), "asset_report": str(files[-2]), "environment_report": str(files[-1]),
            "file_sha256": {str(path): sha256(path) for path in files}, "ledger_root": str(tmp_path / "ledger"),
            "simulator_python": "/unused/python"}
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    return path


def test_engineering_plan_binds_scene_checkpoint_and_original_limit(plan_path):
    plan, row, identity = module.load_plan(plan_path)
    assert identity["protocol_sha256"] == sha256(plan_path)
    assert identity["scene_manifest_sha256"] == plan["scene_row_sha256"]
    assert identity["scene_id"] == row["scene_id"] == "engineering/adjust_bottle/demo_clean/200001"
    assert plan["max_messages"] == 401


@pytest.mark.parametrize("field,value", [("purpose", "final"), ("group", "E3"), ("max_episodes", 2),
    ("automatic_retries", 1), ("checkpoint_name", "checkpoint_step_100.safetensors"),
    ("max_messages", 40), ("episode_seconds", 0), ("startup_seconds", 3601), ("idle_seconds", True),
    ("preflight_seconds", 900), ("provenance_seconds", 901),
    ("scene_row_sha256", "b" * 64), ("file_sha256", {})])
def test_engineering_rejects_scope_or_identity_drift(plan_path, field, value):
    plan = json.loads(plan_path.read_text())
    atomic_json(plan_path, {**plan, field: value})
    with pytest.raises(ValueError):
        module.load_plan(plan_path)


def test_engineering_rejects_final_scene(plan_path):
    plan = json.loads(plan_path.read_text())
    path = module.Path(plan["scene_probe"]) / "scene_probe.json"
    record = json.loads(path.read_text())
    atomic_json(path, {**record, "seed": 400001})
    with pytest.raises(ValueError, match="final scenes"):
        module.load_plan(plan_path)


def test_child_lease_partition_is_explicit_and_does_not_mutate_parent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setenv("R2_GPU_UUIDS", "uuid0,uuid2")
    result = module.child_environment(1, indices=["0", "2"], uuids=["uuid0", "uuid2"])
    assert result["CUDA_VISIBLE_DEVICES"] == "2" and result["R2_GPU_UUIDS"] == "uuid2"
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "0,2"


def test_environment_recheck_uses_fresh_same_interpreter_and_fixed_cwd(tmp_path, monkeypatch):
    output = tmp_path / "provenance.json"
    plan = {"asset_report": "/assets.json", "environment_report": "/environment.json"}

    def run(command, **kwargs):
        assert command[0] == module.sys.executable and command[1] == "-c"
        assert command[-3:] == ["/assets.json", "/environment.json", str(output)]
        assert kwargs["cwd"] == module.ROOT
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["check"] and kwargs["timeout"] == 900
        atomic_json(output, {"verified": True})

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.verify_simulator_provenance(plan, output) == {"verified": True}
    with pytest.raises(FileExistsError):
        module.verify_simulator_provenance(plan, output)


@pytest.mark.parametrize("service_fails", [False, True])
def test_coordinator_closes_both_children_and_preserves_failure(plan_path, monkeypatch, service_fails):
    from scripts.statewam.r2 import gpu_resources

    plan, _, identity = module.load_plan(plan_path)
    output = plan_path.parent / "output"
    stop = plan_path.parent / "STOP"
    monkeypatch.setenv("R2_STOP_FILE", str(stop))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(module, "verify_cuda_devices", lambda: ["uuid0", "uuid1"])
    monkeypatch.setattr(module, "snapshot_sources", lambda path: {})
    monkeypatch.setattr(module, "assert_sources_unchanged", lambda sources: None)
    monkeypatch.setattr(module, "process_identity", lambda pid: {"pid": pid, "created_at": 1})

    class Monitor:
        def snapshot(self):
            return [{"index": 0, "uuid": "uuid0"}]

        def close(self):
            pass

    children = []

    class Child:
        def __init__(self, command, **kwargs):
            self.role, self.pid = command[2], 123 + len(children)
            self.returncode = (1 if service_fails else None) if self.role == "service" else 0
            self.waited = False
            children.append(self)
            if self.role == "preflight":
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
                atomic_json(output / "environment_before.json", {})
            elif self.role == "service" and not service_fails:
                assert children[0].role == "preflight" and children[0].returncode == 0
                atomic_json(output / "service/ready.json", {})
            elif self.role == "episode":
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
                atomic_json(output / "episode_result.json", {"outcome": "failure"})

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.waited = True
            self.returncode = self.returncode or 0
            return self.returncode

    monkeypatch.setattr(gpu_resources, "DeviceMonitor", Monitor)
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    if service_fails:
        with pytest.raises(RuntimeError, match="Service failed"):
            module.coordinate(plan_path, plan, identity, output)
    else:
        module.coordinate(plan_path, plan, identity, output)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == ("failed" if service_fails else "completed")
    assert len(children) == (2 if service_fails else 3)
    assert all(child.waited for child in children)
    assert (output / "SERVICE_STOP").exists()
    assert stop.exists() == service_fails


def test_legacy_plan_can_be_audited_but_not_relaunched(plan_path):
    plan = json.loads(plan_path.read_text())
    plan["protocol_id"] = module.LEGACY_PROTOCOL
    del plan["preflight_seconds"], plan["provenance_seconds"]
    atomic_json(plan_path, plan)
    loaded, _, identity = module.load_plan(plan_path)
    with pytest.raises(ValueError, match="audit-only"):
        module.coordinate(plan_path, loaded, identity, plan_path.parent / "output")


def test_preflight_failure_never_launches_policy(plan_path, monkeypatch):
    from scripts.statewam.r2 import gpu_resources

    plan, _, identity = module.load_plan(plan_path)
    output = plan_path.parent / "output"
    monkeypatch.setenv("R2_STOP_FILE", str(plan_path.parent / "STOP"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(module, "verify_cuda_devices", lambda: ["uuid0", "uuid1"])
    monkeypatch.setattr(module, "snapshot_sources", lambda path: {})
    monkeypatch.setattr(module, "process_identity", lambda pid: {"pid": pid, "created_at": 1})

    class Monitor:
        def snapshot(self):
            return [{"index": 0, "uuid": "uuid0"}]

        def close(self):
            pass

    class Child:
        pid, returncode = 123, 1

        def __init__(self, command, **kwargs):
            assert command[2] == "preflight"
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(gpu_resources, "DeviceMonitor", Monitor)
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="Preflight failed"):
        module.coordinate(plan_path, plan, identity, output)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "failed" and set(state["children"]) == {"preflight"}
