"""Persistent episode identities and bounded retries for paired R2 evaluation."""

import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive, identity as process_identity

IDENTITY_FIELDS = {"goal_id", "protocol_sha256", "checkpoint_sha256", "scene_manifest_sha256",
                   "instruction_sha256", "group", "task", "condition", "split", "scene_id", "policy_seed", "replicate"}
POLICY_FAILURES = {"task_failure", "action_limit", "nonfinite_action", "model_error"}
INFRASTRUCTURE_FAILURES = {"external_interrupt", "transport_disconnect", "simulator_crash",
                           "device_failure", "host_restart"}


class RetryExhausted(RuntimeError):
    """The paired episode remains incomplete after its one permitted retry."""


def validate_identity(value):
    if set(value) != IDENTITY_FIELDS or value["goal_id"] != GOAL_ID:
        raise ValueError("Incomplete or unsupported R2 episode identity")
    for key in ("protocol_sha256", "checkpoint_sha256", "scene_manifest_sha256", "instruction_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError(f"Invalid identity digest: {key}")
    if (value["group"] not in ("E0", "E1", "E2", "E3") or value["task"] not in TASKS
            or value["condition"] not in ("demo_clean", "demo_randomized")
            or value["split"] not in ("development", "final", "repeatability")
            or type(value["replicate"]) is not int
            or value["replicate"] not in ((0, 1) if value["split"] == "repeatability" else (0,))
            or not isinstance(value["scene_id"], str) or not value["scene_id"]
            or type(value["policy_seed"]) is not int or not 0 <= value["policy_seed"] < 2**32):
        raise ValueError("Invalid episode group, scene or seed")
    return dict(value)


def artifact_inventory(directory, artifacts):
    inventory = {}
    for label, relative in artifacts.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or path == directory.resolve() or not path.is_file():
            raise ValueError("Episode artifact must be a file inside its own attempt")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError("Empty episode evidence")
        inventory[label] = {"path": str(path.relative_to(directory.resolve())), "bytes": size, "sha256": sha256(path)}
    return inventory


def validate_outcome(outcome, reason, stats, artifacts):
    if outcome in ("success", "failure"):
        if (outcome == "success" and reason != "goal_reached") or (outcome == "failure" and reason not in POLICY_FAILURES):
            raise ValueError("Unsupported policy outcome reason")
        if not {"video", "rollout"}.issubset(artifacts):
            raise ValueError("Completed policy outcomes require video and rollout evidence")
        if (type(stats.get("policy_actions")) is not int or stats["policy_actions"] < 0
                or type(stats.get("physics_steps")) is not int or stats["physics_steps"] < 0
                or type(stats.get("wall_seconds")) not in (int, float)
                or not math.isfinite(stats["wall_seconds"]) or stats["wall_seconds"] < 0):
            raise ValueError("Missing or invalid rollout counters")
    elif outcome == "invalid":
        if reason not in INFRASTRUCTURE_FAILURES or "evidence" not in artifacts:
            raise ValueError("Invalid requires a documented infrastructure or external interruption")
    else:
        raise ValueError("Unsupported episode outcome")


class EpisodeLedger:
    """Hold one exclusive episode lock throughout execution and publication.

    A terminal attempt result is atomic and includes hashes of closed artifacts.
    The episode COMPLETE pointer can be reconstructed if its publication was
    interrupted after the terminal result. Unknown partial attempts require an
    explicit, evidenced classification; dead processes alone are not invalid.
    """

    def __init__(self, root, identity):
        self.identity = validate_identity(identity)
        self.key = stable_hash(self.identity)
        self.directory = Path(root).resolve() / self.key
        self._lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / "episode.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "identity.json"
            if path.exists():
                if json.loads(path.read_text()) != self.identity:
                    raise ValueError("Episode ledger identity mismatch")
            else:
                atomic_json(path, self.identity)
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(self, *args):
        self._lock.close()
        self._lock = None

    def _locked(self):
        if self._lock is None:
            raise RuntimeError("Episode ledger requires an exclusive context")

    def _attempts(self):
        self._locked()
        paths = sorted(self.directory.glob("attempt_*"))
        if len(paths) > 2 or [path.name for path in paths] != [f"attempt_{i:03d}" for i in range(1, len(paths) + 1)]:
            raise ValueError("Episode attempt ledger has a gap or exceeds one retry")
        for path in paths:
            begin = json.loads((path / "begin.json").read_text())
            if begin["episode_key"] != self.key or begin["attempt"] != path.name:
                raise ValueError("Attempt identity mismatch")
        return paths

    def _result(self, path):
        result_path = path / "result.json"
        if not result_path.exists():
            return None
        result = json.loads(result_path.read_text())
        if result["episode_key"] != self.key or result["attempt"] != path.name:
            raise ValueError("Episode result identity mismatch")
        if result["begin_sha256"] != sha256(path / "begin.json"):
            raise ValueError("Episode attempt origin changed after publication")
        validate_outcome(result["outcome"], result["reason"], result["stats"], result["artifacts"])
        computed = artifact_inventory(path, {key: row["path"] for key, row in result["artifacts"].items()})
        if computed != result["artifacts"]:
            raise ValueError("Episode evidence changed after terminal publication")
        return result

    def _publish_complete(self, path):
        marker = {"episode_key": self.key, "attempt": path.name, "result_sha256": sha256(path / "result.json")}
        target = self.directory / "COMPLETE.json"
        if target.exists() and json.loads(target.read_text()) != marker:
            raise ValueError("Episode completion pointer differs")
        if not target.exists():
            atomic_json(target, marker)

    def begin(self):
        """Return a new attempt directory, or None for a verified policy outcome."""
        paths = self._attempts()
        for index, path in enumerate(paths):
            result = self._result(path)
            if result is None:
                raise RuntimeError("Unclassified partial episode; inspect evidence before retry")
            if result["outcome"] != "invalid":
                if index != len(paths) - 1:
                    raise ValueError("Policy success or failure was improperly retried")
                self._publish_complete(path)
                return None
        if (self.directory / "COMPLETE.json").exists():
            raise ValueError("Complete marker has no completed policy result")
        if len(paths) == 2:
            raise RetryExhausted("Both attempts invalid; retain the missing paired result")
        path = self.directory / f"attempt_{len(paths) + 1:03d}"
        path.mkdir(exist_ok=False)
        atomic_json(path / "begin.json", {"episode_key": self.key, "attempt": path.name,
                    "owner": process_identity(os.getpid()), "started_at": time.time()})
        return path

    def finish(self, path, *, outcome, reason, stats, artifacts):
        """Publish only after the caller closes and validates rollout artifacts."""
        paths = self._attempts()
        if not paths or path != paths[-1] or (path / "result.json").exists():
            raise ValueError("Only the current unpublished attempt may finish")
        begin = json.loads((path / "begin.json").read_text())
        if begin["owner"] != process_identity(os.getpid()) and (outcome != "invalid" or alive(begin["owner"])):
            raise RuntimeError("Only the owning process can publish a policy outcome")
        validate_outcome(outcome, reason, stats, artifacts)
        inventory = artifact_inventory(path, artifacts)
        atomic_json(path / "result.json", {"episode_key": self.key, "attempt": path.name,
                    "begin_sha256": sha256(path / "begin.json"),
                    "outcome": outcome, "reason": reason, "stats": stats, "artifacts": inventory,
                    "ended_at": time.time()})
        if outcome != "invalid":
            self._publish_complete(path)

    def record_interruption(self, *, reason, evidence):
        """Classify a dead attempt using existing evidence, never by death alone."""
        paths = self._attempts()
        if not paths or (paths[-1] / "result.json").exists():
            raise ValueError("No partial attempt to reconcile")
        path = paths[-1]
        begin = json.loads((path / "begin.json").read_text())
        if alive(begin["owner"]):
            raise RuntimeError("Attempt owner is still alive; do not take over")
        self.finish(path, outcome="invalid", reason=reason, stats={}, artifacts={"evidence": evidence})
