"""Freeze RoboTwin instructions once, without consuming the simulator's RNG.

This module does not import SAPIEN or execute a policy. Final evaluation replays
the saved text after checking its identity; it never regenerates alternatives.
"""

import importlib.util
import json
import random
import re
from pathlib import Path

from scripts.statewam.r2.common import sha256, stable_hash


def freeze_instruction(robotwin_root, task, episode_parameters, *, seed, instruction_type="unseen"):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task) or instruction_type not in {"seen", "unseen"}:
        raise ValueError("Invalid task or instruction type")
    if not isinstance(episode_parameters, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in episode_parameters.items()):
        raise ValueError("Expected explicit episode instruction parameters")
    root = Path(robotwin_root).resolve()
    generator = root / "description/utils/generate_episode_instructions.py"
    template = root / f"description/task_instruction/{task}.json"
    objects = root / "description/objects_description"
    sources = {generator, template}
    fallbacks = []
    for value in episode_parameters.values():
        path = (objects / f"{value}.json").resolve()
        if not path.is_relative_to(objects.resolve()):
            raise ValueError("Object description is outside the fixed source directory")
        if path.is_file():
            sources.add(path)
            data = json.loads(path.read_text())
            if instruction_type == "unseen" and not data.get("unseen"):
                fallbacks.append(str(path.relative_to(root)))
        elif "/" in value or "\\" in value:
            raise FileNotFoundError(f"Missing object description: {value}")
    hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(sources)}
    spec = importlib.util.spec_from_file_location("r2_frozen_instruction_generator", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = random.getstate()
    try:
        random.seed(seed)
        rows = module.generate_episode_descriptions(task, [dict(episode_parameters)], max_descriptions=100)
    finally:
        random.setstate(previous)
    if len(rows) != 1 or not rows[0].get(instruction_type):
        raise ValueError("No instruction of the requested type for this scene")
    candidates = rows[0][instruction_type]
    # Selection is isolated from both the simulator and the generator stream.
    selection_seed = int(stable_hash(["r2-instruction-selection-v1", seed])[:16], 16)
    index = random.Random(selection_seed).randrange(len(candidates))
    text = candidates[index]
    if not isinstance(text, str) or not text.strip() or re.search(r"\{[^}]+\}", text):
        raise ValueError("Empty or unresolved instruction")
    if hashes != {name: sha256(root / name) for name in hashes}:
        raise ValueError("Instruction sources changed during generation")
    record = {"text": text, "text_sha256": stable_hash(text), "type": instruction_type,
              "generator_seed": seed, "selection_seed": selection_seed,
              "candidate_count": len(candidates), "selected_index": index,
              "task": task, "episode_parameters": dict(episode_parameters),
              "source_sha256": hashes, "unseen_object_fallback_to_seen": sorted(set(fallbacks)),
              "generation_limit": 100, "schema": "r2-frozen-instruction-v1"}
    record["identity_sha256"] = stable_hash(record)
    return record


def replay_instruction(record, *, robotwin_root, task, episode_parameters):
    data = {key: value for key, value in record.items() if key != "identity_sha256"}
    if stable_hash(data) != record["identity_sha256"] or stable_hash(record["text"]) != record["text_sha256"]:
        raise ValueError("Frozen instruction integrity mismatch")
    if record["schema"] != "r2-frozen-instruction-v1" or record["task"] != task:
        raise ValueError("Frozen instruction task/schema mismatch")
    if record["episode_parameters"] != episode_parameters:
        raise ValueError("Frozen instruction scene parameters mismatch")
    root = Path(robotwin_root).resolve()
    for name, digest in record["source_sha256"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or sha256(source) != digest:
            raise ValueError("Frozen instruction source mismatch")
    return record["text"]
