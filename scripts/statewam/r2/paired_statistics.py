"""Scene-paired, task-equal R2 pilot estimates; never infer training-seed variance."""

import numpy as np

from scripts.statewam.r2.common import GOAL_ID, TASKS, stable_hash
from scripts.statewam.r2.evaluation_plan import CONDITIONS, GROUPS, scheduled_episodes

COMPARISONS = (("E3", "E0"), ("E3", "E1"), ("E2", "E1"), ("E3", "E2"))
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260912


def interval(samples):
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {"lower": float(lower), "upper": float(upper), "degenerate": bool(lower == upper)}


def paired_bootstrap(values, *, rng, samples=BOOTSTRAP_SAMPLES, resample_tasks=True):
    values = np.asarray(values)
    if (values.ndim != 3 or values.shape[0] != 4 or min(values.shape[1:]) < 1
            or not np.isin(values, [0, 1]).all() or type(samples) is not int or samples < 1):
        raise ValueError("Require four groups of matched task/scene binary outcomes")
    _, tasks, scenes = values.shape
    task_index = (rng.integers(tasks, size=(samples, tasks)) if resample_tasks
                  else np.broadcast_to(np.arange(tasks), (samples, tasks)))
    scene_index = rng.integers(scenes, size=(samples, tasks, scenes))
    # The same task and scene indices select all four groups, preserving pairs.
    draws = values[:, task_index[:, :, None], scene_index].mean(axis=(2, 3))
    point = values.mean(axis=(1, 2))
    return {
        "rates": {group: {"estimate": float(point[index]), "ci95": interval(draws[index])}
                  for index, group in enumerate(GROUPS)},
        "differences_pp": {f"{left}-{right}": {
            "estimate": float(100 * (point[GROUPS.index(left)] - point[GROUPS.index(right)])),
            "ci95": interval(100 * (draws[GROUPS.index(left)] - draws[GROUPS.index(right)]))}
            for left, right in COMPARISONS},
    }


def summarize(schedule, records):
    """Summarize independently audited complete results, refusing paired gaps.

    The filesystem entry point must independently validate each episode before
    supplying records. A result flag alone is not physical execution evidence.
    """
    episodes = scheduled_episodes(schedule)
    expected = {row["episode_key"]: row for row in episodes}
    if len(expected) != 2040 or set(records) != set(expected):
        raise ValueError("Missing, duplicate or extra paired episode results")
    for key, item in records.items():
        if (item["identity"] != expected[key]["identity"] or stable_hash(item["identity"]) != key
                or item["outcome"] not in ("success", "failure") or item["audited"] is not True):
            raise ValueError("Unaudited, invalid or mismatched paired result")
    report = {"goal_id": GOAL_ID, "protocol_sha256": schedule["protocol_sha256"],
              "counts": schedule["counts"], "bootstrap_samples": BOOTSTRAP_SAMPLES,
              "bootstrap_seed": BOOTSTRAP_SEED, "significance_tests": False,
              "method": "paired percentile bootstrap; task-equal macro, tasks then scenes with replacement",
              "conditions": {}, "repeatability": [],
              "limits": "Single training seed; intervals omit training variance. Degenerate bootstrap intervals can occur with few or no discordant scenes and do not prove certainty, equivalence or generalization."}
    for split_index, split in enumerate(("development", "final")):
        count = 5 if split == "development" else 20
        for condition_index, condition in enumerate(CONDITIONS):
            values = np.zeros((4, len(TASKS), count), dtype=np.uint8)
            paired_rows = []
            for task_index, task in enumerate(TASKS):
                selected = [row for row in episodes if row["identity"]["split"] == split
                            and (row["identity"]["task"], row["identity"]["condition"]) == (task, condition)]
                scenes = sorted({row["identity"]["scene_id"] for row in selected})
                if len(scenes) != count or len(selected) != count * 4:
                    raise ValueError("Task/condition denominator differs")
                for scene_index, scene in enumerate(scenes):
                    group_results = {row["identity"]["group"]: records[row["episode_key"]]
                                     for row in selected if row["identity"]["scene_id"] == scene}
                    if set(group_results) != set(GROUPS):
                        raise ValueError("Scene is not paired across all groups")
                    outcomes = {group: group_results[group]["outcome"] for group in GROUPS}
                    paired_rows.append({"task": task, "scene_id": scene, "outcomes": outcomes})
                    values[:, task_index, scene_index] = [outcomes[group] == "success" for group in GROUPS]
            rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, split_index, condition_index]))
            estimate = paired_bootstrap(values, rng=rng)
            per_task = {task: {"denominator_per_group": count,
                              **paired_bootstrap(values[:, index:index + 1], rng=rng, resample_tasks=False)}
                        for index, task in enumerate(TASKS)}
            report["conditions"][f"{split}/{condition}"] = {
                "denominator_per_group": count * len(TASKS), "macro": estimate,
                "per_task": per_task, "paired_scenes": paired_rows}
    for task in TASKS:
        for condition in CONDITIONS:
            selected = sorted((row for row in episodes if row["identity"]["split"] == "repeatability"
                               and (row["identity"]["task"], row["identity"]["condition"]) == (task, condition)),
                              key=lambda row: row["identity"]["replicate"])
            if len(selected) != 2 or [r["identity"]["replicate"] for r in selected] != [0, 1]:
                raise ValueError("Repeatability diagnostic matrix differs")
            outcomes = [records[row["episode_key"]]["outcome"] for row in selected]
            report["repeatability"].append({"task": task, "condition": condition,
                "scene_id": selected[0]["identity"]["scene_id"], "episode_keys": [row["episode_key"] for row in selected],
                "outcomes": outcomes, "outcomes_equal": outcomes[0] == outcomes[1],
                "limits": "Outcome comparison only; initial/action residual audits are separate evidence"})
    return report
