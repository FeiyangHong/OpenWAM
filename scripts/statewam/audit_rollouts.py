"""Check completed smoke episodes and their actual recorded video frames."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def validate_frame_count(episode, frame_count):
    # Pinned RoboTwin 0aeea2d writes before every action and once more on success.
    expected = episode["action_steps"] + int(episode["success"])
    if not 0 < episode["action_steps"] <= episode["step_limit"] or frame_count != expected:
        raise ValueError(f"Video has {frame_count} frames; expected {expected} for {episode}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for run in args.runs:
        result = json.loads((run / "smoke_result.json").read_text())
        episodes = json.loads((run / "episodes.json").read_text())
        assert result["pass"] and result["episodes"] == episodes and 1 <= len(episodes) <= 5
        assert result["count"] == len(episodes) and result["successes"] == sum(e["success"] for e in episodes)
        videos = sorted((run / "runtime/eval_result").rglob("episode*.mp4"))
        assert len(videos) == len(episodes)
        details = []
        for episode, video in zip(episodes, videos):
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(video),
            ], text=True))["streams"][0]
            expected = validate_frame_count(episode, int(probe["nb_frames"]))
            pixels = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ])
            frames = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, probe["height"], probe["width"], 3)
            validate_frame_count(episode, len(frames))
            first_std, last_std = float(frames[0].std()), float(frames[-1].std())
            change = float(np.abs(frames[-1].astype(np.float32) - frames[0]).mean())
            assert first_std > 1 and last_std > 1 and change > 0
            details.append({"path": str(video), "probe": probe, "expected_frames": expected,
                            "decoded_frames": len(frames), "first_frame_std": first_std,
                            "last_frame_std": last_std, "first_last_mean_absolute_change": change,
                            "sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
        records.append({"run": str(run), "result": result, "videos": details})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"goal_id": "r1/robotwin-smoke", "pass": True, "runs": records}, indent=2) + "\n")
    print(f"Rollout and video audit passed: {args.output}")


if __name__ == "__main__":
    main()
