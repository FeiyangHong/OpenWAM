"""Apply the pinned RoboTwin installer's two dependency patches, with an audit."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patches = [
        ("sapien", "3.0.0b1", "sapien/wrapper/urdf_loader.py", [
            ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
            ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
        ]),
        ("mplib", "0.2.1", "mplib/planner.py", [
            ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
             "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
        ]),
    ]
    records = []
    for package, version, relative, replacements in patches:
        dist = importlib.metadata.distribution(package)
        if dist.version != version:
            raise RuntimeError(f"Expected {package}=={version}, found {dist.version}")
        path = Path(dist.locate_file(relative))
        original = path.read_text()
        patched = original
        for old, new in replacements:
            if new in patched and old not in patched:
                continue
            if patched.count(old) != 1:
                raise RuntimeError(f"Unexpected patch target in {path}: {old}")
            patched = patched.replace(old, new)
        backup = path.with_suffix(path.suffix + ".statewam-original")
        if patched != original:
            if not backup.exists():
                backup.write_text(original)
            path.write_text(patched)
        records.append({
            "package": package, "version": version, "path": str(path),
            "backup": str(backup), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "goal_id": "r1/robotwin-smoke",
        "source": "RoboTwin 0aeea2d script/_install.sh (not the discrepant SRDF comment)",
        "patches": records,
    }, indent=2) + "\n")
    print(f"Verified official dependency patches: {args.output}")


if __name__ == "__main__":
    main()
