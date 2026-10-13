"""Read-only inventory of extracted simulator assets against pinned archives."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash

ASSET_NAMES = ("objects", "background_texture", "embodiments")


def archive_inventory(path, name):
    records, ignored = {}, []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe simulator archive member")
            if info.is_dir():
                continue
            if relative.parts[0] == "__MACOSX":
                ignored.append(info.filename)
                continue
            if relative.parts[0] != name or len(relative.parts) < 2 or str(relative) in records:
                raise ValueError("Unexpected or duplicate simulator archive member")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != info.file_size:
                raise ValueError("Archive member size differs")
            records[str(relative)] = {"bytes": size, "sha256": digest.hexdigest()}
    return records, ignored


def extracted_inventory(root, name):
    directory = root / name
    if not directory.is_dir():
        raise ValueError(f"Missing simulator asset directory: {directory}")
    records = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Nested asset symlinks require explicit provenance")
        if not path.is_file():
            continue
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Simulator asset changed while hashing")
        records[f"{name}/{path.relative_to(directory).as_posix()}"] = {"bytes": after.st_size, "sha256": digest}
    return records


def compare_inventory(expected, actual):
    return {"missing": sorted(set(expected) - set(actual)), "extra": sorted(set(actual) - set(expected)),
            "changed": sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])}


def generated_configuration_inventory(archive_path, robotwin_root):
    records, derivations = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not info.filename.startswith("embodiments/") or not info.filename.endswith("_tmp.yml"):
                continue
            with archive.open(info) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8") as text:
                    template = text.read()
            # Replay the pinned installer without executing it or writing assets.
            generated = template.replace("${ASSETS_PATH}", str(robotwin_root)).replace("$ASSETS_PATH", str(robotwin_root))
            data = generated.encode("utf-8")
            target = info.filename.replace("_tmp.yml", ".yml")
            if target in records:
                raise ValueError("Duplicate generated embodiment configuration")
            records[target] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            derivations[target] = {"template": info.filename, "assets_path": str(robotwin_root),
                                   "recipe": "RoboTwin update_embodiment_config_path.py literal ASSETS_PATH replacement"}
    return records, derivations


def audit(source_report, asset_root, robotwin_root):
    source = json.loads(source_report.read_text())
    if source["goal_id"] != GOAL_ID or source["status"] != "source_identity_passed":
        raise ValueError("Require passed R2 source audit")
    archives = {Path(row["path"]).stem: row for row in source["asset_archives"]}
    if set(archives) != set(ASSET_NAMES):
        raise ValueError("Incomplete simulator asset archive set")
    from benchmarks.robotwin.eval_policy_wrapper import VERIFIED_ROBOTWIN_COMMIT
    if source["simulator_commit"] != VERIFIED_ROBOTWIN_COMMIT:
        raise ValueError("Source report has another simulator version")
    generator_relative = "script/update_embodiment_config_path.py"
    pinned = subprocess.check_output(["git", "-C", str(robotwin_root), "show", f"{VERIFIED_ROBOTWIN_COMMIT}:{generator_relative}"])
    if (robotwin_root / generator_relative).read_bytes() != pinned:
        raise ValueError("Embodiment configuration generator differs from pinned source")
    report = {"goal_id": GOAL_ID, "protocol_id": "r2-scene-assets-v2", "started_at": time.time(),
              "robotwin_root": str(robotwin_root), "generator_sha256": hashlib.sha256(pinned).hexdigest(),
              "source_report": str(source_report.resolve()), "source_report_sha256": sha256(source_report),
              "asset_root": str(asset_root.resolve()), "groups": {}}
    for name in ASSET_NAMES:
        row = archives[name]
        archive = Path(row["path"])
        if archive.stat().st_size != row["bytes"] or sha256(archive) != row["sha256"]:
            raise ValueError("Asset archive differs from pinned P0 identity")
        expected, ignored = archive_inventory(archive, name)
        archive_members = len(expected)
        generated, derivations = generated_configuration_inventory(archive, robotwin_root) if name == "embodiments" else ({}, {})
        if set(generated) & set(expected):
            raise ValueError("Generated asset unexpectedly overwrites an archived asset")
        expected.update(generated)
        actual = extracted_inventory(asset_root, name)
        differences = compare_inventory(expected, actual)
        report["groups"][name] = {"archive": row, "archive_members": archive_members,
                                 "generated_configuration_derivations": derivations,
                                 "ignored_nonruntime_archive_members": ignored,
                                 "extracted_directory": str((asset_root / name).resolve()),
                                 "files": actual, "inventory_sha256": stable_hash(actual),
                                 "logical_bytes": sum(record["bytes"] for record in actual.values()),
                                 "differences": differences}
        print(f"{name}: files={len(actual)} differences="
              f"{ {key: len(value) for key, value in differences.items()} }", flush=True)
    report["passed"] = all(not values for group in report["groups"].values() for values in group["differences"].values())
    report["ended_at"] = time.time()
    report["limits"] = "Expanded runtime asset content only; package patches, driver/backend and scene behavior require separate checks. No assets were modified."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing asset audit")
    report = audit(args.source_report, args.asset_root, args.robotwin_root.resolve())
    atomic_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
