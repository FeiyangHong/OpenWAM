"""Preserve executable source evidence and reject in-run source drift."""

import shutil
from pathlib import Path

from scripts.statewam.r2.common import ROOT, atomic_json, sha256


def source_paths(root=ROOT):
    root = Path(root)
    files = []
    for directory, suffixes in (("openwam", {".py"}), ("scripts", {".py"}),
                                ("configs", {".yaml", ".yml"})):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file() and path.suffix in suffixes)
    return sorted(files)


def capture_sources(destination, *, root=ROOT):
    destination, root = Path(destination), Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for source in source_paths(root):
        relative = source.relative_to(root)
        copy = destination / relative
        before = sha256(source)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy)
        if before != sha256(copy) or before != sha256(source):
            raise ValueError(f"Source changed while taking snapshot: {relative}")
        manifest[str(relative)] = before
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def assert_sources_unchanged(manifest, *, root=ROOT):
    root = Path(root)
    for name, digest in manifest.items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Executable source changed during the run: {name}")
