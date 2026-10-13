#!/usr/bin/env python3
"""Manually apply the reviewed patch only after this repository's training stops."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def validate(root, bundle, manifest):
    entries = manifest['files']
    for name in entries:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise RuntimeError('Unsafe patch path: ' + name)
        if digest(bundle / 'staged' / name) != entries[name]['after']:
            raise RuntimeError('Prepared payload changed: ' + name)
    applied = all(digest(root / n) == v['after'] for n, v in entries.items())
    if applied:
        return True
    for name, values in entries.items():
        if digest(root / name) != values['before']:
            raise RuntimeError('Source changed since preparation; review/rebase required: ' + name)
    for name, expected in manifest['context'].items():
        if digest(root / name) != expected:
            raise RuntimeError('Training dependency changed; review required: ' + name)
    if digest(bundle / 'changes.patch') != manifest['patch_sha256']:
        raise RuntimeError('Prepared patch changed')
    return False


def training_pids(root):
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if (proc / 'cwd').resolve() != root.resolve():
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if any(a == b'train.py' or a.endswith(b'/train.py') for a in args):
                found.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    root = bundle.parent.parent
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if validate(root, bundle, manifest):
        print('Already applied; no files changed.')
        return
    def git(*argv, env=None):
        return subprocess.check_output(['git', *argv], cwd=root, env=env).decode().strip()
    git('apply', '--check', str(bundle / 'changes.patch'))
    lock = root / 'outputs/wuji_rtc_history/manual_training.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        busy = False
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            busy = True
        pids = training_pids(root)
        if args.check:
            print(json.dumps({'patch_valid': True, 'training_lock_busy': busy,
                              'training_pids': pids, 'can_apply_now': not busy and not pids}))
            return
        if busy or pids:
            raise RuntimeError('Training is still running; nothing applied. PIDs=' + str(pids))
        # Revalidate under the lock, then save all existing dirty work in Git.
        validate(root, bundle, manifest)
        head = git('rev-parse', 'HEAD')
        with tempfile.TemporaryDirectory(prefix='rtc-prefix-apply-') as td:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / 'index'))
            git('read-tree', head, env=env)
            git('add', '-A', env=env)
            before = git('commit-tree', git('write-tree', env=env), '-p', head,
                         '-m', 'Save dirty work before applying RTC prefix PMF', env=env)
            git('update-ref', 'refs/heads/codex/rtc-prefix-pmf-activation-before-20261012', before)
            git('apply', str(bundle / 'changes.patch'))
            git('add', '--', *manifest['files'], env=env)
            after = git('commit-tree', git('write-tree', env=env), '-p', before,
                        '-m', 'Apply optional RTC prefix PMF and manual training entry', env=env)
            git('update-ref', 'refs/heads/codex/rtc-prefix-pmf-applied-20261012', after)
        print('Applied. Git commit: ' + after + '. Training was NOT started.')


if __name__ == '__main__':
    main()
