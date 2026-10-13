#!/usr/bin/env python3
"""Edit locally, synchronize and execute in the original H20 repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.local-h20'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def collect():
    names = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    modes = {}
    for row in git('ls-files', '--stage', '-z').decode().split('\0'):
        if row:
            meta, name = row.split('\t', 1)
            modes[name] = 0o755 if meta.startswith('100755 ') else 0o644
    files = {}
    for name in sorted(set(names) - {''}):
        path = ROOT / name
        if path.is_symlink():
            raise RuntimeError('Source symlink needs explicit handling: ' + name)
        if not path.is_file():
            continue  # Deleted files and uninitialized gitlinks.
        if path.stat().st_size > 32 * 1024 * 1024:
            raise RuntimeError('Refusing source file over 32 MiB: ' + name)
        if any(p in {'.local-h20', '.h20-local', '.codex', '.ssh', 'wandb'} for p in path.relative_to(ROOT).parts):
            raise RuntimeError('Private runtime file in Git source list: ' + name)
        if path.name in {'.env', '.netrc', 'auth.json'}:
            raise RuntimeError('Credential file in Git source list: ' + name)
        files[name] = {'sha256': digest(path), 'mode': modes.get(name, 0o755 if name.endswith('.sh') else 0o644)}
    return files


def changed(before, after):
    return [name for name in sorted(before.keys() | after.keys())
            if before.get(name, {}).get('sha256') != after.get(name, {}).get('sha256')]


def ssh(config):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ForwardAgent=no', config['host']]


def remote(config, operation, **kwargs):
    request = dict(config, operation=operation, **kwargs)
    import base64
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    helper = (ROOT / 'scripts/dev/h20_remote.py').read_bytes()
    return subprocess.check_output(ssh(config) + ['python - ' + shlex.quote(payload)], input=helper).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('status', help='Compare local/server files to the last synchronization')
    sub.add_parser('sync', help='Synchronize original worktree after conflict checks, backing up replaced files')
    run = sub.add_parser('run', help='Execute the last synced version in the H20 environment')
    run.add_argument('--background', metavar='LABEL', help='Run in tmux and save durable log/exit status')
    run.add_argument('command', nargs=argparse.REMAINDER)
    logs = sub.add_parser('logs', help='Tail a background run and show its exit status')
    logs.add_argument('run_id')
    args = parser.parse_args()
    config = json.loads((STATE / 'config.json').read_text(encoding='utf-8'))
    baseline = json.loads((STATE / 'initial-snapshot/baseline.json').read_text())
    last_path = STATE / 'last-sync.json'
    if last_path.exists():
        previous = json.loads(last_path.read_text())
        if previous.get('mode') == 'in-place':
            baseline = {'files': previous['files'], 'head': previous['remote_head']}
    if args.operation == 'status':
        original = json.loads(remote(config, 'inspect'))
        local = collect()
        local_changes = changed(baseline['files'], local)
        remote_changes = changed(baseline['files'], original['files'])
        conflicts = [p for p in set(local_changes) & set(remote_changes)
                     if local.get(p, {}).get('sha256') != original['files'].get(p, {}).get('sha256')]
        print(json.dumps({'local_changes_since_sync': local_changes,
                          'server_changes_since_sync': remote_changes,
                          'conflicts': sorted(conflicts), 'remote_head': original['head'],
                          'baseline_remote_head': baseline['head']}, indent=2))
    elif args.operation == 'sync':
        files = collect()
        head = git('rev-parse', 'HEAD').decode().strip()
        identity = {'files': files, 'head': head, 'source_root': config['source_root']}
        snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        manifest = dict(identity, snapshot_id=snapshot_id,
                        baseline=baseline,
                        branch=git('branch', '--show-current').decode().strip(),
                        git_status=git('status', '--short', '--branch').decode())
        with tempfile.TemporaryDirectory(prefix='openwam-upload-') as tmp:
            archive = Path(tmp) / 'source.tar.gz'
            metadata = Path(tmp) / 'manifest.json'
            metadata.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            with tarfile.open(archive, 'w:gz') as tf:
                for name, entry in files.items():
                    member = tf.gettarinfo(str(ROOT / name), arcname=name)
                    member.mode = entry['mode']
                    with (ROOT / name).open('rb') as f:
                        tf.addfile(member, f)
                tf.add(metadata, arcname='.h20-manifest.json')
            if collect() != files or git('rev-parse', 'HEAD').decode().strip() != head:
                raise RuntimeError('Local files changed while packaging; rerun sync')
            incoming = json.loads(remote(config, 'prepare'))['incoming']
            subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ForwardAgent=no',
                            str(archive), config['host'] + ':' + incoming], check=True)
            result = json.loads(remote(config, 'install', incoming=incoming, snapshot_id=snapshot_id))
        (STATE / 'last-sync.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
    elif args.operation == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('run requires a command after --')
        last = json.loads((STATE / 'last-sync.json').read_text())
        current = collect()
        if last.get('mode') != 'in-place' or current != last['files'] or git('rev-parse', 'HEAD').decode().strip() != last['head']:
            raise RuntimeError('Local source differs from last in-place sync; run sync first')
        result = json.loads(remote(config, 'launch', snapshot_id=last['snapshot_id'],
                                   command=command, background=args.background))
        if args.background:
            print(json.dumps(result, indent=2))
        else:
            completed = subprocess.run(ssh(config) + ['bash ' + shlex.quote(result['script'])])
            print('\nRemote run: ' + result['run_id'], flush=True)
            raise SystemExit(completed.returncode)
    elif args.operation == 'logs':
        print(remote(config, 'logs', run_id=args.run_id), end='')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print('h20: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
