"""Remote half of h20.py; sent over SSH stdin, no Codex credentials needed."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import fcntl
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise RuntimeError('Unsafe source path: ' + name)
    if path.parts[0] in {'.git', '.codex', '.local-h20', '.h20-local'}:
        raise RuntimeError('Private metadata in source archive: ' + name)


def verify(root, manifest):
    for name, entry in manifest['files'].items():
        validate_name(name)
        path = root / name
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise RuntimeError('Missing or redirected source file: ' + name)
        if digest(path) != entry['sha256']:
            raise RuntimeError('Source was edited remotely; refusing to run: ' + name)


def inspect(source):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args]).decode()
    names = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0')
    files = {}
    for name in set(names) - {''}:
        if name.startswith('.h20-local/'):
            continue
        path = source / name
        if path.is_symlink():
            raise RuntimeError('Source symlink requires manual handling: ' + name)
        if path.is_file():
            files[name] = {'sha256': digest(path)}
    return {'head': git('rev-parse', 'HEAD').strip(), 'files': files}


def hash_of(files, name):
    return files.get(name, {}).get('sha256')


def install(source, base, incoming, manifest):
    baseline, desired = manifest['baseline'], manifest['files']
    current = inspect(source)
    if current['head'] != baseline['head']:
        raise RuntimeError('Server HEAD changed since sync/import; reconcile Git history first')
    conflicts = [name for name in sorted(current['files'].keys() | desired.keys() | baseline['files'].keys())
                 if hash_of(current['files'], name) not in
                 (hash_of(baseline['files'], name), hash_of(desired, name))]
    if conflicts:
        raise RuntimeError('Server edits must be merged locally before sync: ' + ', '.join(conflicts))
    changed = [name for name in sorted(desired.keys() | baseline['files'].keys())
               if hash_of(current['files'], name) != hash_of(desired, name)]
    for name in changed:
        validate_name(name)
        path = source / name
        if path.is_symlink() or source not in path.resolve().parents or (path.exists() and not path.is_file()):
            raise RuntimeError('Refusing to replace non-regular/redirected path: ' + name)
    backup = base / 'backups' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    (backup / 'before.json').write_text(json.dumps({'files': current['files'], 'changed': changed}, indent=2))
    with tarfile.open(incoming) as tf, tempfile.TemporaryDirectory(prefix='staging-', dir=base) as temp:
        staging = Path(temp)
        for name in desired:
            p = staging / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tf.extractfile(name).read())
        verify(staging, manifest)
        # Back up all affected files before changing any; leave Git index/HEAD untouched.
        for name in changed:
            path = source / name
            if path.is_file():
                dest = backup / 'files' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        for name in changed:
            path = source / name
            if name not in desired:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = path.stat().st_mode & 0o777 if path.exists() else desired[name]['mode']
                (staging / name).chmod(mode)
                os.replace(staging / name, path)
    verify(source, manifest)
    result = dict(mode='in-place', snapshot_id=manifest['snapshot_id'], path=str(source),
                  head=manifest['head'], remote_head=current['head'], files=desired,
                  changed=changed, backup=str(backup))
    (base / 'last-sync.json').write_text(json.dumps(result, indent=2))
    return result


def main(request):
    source = Path(request['source_root']).resolve()
    if Path(request['remote_root']).resolve() != source:
        raise RuntimeError('remote_root must be the original source repository')
    base = source / '.h20-local'
    if base.is_symlink():
        raise RuntimeError('Runtime directory must not be a symlink')
    operation = request['operation']
    if operation == 'inspect':
        return inspect(source)
    if operation == 'prepare':
        (base / 'incoming').mkdir(parents=True, exist_ok=True)
        exclude = source / '.git/info/exclude'
        if '.h20-local/' not in exclude.read_text():
            with exclude.open('a') as f:
                f.write('\n# Local-editor synchronization state, backups and logs\n.h20-local/\n')
        fd, name = tempfile.mkstemp(prefix='source-', suffix='.tar.gz', dir=base / 'incoming')
        os.close(fd)
        return {'incoming': name}
    if operation == 'install':
        incoming = Path(request['incoming']).resolve()
        if incoming.parent != base / 'incoming' or not incoming.name.startswith('source-'):
            raise RuntimeError('Invalid incoming archive')
        snapshot_id = request['snapshot_id']
        if not re.fullmatch('[a-f0-9]{20}', snapshot_id):
            raise RuntimeError('Invalid snapshot ID')
        with tarfile.open(incoming) as tf:
            members = tf.getmembers()
            if len({m.name for m in members}) != len(members):
                raise RuntimeError('Duplicate archive entries')
            manifest = json.load(tf.extractfile('.h20-manifest.json'))
            identity = {k: manifest[k] for k in ('files', 'head', 'source_root')}
            expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            if manifest['snapshot_id'] != snapshot_id or expected != snapshot_id:
                raise RuntimeError('Snapshot identity mismatch')
            if manifest['source_root'] != str(source):
                raise RuntimeError('Manifest source root differs from destination repository')
            allowed = set(manifest['files']) | {'.h20-manifest.json'}
            if {m.name for m in members} != allowed:
                raise RuntimeError('Archive differs from file manifest')
            for member in members:
                validate_name(member.name)
                if not member.isfile():
                    raise RuntimeError('Only regular source files are accepted')
        with (base / 'sync.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another sync or managed run is active; retry after it finishes')
            result = install(source, base, incoming, manifest)
        incoming.unlink()
        return result
    if operation == 'launch':
        snapshot_id = request['snapshot_id']
        if not re.fullmatch('[a-f0-9]{20}', snapshot_id):
            raise RuntimeError('Invalid snapshot ID')
        root = source
        manifest = json.loads((base / 'last-sync.json').read_text())
        if manifest['snapshot_id'] != snapshot_id:
            raise RuntimeError('Server sync version differs from local state; sync first')
        current = inspect(source)
        if current['head'] != manifest['remote_head'] or any(
            hash_of(current['files'], name) != hash_of(manifest['files'], name)
            for name in current['files'].keys() | manifest['files'].keys()
        ):
            raise RuntimeError('Server source changed after synchronization; reconcile and sync first')
        verify(root, manifest)
        env = Path(request['env_prefix'])
        if not (env / 'bin/python').is_file():
            raise RuntimeError('Configured Python environment does not exist')
        label = request.get('background') or 'foreground'
        if not re.fullmatch('[a-zA-Z0-9_-]{1,48}', label):
            raise RuntimeError('Run label must use letters, digits, underscore or hyphen')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + label + '-' + uuid.uuid4().hex[:6]
        run = base / 'runs' / run_id
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text(json.dumps(dict(snapshot_id=snapshot_id, source_head=manifest['head'],
                command=request['command'], env_prefix=str(env), source_root=str(root), files=manifest['files']), indent=2))
        command = shlex.join(request['command'])
        q = shlex.quote
        script = run / 'run.sh'
        script.write_text(
            '#!/usr/bin/env bash\nset -uo pipefail\n'
            'trap ' + q('rc=$?; printf "%s\\n" "$rc" > ' + q(str(run / 'exit-code'))) + ' EXIT\n'
            'exec 9>' + q(str(base / 'sync.lock')) + '\nflock -n 9 || exit 125\n'
            'cd ' + q(str(root)) + ' || exit 125\n'
            'export PATH=' + q(str(env / 'bin')) + ':"$PATH"\n'
            'export CONDA_PREFIX=' + q(str(env)) + '\n'
            'export PYTHONPATH=' + q(str(root)) + '\n'
            'export PYTHONDONTWRITEBYTECODE=1\n'
            'export OPENWAM_RUN_DIR=' + q(str(run)) + '\n'
            # W&B can explicitly opt in to the original server-side credentials; no file copied.
            'export WANDB_MODE="${WANDB_MODE:-disabled}"\n'
            'export WANDB_CONFIG_DIR=' + q(str(run / 'wandb-config')) + '\n'
            'python -c ' + q('import json,hashlib; from pathlib import Path; m=json.loads(Path(' + repr(str(run / 'manifest.json')) + ').read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v["sha256"] for p,v in m["files"].items()), "Source changed before execution"') + ' || exit 125\n'
            'set +e\n' + command + ' 2>&1 | tee ' + q(str(run / 'output.log')) + '\n'
            'rc=${PIPESTATUS[0]}\nprintf "%s\\n" "$rc" > ' + q(str(run / 'exit-code')) + '\nexit "$rc"\n')
        result = {'run_id': run_id, 'script': str(script), 'worktree': str(root), 'log': str(run / 'output.log')}
        if request.get('background'):
            session = 'openwam-' + run_id
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash ' + q(str(script))], check=True)
            result['tmux_session'] = session
        return result
    if operation == 'logs':
        run_id = request['run_id']
        if not re.fullmatch('[a-zA-Z0-9_-]+', run_id):
            raise RuntimeError('Invalid run ID')
        run = base / 'runs' / run_id
        if not (run / 'manifest.json').is_file():
            raise RuntimeError('Unknown run')
        log = run / 'output.log'
        if log.exists():
            subprocess.run(['tail', '-n', '80', str(log)], check=True)
        exit_file = run / 'exit-code'
        print('exit_code=' + (exit_file.read_text().strip() if exit_file.exists() else 'pending (or interrupted)'))
        return None
    raise RuntimeError('Unknown operation')


if __name__ == '__main__':
    result = main(json.loads(base64.b64decode(sys.argv[1])))
    if result is not None:
        print(json.dumps(result))
