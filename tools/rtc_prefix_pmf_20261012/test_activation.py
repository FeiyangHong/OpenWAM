"""Exercise activation in a disposable Git repository, never the training checkout."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def staged_repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    git('init', '-q')
    git('config', 'user.name', 'RTC test')
    git('config', 'user.email', 'rtc-test@example.invalid')
    (root / 'source.py').write_text('before\n')
    git('add', 'source.py')
    git('commit', '-qm', 'initial')
    head = git('rev-parse', 'HEAD')
    bundle = root / 'tools/rtc_prefix_pmf_20261012'
    (bundle / 'staged').mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name('activate.py'), bundle / 'activate.py')
    (bundle / 'staged/source.py').write_text('after\n')
    patch = '--- a/source.py\n+++ b/source.py\n@@ -1 +1 @@\n-before\n+after\n'
    (bundle / 'changes.patch').write_text(patch)
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    (bundle / 'manifest.json').write_text(json.dumps({
        'files': {'source.py': {'before': sha('before\n'), 'after': sha('after\n')}},
        'context': {}, 'patch_sha256': sha(patch),
    }))
    return root, bundle, git, head


def invoke(bundle, mode):
    return subprocess.run([sys.executable, str(bundle / 'activate.py'), mode], text=True, capture_output=True)


def test_check_is_readonly_and_apply_preserves_head_index(staged_repo):
    root, bundle, git, head = staged_repo
    (root / 'unrelated.txt').write_text('keep me\n')
    git('add', 'unrelated.txt')
    index = git('diff', '--cached', '--binary')
    checked = invoke(bundle, '--check')
    assert checked.returncode == 0, checked.stderr
    assert (root / 'source.py').read_text() == 'before\n'
    applied = invoke(bundle, '--apply')
    assert applied.returncode == 0, applied.stderr
    assert (root / 'source.py').read_text() == 'after\n'
    assert (root / 'unrelated.txt').read_text() == 'keep me\n'
    assert git('rev-parse', 'HEAD') == head
    assert git('diff', '--cached', '--binary') == index
    assert git('show', 'codex/rtc-prefix-pmf-applied-20261012:source.py') == 'after'
    assert invoke(bundle, '--apply').returncode == 0  # Idempotent.


def test_busy_training_lock_blocks_apply(staged_repo):
    root, bundle, _, _ = staged_repo
    lock = root / 'outputs/wuji_rtc_history/manual_training.lock'
    lock.parent.mkdir(parents=True)
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(bundle, '--apply')
    assert result.returncode != 0 and 'Training is still running' in result.stderr
    assert (root / 'source.py').read_text() == 'before\n'


def test_source_conflict_blocks_all_writes(staged_repo):
    root, bundle, _, _ = staged_repo
    (root / 'source.py').write_text('someone else changed this\n')
    result = invoke(bundle, '--apply')
    assert result.returncode != 0 and 'Source changed' in result.stderr
    assert (root / 'source.py').read_text() == 'someone else changed this\n'


def test_payload_corruption_blocks_apply(staged_repo):
    root, bundle, _, _ = staged_repo
    (bundle / 'staged/source.py').write_text('corrupted\n')
    result = invoke(bundle, '--apply')
    assert result.returncode != 0 and 'payload changed' in result.stderr
    assert (root / 'source.py').read_text() == 'before\n'
