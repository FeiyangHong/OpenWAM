"""Run on Linux: python -m unittest discover -s tests/dev -v."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from scripts.dev.h20_remote import main, validate_name


class InPlaceSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original'
        self.source.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        (self.source / 'untouched.txt').write_text('original work')
        subprocess.run(['git', '-C', str(self.source), 'add', 'untouched.txt'], check=True)
        subprocess.run(['git', '-C', str(self.source), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'test fixture'], check=True)
        (self.source / 'data').mkdir()
        (self.source / 'assets/openwam_ckpt').mkdir(parents=True)
        self.config = dict(source_root=str(self.source), remote_root=str(self.source),
                           env_prefix=sys.prefix)
        self.baseline = main(dict(self.config, operation='inspect'))

    def install(self, content=b'print("snapshot")\n', name='hello.py'):
        desired = {p: dict(v, mode=0o644) for p, v in self.baseline['files'].items()}
        desired[name] = dict(sha256=hashlib.sha256(content).hexdigest(), mode=0o644)
        identity = dict(files=desired,
                        head='example-commit', source_root=str(self.source))
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        manifest = dict(identity, snapshot_id=sid, baseline=self.baseline)
        incoming = main(dict(self.config, operation='prepare'))['incoming']
        with tarfile.open(incoming, 'w:gz') as tf:
            entries = [(p, content if p == name else (self.source / p).read_bytes()) for p in desired]
            for member_name, data in entries + [('.h20-manifest.json', json.dumps(manifest).encode())]:
                member = tarfile.TarInfo(member_name)
                member.size = len(data)
                tf.addfile(member, io.BytesIO(data))
        return main(dict(self.config, operation='install', incoming=incoming, snapshot_id=sid))

    def test_install_uses_original_and_preserves_data(self):
        result = self.install()
        snapshot = Path(result['path'])
        self.assertEqual((snapshot / 'hello.py').read_text(), 'print("snapshot")\n')
        self.assertEqual((snapshot / 'data').resolve(), self.source / 'data')
        self.assertEqual((snapshot / 'assets/openwam_ckpt').resolve(), self.source / 'assets/openwam_ckpt')
        self.assertFalse((snapshot / 'wandb').exists())
        self.assertFalse((snapshot / 'outputs').exists())
        self.assertEqual(snapshot, self.source)
        self.assertFalse((snapshot / 'data').is_symlink())
        self.assertTrue((self.source / 'hello.py').exists())
        self.assertEqual((self.source / 'untouched.txt').read_text(), 'original work')

    def test_repeat_sync_does_not_change_identical_files(self):
        self.assertEqual(self.install()['path'], self.install()['path'])

    def test_remote_edits_are_refused_without_overwrite(self):
        result = self.install()
        edited = Path(result['path']) / 'hello.py'
        edited.write_text('remote work')
        with self.assertRaisesRegex(RuntimeError, 'must be merged'):
            self.install()
        self.assertEqual(edited.read_text(), 'remote work')

    def test_local_edit_backs_up_previous_contents(self):
        first = self.install()
        self.baseline = main(dict(self.config, operation='inspect'))
        second = self.install(b'print("new")\n')
        self.assertEqual(first['path'], second['path'])
        self.assertEqual((Path(second['backup']) / 'files/hello.py').read_text(), 'print("snapshot")\n')
        self.assertEqual((self.source / 'hello.py').read_text(), 'print("new")\n')

    def test_traversal_and_private_metadata_are_rejected(self):
        for name in ('../outside', '/absolute', '.git/config', '.codex/auth.json', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                validate_name(name)
        with self.assertRaises(RuntimeError):
            self.install(name='../outside')
        self.assertFalse((self.root / 'outside').exists())

    def test_run_records_nonzero_exit_and_uses_original_imports(self):
        result = self.install()
        launched = main(dict(self.config, operation='launch', snapshot_id=result['snapshot_id'],
                            command=['python', '-c', 'import hello; raise SystemExit(7)'], background=None))
        process = subprocess.run(['bash', launched['script']], capture_output=True, text=True)
        self.assertEqual(process.returncode, 7, process.stderr)
        self.assertIn('snapshot', process.stdout)
        run = Path(launched['script']).parent
        self.assertEqual((run / 'exit-code').read_text().strip(), '7')
        self.assertIn('snapshot', (run / 'output.log').read_text())

    def test_unrelated_remote_addition_blocks_sync(self):
        (self.source / 'server-only.py').write_text('server work')
        with self.assertRaisesRegex(RuntimeError, 'must be merged'):
            self.install()
        self.assertFalse((self.source / 'hello.py').exists())

    def test_active_managed_run_blocks_sync(self):
        import fcntl
        main(dict(self.config, operation='prepare'))
        with (self.source / '.h20-local/sync.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'active'):
                self.install()


if __name__ == '__main__':
    unittest.main()
