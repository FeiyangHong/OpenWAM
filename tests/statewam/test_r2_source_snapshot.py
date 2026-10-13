import pytest

from scripts.statewam.r2.source_snapshot import assert_sources_unchanged, capture_sources


def test_snapshot_preserves_original_and_detects_drift(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    source = root / "scripts/train.py"
    source.write_text("print('original')\n")
    snapshot = tmp_path / "snapshot"
    manifest = capture_sources(snapshot, root=root)
    assert_sources_unchanged(manifest, root=root)
    source.write_text("print('changed')\n")
    assert (snapshot / "scripts/train.py").read_text() == "print('original')\n"
    with pytest.raises(ValueError, match="source changed"):
        assert_sources_unchanged(manifest, root=root)
    with pytest.raises(FileExistsError):
        capture_sources(snapshot, root=root)
