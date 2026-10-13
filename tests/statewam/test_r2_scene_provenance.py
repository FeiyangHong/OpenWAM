import zipfile

import pytest

from benchmarks.robotwin.r2_scene_provenance import archive_inventory, compare_inventory, extracted_inventory, generated_configuration_inventory


def test_actual_assets_match_archive_content_and_metadata_is_explicit(tmp_path):
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("objects/model.bin", b"model contents")
        stream.writestr("__MACOSX/._objects", b"metadata")
    root = tmp_path / "assets"
    (root / "objects").mkdir(parents=True)
    target = root / "objects/model.bin"
    target.write_bytes(b"model contents")
    expected, ignored = archive_inventory(archive, "objects")
    assert ignored == ["__MACOSX/._objects"]
    assert not any(compare_inventory(expected, extracted_inventory(root, "objects")).values())
    target.write_bytes(b"different contents")
    assert compare_inventory(expected, extracted_inventory(root, "objects"))["changed"] == ["objects/model.bin"]


@pytest.mark.parametrize("member", ["../model", "/objects/model", "other/model", "objects/../model", "objects\\model"])
def test_archive_paths_must_belong_to_expected_asset_tree(tmp_path, member):
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, b"content")
    with pytest.raises(ValueError):
        archive_inventory(archive, "objects")


def test_inventory_reports_missing_and_extra_and_refuses_nested_symlink(tmp_path):
    (tmp_path / "objects").mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"asset")
    (tmp_path / "objects/link").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        extracted_inventory(tmp_path, "objects")
    assert compare_inventory({"a": {}}, {"b": {}}) == {"missing": ["a"], "extra": ["b"], "changed": []}


def test_generated_configs_are_replayed_from_archived_templates_without_writes(tmp_path):
    import hashlib

    archive = tmp_path / "embodiments.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("embodiments/robot/curobo_tmp.yml", "urdf: ${ASSETS_PATH}/assets/robot\nroot: $ASSETS_PATH\n")
    root = tmp_path / "robotwin"
    expected = f"urdf: {root}/assets/robot\nroot: {root}\n".encode()
    records, derivations = generated_configuration_inventory(archive, root)
    assert records == {"embodiments/robot/curobo.yml": {"bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}}
    assert derivations["embodiments/robot/curobo.yml"]["template"] == "embodiments/robot/curobo_tmp.yml"
    assert not root.exists()
