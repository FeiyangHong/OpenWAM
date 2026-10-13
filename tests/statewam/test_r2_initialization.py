import pytest
import torch
from safetensors.torch import save_file

from scripts.statewam.r2.initialize_matrix import load_exact_initialization
from scripts.statewam.r2.paired_data import tensor_hash


def test_initialization_retains_checkpoint_dtypes_and_values(tmp_path):
    original = torch.nn.Linear(3, 2).to(torch.bfloat16)
    path = tmp_path / "weights.safetensors"
    save_file(original.state_dict(), str(path))
    restored = torch.nn.Linear(3, 2)
    report = load_exact_initialization(restored, path)
    assert report["all_values_shapes_dtypes_identical"]
    assert report["tensor_count"] == 2
    for key, value in original.state_dict().items():
        assert tensor_hash(value) == tensor_hash(restored.state_dict()[key])


def test_initialization_rejects_missing_keys(tmp_path):
    path = tmp_path / "incomplete.safetensors"
    save_file({"weight": torch.zeros(2, 3)}, str(path))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_exact_initialization(torch.nn.Linear(3, 2), path)
