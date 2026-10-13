import pytest
import torch
from safetensors.torch import save_file

from scripts.statewam.r2.export_checkpoint import deployment_tensors


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "initial.safetensors"
    save_file({"layer.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}, str(path))
    state = {"module": {"architecture.layer.weight": torch.ones(2, 3, dtype=torch.bfloat16)},
             "mp_world_size": 1, "dp_world_size": 4}
    return path, state


def test_export_preserves_trained_tensor_values_and_strips_wrapper(weights):
    path, state = weights
    exported = deployment_tensors(state, path)
    assert set(exported) == {"layer.weight"}
    assert torch.equal(exported["layer.weight"], state["module"]["architecture.layer.weight"])


@pytest.mark.parametrize("case", ["missing", "extra", "shape", "dtype", "nonfinite", "prefix", "world"])
def test_export_rejects_partial_or_invalid_model_state(weights, case):
    path, state = weights
    if case == "missing":
        state["module"].clear()
    elif case == "extra":
        state["module"]["architecture.extra"] = torch.zeros(1, dtype=torch.bfloat16)
    elif case == "shape":
        state["module"]["architecture.layer.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    elif case == "dtype":
        state["module"]["architecture.layer.weight"] = torch.ones(2, 3)
    elif case == "nonfinite":
        state["module"]["architecture.layer.weight"][0, 0] = float("nan")
    elif case == "prefix":
        state["module"]["layer.weight"] = state["module"].pop("architecture.layer.weight")
    else:
        state["dp_world_size"] = 2
    with pytest.raises(ValueError):
        deployment_tensors(state, path)
