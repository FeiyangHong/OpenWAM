import numpy as np
import pytest
import torch

from scripts.statewam.r2.audit_model_resume import require_exact


def test_nested_full_state_checks_rng_and_optimizer_not_only_weights():
    def state():
        return {"model": torch.ones(3, dtype=torch.bfloat16),
                "optimizer": {"moments": [torch.arange(4), 2]},
                "rng": (np.arange(5, dtype=np.uint32), None)}
    a, b = state(), state()
    require_exact(a, b)
    b["optimizer"]["moments"][0][2] += 1
    with pytest.raises(ValueError, match="optimizer.*tensor bytes differ"):
        require_exact(a, b)
    b = state()
    b["rng"][0][1] += 1
    with pytest.raises(ValueError, match="rng.*array differs"):
        require_exact(a, b)


def test_tensor_dtype_and_signed_zero_are_exact():
    with pytest.raises(ValueError, match="dtype/shape"):
        require_exact(torch.ones(1), torch.ones(1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bytes differ"):
        require_exact(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_deepspeed_fragment_addresses_compare_each_field():
    from deepspeed.utils.tensor_fragment import fragment_address

    a = fragment_address(numel=32, start=16)
    require_exact(a, fragment_address(numel=32, start=16))
    with pytest.raises(ValueError, match="attributes.start"):
        require_exact(a, fragment_address(numel=32, start=17))
    with pytest.raises(ValueError, match="attributes.numel"):
        require_exact(a, fragment_address(numel=31, start=16))
    b = fragment_address(numel=32, start=16)
    b.extra = 1
    with pytest.raises(ValueError, match="unexpected fragment"):
        require_exact(a, b)
