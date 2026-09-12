import pytest

from scripts.statewam.initialize_checkpoint import load_original_parameters
from tests.statewam.test_statewam import tiny_architecture


def test_initialization_allows_only_state_keys():
    model = tiny_architecture()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("interaction_state.")}
    added = load_original_parameters(model, state)
    assert added and all(k.startswith("interaction_state.") for k in added)
    damaged = dict(state)
    damaged.pop(next(iter(damaged)))
    with pytest.raises(ValueError, match="key mismatch"):
        load_original_parameters(model, damaged)
    damaged = dict(state)
    name = next(iter(damaged))
    damaged[name] = damaged[name].flatten()[:1]
    with pytest.raises(ValueError, match="shape mismatch"):
        load_original_parameters(model, damaged)
