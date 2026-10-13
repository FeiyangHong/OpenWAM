import pytest
import torch
from torch.nn import functional as F

from openwam.model.architectures.statewam.state_expert import CrossAttention
from scripts.statewam.r2.workload import WorkloadTrace, attention_geometry


def test_attention_workload_counts_broadcast_and_causal_masks():
    q = torch.zeros(2, 3, 4, 8)
    k = torch.zeros(2, 3, 5, 8)
    mask = torch.tensor([True, True, False, False, False])[None, None, None, :]
    row = attention_geometry(q, k, k, mask)
    assert row["dense_pairs"] == 120 and row["allowed_pairs"] == 48
    assert row["dense_qk_av_flops"] == 3840
    assert attention_geometry(q, k, k, causal=True)["allowed_pairs"] == 60
    additive = torch.where(mask, 0.0, -torch.inf)
    assert attention_geometry(q, k, k, additive)["allowed_pairs"] == 48
    with pytest.raises(ValueError, match="Combined"):
        attention_geometry(q, k, k, mask, causal=True)


def test_instrumentation_preserves_output_gradients_rng_and_restores_function():
    torch.manual_seed(10)
    model = CrossAttention(8, 2)
    q = torch.randn(2, 4, 8, requires_grad=True)
    source = torch.randn(2, 5, 8, requires_grad=True)
    output = model(q, source)
    expected = torch.autograd.grad(output.sum(), [q, source, *model.parameters()])
    original = F.scaled_dot_product_attention
    state = torch.get_rng_state().clone()
    with WorkloadTrace(model) as trace:
        with trace.during("forward"):
            observed = model(q, source)
        actual = torch.autograd.grad(observed.sum(), [q, source, *model.parameters()])
    assert F.scaled_dot_product_attention is original
    assert torch.equal(state, torch.get_rng_state()) and torch.equal(output, observed)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert len(trace.calls) == 1 and trace.calls[0]["source_tokens"] == 5
    assert trace.calls[0]["phase"] == "forward"
    with pytest.raises(RuntimeError):
        with WorkloadTrace(model):
            raise RuntimeError("abort")
    assert F.scaled_dot_product_attention is original
