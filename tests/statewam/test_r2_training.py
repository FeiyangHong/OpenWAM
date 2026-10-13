import torch

from openwam.model.action_backbone.scheduler import ActionScheduler
from scripts.statewam.r2.matrix import apply_parameter_policy
from scripts.statewam.r2.paired_data import flow_randomness
from scripts.statewam.r2.train import LossModule
from scripts.statewam.r2.workload import WorkloadTrace
from tests.statewam.test_statewam import inputs, tiny_architecture


def test_loss_wrapper_uses_paired_flow_and_preserves_gradient_graph():
    arch = tiny_architecture()
    apply_parameter_policy(arch)
    arch.video_backbone._scheduler = ActionScheduler()
    arch.init_training_schedulers(1000)
    batch = inputs()
    batch["input_latents"] = batch.pop("latents")
    batch["actions"] = batch.pop("noisy_actions")
    batch.pop("action_timestep")
    batch.pop("timestep")
    randomness = flow_randomness(42, [0, 1], video=batch["input_latents"], actions=batch["actions"])
    wrapped = LossModule(arch)
    result = wrapped(batch, randomness)
    assert result["loss"].requires_grad
    result["loss"].backward()
    assert all(p.grad is None for p in arch.video_backbone.parameters())
    assert all(layer.gate.grad is not None and torch.isfinite(layer.gate.grad)
               for layer in arch.interaction_state.layers.values())
    assert any(p.grad is not None for p in arch.action_backbone.parameters())


def test_real_architecture_trace_captures_state_sources_and_checkpoint_recomputation():
    model = tiny_architecture()
    data = inputs()
    with WorkloadTrace(model) as trace:
        with trace.during("forward"):
            outputs = model(**data, use_gradient_checkpointing=True)
        with trace.during("backward_recomputation"):
            sum(value.square().mean() for value in outputs).backward()
    assert trace.sources and all(row["action_source_shape"] is not None for row in trace.sources)
    assert {row["phase"] for row in trace.calls} == {"forward", "backward_recomputation"}
    assert any("read_action" in row["scope"] for row in trace.calls)
    assert trace.scope == []
