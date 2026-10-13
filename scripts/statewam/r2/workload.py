"""Diagnostic SDPA geometry and State-source tracing; not a kernel FLOPs profiler."""

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def attention_geometry(query, key, value, mask=None, *, causal=False):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Workload audit requires explicit batch/head/token/channel axes")
    batch, heads, queries, dim = query.shape
    keys = key.shape[-2]
    if key.shape[:2] != query.shape[:2] or value.shape[:3] != key.shape[:3] or key.shape[-1] != dim:
        raise ValueError("Unexpected SDPA or grouped-query geometry")
    dense_pairs = batch * heads * queries * keys
    if mask is not None:
        if causal:
            raise ValueError("Combined explicit and causal mask not audited")
        if mask.dtype == torch.bool:
            allowed = mask
        else:
            if torch.isnan(mask).any() or torch.isposinf(mask).any():
                raise ValueError("Invalid additive attention mask")
            allowed = ~torch.isneginf(mask)
        target = (batch, heads, queries, keys)
        if torch.broadcast_shapes(allowed.shape, target) != target:
            raise ValueError("Mask does not broadcast to attention geometry")
        factor = math.prod(target) // allowed.numel()
        valid_pairs = int(allowed.detach().sum().item()) * factor
    elif causal:
        triangle = min(queries, keys)
        valid_pairs = batch * heads * (triangle * (triangle + 1) // 2 + max(queries - keys, 0) * keys)
    else:
        valid_pairs = dense_pairs
    return {"batch": batch, "heads": heads, "query_tokens": queries, "source_tokens": keys,
            "query_key_dim": dim, "value_dim": value.shape[-1], "dtype": str(query.dtype),
            "dense_pairs": dense_pairs, "allowed_pairs": valid_pairs,
            "dense_qk_av_flops": 2 * dense_pairs * (dim + value.shape[-1]),
            "allowed_qk_av_flops": 2 * valid_pairs * (dim + value.shape[-1])}


class WorkloadTrace:
    """Process-local instrumentation, restored even when the model raises."""

    def __init__(self, architecture):
        self.architecture = architecture
        self.calls = []
        self.sources = []
        self.phase = "unset"
        self.scope = []
        self.hooks = []
        self.original = None

    @contextmanager
    def during(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def __enter__(self):
        from openwam.model.architectures.statewam.state_expert import CrossAttention, InteractionStateLayer

        if getattr(F.scaled_dot_product_attention, "_r2_trace", False):
            raise ValueError("Nested SDPA instrumentation is not supported")
        self.original = F.scaled_dot_product_attention

        def traced(query, key, value, *args, **kwargs):
            mask = args[0] if args else kwargs.get("attn_mask")
            causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
            if len(self.calls) >= 10000:
                raise ValueError("Diagnostic attention call limit exceeded")
            row = attention_geometry(query, key, value, mask, causal=causal)
            row.update(phase=self.phase, scope=self.scope[-1] if self.scope else "backbone_or_preprocessing")
            self.calls.append(row)
            return self.original(query, key, value, *args, **kwargs)

        traced._r2_trace = True
        F.scaled_dot_product_attention = traced
        for name, module in self.architecture.named_modules():
            if isinstance(module, CrossAttention):
                self.hooks.append(module.register_forward_pre_hook(lambda _module, _args, name=name: self.scope.append(name)))
                self.hooks.append(module.register_forward_hook(lambda _module, _args, _output: self._leave(), always_call=True))
            elif isinstance(module, InteractionStateLayer):
                def source(_module, args, kwargs, name=name):
                    video, context, actions, writer = args
                    row = {"layer": name, "phase": self.phase, "video_shape": list(video.shape),
                           "context_shape": list(context.shape) if context is not None else None,
                           "action_source_shape": list(actions.shape) if actions is not None else None,
                           "writer_shape": list(writer.shape)}
                    for key in ("video_valid", "context_valid", "action_valid"):
                        mask = kwargs.get(key)
                        row[key] = mask.detach().sum(dim=-1).cpu().tolist() if mask is not None else None
                    self.sources.append(row)
                self.hooks.append(module.register_forward_pre_hook(source, with_kwargs=True))
        return self

    def _leave(self):
        self.scope.pop()

    def __exit__(self, *error):
        F.scaled_dot_product_attention = self.original
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def report(self):
        return {"sdpa_calls": self.calls, "state_sources": self.sources,
                "limits": "Observed SDPA calls only. Backward entries are checkpoint forward recomputation, not backward kernel FLOPs. QK/AV arithmetic excludes projection, softmax, normalization and FFN. Allowed pairs describe the mask, not sparse kernel execution. Instrumented timings are not steady throughput."}
