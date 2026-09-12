"""Small invocation-local state readers and action attention writers."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("State width must be divisible by its head count")
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query, source, valid=None):
        b, nq, dim = query.shape
        if valid is None:
            valid = torch.ones(source.shape[:2], dtype=torch.bool, device=source.device)
        # Mask before projecting: padded values (including NaNs) must be inert.
        source = source.masked_fill(~valid[..., None], 0)
        q = self.q(query).view(b, nq, self.heads, -1).transpose(1, 2)
        k = self.k(source).view(b, -1, self.heads, dim // self.heads).transpose(1, 2)
        v = self.v(source).view(b, -1, self.heads, dim // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        out = self.out(out.transpose(1, 2).reshape(b, nq, dim))
        return out * valid.any(dim=-1)[:, None, None].to(out.dtype)


class InteractionStateLayer(nn.Module):
    def __init__(self, video_dim, action_dim, attention_dim, dim, num_tokens, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_tokens, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.video_proj = nn.Linear(video_dim, dim)
        self.context_proj = nn.Linear(video_dim, dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.position_proj = nn.Linear(6, dim, bias=False)
        self.noise_proj = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.read = CrossAttention(dim, heads)
        self.read_context = CrossAttention(dim, heads)
        self.read_action = CrossAttention(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.write_query = nn.Linear(action_dim, dim)
        self.write = CrossAttention(dim, heads)
        self.output = nn.Linear(dim, attention_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def read_state(self, video, context, actions, *, positions, noise, video_valid, context_valid, action_valid):
        video = video.masked_fill(~video_valid[..., None], 0)
        video = self.video_proj(video) + self.position_proj(positions).unsqueeze(0)
        z = self.queries.unsqueeze(0).expand(video.shape[0], -1, -1)
        z = z + self.noise_proj(noise).unsqueeze(1)
        z = z + self.read(z, video, video_valid)
        if context is not None:
            if context_valid is not None:
                context = context.masked_fill(~context_valid[..., None], 0)
            z = z + self.read_context(z, self.context_proj(context), context_valid)
        if actions is not None:
            if action_valid is not None:
                actions = actions.masked_fill(~action_valid[..., None], 0)
            z = z + self.read_action(z, self.action_proj(actions), action_valid)
        z = self.norm(z)
        return z + self.ffn(z)

    def forward(self, video, context, actions, writer_actions, **kwargs):
        z = self.read_state(video, context, actions, **kwargs)
        delta = self.output(self.write(self.write_query(writer_actions), z))
        return self.gate.tanh() * delta


class InteractionStateExpert(nn.Module):
    SOURCES = ("off", "current", "future", "future_action")

    def __init__(self, *, video_dim, action_dim, attention_dim, num_layers, cfg):
        super().__init__()
        self.source = str(cfg.get("source", "future_action"))
        if self.source not in self.SOURCES:
            raise ValueError(f"Unknown State source: {self.source}")
        self.capacity_control = str(cfg.get("capacity_control", "none"))
        if self.capacity_control not in ("none", "current_resample"):
            raise ValueError(f"Unknown State capacity control: {self.capacity_control}")
        if self.capacity_control == "current_resample" and self.source not in ("current", "future"):
            raise ValueError("current_resample is a current-only control for current/future readers")
        if cfg.get("mode", "residual") != "residual":
            raise ValueError("R1 supports residual State only")
        if cfg.get("cache_across_denoise_steps", False) or cfg.get("semantic_loss_weight", 0):
            raise ValueError("R1 does not enable State caching or semantic loss")
        if float(cfg.get("gate_init", 0)) != 0:
            raise ValueError("State gates must initialize at zero")
        layers = cfg.get("state_bridge_layers")
        if layers is None:
            if num_layers != 30:
                raise ValueError("Non-30-layer backbones require explicit state_bridge_layers")
            layers = [4, 9, 14, 19, 24, 29]
        if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= num_layers for i in layers):
            raise ValueError("state_bridge_layers must be unique, nonempty and inside the backbone")
        self.layers = nn.ModuleDict({
            str(i): InteractionStateLayer(
                video_dim, action_dim, attention_dim, int(cfg.get("dim", 512)),
                int(cfg.get("num_tokens", 16)), int(cfg.get("num_heads", 8)),
            ) for i in layers
        })

    @staticmethod
    def current_capacity_source(video, valid, *, tokens, dim):
        """Parameter-free, masked pooling into the action reader's input geometry."""
        video = video.masked_fill(~valid[..., None], 0)
        pooled = F.adaptive_avg_pool2d(video.unsqueeze(1), (tokens, dim)).squeeze(1)
        weights = F.adaptive_avg_pool1d(valid.to(video.dtype).unsqueeze(1), tokens).transpose(1, 2)
        pooled = pooled / weights.clamp_min(torch.finfo(video.dtype).tiny)
        return pooled, weights.squeeze(-1) > 0

    @staticmethod
    def positions(frames, height, width, *, device, dtype):
        axes = [torch.linspace(0, 1, n, device=device, dtype=dtype) for n in (frames, height, width)]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        return torch.cat([torch.sin(math.pi * coordinates), torch.cos(math.pi * coordinates)], dim=-1)
