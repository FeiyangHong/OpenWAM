"""Opt-in TrainingRTC: clean committed actions, token times, suffix supervision.

OpenWAM uses sigma=0 for clean data. No trainable parameters are added.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingRTCConfig:
    enabled: bool = False
    max_prefix_steps: int = 12
    zero_prefix_probability: float = 0.25

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            return cls()
        result = cls(**dict(cfg))
        if type(result.enabled) is not bool:
            raise ValueError("training.rtc.enabled must be boolean")
        if type(result.max_prefix_steps) is not int or result.max_prefix_steps < 0:
            raise ValueError("training.rtc.max_prefix_steps must be a non-negative integer")
        if not 0 <= result.zero_prefix_probability <= 1:
            raise ValueError("training.rtc.zero_prefix_probability must be in [0, 1]")
        return result


def require_rtc_architecture(architecture):
    # Exact type intentionally excludes StateWAM (a subclass of this class).
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

    if type(architecture) is not DualSystemSelfAttnArchitecture:
        raise ValueError("TrainingRTC currently supports only dual_system_self_attn, not StateWAM or other architectures")


def configure_training_rtc(architecture, cfg=None, *, saved_cfg=None, resume=False):
    config = TrainingRTCConfig.from_config(cfg)
    if resume:
        saved = TrainingRTCConfig.from_config(saved_cfg)
        if (config.enabled or saved.enabled) and config != saved:
            raise ValueError("Resume must keep the saved training.rtc config; use finetune_ckpt_path for a new RTC run")
    if config.enabled:
        require_rtc_architecture(architecture)
    architecture.training_rtc = config
    return config


def make_training_prefix(actions, noisy_actions, timesteps, is_pad, config, prefix_lengths=None, *, sampling_config=None):
    """Return noisy input, forward times and loss mask without changing loss weights.

    Prefixes never include padding or consume the last valid supervised action.
    Random draws use the standard CPU generator, captured by checkpoint RNG state.
    """
    batch, horizon, _ = actions.shape
    if config.max_prefix_steps >= horizon:
        raise ValueError("training.rtc.max_prefix_steps must be smaller than the action horizon")
    if is_pad is None:
        pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
    else:
        pad = is_pad.to(device=actions.device, dtype=torch.bool)
        if pad.shape not in (actions.shape, actions.shape[:2]):
            raise ValueError("RTC action_is_pad must have shape [B,T] or [B,T,D]")
    valid = ~(pad.all(-1) if pad.ndim == 3 else pad)
    # Length of the contiguous valid prefix; robust to episode tails/interior holes.
    usable = valid.long().cumprod(dim=1).sum(dim=1)
    limits = (usable - 1).clamp(min=0, max=config.max_prefix_steps)
    if prefix_lengths is None:
        if sampling_config is not None and sampling_config.enabled:
            from .rtc_prefix_sampling import sample_prefix_lengths
            lengths = sample_prefix_lengths(batch, config.max_prefix_steps, sampling_config)
            lengths = torch.minimum(lengths.to(actions.device), limits)
        elif config.max_prefix_steps == 0 or config.zero_prefix_probability == 1:
            lengths = torch.zeros(batch, dtype=torch.long, device=actions.device)
        else:
            lengths = torch.randint(1, config.max_prefix_steps + 1, (batch,))
            lengths[torch.rand(batch) < config.zero_prefix_probability] = 0
            lengths = torch.minimum(lengths.to(actions.device), limits)
    else:
        lengths = torch.as_tensor(prefix_lengths, device=actions.device)
        if lengths.shape != (batch,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("rtc_prefix_lengths must be integer [B]")
        if ((lengths < 0) | (lengths > limits)).any():
            raise ValueError("rtc_prefix_lengths exceeds training limit or valid action suffix")
    if not lengths.any():
        # Preserve the exact original scalar-time path for all-zero prefixes.
        return noisy_actions, timesteps, is_pad
    prefix = torch.arange(horizon, device=actions.device)[None, :] < lengths[:, None]
    noisy_actions = torch.where(prefix[..., None], actions, noisy_actions)
    token_times = timesteps[:, None].expand(batch, horizon).masked_fill(prefix, 0)
    loss_pad = pad | (prefix[..., None] if pad.ndim == 3 else prefix)
    return noisy_actions, token_times, loss_pad


def prepare_inference_prefix(architecture, prefix, horizon, device, dtype):
    """Validate an already normalized model-space [d,D] prefix."""
    require_rtc_architecture(architecture)
    config = getattr(architecture, "training_rtc", TrainingRTCConfig())
    if not config.enabled:
        raise ValueError("RTC prefix requires a checkpoint trained with training.rtc.enabled=true")
    prefix = torch.as_tensor(prefix, device=device, dtype=dtype)
    if prefix.ndim != 2 or prefix.shape[1] != architecture.action_dim:
        raise ValueError("rtc_prefix must have shape [d, model_action_dim]")
    if not 0 < len(prefix) <= config.max_prefix_steps or len(prefix) >= horizon:
        raise ValueError("RTC prefix must be nonempty, within the trained limit, and leave an action suffix")
    if not torch.isfinite(prefix).all():
        raise ValueError("RTC prefix must be finite")
    return prefix.unsqueeze(0)
