"""Optional, training-only weighting of the first actions after a clean RTC prefix."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RTCSeamLossConfig:
    enabled: bool = False
    steps: int = 3
    weight: float = 2.0

    @classmethod
    def from_config(cls, cfg=None):
        value = cfg if isinstance(cfg, cls) else cls(**dict(cfg or {}))
        if type(value.enabled) is not bool:
            raise ValueError('rtc_seam_loss.enabled must be boolean')
        if type(value.steps) is not int or value.steps < 1:
            raise ValueError('rtc_seam_loss.steps must be a positive integer')
        if isinstance(value.weight, bool) or not isinstance(value.weight, (int, float)) or not math.isfinite(value.weight) or value.weight < 1:
            raise ValueError('rtc_seam_loss.weight must be finite and >= 1')
        return value


def configure_rtc_seam_loss(architecture, cfg=None, *, saved_cfg=None, resume=False):
    config = RTCSeamLossConfig.from_config(cfg)
    if resume:
        saved = RTCSeamLossConfig.from_config(saved_cfg)
        if (config.enabled or saved.enabled) and config != saved:
            raise ValueError('Resume must keep rtc_seam_loss; use finetune_ckpt_path for a new experiment')
    if config.enabled:
        from .training_rtc import require_rtc_architecture
        require_rtc_architecture(architecture)
        if not getattr(getattr(architecture, 'training_rtc', None), 'enabled', False):
            raise ValueError('RTC seam loss requires TrainingRTC enabled')
    architecture.rtc_seam_loss = config
    return config


def make_seam_weights(actions, original_pad, rtc_pad, config):
    """Infer committed length from newly masked cells, not from diffusion time=0.

    Returns None on the original path. No RNG or action/target/mask mutation.
    Empty-prefix samples keep unit weights, including in a mixed batch.
    """
    if config is None or not config.enabled or config.weight == 1 or rtc_pad is None:
        return None
    added = rtc_pad.to(device=actions.device, dtype=torch.bool)
    if original_pad is not None:
        added = added & ~original_pad.to(device=actions.device, dtype=torch.bool)
    if added.ndim == 3:
        added = added.any(-1)
    lengths = added.long().sum(-1)
    if not lengths.any():
        return None
    t = torch.arange(actions.shape[1], device=actions.device)[None, :]
    seam = (lengths[:, None] > 0) & (t >= lengths[:, None]) & (t < lengths[:, None] + config.steps)
    return torch.where(seam, config.weight, 1.0).float()


def weighted_action_mean(per_element, is_pad, step_weights):
    """Per-sample MSE: sum(valid * weight * error) / sum(valid * weight).

    This redistributes action supervision without multiplying its global scale.
    Scheduler weights and video loss stay with the existing caller.
    """
    if step_weights.shape != per_element.shape[:2]:
        raise ValueError('RTC seam weights must have shape [B,T]')
    weights = step_weights.to(device=per_element.device, dtype=torch.float32)[..., None].expand_as(per_element)
    if is_pad is not None:
        valid = ~is_pad.to(device=per_element.device, dtype=torch.bool)
        if valid.shape != per_element.shape:
            if valid.ndim == 3:
                valid = valid.any(-1)
            valid = valid[..., None].expand_as(per_element)
        weights = weights * valid.float()
    return (per_element * weights).sum((1, 2)) / weights.sum((1, 2)).clamp(min=1)
