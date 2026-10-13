"""Opt-in training-only, shared bimanual XYZ observation perturbation.

The same physical translation is applied to both EEF positions. Actions,
rotations, hands and images are untouched. This is observation-bias augmentation,
not a simulation of mechanical lag or a correction for emergency hardware.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RTCStateAugmentationConfig:
    enabled: bool = False
    probability: float = 0.10
    max_translation_m: float = 0.005

    @classmethod
    def from_config(cls, cfg=None):
        result = cfg if isinstance(cfg, cls) else cls(**dict(cfg or {}))
        if type(result.enabled) is not bool:
            raise ValueError("rtc_state_augmentation.enabled must be boolean")
        if not math.isfinite(result.probability) or not 0 <= result.probability <= 1:
            raise ValueError("rtc_state_augmentation.probability must be in [0, 1]")
        if not math.isfinite(result.max_translation_m) or result.max_translation_m < 0:
            raise ValueError("rtc_state_augmentation.max_translation_m must be finite and non-negative")
        return result


class RTCStateAugmenter:
    """Plain callable: no parameters/buffers added to model checkpoints."""

    def __init__(self, config, xyz_indices, normalized_units_per_metre, state_dim):
        self.config = RTCStateAugmentationConfig.from_config(config)
        self.xyz_indices = torch.as_tensor(xyz_indices, dtype=torch.long)
        self.scale = torch.as_tensor(normalized_units_per_metre, dtype=torch.float32)
        self.state_dim = int(state_dim)
        if self.xyz_indices.shape != (2, 3) or self.scale.shape != (2, 3):
            raise ValueError("RTC XYZ augmentation requires exactly two EEF XYZ groups")
        if self.xyz_indices.unique().numel() != 6 or self.xyz_indices.min() < 0 or self.xyz_indices.max() >= state_dim:
            raise ValueError("Invalid RTC XYZ indices")
        if not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("Invalid RTC XYZ normalization scale")

    def __call__(self, proprio, has_prefix, proprio_mask=None):
        cfg = self.config
        # Preserve tensor identity and RNG when augmentation is inactive.
        if not cfg.enabled or cfg.probability == 0 or cfg.max_translation_m == 0:
            return proprio
        if proprio is None:
            raise ValueError("RTC state augmentation requires proprioception")
        if proprio.ndim not in (2, 3) or (proprio.ndim == 3 and proprio.shape[1] != 1) or proprio.shape[-1] != self.state_dim:
            raise ValueError("RTC state augmentation requires [B,D] or [B,1,D] state")
        batch = proprio.shape[0]
        eligible = torch.as_tensor(has_prefix, dtype=torch.bool, device='cpu').reshape(-1).clone()
        if eligible.shape != (batch,):
            raise ValueError("RTC prefix eligibility must have shape [B]")
        if proprio_mask is not None:
            mask = proprio_mask.detach().to(device='cpu', dtype=torch.bool)
            if mask.numel() == batch:
                eligible &= mask.reshape(batch)
            elif mask.shape in ((batch, self.state_dim), (batch, 1, self.state_dim)):
                eligible &= mask.reshape(batch, -1)[:, self.xyz_indices.flatten()].all(dim=-1)
            else:
                raise ValueError("Unsupported RTC proprio mask shape")
        if not eligible.any():
            return proprio
        selected = eligible & (torch.rand(batch) < cfg.probability)
        if not selected.any():
            return proprio
        # CPU RNG participates in existing rank/per-step seeds and resume state.
        # Uniform direction and uniform radius: more weight on small biases.
        direction = torch.randn(batch, 3)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta_m = direction * (torch.rand(batch, 1) * cfg.max_translation_m)
        delta_m *= selected[:, None]
        delta = (delta_m[:, None, :] * self.scale[None]).reshape(batch, 6)
        result = proprio.clone()
        flat = result.reshape(batch, self.state_dim)
        idx = self.xyz_indices.flatten().to(proprio.device)
        values = (flat[:, idx].float() + delta.to(proprio.device)).clamp(-1, 1).to(proprio.dtype)
        # Unselected rows remain bitwise identical, including out-of-range values.
        flat[:, idx] = torch.where(selected.to(proprio.device)[:, None], values, flat[:, idx])
        return result


def build_rtc_state_augmenter(dataset, config):
    """Derive physical-unit scales and scatter indices from the actual reader."""
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset

    if not isinstance(dataset, WujiRealTaskDataset) or dataset.rot6d_convention != 'column':
        raise ValueError("RTC state augmentation currently requires Wuji absolute EEF column data")
    if dataset._normalize_mode != 'quantile':
        raise ValueError("RTC state augmentation currently requires quantile normalization")
    raw_xyz = np.array([[0, 1, 2], [29, 30, 31]])
    stats = dataset._normalization_stats
    span = np.asarray(stats['q99']) - np.asarray(stats['q01'])
    if span.shape != (58,) or not np.isfinite(span[raw_xyz]).all() or not (span[raw_xyz] > 1e-6).all():
        raise ValueError("Invalid Wuji XYZ normalization statistics")
    if dataset._unify:
        mapping = np.asarray(dataset._unify_state_dst_index)
        if mapping.shape != (58,):
            raise ValueError("Invalid Wuji action/state scatter map")
        xyz_indices = mapping[raw_xyz]
        state_dim = dataset._unify_dim
    else:
        xyz_indices, state_dim = raw_xyz, 58
    return RTCStateAugmenter(config, xyz_indices, 2.0 / span[raw_xyz], state_dim)


def configure_rtc_state_augmentation(architecture, cfg=None, *, dataset=None, saved_cfg=None, resume=False):
    config = RTCStateAugmentationConfig.from_config(cfg)
    if resume and config != RTCStateAugmentationConfig.from_config(saved_cfg):
        raise ValueError("Resume must keep rtc_state_augmentation; use finetune_ckpt_path for a new experiment")
    if config.enabled and not getattr(getattr(architecture, 'training_rtc', None), 'enabled', False):
        raise ValueError("RTC state augmentation requires training.rtc.enabled=true")
    architecture.rtc_state_augmentation = config
    architecture.rtc_state_augmenter = build_rtc_state_augmenter(dataset, config) if config.enabled else None
    return config
