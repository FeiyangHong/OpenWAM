"""Opt-in, prefix-conditioned replacement by a historical full state.

History is read within the same episode, normalized by the existing reader,
and selected only after RTC prefix sampling. No action/image/weight mutation.
"""
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class RTCStateHistoryConfig:
    enabled: bool = False
    probability: float = 0.30
    max_history_steps: int = 3
    max_translation_m: Optional[float] = 0.025
    max_rotation_deg: Optional[float] = 3.0
    max_joint_rad: Optional[float] = 0.05

    @classmethod
    def from_config(cls, cfg=None):
        value = cfg if isinstance(cfg, cls) else cls(**dict(cfg or {}))
        if type(value.enabled) is not bool:
            raise ValueError('rtc_state_history.enabled must be boolean')
        if not math.isfinite(value.probability) or not 0 <= value.probability <= 1:
            raise ValueError('rtc_state_history.probability must be in [0, 1]')
        if type(value.max_history_steps) is not int or not 1 <= value.max_history_steps <= 3:
            raise ValueError('rtc_state_history.max_history_steps must be 1, 2 or 3')
        for name in ('max_translation_m', 'max_rotation_deg', 'max_joint_rad'):
            limit = getattr(value, name)
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError('rtc_state_history.'+name+' must be null or finite and positive')
        return value


def rotation_distance_deg(a, b):
    """Geodesic angle after Gram-Schmidt for column ROT6D; invalid -> inf."""
    def matrix(v):
        v = np.asarray(v, np.float64)
        x = v[..., :3]
        nx = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.maximum(nx, 1e-12)
        y = v[..., 3:] - (v[..., 3:] * x).sum(-1, keepdims=True) * x
        ny = np.linalg.norm(y, axis=-1, keepdims=True)
        y = y / np.maximum(ny, 1e-12)
        return np.stack((x, y, np.cross(x, y)), -1), ((nx > 1e-8) & (ny > 1e-8))[..., 0]
    ra, va = matrix(a)
    rb, vb = matrix(b)
    angle = np.degrees(np.arccos(np.clip(((ra * rb).sum((-2, -1))-1)/2, -1, 1)))
    return np.where(va & vb, angle, np.inf)


def bounded_history(current, past, available, config):
    """past[k-1]=S(t-k). Optional symmetric caps; None disables a cap."""
    cfg = RTCStateHistoryConfig.from_config(config)
    current, past = np.asarray(current), np.asarray(past)
    valid = np.asarray(available, dtype=bool).copy()
    valid &= np.isfinite(past).all(-1) & np.isfinite(current).all()
    for base in (0, 29):
        if cfg.max_translation_m is not None:
            valid &= np.linalg.norm(past[:, base:base+3]-current[base:base+3], axis=-1) <= cfg.max_translation_m
        angle = rotation_distance_deg(past[:, base+3:base+9], current[base+3:base+9])
        valid &= np.isfinite(angle)
        if cfg.max_rotation_deg is not None:
            valid &= angle <= cfg.max_rotation_deg
        if cfg.max_joint_rad is not None:
            valid &= np.max(np.abs(past[:, base+9:base+29]-current[base+9:base+29]), axis=-1) <= cfg.max_joint_rad
    return valid


def read_history(dataset, idx):
    """Read at most four cached parquet rows; never decode historical images."""
    cfg = dataset.rtc_state_history
    kmax = cfg.max_history_steps
    ep = int(np.searchsorted(dataset._cum_n_starts, idx, side='right')-1)
    start = int(dataset._ep_valid_start[ep])
    offset = start + (idx-int(dataset._cum_n_starts[ep])) * dataset._window_stride
    row = dataset._eps_df.iloc[ep]
    n = min(kmax, offset-start)
    table = dataset._load_data_table(int(row['data/chunk_index']), int(row['data/file_index']))
    win = table.slice(int(dataset._ep_data_row_offset[ep])+offset-n, n+1).to_pandas()
    states = dataset._read_vector(win, 'observation.state')
    current = states[-1]
    past = np.repeat(current[None], kmax, axis=0)
    available = np.arange(kmax) < n
    if n:
        past[:n] = states[-2::-1]
        # Do not cross an instruction change even within an episode.
        tasks = np.asarray(win['task_index'])
        available[:n] &= np.array([np.all(tasks[-(k+2):] == tasks[-1]) for k in range(n)])
    valid = bounded_history(current, past, available, cfg)
    normalized = dataset._normalize_array(past)
    candidates = np.stack([dataset._finalize_proprio(v[None])[0][0] for v in normalized])
    return dict(rtc_history_states=torch.from_numpy(candidates),
                rtc_history_valid=torch.from_numpy(valid))


def collate_history(samples):
    keys = ('rtc_history_states', 'rtc_history_valid')
    if not any(keys[0] in s for s in samples):
        return {}
    if not all(all(k in s for k in keys) for s in samples):
        raise ValueError('Mixed or incomplete RTC state history batch')
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def apply_history(proprio, has_prefix, states, valid, config):
    """Uniform requested k; fall back toward present if bounded k is invalid.

    Sampling is isolated from the existing flow/prefix/dropout RNG stream.
    All components use the same k. Invalid/unselected rows retain exact bits.
    """
    cfg = RTCStateHistoryConfig.from_config(config)
    if not cfg.enabled:
        return proprio, {}
    if proprio is None or states is None or valid is None:
        raise ValueError('Enabled RTC state history requires dataset history and proprioception')
    batch, kmax = valid.shape
    if kmax != cfg.max_history_steps or states.shape[:2] != (batch, kmax):
        raise ValueError('Invalid RTC state history shape')
    if proprio.ndim not in (2, 3) or (proprio.ndim == 3 and proprio.shape[1] != 1):
        raise ValueError('RTC history requires [B,D] or [B,1,D] proprio')
    if proprio.shape[0] != batch or proprio.shape[-1] != states.shape[-1]:
        raise ValueError('RTC history/proprio shape mismatch')
    eligible = torch.as_tensor(has_prefix, dtype=torch.bool, device='cpu').reshape(batch)
    with torch.random.fork_rng(devices=[]):
        requested = eligible & (torch.rand(batch) < cfg.probability)
        requested_k = torch.randint(1, kmax+1, (batch,))
    valid = valid.detach().to(device='cpu', dtype=torch.bool)
    ks = torch.arange(1, kmax+1)[None].expand(batch, -1)
    chosen = torch.where(valid & (ks <= requested_k[:, None]), ks, 0).max(-1).values
    chosen = torch.where(requested, chosen, 0)
    selected = chosen > 0
    result = proprio
    changed = torch.zeros(batch, dtype=torch.bool)
    if selected.any():
        candidate = states[torch.arange(batch, device=states.device), (chosen-1).clamp_min(0).to(states.device)]
        candidate = candidate.to(device=proprio.device, dtype=proprio.dtype).reshape_as(proprio)
        mask = selected.to(proprio.device).reshape(batch, *([1]*(proprio.ndim-1)))
        result = torch.where(mask, candidate, proprio)
        changed = result.detach().reshape(batch, -1).ne(proprio.detach().reshape(batch, -1)).any(-1).cpu()
    stats = {'prefix_fraction': eligible.float().mean().item(),
             'requested_fraction': requested.float().mean().item(),
             'selected_fraction': selected.float().mean().item(),
             'changed_fraction': changed.float().mean().item(),
             'fallback_fraction': (requested & ~selected).float().mean().item()}
    stats.update({f'k{k}_fraction': (chosen == k).float().mean().item() for k in range(1, kmax+1)})
    return result, stats


def configure_rtc_state_history(architecture, cfg=None, *, dataset=None, saved_cfg=None, resume=False):
    config = RTCStateHistoryConfig.from_config(cfg)
    if resume and config != RTCStateHistoryConfig.from_config(saved_cfg):
        raise ValueError('Resume must keep rtc_state_history; use finetune for a new experiment')
    if config.enabled:
        from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
        if not getattr(getattr(architecture, 'training_rtc', None), 'enabled', False):
            raise ValueError('RTC state history requires training.rtc.enabled=true')
        if getattr(getattr(architecture, 'rtc_state_augmentation', None), 'enabled', False):
            raise ValueError('Do not combine legacy XYZ augmentation and RTC state history')
        if not isinstance(dataset, WujiRealTaskDataset) or dataset.rot6d_convention != 'column':
            raise ValueError('RTC state history requires Wuji absolute EEF column data')
        if dataset._split != 'train' or dataset._normalize_mode != 'quantile':
            raise ValueError('RTC state history requires quantile-normalized training data')
        if not math.isclose(dataset._fps, 30.0):
            raise ValueError('RTC state history experiment requires 30 Hz data')
    architecture.rtc_state_history = config
    if dataset is not None:
        dataset.rtc_state_history = config
    return config
