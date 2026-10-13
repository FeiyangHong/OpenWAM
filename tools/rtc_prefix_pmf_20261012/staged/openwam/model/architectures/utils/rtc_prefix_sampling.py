"""Training-only opt-in prefix PMF; default RTC and inference contract stay intact."""
from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class RTCPrefixSamplingConfig:
    enabled: bool = False
    probabilities: tuple | None = None

    @classmethod
    def from_config(cls, cfg=None):
        if isinstance(cfg, cls):
            value = cfg
        else:
            fields = dict(cfg or {})
            if fields.get('probabilities') is not None:
                fields['probabilities'] = tuple(fields['probabilities'])
            value = cls(**fields)
        if type(value.enabled) is not bool:
            raise ValueError('rtc_prefix_sampling.enabled must be boolean')
        probs = value.probabilities
        if value.enabled and probs is None:
            raise ValueError('Enabled RTC prefix sampling requires probabilities for d=0..max_prefix_steps')
        if probs is not None:
            if not probs or any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or p < 0 for p in probs):
                raise ValueError('RTC prefix probabilities must be finite nonnegative numbers')
            if not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError('RTC prefix probabilities must sum to 1')
        return value


def configure_rtc_prefix_sampling(architecture, cfg=None, *, saved_cfg=None, resume=False):
    config = RTCPrefixSamplingConfig.from_config(cfg)
    if resume:
        saved = RTCPrefixSamplingConfig.from_config(saved_cfg)
        if (config.enabled or saved.enabled) and config != saved:
            raise ValueError('Resume must keep rtc_prefix_sampling; use finetune_ckpt_path for a new experiment')
    if config.enabled:
        from .training_rtc import require_rtc_architecture
        require_rtc_architecture(architecture)
        rtc = getattr(architecture, 'training_rtc', None)
        if rtc is None or not rtc.enabled:
            raise ValueError('RTC prefix sampling requires TrainingRTC enabled')
        if len(config.probabilities) != rtc.max_prefix_steps + 1:
            raise ValueError('RTC prefix probabilities must cover exactly d=0..max_prefix_steps')
    architecture.rtc_prefix_sampling = config
    return config


def sample_prefix_lengths(batch_size, max_prefix_steps, config):
    """CPU RNG is checkpointed by the existing trainer. Does not draw GPU RNG.

    This PMF includes d=0; zero_prefix_probability is not applied a second time.
    The caller still caps lengths to each sample's valid action window.
    """
    if len(config.probabilities) != max_prefix_steps + 1:
        raise ValueError('RTC prefix probabilities must cover exactly d=0..max_prefix_steps')
    if config.probabilities[0] == 1:
        return torch.zeros(batch_size, dtype=torch.long)
    probs = torch.tensor(config.probabilities, dtype=torch.float64, device='cpu')
    return torch.multinomial(probs, batch_size, replacement=True)
