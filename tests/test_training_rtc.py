"""CPU integration tests: real Wan/ActionDiT, flow loss, sampling, execution."""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openwam.model.architectures.utils.training_rtc import (
    TrainingRTCConfig, configure_training_rtc, make_training_prefix,
)
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21
from tests.test_openwam_trainer import _MockScheduler


def tiny_architecture(cls=DualSystemSelfAttnArchitecture):
    cfg = dict(
        video_dim=96, dim=32, ffn_dim=64, num_heads=4, attn_head_dim=24,
        num_dit_layers=2, bridge_interval=1, text_dim=32, action_dim=7,
        state_dim=7, use_proprioception=True, attention_mask_mode="mutual",
        video_attention_mask_mode="first_frame_causal", mot_checkpoint_mixed_attn=True,
    )
    arch = cls(cfg)
    dit = WanModel(
        dim=96, in_dim=4, ffn_dim=128, out_dim=4, text_dim=32, freq_dim=32,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
        has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True,
    )
    arch.video_backbone = Wan21(SimpleNamespace(dit=dit, vace=None, motion_controller=None))
    arch.build_mot_driver()
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    return arch


def inputs():
    return dict(
        noisy_actions=torch.randn(2, 3, 7), action_timestep=torch.tensor([100., 800.]),
        latents=torch.randn(2, 4, 3, 4, 4), timestep=torch.tensor([700., 200.]),
        first_frame_latents=torch.randn(2, 4, 1, 4, 4), context=torch.randn(2, 4, 32),
        context_mask=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        proprio=torch.randn(2, 7),
    )


def model_and_data():
    model = tiny_architecture(DualSystemSelfAttnArchitecture)
    model.video_backbone._scheduler = _MockScheduler()
    model.action_backbone.scheduler.set_timesteps(1000, training=True)
    data = inputs()
    data["actions"] = data.pop("noisy_actions")
    data["input_latents"] = data.pop("latents")
    data.pop("action_timestep")
    data.pop("timestep")
    return model, data


def test_disabled_and_zero_prefix_preserve_loss_rng_and_weights():
    model, data = model_and_data()
    keys = tuple(model.state_dict())
    torch.manual_seed(100)
    original = model.compute_loss(**data)
    rng = torch.get_rng_state()
    configure_training_rtc(model, {"enabled": False})
    torch.manual_seed(100)
    disabled = model.compute_loss(**data)
    assert torch.equal(rng, torch.get_rng_state())
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    torch.manual_seed(100)
    zero = model.compute_loss(**data, rtc_prefix_lengths=torch.zeros(2, dtype=torch.long))
    assert torch.equal(rng, torch.get_rng_state())
    for key in original:
        assert torch.equal(original[key], disabled[key])
        assert torch.equal(original[key], zero[key])
    assert keys == tuple(model.state_dict())


@pytest.mark.parametrize("checkpointed", [False, True])
def test_token_forward_loss_backward_and_checkpoint_roundtrip(checkpointed, tmp_path):
    model, data = model_and_data()
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    data["use_gradient_checkpointing"] = checkpointed
    data["action_is_pad"] = torch.zeros_like(data["actions"], dtype=torch.bool)
    data["action_is_pad"][1, 2] = True
    observed = {}

    def capture(module, args, kwargs):
        observed["actions"], observed["times"] = args[:2]

    handle = model.register_forward_pre_hook(capture, with_kwargs=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([2, 1]))["loss"]
    assert loss.isfinite()
    assert torch.equal(observed["actions"][0, :2], data["actions"][0, :2])
    assert observed["times"].shape == (2, 3)
    assert observed["times"][0, :2].eq(0).all()
    loss.backward()
    grads = [p.grad for p in model.action_backbone.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    optimizer.step()
    handle.remove()
    path = str(tmp_path / "rtc.safetensors")
    model.save_checkpoint(path)
    restored, _ = model_and_data()
    restored.load_checkpoint(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_uniform_token_times_match_scalar_forward():
    model = tiny_architecture(DualSystemSelfAttnArchitecture).eval()
    data = inputs()
    scalar = model(**data)
    data["action_timestep"] = data["action_timestep"][:, None].expand(2, 3)
    token = model(**data)
    for a, b in zip(scalar, token):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim_mask", [False, True])
def test_suffix_loss_excludes_prefix_and_padding(dim_mask):
    actions = torch.randn(2, 5, 7)
    pad = torch.zeros_like(actions, dtype=torch.bool) if dim_mask else torch.zeros(2, 5, dtype=torch.bool)
    pad[1, 4] = True
    if dim_mask:
        pad[..., -1] = True
    original_pad = pad.clone()
    noisy = torch.randn_like(actions)
    x, t, mask = make_training_prefix(actions, noisy, torch.tensor([100., 900.]), pad,
                                     TrainingRTCConfig(True, 3), torch.tensor([3, 2]))
    assert torch.equal(pad, original_pad)
    assert torch.equal(x[0, :3], actions[0, :3]) and torch.equal(x[0, 3:], noisy[0, 3:])
    assert t[1, :2].eq(0).all() and t[1, 2:].eq(900).all()
    from openwam.model.architectures.base import BaseWAMArchitecture
    pred = torch.randn_like(actions, requires_grad=True)
    loss = BaseWAMArchitecture._compute_action_loss(None, pred, actions, torch.tensor([1, 2]),
                                                   _MockScheduler(), {"action_is_pad": mask}, "cpu")
    loss.backward()
    fullmask = mask if dim_mask else mask[..., None].expand_as(pred)
    assert pred.grad[fullmask].eq(0).all()
    assert pred.grad[~fullmask].abs().sum() > 0


def test_padding_limits_and_rng_replay():
    actions = torch.randn(100, 32, 80)
    pad = torch.zeros_like(actions, dtype=torch.bool)
    pad[:, 6:] = True
    config = TrainingRTCConfig(True, 12)
    rng = torch.get_rng_state()
    a = make_training_prefix(actions, actions + 1, torch.ones(100), pad, config)
    torch.set_rng_state(rng)
    b = make_training_prefix(actions, actions + 1, torch.ones(100), pad, config)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert a[1][:, 5].eq(1).all()  # Always leave at least one valid supervised token.
    assert (a[1][:, 0] == 1).any() and (a[1][:, 0] == 0).any()
    with pytest.raises(ValueError, match="valid action suffix"):
        make_training_prefix(actions, actions, torch.ones(100), pad, config, torch.full((100,), 6))


def test_resume_and_architecture_guard():
    model, _ = model_and_data()
    config = {"enabled": True, "max_prefix_steps": 2}
    with pytest.raises(ValueError, match="Resume"):
        configure_training_rtc(model, config, resume=True)
    configure_training_rtc(model, config, saved_cfg=config, resume=True)
    class UnsupportedSubclass(DualSystemSelfAttnArchitecture):
        pass

    unsupported = tiny_architecture(UnsupportedSubclass)
    with pytest.raises(ValueError, match="only dual_system"):
        configure_training_rtc(unsupported, config)
    configure_training_rtc(unsupported, {"enabled": False})


@pytest.mark.parametrize("cfg_scale,cfg_merge", [(1., False), (2., False), (2., True)])
def test_sampling_clamps_every_forward_and_final_prefix(cfg_scale, cfg_merge):
    model, _ = model_and_data()
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    # Use the real architecture's generate/CFG/scheduler and inspect every forward.
    model.video_backbone.preprocess_input_for_inference = lambda **kw: {
        "latents": torch.ones(1, 4, 3, 4, 4), "context": torch.ones(1, 4, 32),
        "uncond_context": torch.zeros(1, 4, 32),
    }
    prefix = torch.randn(2, 7)
    prefix[:, -1] = 0  # Canonical inactive padding.
    seen = []

    def forward(actions, action_timestep, **kw):
        assert torch.equal(actions[:, :2], prefix[None].expand(actions.shape[0], -1, -1))
        assert action_timestep.shape == actions.shape[:2] and action_timestep[:, :2].eq(0).all()
        seen.append(actions.clone())
        return torch.ones_like(kw["latents"]), torch.full_like(actions, 9.)

    model.forward = forward
    result = model.generate([(1000, 1000), (500, 500), (0, 0)], "test", action_num_frames=6,
                            rtc_prefix=prefix, proprio=torch.zeros(1, 7), decode_video=False,
                            active_action_mask=torch.tensor([True] * 6 + [False]),
                            cfg_scale=cfg_scale, cfg_merge=cfg_merge)
    np.testing.assert_array_equal(result["actions"][:2], prefix.numpy())
    assert (result["actions"][:, -1] == 0).all()
    assert len(seen) == (4 if cfg_scale > 1 and not cfg_merge else 2)


def test_action_normalization_uses_action_stats_and_map():
    from openwam.deploy.model_loader import _UnifyAwareNormalizer, _CommandAwareNormalizer
    # Separate state/action transforms intentionally differ.
    action = SimpleNamespace(normalize=lambda x: x / 2, unnormalize=lambda x: x * 2)
    state = SimpleNamespace(normalize=lambda x: x / 10)
    inner = _CommandAwareNormalizer(action, state, ())
    norm = _UnifyAwareNormalizer(inner, np.array([1, 3]), 5, np.array([0, 2]))
    raw = np.array([[2., 4.]])
    np.testing.assert_array_equal(norm.normalize_action(raw), [[0., 1., 0., 2., 0.]])
    np.testing.assert_array_equal(norm.unnormalize(norm.normalize_action(raw)), raw)


def test_real_normalizer_action_roundtrip_with_distinct_state_stats_and_outliers():
    from openwam.dataloader.transforms.normalize import Normalizer
    norm = Normalizer(mode="q99", stats={"q01": [-2., 1.], "q99": [2., 3.]})
    # Deliberately make proprio transforms incompatible with action transforms.
    norm._normalize_mode_stats = {"q01": np.array([-10., -10.]), "q99": np.array([10., 10.])}
    raw = np.array([[8., 9.]], dtype=np.float32)
    normalized = norm.normalize_action(raw)
    assert (normalized > 1).all()
    np.testing.assert_allclose(norm.unnormalize(normalized), raw)
    assert not np.array_equal(norm.normalize(raw), normalized)


@pytest.mark.parametrize("prefix", [np.zeros((0, 7)), np.zeros((3, 7)), np.zeros((1, 8)), np.full((1, 7), np.nan)])
def test_invalid_inference_prefix_rejected(prefix):
    from openwam.model.architectures.utils.training_rtc import prepare_inference_prefix
    model, _ = model_and_data()
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    with pytest.raises(ValueError):
        prepare_inference_prefix(model, prefix, 5, "cpu", torch.float32)


class FakeEngine:
    def __init__(self):
        self.architecture = SimpleNamespace(training_rtc=TrainingRTCConfig(True, 12))
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def generate(self, conditions):
        self.calls.append(copy.deepcopy(conditions))
        if "rtc_prefix" in conditions:
            self.started.set()
            assert self.release.wait(5)
        offset = 100 * (len(self.calls) - 1)
        return {"actions": np.arange(32)[:, None] + offset}


def test_executor_prefix_alignment_late_result_and_reset():
    from openwam.deploy.executors.rtc_executor import TrainingRTCExecutor
    engine = FakeEngine()
    executor = TrainingRTCExecutor(engine, 16, 10)
    try:
        engine.release.clear()
        first = [executor.predict_action({"tick": i})[0] for i in range(16)]
        assert first == list(range(16))
        assert engine.started.wait(5)
        np.testing.assert_array_equal(engine.calls[1]["rtc_prefix"][:, 0], np.arange(6, 16))
        assert engine.calls[1]["tick"] == 6
        # Release from a helper after the caller has reached the empty boundary.
        timer = threading.Timer(0.05, engine.release.set)
        timer.start()
        assert executor.predict_action({"tick": 16})[0] == 110
        timer.join()
        assert executor.late_chunks == 1
        executor.reset()
        assert executor.predict_action({"tick": 0})[0] == 200
    finally:
        engine.release.set()
        executor.shutdown()


def test_executor_failed_pending_is_cleared_and_reset_discards_pending():
    from openwam.deploy.executors.rtc_executor import TrainingRTCExecutor
    engine = FakeEngine()
    executor = TrainingRTCExecutor(engine, 16, 10)
    try:
        for i in range(7):
            executor.predict_action({"tick": i})
        assert engine.started.wait(5)
        executor.reset()
        assert executor.predict_action({"tick": 0})[0] == 200
        from concurrent.futures import Future
        failed = Future()
        failed.set_exception(RuntimeError("test failure"))
        executor._buffer.clear()
        executor._pending = failed
        with pytest.raises(RuntimeError, match="test failure"):
            executor.predict_action({})
        assert executor._pending is None
    finally:
        executor.shutdown()


def test_engine_rejects_untrained_checkpoint_and_unsupported_optimizations():
    from openwam.deploy.engine import JointInferenceEngine
    model, _ = model_and_data()
    cfg = OmegaConf.create({"inference": {"rtc": {"enabled": True}}})
    with pytest.raises(ValueError, match="RTC-trained"):
        JointInferenceEngine(cfg, architecture=model)
    configure_training_rtc(model, {"enabled": True})
    for option in ("dit_cache",):
        cfg.optimization = {option: {"enabled": True}}
        with pytest.raises(ValueError, match=option):
            JointInferenceEngine(cfg, architecture=model)


@pytest.mark.parametrize("trained", [False, True])
def test_saved_checkpoint_binds_rtc_capability_before_deploy_merge(trained, tmp_path, monkeypatch):
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from tests.test_finetune_config_override import CKPT_CFG
    model, _ = model_and_data()
    model.save_checkpoint(str(tmp_path / "checkpoint_step_1.safetensors"))
    cfg = OmegaConf.create(CKPT_CFG)
    cfg.training.mixed_precision = "no"
    if trained:
        cfg.training.rtc = {"enabled": True, "max_prefix_steps": 2, "zero_prefix_probability": 0.25}
    cfg.dataloader = {"normalize_mode": None}
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    monkeypatch.setattr("openwam.model.build_architecture", lambda *args, **kwargs: model_and_data()[0])
    loaded_cfg, loaded = load_from_checkpoint_dir(str(tmp_path), device="cpu")
    assert loaded.training_rtc.enabled == trained
    # Deploy override cannot turn a legacy checkpoint into an RTC checkpoint.
    loaded_cfg = OmegaConf.merge(loaded_cfg, {"training": {"rtc": {"enabled": not trained}}})
    assert loaded.training_rtc.enabled == trained
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_engine_normalizes_committed_raw_prefix_once(monkeypatch):
    from openwam.deploy import engine as module
    from openwam.deploy.model_loader import _UnifyAwareNormalizer
    model, _ = model_and_data()
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    normalizer = _UnifyAwareNormalizer(
        SimpleNamespace(normalize=lambda x: x / 100, normalize_action=lambda x: x / 2),
        np.array([0, 2, 4, 6]), 7,
    )
    model.attach_normalizer(normalizer)
    captured = {}
    model.generate = lambda **kwargs: captured.update(kwargs) or {"actions": np.zeros((5, 4))}
    monkeypatch.setattr(module, "make_schedule", lambda *args, **kwargs: [(1000, 1000), (0, 0)])
    cfg = OmegaConf.create({"inference": {"rtc": {"enabled": True}, "denoise_steps": 10},
                           "optimization": {"compile": {"enabled": False}, "dit_cache": {"enabled": False}}})
    engine = module.JointInferenceEngine(cfg, architecture=model)
    raw = np.array([[2., 4., 6., 8.]])
    engine.generate({"rtc_prefix": raw, "proprio": np.zeros(4)})
    np.testing.assert_array_equal(captured["rtc_prefix"], [[1., 0., 2., 0., 3., 0., 4.]])
    assert np.array_equal(raw, [[2., 4., 6., 8.]])


def test_rtc_compile_scalar_and_prefix_paths():
    """Exercise actual CPU Inductor, including alternating the two warmup shapes."""
    model = tiny_architecture(DualSystemSelfAttnArchitecture).eval()
    configure_training_rtc(model, {"enabled": True, "max_prefix_steps": 2})
    scalar = inputs()
    prefix = dict(scalar, action_timestep=torch.tensor([[0., 0., 100.], [0., 800., 800.]]))
    with torch.no_grad():
        expected = [model(**data) for data in (scalar, prefix)]
        model.apply_compile_optimizations(OmegaConf.create({
            "enabled": True, "self_attn": {"torch_mode": "default", "dynamic": False},
        }))
        for _ in range(2):
            for data, reference in zip((scalar, prefix), expected):
                actual = model(**data)
                assert model._compiled_mot_run_joint_loop is not None, "compile silently fell back"
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
