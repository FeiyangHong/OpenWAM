"""Common configuration and parameter-freezing policy for E0--E3."""

from pathlib import Path

from omegaconf import OmegaConf

from scripts.statewam.r2.common import GOAL_ID, ROOT, sha256

GROUPS = {"E0": None, "E1": "current", "E2": "future", "E3": "future_action"}


def group_config(source, data, group):
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    source, data = Path(source), Path(data)
    cfg = OmegaConf.load(ROOT / "configs/train.yaml")
    del cfg.defaults
    cfg = OmegaConf.merge(cfg, OmegaConf.load(source / "config.yaml"))
    reference = ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/config.yaml"
    cfg.dataloader = OmegaConf.load(reference).dataloader
    cfg.dataloader.dataset_dir = str((data / "dataset").resolve())
    cfg.dataloader.episode_manifest = str((data / "episode_manifest.json").resolve())
    cfg.dataloader.normalization_stats_path = str((data / "normalization_stats.npy").resolve())
    cfg.dataloader.variant = "clean_50"
    cfg.dataloader.window_stride = 1
    cfg.dataloader.color_jitter = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0}
    cfg.model.architecture.parameter_freeze = []
    if group != "E0":
        cfg.model.architecture = OmegaConf.merge(
            cfg.model.architecture, OmegaConf.load(ROOT / "configs/model/statewam.yaml").architecture,
        )
        cfg.model.architecture.interaction_state.source = GROUPS[group]
        cfg.model.architecture.interaction_state.capacity_control = "current_resample" if group in ("E1", "E2") else "none"
    cfg.training = OmegaConf.merge(cfg.training, {
        "debug": False, "num_epochs": None, "max_steps": None, "batch_size": 4,
        "gradient_accumulation_steps": 1, "dataset_num_workers": 4,
        "mixed_precision": "bf16", "zero_stage": 2, "initialize_model_on_cpu": True,
        "use_gradient_checkpointing": True, "use_gradient_checkpointing_offload": False,
        "learning_rate": 1e-4, "action_lr": 1e-5, "video_lr": None,
        "adam_betas": [0.9, 0.95], "adam_eps": 1e-8, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "lr_scheduler": None, "lambda_video": 1.0, "lambda_action": 1.0,
        "finetune_ckpt_path": None, "resume_ckpt_path": None,
        "save_steps": None, "save_full_states_for_resume": True, "keep_last_k_ckpts": 2,
    })
    cfg.project.seed = 42
    cfg.project.output_dir = "statewam_artifacts/robotwin/r2-robotwin-pilot/runs"
    cfg.project.wandb.entity = None
    cfg.r2 = {"goal_id": GOAL_ID, "group": group, "protocol_id": "r2-initialization-v1",
              "world_freeze": "parameters_only_input_differentiable", "budget_frozen": False,
              "sampling": "task_uniform_window_uniform_with_replacement_by_occurrence",
              "rng": "r2-paired-rng-v1", "episode_manifest_sha256": sha256(data / "episode_manifest.json"),
              "statistics_sha256": sha256(data / "normalization_stats.npy")}
    OmegaConf.resolve(cfg)
    return cfg


def apply_parameter_policy(architecture):
    """Use identical parameter-only world freezing for native E0 and State groups."""
    differentiable_world = [architecture.video_backbone, *architecture.video_backbone.dit.modules()]
    if any(getattr(module, "_openwam_no_grad_wrapped", False) for module in differentiable_world):
        raise ValueError("World DiT no_grad wrappers would break the input-differentiable policy")
    architecture.video_backbone.requires_grad_(False)
    architecture.action_backbone.requires_grad_(True)
    architecture.proprio_encoder.requires_grad_(True)
    if hasattr(architecture, "interaction_state"):
        architecture.interaction_state.requires_grad_(True)
    allowed = ("action_backbone.", "proprio_encoder.", "interaction_state.")
    unexpected = [name for name, parameter in architecture.named_parameters() if parameter.requires_grad and not name.startswith(allowed)]
    if unexpected:
        raise ValueError(f"Unexpected trainable parameters: {unexpected}")
    return {"total": sum(p.numel() for p in architecture.parameters()),
            "trainable": sum(p.numel() for p in architecture.parameters() if p.requires_grad),
            "by_module": {prefix: sum(p.numel() for name, p in architecture.named_parameters()
                                      if name.startswith(prefix) and p.requires_grad) for prefix in allowed}}
