"""Residual interaction state with the original OpenWAM external contract."""

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture

from .mot_driver import StateWAMMoTDriver
from .state_expert import InteractionStateExpert


@register_architecture(
    "statewam_interaction", framework="statewam", variant="interaction", status="experimental",
    note="R1 residual interaction-state research architecture.",
)
class StateWAMArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if self.video_backbone is not None:
            self._init_state_expert()
            self.freeze_parameters(self._cfg_get(cfg, "parameter_freeze", []))

    def _init_state_expert(self):
        vb, ab = self.video_backbone, self.action_backbone
        self.interaction_state = InteractionStateExpert(
            video_dim=vb.dim, action_dim=ab.dim, attention_dim=ab.num_heads * ab.head_dim,
            num_layers=vb.num_layers, cfg=self._cfg_get(self.cfg, "interaction_state", {}),
        )
        self.build_mot_driver()

    def build_mot_driver(self):
        if not hasattr(self, "interaction_state"):
            return super().build_mot_driver()
        self._mot_driver = StateWAMMoTDriver(
            self.video_backbone, self.action_backbone, state_expert=self.interaction_state,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        self.interaction_state.to(device=device, dtype=dtype)

    def freeze_parameters(self, names):
        """Freeze weights while preserving gradients through module inputs."""
        for name in names:
            self.get_submodule(name).requires_grad_(False)

    def compute_loss(self, *, actions=None, **inputs):
        inputs["_state_video_is_pad"] = inputs.get("video_is_pad")
        inputs["_state_action_is_pad"] = inputs.get("action_is_pad")
        return super().compute_loss(actions=actions, **inputs)

    def prepare_joint_states(self, vstate, astate, pipeline_inputs):
        if self.interaction_state.source == "off":
            return vstate, astate
        b = vstate.hidden_states.shape[0]
        f, h, w = vstate.grid_frames, vstate.grid_height, vstate.grid_width
        device, dtype = vstate.hidden_states.device, vstate.hidden_states.dtype
        frame_valid = torch.ones((b, f), device=device, dtype=torch.bool)
        pad = pipeline_inputs.get("_state_video_is_pad")
        if pad is not None:
            pad = pad.to(device=device, dtype=torch.bool)
            if pad.shape == (b, f - 1):
                frame_valid = torch.cat([frame_valid[:, :1], ~pad], dim=1)
            elif pad.shape == (b, f):
                frame_valid = ~pad
            else:
                raise ValueError("State video padding must match latent frame count")
        action_pad = pipeline_inputs.get("_state_action_is_pad")
        action_valid = None
        if action_pad is not None:
            action_valid = ~action_pad.to(device=device, dtype=torch.bool)
            if action_valid.ndim == 3:
                action_valid = action_valid.any(dim=-1)
            if action_valid.shape != astate.payload.x_action.shape[:2]:
                raise ValueError("State action padding must match action token count")
        tv = pipeline_inputs["timestep"].reshape(-1).expand(b).to(device=device, dtype=dtype) / 1000
        ta = astate.timestep.reshape(-1).expand(b).to(device=device, dtype=dtype) / 1000
        metadata = {
            "video_valid": frame_valid.repeat_interleave(h * w, dim=1),
            "action_valid": action_valid,
            "positions": self.interaction_state.positions(f, h, w, device=device, dtype=dtype),
            "noise": torch.stack([tv, ta, tv.sin(), ta.sin()], dim=-1),
        }
        # New metadata per forward; checkpoint closures only read these tensors.
        vstate.extras = {**vstate.extras, "interaction_state": metadata}
        return vstate, astate
