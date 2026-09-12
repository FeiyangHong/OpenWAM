"""State read/write extension of the unchanged world/action mixed attention."""

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver


class StateWAMMoTDriver(DualSystemMoTDriver):
    def __init__(self, vb, ab, *, state_expert, **kwargs):
        super().__init__(vb, ab, **kwargs)
        self.state_expert = state_expert  # Owned and registered by the architecture.
        if vb.video_attention_mask_mode != "first_frame_causal":
            raise ValueError("StateWAM R1 requires first_frame_causal video attention")

    def augment_action_attention(self, layer_id, vstate, astate, attn_a):
        expert = self.state_expert
        if expert.source == "off" or str(layer_id) not in expert.layers:
            return attn_a
        if vstate.hidden_states.ndim != 3 or vstate.prefix_kv_len:
            raise ValueError("StateWAM R1 requires Wan residual tokens without prefix KV")
        metadata = vstate.extras["interaction_state"]
        # Read pre-attention residuals, not the RoPE-transformed mixed Q/K.
        video = vstate.hidden_states
        valid = metadata["video_valid"]
        positions = metadata["positions"]
        if expert.source == "current":
            n = self._video_tokens_per_frame(vstate)
            video, valid, positions = video[:, :n], valid[:, :n], positions[:n]
        actions = astate.payload.x_action if expert.source == "future_action" else None
        delta = expert.layers[str(layer_id)](
            video, vstate.context, actions, astate.payload.x_action,
            positions=positions, noise=metadata["noise"], video_valid=valid,
            context_valid=vstate.context_mask, action_valid=metadata["action_valid"],
        )
        return attn_a + delta
