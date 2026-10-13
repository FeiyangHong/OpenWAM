"""Thin R2 client: explicit episode reset and no ambiguous request resends."""

import numpy as np

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from benchmarks.robotwin.r2_policy_protocol import reset_identity, validate_action_response, validate_reset_ack
from benchmarks.utils.client import build_payload, encode_numpy_b64
from benchmarks.utils.transport import WSPolicyClient
from scripts.statewam.r2.common import stable_hash


class PairedClient:
    def __init__(self, url, *, timeout=300):
        self.transport = WSPolicyClient(url, timeout=timeout)
        self.expected = self.instruction = None
        self.next_action = 0
        self.failed = False

    def reset(self, identity, *, attempt, instruction):
        expected = reset_identity(identity, attempt=attempt)
        if stable_hash(instruction) != identity["instruction_sha256"]:
            raise ValueError("Instruction differs from manifest identity")
        self.failed = True
        ack = self.transport._roundtrip({"type": "reset", "identity": identity, "attempt": attempt,
                                         "reset": expected, "instruction": instruction}, reconnect=False)
        validate_reset_ack(ack, expected)
        self.expected, self.instruction = expected, instruction
        self.next_action, self.failed = 0, False
        return ack

    def predict(self, *, cameras, state, instruction):
        if self.expected is None or self.failed:
            raise ValueError("Explicit reset required; do not resend an uncertain action")
        if instruction != self.instruction:
            raise ValueError("Instruction changed; no automatic mid-episode reset")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("Require finite raw 20D EEF proprio")
        payload = build_payload(*(encode_numpy_b64(cameras[name]) for name in ("head_camera", "left_camera", "right_camera")),
                                prompt=format_prompt_for_inference(instruction), state=state.tolist())
        payload.update(episode_key=self.expected["episode_key"], attempt=self.expected["attempt"], action_index=self.next_action)
        try:
            result = self.transport.predict_once(payload)
            validate_action_response(result, self.expected, self.next_action)
            action = np.asarray(result["action"], dtype=np.float32)
            if action.shape != (20,) or not np.isfinite(action).all():
                raise ValueError("Nonfinite or wrong-width policy action")
        except BaseException:
            self.failed = True
            self.transport.close()
            raise
        self.next_action += 1
        return action, result

    def close(self):
        self.transport.close()
        self.expected = self.instruction = None
        self.failed = True
