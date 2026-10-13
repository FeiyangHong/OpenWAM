"""Paired scene/chunk noise and explicit episode reset identities for R2."""

from benchmarks.robotwin.r2_episode_ledger import validate_identity
from scripts.statewam.r2.common import stable_hash

NOISE_SCHEME = "r2-policy-independent-streams-v1"


def chunk_seeds(identity, chunk_index):
    identity = validate_identity(identity)
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("Chunk index must be a nonnegative integer")
    # Model group, checkpoint, repeatability replicate and attempt do not enter
    # the noise key. The two modalities have domain-separated local generators.
    context = [identity[key] for key in ("protocol_sha256", "scene_manifest_sha256", "task", "condition",
                                        "scene_id", "policy_seed")]
    return {modality: int(stable_hash([NOISE_SCHEME, context, chunk_index, modality])[:16], 16) & (2**63 - 1)
            for modality in ("world", "action")}


def reset_identity(identity, *, attempt):
    identity = validate_identity(identity)
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError("An episode permits only an initial attempt and one retry")
    return {"episode_key": stable_hash(identity), "identity": identity, "attempt": attempt,
            "noise_scheme": NOISE_SCHEME}


def validate_reset_ack(ack, expected):
    if ack != {"type": "reset_ack", **expected, "next_action": 0, "next_chunk": 0,
               "cleared": {"executor": True, "prompt": True, "vace": True, "velocity": True, "temporary_state": True}}:
        raise ValueError("R2 reset acknowledgement does not confirm identity and cleared state")


def validate_action_response(response, expected, action_index):
    if type(action_index) is not int or action_index < 0:
        raise ValueError("Action index must be a nonnegative integer")
    if (response.get("type") != "action" or response.get("episode_key") != expected["episode_key"]
            or response.get("attempt") != expected["attempt"]
            or type(response.get("action_index")) is not int or response["action_index"] != action_index
            or type(response.get("chunk_index")) is not int or response["chunk_index"] != action_index // 32
            or response.get("seeds") != chunk_seeds(expected["identity"], action_index // 32)):
        raise ValueError("R2 action response has stale episode, order or noise identity")
