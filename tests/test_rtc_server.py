"""RTC service tests without checkpoints, GPU allocation or robot hardware."""
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.deploy.rtc_server import RTCChunkServer, PROTOCOL
from openwam.deploy.server import _decode_wire_message, _encode_wire_message


CONTRACT = dict(representation="eef", raw_action_dim=58, rot6d_convention="column",
                action_layout="left_eef9,left_hand20,right_eef9,right_hand20")


def make_server():
    calls = []
    engine = SimpleNamespace(
        architecture=SimpleNamespace(training_rtc=SimpleNamespace(enabled=True, max_prefix_steps=12),
                                     repr_contract=CONTRACT),
        reset_video_saving=lambda session: None,
        generate=lambda conditions: calls.append(conditions) or {"actions": np.zeros((32, 58))},
    )
    cfg = OmegaConf.create({"inference": {"num_frames": 33}, "dataloader": {"multiview": False}})
    server = RTCChunkServer(engine, cfg)
    server.preprocessor = SimpleNamespace(preprocess=lambda request: dict(request, image="decoded"))
    return server, calls


def payload(server, request_id=1, start=0, prefix=0, kind="rtc_chunk"):
    return dict(type=kind, session_id=server.session_id, episode_id=server.episode_id,
                request_id=request_id, start_step=start, execute_steps=16,
                rtc_prefix=np.ones((prefix, 58)).tolist(), images={"head_camera": b"jpeg"},
                state=np.zeros(58).tolist(), prompt="pick")


def test_warmup_bootstrap_and_committed_prefix_timeline():
    server, calls = make_server()
    assert server.dispatch({"type": "ping"})["rtc_protocol"] == PROTOCOL
    server.dispatch({"type": "reset"})
    reply = server.dispatch(payload(server, prefix=10, kind="rtc_warmup"))
    assert reply["type"] == "rtc_warmup_ack" and len(calls) == 2
    assert "rtc_prefix" not in calls[0] and calls[1]["rtc_prefix"].shape == (10, 58)
    assert server.last_start == -1  # Warmup must not advance action execution.
    first = server.dispatch(payload(server, request_id=2))
    assert len(first["actions"]) == 32 and first["prefix_steps"] == 0
    next_chunk = server.dispatch(payload(server, request_id=3, start=6, prefix=10))
    assert next_chunk["start_step"] == 6 and np.array_equal(calls[-1]["rtc_prefix"], np.ones((10, 58)))
    server.dispatch(payload(server, request_id=4, start=22, prefix=10))
    assert calls[-1]["video_save_episode_id"] == server.episode_id


def test_binary_wire_and_legacy_endpoint_rejected():
    server, _ = make_server()
    request = payload(server)
    decoded, binary = _decode_wire_message(_encode_wire_message(request, True))
    assert binary and decoded["images"]["head_camera"] == b"jpeg"
    with pytest.raises(ValueError, match="legacy"):
        server.dispatch({"type": "obs_chunk"})


def test_reset_drops_episode_and_replayed_requests():
    server, calls = make_server()
    server.reset()
    request = payload(server)
    server.dispatch(request)
    with pytest.raises(ValueError, match="IDs"):
        server.dispatch(request)
    server.reset()
    with pytest.raises(ValueError, match="Stale"):
        server.dispatch(request)
    server.dispatch(payload(server))
    assert len(calls) == 2


@pytest.mark.parametrize("change", [{"rtc_prefix": [[0.] * 80]}, {"rtc_prefix": [[float('nan')] * 58]},
                                    {"rtc_prefix": [[], []]},
                                    {"execute_steps": True}, {"request_id": -1}, {"start_step": 1},
                                    {"rtc_prefix": [[0.] * 58] * 13}])
def test_invalid_request_is_not_run(change):
    server, calls = make_server()
    server.reset()
    with pytest.raises(ValueError):
        server.dispatch(dict(payload(server), **change))
    assert not calls


def test_client_cannot_skip_uncommitted_actions():
    server, _ = make_server()
    server.reset()
    server.dispatch(payload(server))
    with pytest.raises(ValueError, match="timeline"):
        server.dispatch(payload(server, request_id=2, start=16, prefix=10))


def test_requires_trained_checkpoint():
    server, _ = make_server()
    server.engine.architecture.training_rtc.enabled = False
    with pytest.raises(ValueError, match="RTC-trained"):
        RTCChunkServer(server.engine, server.cfg)
