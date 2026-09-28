"""No SDK imports/connections: control queue, immutable commands and stale replies."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from openwam_robot_client.config import parse_args as sync_args
from openwam_robot_client.rtc_main import parse_args
from openwam_robot_client.rtc_runtime import RTCScheduler, NetworkWorker
from openwam_robot_client.rtc_controller import RTCController, RTCWujiHands
from openwam_robot_client.rtc_transport import RTCTransport
from openwam_robot_client.transport import PolicyServerError, EXPECTED_CONTRACT
from openwam_robot_client.geometry import pose7_to_eef9_column


def raw_action(x=0.):
    pose = pose7_to_eef9_column(np.array([x, 0, 0, 0, 0, 0, 1.], dtype=np.float32))
    return np.concatenate((pose, np.zeros(20), pose, np.zeros(20))).astype(np.float32)


class Controller:
    def __init__(self):
        self.executed = []

    def prepare(self, actions):
        return [SimpleNamespace(raw=a.copy()) for a in actions]

    def execute_prepared(self, action):
        self.executed.append(action.raw.copy())


class Worker:
    def __init__(self):
        self.requests = []
        self.future = None

    def submit(self, payload):
        self.requests.append(payload)
        self.future = Future()
        return self.future

    def finish(self, offset=0, **overrides):
        req = self.requests[-1]
        actions = np.stack([raw_action(offset + i) for i in range(32)])
        self.future.set_result(dict(type="rtc_action_chunk", request_id=req["request_id"],
                                    start_step=req["start_step"], prefix_steps=len(req["rtc_prefix"]),
                                    execute_steps=16, actions=actions.tolist(), **overrides))


def setup_scheduler():
    controller, worker = Controller(), Worker()
    scheduler = RTCScheduler(controller, worker)
    scheduler.request({"state": raw_action().tolist()})
    worker.finish()
    scheduler.adopt()
    return scheduler, controller, worker


def test_independent_entry_defaults_and_sync_unchanged():
    assert sync_args(["--task", "pick"]).execution_horizon == 32
    rtc = parse_args(["--task", "pick"])
    assert rtc.execution_horizon == 16 and rtc.rtc_prefix_steps == 10 and rtc.rtc_warmup_rounds == 2
    assert not rtc.enable_arms and not rtc.enable_hands


def test_exact_prefix_and_suffix_adoption():
    scheduler, controller, worker = setup_scheduler()
    for _ in range(6):
        scheduler.execute()
    observation = {"images": {"head_camera": b"jpeg"}, "state": raw_action().tolist()}
    scheduler.request(observation)
    np.testing.assert_array_equal(np.asarray(worker.requests[-1]["rtc_prefix"])[:, 0], np.arange(6, 16))
    assert worker.requests[-1]["start_step"] == 6
    worker.finish(offset=100)
    for _ in range(10):
        scheduler.execute()
    assert scheduler.step == 16
    scheduler.execute()
    assert controller.executed[-1][0] == 110
    assert scheduler.step == 17


def test_late_reply_stops_before_unconditioned_action():
    scheduler, controller, worker = setup_scheduler()
    for _ in range(6):
        scheduler.execute()
    scheduler.request({})
    for _ in range(10):
        scheduler.execute()
    with pytest.raises(TimeoutError, match="prefix budget"):
        scheduler.execute()
    assert len(controller.executed) == 16 and scheduler.step == 16


def test_mismatched_response_is_never_executed():
    scheduler, controller, worker = setup_scheduler()
    for _ in range(6):
        scheduler.execute()
    scheduler.request({})
    worker.finish()
    worker.future.result()["request_id"] = -99
    for _ in range(10):
        scheduler.execute()
    with pytest.raises(RuntimeError, match="mismatch"):
        scheduler.execute()
    assert len(controller.executed) == 16


def test_failed_publication_does_not_advance_clock():
    scheduler, controller, _ = setup_scheduler()
    controller.execute_prepared = lambda a: (_ for _ in ()).throw(RuntimeError("sdk failed"))
    with pytest.raises(RuntimeError, match="sdk failed"):
        scheduler.execute()
    assert scheduler.step == 0 and len(scheduler.buffer) == 16


def test_prepared_targets_are_projected_once_without_mutating_state():
    pose = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)
    robot = SimpleNamespace(arm_left_name="left", arm_right_name="right",
                            get_current_cartesian_pose=lambda *a, **kw: [pose.copy(), pose.copy()])
    calls = []
    hands = SimpleNamespace(left=SimpleNamespace(get=lambda: np.zeros(20)),
                            right=SimpleNamespace(get=lambda: np.zeros(20)), max_step=.1,
                            validate_prepared=lambda *args: None,
                            send_prepared=lambda side, target: calls.append((side, target.copy())))
    controller = RTCController(robot, hands, enable_arms=False, eef_frame="chassis", control_way="filter",
                               use_wbc=False, workspace_radius=np.ones(3), max_eef_step=.03,
                               max_rotation_step_rad=.1)
    plan = controller.prepare(np.stack([raw_action(1.), raw_action(1.)]))
    assert controller.previous[0][0] == 0
    np.testing.assert_allclose([p.raw[0] for p in plan], [.03, .06])
    with pytest.raises(ValueError):
        plan[0].raw[0] = 999
    controller.execute_prepared(plan[0])
    np.testing.assert_allclose(controller.previous[0][0], .03)
    assert len(calls) == 2


def test_hand_guard_preserved_instead_of_silent_reclipping():
    hands = object.__new__(RTCWujiHands)
    hands.enable_commands, hands.max_step = True, .1
    hands._connections = {"left": SimpleNamespace(state=SimpleNamespace(get=lambda: np.zeros(20)))}
    hands.validate_prepared("left", np.full(20, .05))
    with pytest.raises(RuntimeError, match="re-clipping"):
        hands.validate_prepared("left", np.full(20, .2))


def test_transport_contract_and_stale_episode():
    transport = RTCTransport("ws://unused")
    transport._roundtrip = lambda payload: dict(type="pong", **EXPECTED_CONTRACT)
    with pytest.raises(PolicyServerError, match="dedicated"):
        transport.ping_and_validate()
    transport.session_id, transport.episode_id = "session", 1
    transport._roundtrip = lambda payload: dict(type="rtc_action_chunk", session_id="session", episode_id=0, request_id=1)
    with pytest.raises(PolicyServerError, match="episode"):
        transport.request({"type": "rtc_chunk", "request_id": 1})


def test_network_worker_preserves_binary_images_and_reports_failure():
    received = []
    transport = SimpleNamespace(request=lambda request: received.append(request) or {"type": "ack"}, close=lambda: None)
    worker = NetworkWorker(transport)
    try:
        result = worker.submit({"images": {"head_camera": b"jpeg"}}).result(timeout=2)
        assert received[0]["images"]["head_camera"] == b"jpeg" and "_received_at" in result
        transport.request = lambda request: (_ for _ in ()).throw(RuntimeError("network failed"))
        with pytest.raises(RuntimeError, match="network failed"):
            worker.submit({}).result(timeout=2)
    finally:
        worker.close()
