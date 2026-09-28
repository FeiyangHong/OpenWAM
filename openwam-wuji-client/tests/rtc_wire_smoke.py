"""Run only against rtc_mock_server.py. No ROS, SDK or actuator access."""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam_robot_client.rtc_transport import RTCTransport
from openwam_robot_client.rtc_runtime import NetworkWorker, RTCScheduler
from openwam_robot_client.transport import PolicyServerError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:18848")
    args = parser.parse_args()
    transport = RTCTransport(args.server, timeout=5.)
    worker = None
    published = []
    try:
        assert transport.ping_and_validate()["action_horizon"] == 32
        transport.reset()
        controller = SimpleNamespace(prepare=lambda actions: [SimpleNamespace(raw=a.copy()) for a in actions],
                                     execute_prepared=lambda action: published.append(action.raw.copy()))
        worker = NetworkWorker(transport)
        scheduler = RTCScheduler(controller, worker)
        observation = {"images": {"head_camera": b"mock-compressed-image"},
                       "state": [0.] * 58, "prompt": "mock only"}
        scheduler.request(observation, warmup=True)
        scheduler.pending.result(timeout=10.)
        scheduler.finish_warmup()
        scheduler.request(observation)
        scheduler.pending.result(timeout=10.)
        scheduler.adopt()
        while scheduler.step < 48:
            if not scheduler.buffer:
                scheduler.adopt()
            if len(scheduler.buffer) == scheduler.d and scheduler.chunks < 3:
                scheduler.request(observation)
            scheduler.execute()
            time.sleep(1. / 30.)
        assert scheduler.chunks == 3 and len(published) == 48
        # Replay/old episode is rejected over the actual wire.
        worker.close()
        worker = None
        transport.ping_and_validate()
        transport.reset()
        old_episode = transport.episode_id
        transport.reset()
        transport.episode_id = old_episode
        try:
            transport.request(dict(observation, type="rtc_chunk", request_id=999, start_step=0,
                                   execute_steps=16, rtc_prefix=[]))
        except PolicyServerError:
            pass
        else:
            raise AssertionError("stale episode accepted")
        print("PASS: binary images, warmup, 3 chunks / 48 mock commands at 30 Hz, stale episode rejected")
    finally:
        if worker is not None:
            worker.close()
        else:
            transport.close()


if __name__ == "__main__":
    main()
