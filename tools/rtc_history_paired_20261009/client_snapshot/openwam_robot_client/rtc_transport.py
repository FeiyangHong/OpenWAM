"""RTC wire protocol over the existing compressed-image MsgPack transport."""
from __future__ import annotations

import inspect

from .transport import PolicyTransport, PolicyServerError

PROTOCOL = "openwam-training-rtc-v1"


class RTCTransport(PolicyTransport):
    def connect(self):
        if self._socket is not None:
            return
        from websockets.sync.client import connect
        options = dict(max_size=None, compression=None, open_timeout=self.open_timeout,
                       close_timeout=1., ping_interval=None, proxy=None)
        accepted = inspect.signature(connect).parameters
        self._socket = connect(self.url, **{k: v for k, v in options.items() if k in accepted})

    def ping_and_validate(self):
        response = super().ping_and_validate()
        if response.get("rtc_protocol") != PROTOCOL:
            raise PolicyServerError("Connect this client to the dedicated TrainingRTC server")
        self.session_id = response["session_id"]
        return response

    def reset(self):
        response = self._roundtrip({"type": "reset"})
        if response.get("type") != "reset_ack" or response.get("session_id") != self.session_id:
            raise PolicyServerError("Invalid RTC reset acknowledgement")
        self.episode_id = response["episode_id"]

    def request(self, payload):
        response = self._roundtrip(dict(payload, session_id=self.session_id, episode_id=self.episode_id))
        for name, value in (("session_id", self.session_id), ("episode_id", self.episode_id),
                            ("request_id", payload["request_id"])):
            if response.get(name) != value:
                raise PolicyServerError(f"Stale or mismatched RTC response: {name}")
        expected = "rtc_warmup_ack" if payload["type"] == "rtc_warmup" else "rtc_action_chunk"
        if response.get("type") != expected:
            raise PolicyServerError(f"Expected {expected}")
        return response
