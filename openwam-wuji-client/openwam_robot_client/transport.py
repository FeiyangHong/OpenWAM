from __future__ import annotations

import inspect

import msgpack

EXPECTED_CONTRACT = {
    "representation": "eef",
    "raw_action_dim": 58,
    "rot6d_convention": "column",
    "action_layout": "left_eef9,left_hand20,right_eef9,right_hand20",
}


class PolicyServerError(RuntimeError):
    pass


class PolicyTransport:
    def __init__(self, url: str, timeout: float = 300.0, open_timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.open_timeout = open_timeout
        self._socket = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        from websockets.sync.client import connect

        options = {
            "max_size": None,
            "compression": None,
            "open_timeout": self.open_timeout,
            "ping_interval": None,
            "proxy": None,
        }
        supported = set(inspect.signature(connect).parameters)
        self._socket = connect(self.url, **{key: value for key, value in options.items() if key in supported})

    def _roundtrip(self, payload: dict) -> dict:
        self.connect()
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        try:
            raw = self._socket.recv(timeout=self.timeout)
        except TypeError:
            raw = self._socket.recv()
        if not isinstance(raw, bytes):
            raise PolicyServerError("server returned text to a binary MsgPack request")
        response = msgpack.unpackb(raw, raw=False)
        if not isinstance(response, dict):
            raise PolicyServerError("server response is not a mapping")
        if response.get("type") == "error":
            raise PolicyServerError(f"{response.get('code', 'server_error')}: {response.get('message', '')}")
        return response

    def ping_and_validate(self) -> dict:
        response = self._roundtrip({"type": "ping"})
        if response.get("type") != "pong":
            raise PolicyServerError(f"expected pong, got {response.get('type')!r}")
        mismatches = {
            key: (expected, response.get(key))
            for key, expected in EXPECTED_CONTRACT.items()
            if response.get(key) != expected
        }
        if mismatches:
            raise PolicyServerError(f"checkpoint contract mismatch: {mismatches}")
        return response

    def reset(self) -> None:
        response = self._roundtrip({"type": "reset"})
        if response.get("type") != "reset_ack":
            raise PolicyServerError(f"expected reset_ack, got {response.get('type')!r}")

    def request_chunk(self, *, images: dict[str, bytes], prompt: str, state: list[float], count: int) -> dict:
        # Never retry this request: after a connection drop it is ambiguous
        # whether inference ran and advanced the server-side action buffer.
        response = self._roundtrip(
            {"type": "obs_chunk", "images": images, "prompt": prompt, "state": state, "count": count}
        )
        if response.get("type") != "action_chunk":
            raise PolicyServerError(f"expected action_chunk, got {response.get('type')!r}")
        return response

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
