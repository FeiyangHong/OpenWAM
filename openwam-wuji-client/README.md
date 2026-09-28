# OpenWAM Astribot + Wuji Client

This directory is the hardware-only side of the deployment. It talks to the
OpenWAM WebSocket server, receives raw 58-D actions, and owns all robot safety
projection and SDK calls.

The checkpoint contract is fixed to:

```text
[left EEF9, left hand20, right EEF9, right hand20]
EEF9 = xyz + first two columns of the rotation matrix
```

## Why this client is smaller than the GR00T reference

- ROS 2 is used only for the three camera subscriptions.
- Wuji state and commands use `wuji_sdk` directly.
- Camera callbacks retain the existing compressed bytes. There is no OpenCV
  decode, PNG re-encode, or base64 expansion.
- MsgPack binary frames carry one observation and a configurable action chunk.
- There is no continuous state logger, replay diagnostic path, schema aliasing,
  RTC scheduler, or second ROS executor in the control path.
- The GR00T virtual environment is not activated, avoiding package and ABI
  conflicts. Only the Wuji wheel path and Astribot/ROS runtime are added.

The removed features are useful for GR00T experiments, but they add threads,
copies, disk I/O, and branches that are not part of this OpenWAM checkpoint's
wire or action contract.

## Run

Start the server on `192.168.0.100` after the checkpoint download completes:

```bash
cd /home/ubuntu/astritbot_wuji/OpenWAM
bash scripts/deploy_wuji_server.sh
```

On Orin, first run without publishing arm or hand commands:

```bash
cd /home/astribot/workspace/openwam-wuji-client
./run_client.sh \
  --server ws://192.168.0.100:8848 \
  --task "your exact training prompt" \
  --max-chunks 2
```

After checking the server contract, camera freshness, action values, frames,
and physical workspace, explicitly enable both command paths:

```bash
./run_client.sh \
  --server ws://192.168.0.100:8848 \
  --task "your exact training prompt" \
  --enable-arms --enable-hands
```

Space is a software stop while the terminal is focused. `Ctrl-C`, `SIGTERM`, a
server error, malformed actions, stale cameras, or a dropped connection exits
the loop; when arm commands were enabled, shutdown calls `stop_robot()`. Space
is checked between network/action steps and is not a substitute for the
robot's physical emergency stop.

The training window contains 33 frames and therefore 32 action steps. The
client defaults to `--execution-horizon 32`, matching the server launcher's
synchronous `inference_horizon=32`. `--chunk-size` remains an alias for older
commands. When enabling the server's async inference executor, use
`--execution-horizon 1` so its internal step counter follows physical control
steps instead of advancing an entire batch at once.
