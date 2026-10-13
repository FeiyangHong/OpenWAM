"""CPU-only RTC wire smoke server: synthetic actions, no checkpoint or robot."""
import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party"))

import numpy as np
from omegaconf import OmegaConf
from openwam.deploy.rtc_server import RTCChunkServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18848)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    def preprocess(request):
        assert request["images"]["head_camera"] == b"mock-compressed-image"
        return dict(request, image="mock-image")

    def generate(conditions):
        prefix = conditions.get("rtc_prefix", np.empty((0, 58)))
        actions = np.zeros((32, 58), dtype=np.float32)
        actions[:, [3, 7, 32, 36]] = 1.  # Valid column-rot6d identity rotations.
        actions[:len(prefix)] = prefix
        return {"actions": actions}

    engine = SimpleNamespace(
        architecture=SimpleNamespace(
            training_rtc=SimpleNamespace(enabled=True, max_prefix_steps=12),
            repr_contract=dict(representation="eef", raw_action_dim=58, rot6d_convention="column",
                               action_layout="left_eef9,left_hand20,right_eef9,right_hand20")),
        generate=generate, reset_video_saving=lambda session: None, shutdown=lambda: None,
    )
    server = RTCChunkServer(engine, OmegaConf.create({
        "inference": {"num_frames": 33}, "dataloader": {"multiview": False},
    }))
    server.preprocessor = SimpleNamespace(preprocess=preprocess)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
