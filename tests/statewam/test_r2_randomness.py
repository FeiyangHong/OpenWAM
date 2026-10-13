import copy
import random

import numpy as np
import pytest
import torch

from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.utils.flow_randomness import validate_flow_randomness
from scripts.statewam.r2.paired_data import (
    PairedExposureDataset,
    flow_randomness,
    randomness_hashes,
    stream_seed,
)
from tests.statewam import test_r2_data
from tests.statewam.test_statewam import inputs, tiny_architecture

isolated_dataset = test_r2_data.isolated_dataset


def test_named_streams_and_global_rng_independence():
    seeds = {stream_seed(42, i, stream) for i in range(20) for stream in (
        "sample", "instruction", "augmentation", "video_noise", "action_noise", "video_timestep", "action_timestep")}
    assert len(seeds) == 140
    video, action = torch.zeros(3, 4, 3, 4, 4), torch.zeros(3, 3, 7)
    before = torch.get_rng_state()
    first = flow_randomness(42, [7, 8, 9], video=video, actions=action)
    assert torch.equal(before, torch.get_rng_state())
    torch.randn(2000)
    second = flow_randomness(42, [7, 8, 9], video=video, actions=action)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_noise_is_paired_across_microbatch_splits_and_modalities():
    video, action = torch.zeros(4, 4, 3, 4, 4), torch.zeros(4, 3, 7)
    complete = flow_randomness(42, [40, 41, 42, 43], video=video, actions=action)
    pieces = [flow_randomness(42, [i], video=video[:1], actions=action[:1]) for i in (40, 41, 42, 43)]
    for name in complete:
        torch.testing.assert_close(complete[name], torch.cat([piece[name] for piece in pieces]), rtol=0, atol=0)
    world_only = flow_randomness(42, [40, 41, 42, 43], video=video, actions=None)
    for name in world_only:
        torch.testing.assert_close(world_only[name], complete[name], rtol=0, atol=0)
    changed_world = flow_randomness(42, [40, 41, 42, 43], video=torch.zeros(4, 8, 4, 4, 4), actions=action)
    torch.testing.assert_close(changed_world["action_noise"], complete["action_noise"], rtol=0, atol=0)
    assert len(randomness_hashes(complete)) == 4


def test_explicit_noise_reproduces_original_loss_and_gradients(monkeypatch):
    torch.manual_seed(912)
    model = tiny_architecture()
    model.video_backbone._scheduler = ActionScheduler()
    model.init_training_schedulers(1000)
    data = inputs()
    data["input_latents"] = data.pop("latents")
    data["actions"] = data.pop("noisy_actions")
    data.pop("timestep")
    data.pop("action_timestep")
    other = copy.deepcopy(model)
    captured_ids, captured_noise = [], []
    original_randint, original_randn_like = torch.randint, torch.randn_like

    def randint(*args, **kwargs):
        result = original_randint(*args, **kwargs)
        captured_ids.append(result.clone())
        return result

    def randn_like(*args, **kwargs):
        result = original_randn_like(*args, **kwargs)
        captured_noise.append(result.clone())
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch, "randint", randint)
        patch.setattr(torch, "randn_like", randn_like)
        expected = model.compute_loss(**data)
    assert len(captured_ids) == len(captured_noise) == 2
    explicit = dict(video_timestep_ids=captured_ids[0], video_noise=captured_noise[0],
                    action_timestep_ids=captured_ids[1], action_noise=captured_noise[1])
    before = torch.get_rng_state()
    actual = other.compute_loss(**data, flow_randomness=explicit)
    assert torch.equal(before, torch.get_rng_state())
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    expected["loss"].backward()
    actual["loss"].backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("failure", ["missing", "extra", "range", "shape", "dtype", "nonfinite", "gradient"])
def test_explicit_randomness_validation(failure):
    video, action = torch.zeros(2, 4, 3, 4, 4), torch.zeros(2, 3, 7)
    noise = flow_randomness(42, [0, 1], video=video, actions=action)
    if failure == "missing":
        noise.pop("action_noise")
    elif failure == "extra":
        noise["unexpected"] = 1
    elif failure == "range":
        noise["video_timestep_ids"][0] = 1000
    elif failure == "shape":
        noise["action_noise"] = noise["action_noise"][:1]
    elif failure == "dtype":
        noise["action_timestep_ids"] = noise["action_timestep_ids"].float()
    elif failure == "nonfinite":
        noise["video_noise"][0, 0, 0, 0, 0] = float("nan")
    else:
        noise["video_noise"].requires_grad_(True)
    with pytest.raises(ValueError):
        validate_flow_randomness(noise, video=video, actions=action, video_bounds=(0, 1000), action_count=1000)


def test_actual_reader_pairing_survives_workers_batching_and_replay(isolated_dataset):
    root, manifest, stats = isolated_dataset
    base = MultiTaskRoboTwinDataset(
        dataset_dir=str(root), embodiment="aloha-agilex", variant="clean_50",
        normalization_stats_path=str(stats), episode_manifest=str(manifest),
        split="train", action_mode="eef", multiview=False, height=32, width=32,
        color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    )
    paired = PairedExposureDataset(base, seed=42, exposures=8)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first = [paired[i]["pairing"] for i in range(len(paired))]
    assert random.getstate() == py_state
    np.testing.assert_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    loader = torch.utils.data.DataLoader(paired, batch_size=3, num_workers=2, collate_fn=list, shuffle=False)
    second = [sample["pairing"] for batch in loader for sample in batch]
    assert first == second
    assert [paired[i]["pairing"] for i in range(4, 8)] == first[4:]
    with pytest.raises(IndexError):
        paired[8]
