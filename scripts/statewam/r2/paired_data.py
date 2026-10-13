"""Stateless exposure-indexed sampling and independent flow-matching streams."""

import hashlib
import random

import torch

from scripts.statewam.r2.common import stable_hash


def stream_seed(seed, occurrence, stream):
    if occurrence < 0 or not stream:
        raise ValueError("Occurrence must be nonnegative and stream must be named")
    return int(stable_hash(["r2-paired-rng-v1", int(seed), int(occurrence), stream])[:16], 16) % (2**63)


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return stable_hash({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "bytes": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy()).hexdigest()})


def sample_hashes(sample):
    pixels = hashlib.sha256()
    for frame in sample["video"]:
        pixels.update(frame.mode.encode())
        pixels.update(str(frame.size).encode())
        pixels.update(frame.tobytes())
    return {"video": pixels.hexdigest(), "prompt": stable_hash(sample["prompt"]),
            **{name: tensor_hash(sample[name]) for name in (
                "action", "proprio", "action_mask", "proprio_mask", "video_mask")}}


class PairedExposureDataset(torch.utils.data.Dataset):
    """One index is one global sample occurrence, independent of worker/rank/batch."""

    def __init__(self, dataset, *, seed, exposures):
        if exposures < 1 or not dataset._sub_datasets:
            raise ValueError("Require positive exposures and at least one task")
        if any(ds.split != "train" or not ds.strict_data for ds in dataset._sub_datasets):
            raise ValueError("Paired training requires explicitly isolated strict train data")
        self.dataset, self.seed, self.exposures = dataset, int(seed), int(exposures)
        self.action_dim = dataset.action_dim
        self.normalization_stats = dataset.normalization_stats
        self.normalization_stats_path = dataset.normalization_stats_path

    def __len__(self):
        return self.exposures

    def location(self, occurrence):
        if not 0 <= occurrence < len(self):
            raise IndexError(occurrence)
        sampler = random.Random(stream_seed(self.seed, occurrence, "sample"))
        task = sampler.randrange(len(self.dataset._sub_datasets))
        return task, sampler.randrange(len(self.dataset._sub_datasets[task]))

    def __getitem__(self, occurrence):
        task_index, window_index = self.location(occurrence)
        task = self.dataset._sub_datasets[task_index]
        sample = task.get_sample(
            window_index, instruction_seed=stream_seed(self.seed, occurrence, "instruction"),
            augmentation_seed=stream_seed(self.seed, occurrence, "augmentation"),
        )
        sample["pairing"] = {"occurrence": occurrence, "task_index": task_index, "window_index": window_index,
                             "episode_path": sample["episode_path"], "start_frame": sample["start_frame"],
                             "hashes": sample_hashes(sample)}
        return sample


def flow_randomness(seed, occurrences, *, video, actions, video_bounds=(0, 1000), action_count=1000):
    if len(occurrences) != len(video) or (actions is not None and len(actions) != len(video)):
        raise ValueError("Occurrence count does not match batch")
    result = {}
    for name, reference, bounds in (("video", video, video_bounds), ("action", actions, (0, action_count))):
        if reference is None:
            continue
        if bounds[0] >= bounds[1]:
            raise ValueError("Empty timestep range")
        ids, noises = [], []
        for occurrence in occurrences:
            ids.append(torch.randint(*bounds, (1,), generator=torch.Generator().manual_seed(
                stream_seed(seed, occurrence, f"{name}_timestep"))))
            noises.append(torch.randn(reference.shape[1:], dtype=reference.dtype, device=reference.device,
                                      generator=torch.Generator(device=reference.device).manual_seed(
                                          stream_seed(seed, occurrence, f"{name}_noise"))))
        result[f"{name}_timestep_ids"] = torch.cat(ids)
        result[f"{name}_noise"] = torch.stack(noises)
    return result


def randomness_hashes(randomness):
    return [{name: tensor_hash(tensor[index]) for name, tensor in randomness.items()}
            for index in range(len(randomness["video_timestep_ids"]))]
