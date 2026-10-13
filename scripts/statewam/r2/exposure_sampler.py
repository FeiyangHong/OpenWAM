"""Rank-local batches with one deterministic global exposure cursor."""

from torch.utils.data import Sampler


class ExposureBatchSampler(Sampler):
    def __init__(self, *, updates, micro_batch, accumulation, world_size, rank, start_update=0):
        values = (updates, micro_batch, accumulation, world_size)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Require positive integer training dimensions")
        if not 0 <= rank < world_size or not 0 <= start_update <= updates:
            raise ValueError("Invalid rank or resume update")
        self.updates, self.micro_batch, self.accumulation = updates, micro_batch, accumulation
        self.world_size, self.rank, self.start_update = world_size, rank, start_update

    def __len__(self):
        return (self.updates - self.start_update) * self.accumulation

    def __iter__(self):
        stride = self.world_size * self.micro_batch
        for microstep in range(self.start_update * self.accumulation, self.updates * self.accumulation):
            first = microstep * stride + self.rank * self.micro_batch
            yield list(range(first, first + self.micro_batch))
