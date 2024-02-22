# Adapted from https://github.com/bryant1410/fitclip/blob/a55ec2b/aligner/data/frame_sampler.py
import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .video_utils import resample


@dataclass(frozen=True)
class FrameSampler(ABC):
    """A frame sampler that returns the frame indices to seek for the given clip start and end frame indices."""

    @abstractmethod
    def __call__(self, start_frame: int, end_frame: int, fps: float | None = None) -> Iterable[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class RandomFromUniformIntervalsFrameSampler(FrameSampler):
    max_frames: int

    def __call__(self, start_frame: int, end_frame: int, fps: float | None = None) -> Iterable[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
        return [torch.randint(a, b + 1, size=()).item() for a, b in itertools.pairwise(ticks)]


@dataclass(frozen=True)
class UniformFrameSampler(FrameSampler):
    max_frames: int

    def __call__(self, start_frame: int, end_frame: int, fps: float | None = None) -> Iterable[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
        return [torch.round((a + b) / 2).to(torch.int).item() for a, b in itertools.pairwise(ticks)]


@dataclass(frozen=True)
class FixedFrameFromUniformIntervalsFrameSampler(FrameSampler):
    max_frames: int
    frame_index_from_interval_start: int

    def __call__(self, start_frame: int, end_frame: int, fps: float | None = None) -> Iterable[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame + 1, steps=num_frames + 1, dtype=torch.int)
        return (ticks[:-1] + self.frame_index_from_interval_start).tolist()


@dataclass(frozen=True)
class ConsecutiveFrameSampler(FrameSampler):
    max_frames: int
    fps: float | None = None

    def __call__(self, start_frame: int, end_frame: int, fps: float | None = None) -> Iterable[int]:
        if self.fps:
            indices = resample(num_frames=self.max_frames, original_fps=fps, new_fps=self.fps)
        else:
            indices = range(self.max_frames)

        smallest_possible_end = min(end_frame, start_frame + indices[-1])

        if isinstance(smallest_possible_end, torch.Tensor):
            smallest_possible_end = smallest_possible_end.item()  # To avoid a warning in the floor division.
        start = start_frame + (end_frame - smallest_possible_end) // 2

        return itertools.takewhile(lambda i: i <= end_frame, (start + i for i in indices))
