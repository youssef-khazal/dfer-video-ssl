"""DFEW dataset loading, clip-consistent augmentation and splits.

DFEW ships pre-extracted clips with 16 time-interpolated frames per video:

    <dfew_root>/Clip/clip_224x224_16f/<00001>/1.jpg ... 16.jpg

and 5-fold, subject-independent splits:

    <dfew_root>/EmoLabel_DataSplit/train(single-labeled)/set_<k>.csv
    <dfew_root>/EmoLabel_DataSplit/test(single-labeled)/set_<k>.csv

Each fold's *test* file is kept untouched until the final evaluation. Model selection
(early stopping, best epoch) uses a validation split carved out of the fold's *train* file.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# DFEW label ids are 1..7 in this order; models use 0..6.
EMOTIONS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTIONS)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DIGITS = re.compile(r"(\d+)(?=\.\w+$)")


def frame_index(path: str | Path) -> int:
    """Numeric index of a frame file: '7.jpg' -> 7, 'clip_00012.jpg' -> 12.

    Sorting by this (not alphabetically) keeps frames in time order: 1, 2, ..., 10,
    not 1, 10, 11, ..., 2.
    """
    match = _DIGITS.search(Path(path).name)
    if match is None:
        raise ValueError(f"No frame number in file name: {path}")
    return int(match.group(1))


def sorted_frames(clip_dir: Path) -> list[Path]:
    return sorted((p for p in clip_dir.iterdir() if p.suffix.lower() == ".jpg"), key=frame_index)


class ClipTransform:
    """Apply the *same* random crop / flip to every frame of a clip, then normalise.

    Returns a tensor of shape (T, C, H, W), the layout Hugging Face video models expect.
    """

    def __init__(self, size: int = 224, train: bool = False, scale: tuple[float, float] = (0.8, 1.0)):
        self.size, self.train, self.scale = size, train, scale

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        if self.train:
            i, j, h, w = _random_resized_crop_params(frames[0], self.scale)
            flip = random.random() < 0.5
            frames = [TF.resized_crop(f, i, j, h, w, [self.size, self.size]) for f in frames]
            if flip:
                frames = [TF.hflip(f) for f in frames]
        else:
            frames = [TF.resize(f, [self.size, self.size]) for f in frames]
        clip = torch.stack([TF.to_tensor(f) for f in frames])
        return TF.normalize(clip, IMAGENET_MEAN, IMAGENET_STD)


def _random_resized_crop_params(img: Image.Image, scale: tuple[float, float]) -> tuple[int, int, int, int]:
    """Square-ish random crop covering ``scale`` of the image area (aspect ratio kept near 1)."""
    width, height = img.size
    area = width * height * random.uniform(*scale)
    side = int(round(area**0.5))
    side = min(side, width, height)
    i = random.randint(0, height - side)
    j = random.randint(0, width - side)
    return i, j, side, side


class DFEWClips(Dataset):
    """DFEW clips listed in a dataframe with columns ``video_name`` and ``label`` (1..7)."""

    def __init__(self, clip_root: str | Path, table: pd.DataFrame, transform: ClipTransform, num_frames: int = 16):
        self.clip_root = Path(clip_root)
        self.table = table.reset_index(drop=True)
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.table)

    def clip_dir(self, idx: int) -> Path:
        return self.clip_root / str(self.table.loc[idx, "video_name"]).zfill(5)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        files = sorted_frames(self.clip_dir(idx))
        if len(files) < self.num_frames:
            raise RuntimeError(f"{self.clip_dir(idx)} has {len(files)} frames, need {self.num_frames}")
        # Uniformly sample num_frames over the clip (identity when the clip has exactly num_frames).
        picks = np.linspace(0, len(files) - 1, self.num_frames).round().astype(int)
        frames = []
        for k in picks:
            with Image.open(files[k]) as img:
                frames.append(img.convert("RGB"))
        label = int(self.table.loc[idx, "label"]) - 1
        return self.transform(frames), label


def fold_tables(dfew_root: str | Path, fold: int = 1, val_fraction: float = 0.1, seed: int = 42):
    """Return (train, val, test) dataframes for one DFEW fold.

    ``val`` is a stratified sample of the fold's training file; ``test`` is the official test file.
    """
    split_dir = Path(dfew_root) / "EmoLabel_DataSplit"
    train_full = pd.read_csv(split_dir / "train(single-labeled)" / f"set_{fold}.csv")
    test = pd.read_csv(split_dir / "test(single-labeled)" / f"set_{fold}.csv")
    train, val = train_test_split(
        train_full, test_size=val_fraction, stratify=train_full["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def class_weights(labels: pd.Series | np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights (mean 1) for labels given as 1..num_classes."""
    counts = np.bincount(np.asarray(labels) - 1, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
