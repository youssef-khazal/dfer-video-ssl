"""Data pipeline tests: frame order, clip-consistent augmentation, leakage-free splits."""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from dfer.data import ClipTransform, DFEWClips, class_weights, fold_tables, frame_index, sorted_frames


def make_clip(root, video_id, n=16, size=64):
    """Frames whose brightness equals their frame number, so order is checkable."""
    d = root / str(video_id).zfill(5)
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        Image.new("RGB", (size, size), (k * 10, k * 10, k * 10)).save(d / f"{k}.jpg", quality=100)
    return d


def test_frames_are_sorted_numerically_not_alphabetically(tmp_path):
    d = make_clip(tmp_path, 1)
    names = [p.name for p in sorted_frames(d)]
    assert names == [f"{k}.jpg" for k in range(1, 17)]  # not 1, 10, 11, ..., 2
    assert frame_index("clip_00012.jpg") == 12


def test_dataset_returns_frames_in_time_order(tmp_path):
    make_clip(tmp_path, 7)
    ds = DFEWClips(tmp_path, pd.DataFrame({"video_name": [7], "label": [3]}), ClipTransform(32, train=False))
    clip, label = ds[0]
    assert clip.shape == (16, 3, 32, 32) and label == 2
    brightness = clip.mean(dim=(1, 2, 3))
    assert torch.all(brightness[1:] > brightness[:-1])


def test_train_augmentation_is_identical_across_frames():
    random.seed(0)
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    clip = ClipTransform(48, train=True, scale=(0.3, 0.6))([base] * 16)
    assert all(torch.equal(clip[0], clip[t]) for t in range(1, 16))


def write_split(root, train_labels, test_labels, fold=1):
    split = root / "EmoLabel_DataSplit"
    (split / "train(single-labeled)").mkdir(parents=True)
    (split / "test(single-labeled)").mkdir(parents=True)
    pd.DataFrame({"video_name": range(len(train_labels)), "label": train_labels}).to_csv(
        split / "train(single-labeled)" / f"set_{fold}.csv", index=False)
    pd.DataFrame({"video_name": range(10_000, 10_000 + len(test_labels)), "label": test_labels}).to_csv(
        split / "test(single-labeled)" / f"set_{fold}.csv", index=False)


def test_validation_comes_from_train_and_never_touches_test(tmp_path):
    labels = np.repeat(np.arange(1, 8), 50)
    write_split(tmp_path, labels, np.repeat(np.arange(1, 8), 10))
    train, val, test = fold_tables(tmp_path, fold=1, val_fraction=0.1, seed=0)
    assert set(train.video_name).isdisjoint(val.video_name)
    assert set(val.video_name).isdisjoint(test.video_name)
    assert len(train) + len(val) == len(labels)
    assert set(val.label) == set(range(1, 8))  # stratified: every emotion in validation


def test_class_weights_favour_rare_classes():
    w = class_weights(np.array([1] * 90 + [2] * 10), num_classes=2)
    assert w[1] > w[0]
    assert w.mean().item() == pytest.approx(1.0)
