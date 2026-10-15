"""Run the full training script on a tiny fake DFEW fold (CPU, tiny model, a few seconds)."""

import json

import numpy as np
import pandas as pd
from PIL import Image

import dfer.train as train_module
from dfer.models import build_model

TINY = dict(image_size=32, patch_size=16, num_frames=16, tubelet_size=2, hidden_size=32,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)


def fake_dfew(root, n_train=70, n_test=14):
    clips = root / "Clip" / "clip_224x224_16f"
    split = root / "EmoLabel_DataSplit"
    rng = np.random.default_rng(0)
    for name, n, offset in (("train", n_train, 1), ("test", n_test, 5000)):
        (split / f"{name}(single-labeled)").mkdir(parents=True)
        ids = list(range(offset, offset + n))
        labels = [i % 7 + 1 for i in range(n)]
        pd.DataFrame({"video_name": ids, "label": labels}).to_csv(split / f"{name}(single-labeled)" / "set_1.csv", index=False)
        for vid in ids:
            d = clips / str(vid).zfill(5)
            d.mkdir(parents=True)
            for k in range(1, 17):
                Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(d / f"{k}.jpg")


def test_two_stage_training_writes_metrics(tmp_path, monkeypatch):
    fake_dfew(tmp_path / "dfew")
    monkeypatch.setattr(train_module, "build_model",
                        lambda name, n: build_model(name, n, config_overrides=TINY, pretrained=False))
    args = train_module.parse_args([
        "--dfew-root", str(tmp_path / "dfew"), "--model", "videomae_ssl_2stage", "--img-size", "32",
        "--epochs", "1", "--stage1-epochs", "1", "--unfreeze-blocks", "1", "--batch-size", "4",
        "--grad-accum", "1", "--workers", "0", "--out-dir", str(tmp_path / "runs"),
    ])
    metrics = train_module.train(args)
    run = tmp_path / "runs" / "videomae_ssl_2stage_fold1_seed42"
    saved = json.loads((run / "metrics.json").read_text())
    assert saved["test"]["uar"] == metrics["test"]["uar"]
    assert 0.0 <= saved["test"]["war"] <= 1.0
    assert sum(map(sum, saved["test_confusion_matrix"])) == 14  # every test clip evaluated exactly once
    assert (run / "best.pth").exists() and not (run / "last.pth").exists()
    assert len(pd.read_csv(run / "history.csv")) == 2  # one head epoch + one partial epoch


def test_oom_fallback_keeps_effective_batch(monkeypatch):
    calls = []

    def fake_train(args):
        calls.append((args.batch_size, args.grad_accum, args.resume))
        if len(calls) == 1:
            raise train_module.torch.OutOfMemoryError("CUDA out of memory")
        return {"ok": True}

    monkeypatch.setattr(train_module, "train", fake_train)
    monkeypatch.setattr(train_module.torch.cuda, "empty_cache", lambda: None)
    args = train_module.parse_args(["--dfew-root", "x", "--batch-size", "2", "--grad-accum", "4"])
    assert train_module.train_with_oom_fallback(args) == {"ok": True}
    assert calls == [(2, 4, False), (1, 8, True)]
