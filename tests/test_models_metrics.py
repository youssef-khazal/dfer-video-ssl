"""Model wrapper and metric tests with tiny randomly initialised models (no downloads)."""

import numpy as np
import pytest
import torch

from dfer.metrics import uar, war
from dfer.models import build_model, count_parameters
from dfer.predict import square_box

TINY_VIDEOMAE = dict(image_size=32, patch_size=16, num_frames=4, tubelet_size=2, hidden_size=32,
                     num_hidden_layers=3, num_attention_heads=2, intermediate_size=64)
TINY_VJEPA = dict(crop_size=32, patch_size=16, frames_per_clip=4, tubelet_size=2, hidden_size=32,
                  num_hidden_layers=3, num_attention_heads=2, mlp_ratio=2.0, pred_hidden_size=32,
                  pred_num_hidden_layers=1, pred_num_attention_heads=2)


@pytest.mark.parametrize("name,cfg", [("videomae_k400", TINY_VIDEOMAE), ("vjepa2", TINY_VJEPA)])
def test_forward_shape(name, cfg):
    model = build_model(name, num_classes=7, config_overrides=cfg, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 4, 3, 32, 32)).shape == (2, 7)


def test_set_trainable_head_only_then_last_blocks():
    model = build_model("videomae_ssl_2stage", 7, TINY_VIDEOMAE, pretrained=False)
    model.set_trainable(0)
    head_only = count_parameters(model, trainable_only=True)
    assert head_only == sum(p.numel() for p in model.head().parameters())
    model.set_trainable(1)
    assert all(p.requires_grad for p in model.blocks()[-1].parameters())
    assert not any(p.requires_grad for p in model.blocks()[0].parameters())
    assert count_parameters(model, trainable_only=True) > head_only
    model.set_trainable(None)
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("resnet", 7, pretrained=False)


def test_uar_vs_war_on_imbalanced_data():
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)  # always predicts the majority class
    assert war(y_true, y_pred) == pytest.approx(0.9)
    assert uar(y_true, y_pred, num_classes=2) == pytest.approx(0.5)


def test_face_box_is_square_and_inside_frame():
    x0, y0, x1, y1 = square_box(10, 10, 50, 40, pad=0.3, width=100, height=100)
    assert 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100


def test_partial_finetuning_gets_gradients_with_checkpointing():
    """Frozen early blocks + gradient checkpointing must still train the unfrozen last block."""
    model = build_model("videomae_ssl_2stage", 7, TINY_VIDEOMAE, pretrained=False)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.set_trainable(1)
    model.train()
    model(torch.randn(2, 4, 3, 32, 32)).sum().backward()
    assert all(p.grad is not None for p in model.blocks()[-1].parameters())
    assert all(p.grad is None for p in model.blocks()[0].parameters())
