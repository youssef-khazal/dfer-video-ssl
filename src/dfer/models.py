"""Video backbones compared in this project, all from the Hugging Face Hub.

| name               | checkpoint                                   | pre-training                                  | fine-tuning            |
|--------------------|----------------------------------------------|-----------------------------------------------|------------------------|
| videomae_k400      | MCG-NJU/videomae-base-finetuned-kinetics     | VideoMAE self-supervised + supervised K400    | full                   |
| videomae_ssl       | MCG-NJU/videomae-base                        | VideoMAE self-supervised only (K400 videos)   | full                   |
| videomae_ssl_2stage| MCG-NJU/videomae-base                        | VideoMAE self-supervised only                 | head, then last blocks |
| vjepa2             | facebook/vjepa2-vitl-fpc64-256               | V-JEPA 2 self-supervised (ViT-L)              | head, then last blocks |
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    family: str  # "videomae" or "vjepa2"
    two_stage: bool  # True: train head only, then unfreeze the last `unfreeze_blocks` blocks


MODELS: dict[str, ModelSpec] = {
    "videomae_k400": ModelSpec("MCG-NJU/videomae-base-finetuned-kinetics", "videomae", two_stage=False),
    "videomae_ssl": ModelSpec("MCG-NJU/videomae-base", "videomae", two_stage=False),
    "videomae_ssl_2stage": ModelSpec("MCG-NJU/videomae-base", "videomae", two_stage=True),
    "vjepa2": ModelSpec("facebook/vjepa2-vitl-fpc64-256", "vjepa2", two_stage=True),
}


class VideoClassifier(nn.Module):
    """Uniform wrapper: forward(video[B, T, C, H, W]) -> logits[B, num_classes]."""

    def __init__(self, hf_model: nn.Module, family: str):
        super().__init__()
        self.model, self.family = hf_model, family

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.family == "vjepa2":
            return self.model(pixel_values_videos=video).logits
        return self.model(pixel_values=video).logits

    # --- parts used for freezing -------------------------------------------------
    def backbone(self) -> nn.Module:
        return self.model.vjepa2 if self.family == "vjepa2" else self.model.videomae

    def blocks(self) -> nn.ModuleList:
        return self.backbone().encoder.layer

    def head(self) -> nn.Module:
        return self.model.classifier

    def set_trainable(self, last_blocks: int | None) -> None:
        """``None``: train everything. ``0``: head only. ``k``: head + last k encoder blocks (+ final norm)."""
        if last_blocks is None:
            for p in self.parameters():
                p.requires_grad = True
            return
        for p in self.parameters():
            p.requires_grad = False
        trainable = [self.head()]
        if last_blocks > 0:
            trainable += list(self.blocks()[-last_blocks:])
            for name in ("layernorm", "fc_norm"):
                norm = getattr(self.model, name, None) or getattr(self.backbone(), name, None)
                if isinstance(norm, nn.Module):
                    trainable.append(norm)
        for module in trainable:
            for p in module.parameters():
                p.requires_grad = True


def build_model(name: str, num_classes: int, config_overrides: dict | None = None, pretrained: bool = True) -> VideoClassifier:
    """Load a backbone from the Hub with a fresh ``num_classes`` head.

    ``pretrained=False`` builds a randomly initialised model from the config (used in tests);
    ``config_overrides`` shrinks it.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]
    if spec.family == "vjepa2":
        from transformers import VJEPA2Config as Config, VJEPA2ForVideoClassification as Model
    else:
        from transformers import VideoMAEConfig as Config, VideoMAEForVideoClassification as Model

    if pretrained:
        hf = Model.from_pretrained(spec.checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        cfg = Config(num_labels=num_classes, **(config_overrides or {}))
        hf = Model(cfg)
    return VideoClassifier(hf, spec.family)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
