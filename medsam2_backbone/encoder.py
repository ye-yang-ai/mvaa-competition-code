"""Thin wrapper around the MedSAM2 image encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .build_encoder import DEFAULT_ENCODER_CFG, DEFAULT_ENCODER_CKPT, build_medsam2_image_encoder


class MedSAM2ImageEncoder(nn.Module):
    """Return MedSAM2 backbone_fpn features for an already-normalized image tensor."""

    def __init__(
        self,
        cfg: str = DEFAULT_ENCODER_CFG,
        ckpt_path: str | Path = DEFAULT_ENCODER_CKPT,
        freeze: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.freeze = bool(freeze)
        self.image_encoder = build_medsam2_image_encoder(cfg=cfg, ckpt_path=ckpt_path, device=device)
        self.set_freeze(self.freeze)

    def set_freeze(self, freeze: bool) -> None:
        self.freeze = bool(freeze)
        for param in self.image_encoder.parameters():
            param.requires_grad_(not self.freeze)
        if self.freeze:
            self.image_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.image_encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.freeze:
            with torch.no_grad():
                out = self.image_encoder(x)
        else:
            out = self.image_encoder(x)
        return out["backbone_fpn"]
