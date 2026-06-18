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
        train_mode: str = "frozen",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.image_encoder = build_medsam2_image_encoder(cfg=cfg, ckpt_path=ckpt_path, device=device)
        self.set_train_mode(train_mode)

    def set_train_mode(self, train_mode: str) -> None:
        mode = str(train_mode).lower()
        if mode not in {"frozen", "neck", "full"}:
            raise ValueError(f"Unsupported encoder train mode: {train_mode}")
        self.train_mode = mode

        for param in self.image_encoder.parameters():
            param.requires_grad_(False)

        if mode == "neck":
            for param in self.image_encoder.neck.parameters():
                param.requires_grad_(True)
        elif mode == "full":
            for param in self.image_encoder.parameters():
                param.requires_grad_(True)

        if mode == "frozen":
            self.image_encoder.eval()
        elif mode == "neck":
            self.image_encoder.trunk.eval()
            self.image_encoder.neck.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_mode == "frozen":
            self.image_encoder.eval()
        elif self.train_mode == "neck":
            self.image_encoder.trunk.eval()
            self.image_encoder.neck.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.train_mode == "frozen":
            with torch.no_grad():
                out = self.image_encoder(x)
        else:
            out = self.image_encoder(x)
        return out["backbone_fpn"]
