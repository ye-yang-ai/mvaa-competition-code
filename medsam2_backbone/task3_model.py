"""Task3 automatic segmentation model using MedSAM2 image encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .build_encoder import DEFAULT_ENCODER_CFG, DEFAULT_ENCODER_CKPT
from .decoder import LightFPNDecoder, LightFPNDecoderV2
from .encoder import MedSAM2ImageEncoder


class Task3MedSAM2EncoderSeg(nn.Module):
    """MedSAM2 image_encoder + LightFPNDecoder for Task3 binary masks."""

    def __init__(
        self,
        encoder_cfg: str = DEFAULT_ENCODER_CFG,
        encoder_ckpt: str | Path = DEFAULT_ENCODER_CKPT,
        decoder_channels: int = 128,
        decoder_version: str = "v1",
        freeze_encoder: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        self.decoder_version = str(decoder_version).lower()
        self.encoder = MedSAM2ImageEncoder(
            cfg=encoder_cfg,
            ckpt_path=encoder_ckpt,
            freeze=freeze_encoder,
            device=device,
        )
        if self.decoder_version == "v1":
            self.decoder = LightFPNDecoder(
                in_channels=256,
                decoder_channels=int(decoder_channels),
                out_channels=1,
            )
        elif self.decoder_version == "v2":
            self.decoder = LightFPNDecoderV2(
                in_channels=256,
                decoder_channels=int(decoder_channels),
                out_channels=1,
            )
        else:
            raise ValueError(f"Unsupported decoder_version: {decoder_version}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features, output_size=tuple(x.shape[-2:]))
