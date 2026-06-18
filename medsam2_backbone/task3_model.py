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
        encoder_train_mode: str = "frozen",
        freeze_encoder: bool | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if freeze_encoder is not None:
            encoder_train_mode = "frozen" if bool(freeze_encoder) else "full"
        self.encoder_train_mode = str(encoder_train_mode).lower()
        self.freeze_encoder = self.encoder_train_mode == "frozen"
        self.decoder_version = str(decoder_version).lower()
        self.encoder = MedSAM2ImageEncoder(
            cfg=encoder_cfg,
            ckpt_path=encoder_ckpt,
            train_mode=self.encoder_train_mode,
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
        if self.encoder_train_mode == "frozen":
            self.encoder.eval()
        elif self.encoder_train_mode == "neck":
            self.encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features, output_size=tuple(x.shape[-2:]))
