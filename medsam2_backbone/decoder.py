"""Lightweight FPN decoder for MedSAM2 image encoder features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightFPNDecoder(nn.Module):
    """Fuse stride 4/8/16 MedSAM2 FPN features and produce segmentation logits."""

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: int = 128,
        out_channels: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if decoder_channels % num_groups != 0:
            raise ValueError("decoder_channels must be divisible by num_groups.")

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(in_channels, decoder_channels, kernel_size=1),
                nn.Conv2d(in_channels, decoder_channels, kernel_size=1),
                nn.Conv2d(in_channels, decoder_channels, kernel_size=1),
            ]
        )
        self.refine = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels, num_groups=num_groups),
            ConvNormAct(decoder_channels, decoder_channels, num_groups=num_groups),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} feature levels, got {len(features)}.")

        target_size = features[0].shape[-2:]
        fused = self.projections[0](features[0])
        for feat, proj in zip(features[1:], self.projections[1:]):
            x = proj(feat)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            fused = fused + x

        logits = self.refine(fused)
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, num_groups=num_groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class LightFPNDecoderV2(nn.Module):
    """Stronger decoder with concat fusion and residual refinement for less fragmented masks."""

    def __init__(
        self,
        in_channels: int = 256,
        decoder_channels: int = 128,
        out_channels: int = 1,
        num_groups: int = 8,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        if decoder_channels % num_groups != 0:
            raise ValueError("decoder_channels must be divisible by num_groups.")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=decoder_channels),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=decoder_channels),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=decoder_channels),
                    nn.SiLU(inplace=True),
                ),
            ]
        )
        self.fuse = ConvNormAct(decoder_channels * 3, decoder_channels, num_groups=num_groups)
        blocks = [ResidualConvBlock(decoder_channels, num_groups=num_groups) for _ in range(int(residual_blocks))]
        self.refine = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels, num_groups=num_groups),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} feature levels, got {len(features)}.")

        target_size = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if tuple(x.shape[-2:]) != tuple(target_size):
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.refine(x)
        logits = self.head(x)
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
