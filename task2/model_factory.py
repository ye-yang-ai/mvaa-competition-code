#!/usr/bin/env python3
"""Model and loss factory for full-supervised cardiac segmentation."""

from __future__ import annotations

from typing import Sequence

from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet, UNet


def get_model(
    name: str = "unet3d",
    model_size: str = "large",
    in_channels: int = 1,
    out_channels: int = 3,
    channels: Sequence[int] | None = None,
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int | None = None,
):
    """Create segmentation model by name."""
    name = name.lower()
    model_size = model_size.lower()

    if channels is None or num_res_units is None:
        if model_size == "small":
            channels = (16, 32, 64, 128, 256)
            num_res_units = 2
        elif model_size == "base":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 2
        elif model_size == "large":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 3
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

    if name in {"unet", "unet3d"}:
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
        )

    if name in {"segresnet", "segresnet3d"}:
        if model_size == "small":
            init_filters = 16
            blocks_down = (1, 2, 2, 4)
            blocks_up = (1, 1, 1)
        elif model_size == "base":
            init_filters = 24
            blocks_down = (1, 2, 2, 4)
            blocks_up = (1, 1, 1)
        elif model_size == "large":
            init_filters = 32
            blocks_down = (1, 2, 2, 4)
            blocks_up = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=0.0,
            norm=("GROUP", {"num_groups": 8}),
            act=("RELU", {"inplace": True}),
            upsample_mode="nontrainable",
        )

    raise ValueError(f"Unsupported model name: {name}")


def get_loss_fn():
    """Dice + CE for stable supervised training."""
    return DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.8, lambda_ce=0.2)
