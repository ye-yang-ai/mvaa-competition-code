#!/usr/bin/env python3
"""Model and loss factory for task3 2D segmentation."""

from __future__ import annotations

from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    import segmentation_models_pytorch.encoders as smp_encoders
    from segmentation_models_pytorch.encoders.timm_universal import TimmUniversalEncoder
except Exception as e:  # pragma: no cover
    smp = None
    smp_encoders = None
    TimmUniversalEncoder = None
    _SMP_IMPORT_ERROR = e
else:
    _SMP_IMPORT_ERROR = None

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
LOCAL_SMP_WEIGHTS = {
    "efficientnet-b4": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "efficientnet-b4-imagenet" / "model.safetensors",
    "efficientnet-b5": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "efficientnet-b5-imagenet" / "model.safetensors",
    "efficientnet-b3": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "efficientnet-b3-imagenet" / "model.safetensors",
    "resnet50": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "resnet50-imagenet" / "model.safetensors",
    "resnet34": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "resnet34-imagenet" / "model.safetensors",
    "mit_b2": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "mit_b2-imagenet" / "model.safetensors",
    "mit_b3": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "mit_b3-imagenet" / "model.safetensors",
    "mit_b4": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "mit_b4-imagenet" / "model.safetensors",
    "tu-convnext_tiny.fb_in1k": REPO_ROOT
    / "checkpoints"
    / "pretrained"
    / "smp"
    / "tu-convnext_tiny.fb_in1k"
    / "model.safetensors",
}


if TimmUniversalEncoder is not None:

    class TimmUniversalHalfScaleEncoder(TimmUniversalEncoder):
        def __init__(self, *args, half_scale_channels: int = 32, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.half_scale_channels = int(half_scale_channels)
            self.half_scale = nn.Sequential(
                nn.Conv2d(self._in_channels, self.half_scale_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.half_scale_channels),
                nn.ReLU(inplace=True),
            )
            if getattr(self, "_is_transformer_style", False):
                self._out_channels = [self._in_channels, self.half_scale_channels] + self.model.feature_info.channels()

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            if not getattr(self, "_is_transformer_style", False):
                return super().forward(x)

            features = self.model(x)
            if self._is_channel_last:
                features = [feature.permute(0, 3, 1, 2).contiguous() for feature in features]
            return [x, self.half_scale(x)] + features

else:
    TimmUniversalHalfScaleEncoder = None


def _get_local_smp_weight_path(encoder_name: str, encoder_weights: str | None) -> Path | None:
    if encoder_weights is None:
        return None
    if str(encoder_weights).lower() != "imagenet":
        return None
    return LOCAL_SMP_WEIGHTS.get(str(encoder_name).lower())


def _convert_timm_universal_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("head."):
            continue
        new_key = re.sub(r"^stem\.(\d+)\.", r"model.stem_\1.", key)
        new_key = re.sub(r"^stages\.(\d+)\.", r"model.stages_\1.", new_key)
        if not new_key.startswith("model."):
            new_key = f"model.{new_key}"
        converted[new_key] = value
    return converted


def _load_local_encoder_weights(model: nn.Module, weights_path: Path, encoder_name: str) -> None:
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover
        raise ImportError("safetensors is required to load local SMP encoder weights.") from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"Local encoder weight file not found: {weights_path}")
    if not hasattr(model, "encoder"):
        raise AttributeError("SMP model does not expose an encoder module.")

    state_dict = load_file(str(weights_path), device="cpu")
    if str(encoder_name).lower().startswith("tu-"):
        state_dict = _convert_timm_universal_state_dict(state_dict)
    try:
        model.encoder.load_state_dict(state_dict, strict=False)
    except TypeError:
        model.encoder.load_state_dict(state_dict)


def get_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
):
    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required but not installed. "
            f"Original import error: {_SMP_IMPORT_ERROR!r}"
        )

    arch = arch.lower()
    local_weight_path = _get_local_smp_weight_path(encoder_name, encoder_weights)
    smp_encoder_weights = None if local_weight_path is not None else encoder_weights

    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == "unetplusplus":
        if str(encoder_name).lower().startswith("tu-"):
            if smp_encoders is None or TimmUniversalHalfScaleEncoder is None:
                raise ImportError(
                    "segmentation_models_pytorch timm universal encoder is required for tu-* backbones. "
                    f"Original import error: {_SMP_IMPORT_ERROR!r}"
                )
            original_timm_universal_encoder = smp_encoders.TimmUniversalEncoder
            smp_encoders.TimmUniversalEncoder = TimmUniversalHalfScaleEncoder
            try:
                model = smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=smp_encoder_weights,
                    in_channels=in_channels,
                    classes=classes,
                )
            finally:
                smp_encoders.TimmUniversalEncoder = original_timm_universal_encoder
        else:
            model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=smp_encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == "segformer":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if local_weight_path is not None:
        _load_local_encoder_weights(model, local_weight_path, encoder_name)

    return model


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.75,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp_min(1e-6) ** self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        if self.pos_weight is not None and self.pos_weight != 1.0:
            pos_w = 1.0 + (self.pos_weight - 1.0) * targets
            loss = loss * pos_w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.75,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.focal = BinaryFocalWithLogitsLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


def get_loss_fn(
    loss_type: str = "dice_focal",
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
    focal_weight: float = 0.3,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.75,
    pos_weight: float | None = None,
) -> nn.Module:
    loss_type = str(loss_type).lower()
    if loss_type == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, pos_weight=pos_weight)
    if loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")
