#!/usr/bin/env python3
"""Model and loss factory for task3 2D segmentation."""

from __future__ import annotations

from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
    "efficientnet-b6": REPO_ROOT / "checkpoints" / "pretrained" / "smp" / "efficientnet-b6-imagenet" / "model.safetensors",
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
DEFAULT_LEMONFM_CKPT = REPO_ROOT / "checkpoints" / "pretrained" / "lemonfm" / "lemonfm.pth"


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, int(num_channels))
    while int(num_channels) % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, int(num_channels))


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = int(kernel_size) // 2
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            _group_norm(int(out_channels)),
            nn.GELU(),
        )


class LemonFMConvNeXtLargeFPN(nn.Module):
    """LemonFM ConvNeXt-Large encoder with a lightweight FPN decoder."""

    encoder_channels = (192, 384, 768, 1536)

    def __init__(
        self,
        pretrained_weights: str | Path | None = DEFAULT_LEMONFM_CKPT,
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("LemonFM ConvNeXt-Large expects RGB input with in_channels=3.")

        base = torchvision.models.convnext_large(weights=None)
        base.classifier[2] = nn.Identity()
        if pretrained_weights is not None:
            self._load_lemonfm_weights(base, Path(pretrained_weights))

        self.backbone = base.features
        c = int(decoder_channels)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, c, kernel_size=1) for ch in self.encoder_channels]
        )
        self.smooth_convs = nn.ModuleList([ConvNormAct(c, c, kernel_size=3) for _ in self.encoder_channels])
        self.seg_head = nn.Sequential(
            ConvNormAct(c * 4, c, kernel_size=3),
            nn.Conv2d(c, int(classes), kernel_size=1),
        )

    @staticmethod
    def _load_lemonfm_weights(base: nn.Module, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"LemonFM checkpoint not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "teacher" not in ckpt:
            raise ValueError(f"LemonFM checkpoint must contain a 'teacher' state dict: {weights_path}")
        state_dict = {
            k.replace("backbone.", "", 1): v
            for k, v in ckpt["teacher"].items()
            if k.startswith("backbone.")
        }
        msg = base.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                "Failed to load LemonFM backbone exactly: "
                f"missing={msg.missing_keys[:10]} unexpected={msg.unexpected_keys[:10]}"
            )

    def encoder_parameters(self):
        return self.backbone.parameters()

    def decoder_parameters(self):
        for module in (self.lateral_convs, self.smooth_convs, self.seg_head):
            yield from module.parameters()

    def set_encoder_tail_trainable(self, tail_modules: int) -> list[str]:
        """Freeze the encoder, then unfreeze the last N ConvNeXt feature modules."""
        n_tail = max(0, int(tail_modules))
        for p in self.encoder_parameters():
            p.requires_grad_(False)
        if n_tail <= 0:
            return []

        modules = list(self.backbone.children())
        start = max(0, len(modules) - n_tail)
        names = []
        for idx, module in enumerate(modules[start:], start=start):
            for p in module.parameters():
                p.requires_grad_(True)
            names.append(f"backbone[{idx}]")
        return names

    def _encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        f1 = x
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        f2 = x
        x = self.backbone[4](x)
        x = self.backbone[5](x)
        f3 = x
        x = self.backbone[6](x)
        x = self.backbone[7](x)
        f4 = x
        return [f1, f2, f3, f4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = tuple(int(v) for v in x.shape[-2:])
        feats = self._encoder_features(x)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        p4 = laterals[3]
        p3 = laterals[2] + F.interpolate(p4, size=laterals[2].shape[-2:], mode="bilinear", align_corners=False)
        p2 = laterals[1] + F.interpolate(p3, size=laterals[1].shape[-2:], mode="bilinear", align_corners=False)
        p1 = laterals[0] + F.interpolate(p2, size=laterals[0].shape[-2:], mode="bilinear", align_corners=False)

        pyramid = [p1, p2, p3, p4]
        pyramid = [smooth(feat) for smooth, feat in zip(self.smooth_convs, pyramid)]
        target = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                pyramid[0],
                F.interpolate(pyramid[1], size=target, mode="bilinear", align_corners=False),
                F.interpolate(pyramid[2], size=target, mode="bilinear", align_corners=False),
                F.interpolate(pyramid[3], size=target, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        logits = self.seg_head(fused)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvNormAct(int(in_channels), int(out_channels), kernel_size=1),
                )
                for scale in pool_scales
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        size = tuple(int(v) for v in x.shape[-2:])
        return [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]


class LemonFMConvNeXtLargeUPerNet(nn.Module):
    """LemonFM ConvNeXt-Large encoder with a UPerNet-style decoder."""

    encoder_channels = (192, 384, 768, 1536)

    def __init__(
        self,
        pretrained_weights: str | Path | None = DEFAULT_LEMONFM_CKPT,
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: int = 128,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("LemonFM ConvNeXt-Large expects RGB input with in_channels=3.")

        base = torchvision.models.convnext_large(weights=None)
        base.classifier[2] = nn.Identity()
        if pretrained_weights is not None:
            LemonFMConvNeXtLargeFPN._load_lemonfm_weights(base, Path(pretrained_weights))

        self.backbone = base.features
        c = int(decoder_channels)
        top_channels = int(self.encoder_channels[-1])
        self.ppm = PyramidPoolingModule(top_channels, c, pool_scales=pool_scales)
        self.ppm_bottleneck = ConvNormAct(top_channels + len(pool_scales) * c, c, kernel_size=3)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, c, kernel_size=1) for ch in self.encoder_channels[:-1]])
        self.fpn_convs = nn.ModuleList([ConvNormAct(c, c, kernel_size=3) for _ in self.encoder_channels[:-1]])
        self.fpn_bottleneck = ConvNormAct(c * len(self.encoder_channels), c, kernel_size=3)
        self.seg_head = nn.Conv2d(c, int(classes), kernel_size=1)

    def encoder_parameters(self):
        return self.backbone.parameters()

    def decoder_parameters(self):
        for module in (self.ppm, self.ppm_bottleneck, self.lateral_convs, self.fpn_convs, self.fpn_bottleneck, self.seg_head):
            yield from module.parameters()

    def set_encoder_tail_trainable(self, tail_modules: int) -> list[str]:
        """Freeze the encoder, then unfreeze the last N ConvNeXt feature modules."""
        n_tail = max(0, int(tail_modules))
        for p in self.encoder_parameters():
            p.requires_grad_(False)
        if n_tail <= 0:
            return []

        modules = list(self.backbone.children())
        start = max(0, len(modules) - n_tail)
        names = []
        for idx, module in enumerate(modules[start:], start=start):
            for p in module.parameters():
                p.requires_grad_(True)
            names.append(f"backbone[{idx}]")
        return names

    def _encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        f1 = x
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        f2 = x
        x = self.backbone[4](x)
        x = self.backbone[5](x)
        f3 = x
        x = self.backbone[6](x)
        x = self.backbone[7](x)
        f4 = x
        return [f1, f2, f3, f4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = tuple(int(v) for v in x.shape[-2:])
        feats = self._encoder_features(x)
        ppm_out = self.ppm_bottleneck(torch.cat([feats[-1], *self.ppm(feats[-1])], dim=1))

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats[:-1])]
        laterals.append(ppm_out)
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fpn_outs = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals[:-1])]
        fpn_outs.append(laterals[-1])
        target = tuple(int(v) for v in fpn_outs[0].shape[-2:])
        fused = torch.cat(
            [
                F.interpolate(feat, size=target, mode="bilinear", align_corners=False)
                if tuple(feat.shape[-2:]) != target
                else feat
                for feat in fpn_outs
            ],
            dim=1,
        )
        logits = self.seg_head(self.fpn_bottleneck(fused))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


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
    lemonfm_ckpt: str | Path | None = DEFAULT_LEMONFM_CKPT,
    lemonfm_decoder_channels: int = 128,
):
    arch = arch.lower()
    if arch == "lemonfm_fpn":
        return LemonFMConvNeXtLargeFPN(
            pretrained_weights=lemonfm_ckpt,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=int(lemonfm_decoder_channels),
        )
    if arch == "lemonfm_upernet":
        return LemonFMConvNeXtLargeUPerNet(
            pretrained_weights=lemonfm_ckpt,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=int(lemonfm_decoder_channels),
        )

    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required but not installed. "
            f"Original import error: {_SMP_IMPORT_ERROR!r}"
        )

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
