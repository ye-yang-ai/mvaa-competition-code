"""Build MedSAM2 image encoder from the bundled external repository."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
MEDSAM2_ROOT = REPO_ROOT / "external" / "MedSAM2"
DEFAULT_ENCODER_CFG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_ENCODER_CKPT = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"


def _ensure_medsam2_on_path() -> None:
    root = str(MEDSAM2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_medsam2_image_encoder(
    cfg: str = DEFAULT_ENCODER_CFG,
    ckpt_path: str | Path = DEFAULT_ENCODER_CKPT,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Load MedSAM2 and return only its image_encoder module."""
    _ensure_medsam2_on_path()

    from sam2.build_sam import build_sam2

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MedSAM2 checkpoint not found: {ckpt_path}")

    model = build_sam2(
        config_file=str(cfg),
        ckpt_path=str(ckpt_path),
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    return model.image_encoder
