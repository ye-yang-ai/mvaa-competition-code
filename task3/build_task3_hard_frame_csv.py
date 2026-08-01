#!/usr/bin/env python3
"""Build a hard-frame list from Task3 frame diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-hd", type=float, default=50.0)
    parser.add_argument("--max-dice", type=float, default=0.80)
    parser.add_argument("--min-no-overlap-area", type=float, default=50.0)
    parser.add_argument("--min-area-ratio-dev", type=float, default=0.25)
    parser.add_argument("--min-pred-centroid-shift", type=float, default=100.0)
    return parser.parse_args()


def to_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def clean_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.6f}"


def frame_score(row: dict, args: argparse.Namespace) -> tuple[float, list[str]]:
    hd = to_float(row.get("hd"))
    asd = to_float(row.get("asd"))
    dice = to_float(row.get("dice"))
    area_ratio = to_float(row.get("area_ratio_pred_gt"))
    no_overlap_area = to_float(row.get("components_no_gt_overlap_area"), 0.0)
    pred_shift = to_float(row.get("pred_centroid_shift_prev"))
    pred_components = to_float(row.get("pred_components"), 0.0)

    reasons: list[str] = []
    score = 0.0
    if not math.isnan(hd) and hd >= float(args.min_hd):
        reasons.append("high_hd")
        score += hd / 50.0
    if not math.isnan(asd):
        score += asd / 10.0
    if not math.isnan(dice) and dice <= float(args.max_dice):
        reasons.append("low_dice")
        score += (float(args.max_dice) - dice) * 5.0
    if no_overlap_area >= float(args.min_no_overlap_area):
        reasons.append("no_gt_overlap_component")
        score += no_overlap_area / 200.0
    if not math.isnan(area_ratio) and abs(area_ratio - 1.0) >= float(args.min_area_ratio_dev):
        reasons.append("area_ratio_outlier")
        score += abs(area_ratio - 1.0) * 2.0
    if not math.isnan(pred_shift) and pred_shift >= float(args.min_pred_centroid_shift):
        reasons.append("temporal_centroid_jump")
        score += pred_shift / 100.0
    if pred_components >= 5:
        reasons.append("many_components")
        score += pred_components / 10.0
    return score, reasons


def main() -> int:
    args = parse_args()
    by_case: dict[str, dict] = {}
    for input_csv in args.input_csv:
        with input_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                case_id = str(row.get("case_id", "")).strip()
                if not case_id:
                    continue
                score, reasons = frame_score(row, args)
                if not reasons:
                    continue
                out = {
                    "case_id": case_id,
                    "video_id": row.get("video_id", ""),
                    "frame_idx": row.get("frame_idx", ""),
                    "image_path": row.get("image_path", ""),
                    "score": clean_float(score),
                    "reasons": ";".join(reasons),
                    "dice": row.get("dice", ""),
                    "hd": row.get("hd", ""),
                    "asd": row.get("asd", ""),
                    "gt_area": row.get("gt_area", ""),
                    "pred_area": row.get("pred_area", ""),
                    "area_ratio_pred_gt": row.get("area_ratio_pred_gt", ""),
                    "pred_components": row.get("pred_components", ""),
                    "components_no_gt_overlap_area": row.get("components_no_gt_overlap_area", ""),
                    "pred_centroid_shift_prev": row.get("pred_centroid_shift_prev", ""),
                }
                old = by_case.get(case_id)
                if old is None or to_float(out["score"], 0.0) > to_float(old["score"], 0.0):
                    by_case[case_id] = out

    rows = sorted(by_case.values(), key=lambda r: to_float(r["score"], 0.0), reverse=True)
    if int(args.top_k) > 0:
        rows = rows[: int(args.top_k)]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "video_id",
        "frame_idx",
        "image_path",
        "score",
        "reasons",
        "dice",
        "hd",
        "asd",
        "gt_area",
        "pred_area",
        "area_ratio_pred_gt",
        "pred_components",
        "components_no_gt_overlap_area",
        "pred_centroid_shift_prev",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {args.output_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
