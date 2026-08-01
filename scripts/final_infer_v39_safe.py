#!/usr/bin/env python3
"""Run the MVAA v39-safe final Docker inference pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

TASK1_DATASET = "Dataset111_MVAA_Task1_PseudoTop100"
TASK2_DATASET = "Dataset102_MVAA_Task2"
NNUNET_CONFIGURATION = "3d_fullres"
NNUNET_TRAINER = "nnUNetTrainer"
NNUNET_PLANS = "nnUNetPlans"
NNUNET_FOLDS = ("0", "1", "2", "3", "4")
NNUNET_CHECKPOINT = "checkpoint_best.pth"

TASK1_RESULTS_CANDIDATES = (
    REPO_ROOT / "weights" / "nnunet_task1_v25" / "nnUNet_results",
    REPO_ROOT / "outputs" / "nnunet_task1_pseudo_top100" / "nnUNet_results",
)
TASK2_RESULTS_CANDIDATES = (
    REPO_ROOT / "weights" / "nnunet_task2_v19" / "nnUNet_results",
    REPO_ROOT / "outputs" / "nnunet_task2_fold0" / "nnUNet_results",
)

TASK3_CKPTS = (
    REPO_ROOT / "weights" / "task3_v39" / "resnet34_s42_best.pt",
    REPO_ROOT / "weights" / "task3_v39" / "resnet34_s49_best.pt",
    REPO_ROOT / "weights" / "task3_v39" / "effb4_s42_best.pt",
    REPO_ROOT / "weights" / "task3_v39" / "effb5_s42_best.pt",
    REPO_ROOT / "weights" / "task3_v39" / "effb5_s44_best.pt",
    REPO_ROOT / "weights" / "task3_v39" / "lemonfm_fpn_s52_best.pt",
)
TASK3_LOCAL_CKPTS = (
    REPO_ROOT / "outputs" / "exp" / "task3_unetpp_res34_imagenet_e100_bs2_bestcfg" / "checkpoints" / "best.pt",
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_v36_unetpp_res34_img448x800_s49" / "checkpoints" / "best.pt",
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_sup_unetpp_effb4_img512x896_allframe_s42" / "checkpoints" / "best.pt",
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_v32_unetpp_effb5_sup_allframe_s42" / "checkpoints" / "best.pt",
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_v35_unetpp_effb5_sup_allframe_s44" / "checkpoints" / "best.pt",
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_lemonfm_fpn_formal_tail2_s52" / "checkpoints" / "best.pt",
)
TASK3_PRESENCE_GATE_CKPT = REPO_ROOT / "weights" / "task3_v39" / "lemonfm_presence_gate_s51_best.pt"
TASK3_LOCAL_PRESENCE_GATE_CKPT = (
    REPO_ROOT / "outputs" / "opt" / "task3" / "t3_lemonfm_presence_gate_s51" / "checkpoints" / "best.pt"
)
TASK3_WEIGHTS = ("0.27", "0.19", "0.17", "0.1275", "0.1425", "0.10")
TASK3_THRESHOLD = 0.36

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CaseItem:
    case_id: str
    image_path: Path
    nnunet_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("MVAA_INPUT_DIR", "/input")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MVAA_OUTPUT_DIR", "/output")))
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("MVAA_WORK_DIR", "/work")))
    parser.add_argument("--skip-task1", action="store_true")
    parser.add_argument("--skip-task2", action="store_true")
    parser.add_argument("--skip-task3", action="store_true")
    parser.add_argument("--nnunet-predict-bin", default=os.environ.get("NNUNETV2_PREDICT_BIN", "nnUNetv2_predict"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--task3-device", default="auto")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[final-v39-safe] {message}", flush=True)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def resolve_existing(candidates: Sequence[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n  ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"Missing {label}. Checked:\n  {checked}")


def resolve_task3_ckpts() -> list[Path]:
    ckpts: list[Path] = []
    for docker_path, local_path in zip(TASK3_CKPTS, TASK3_LOCAL_CKPTS):
        if docker_path.exists():
            ckpts.append(docker_path)
        elif local_path.exists():
            ckpts.append(local_path)
        else:
            raise FileNotFoundError(f"Missing Task3 checkpoint. Checked {docker_path} and {local_path}")
    return ckpts


def resolve_task3_presence_gate_ckpt() -> Path:
    if TASK3_PRESENCE_GATE_CKPT.exists():
        return TASK3_PRESENCE_GATE_CKPT
    if TASK3_LOCAL_PRESENCE_GATE_CKPT.exists():
        return TASK3_LOCAL_PRESENCE_GATE_CKPT
    raise FileNotFoundError(
        "Missing Task3 presence gate checkpoint. Checked "
        f"{TASK3_PRESENCE_GATE_CKPT} and {TASK3_LOCAL_PRESENCE_GATE_CKPT}"
    )


def clean_work_dir(path: Path) -> None:
    if path.exists():
        resolved = path.resolve()
        allowed_roots = [Path("/work").resolve(), Path("/tmp").resolve()]
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Refusing to remove work directory outside /work or /tmp: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def existing_dirs(input_dir: Path, candidates: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for rel in candidates:
        path = input_dir / rel if rel else input_dir
        if path.exists() and path.is_dir() and path not in out:
            out.append(path)
    return out


def strip_nii_gz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return Path(name).stem


def task1_case_id(path: Path) -> str:
    case_id = strip_nii_gz(path.name)
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    return case_id


def task2_case_id(path: Path) -> str:
    case_id = strip_nii_gz(path.name)
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    if case_id.endswith("-US"):
        case_id = case_id[: -len("-US")]
    return case_id


def is_likely_label(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("label", "seg", "mask", "pred"))


def discover_nifti_cases(
    input_dir: Path,
    candidates: Sequence[str],
    case_id_fn,
    require_suffix: str | None = None,
    reject_suffixes: Sequence[str] = (),
) -> list[CaseItem]:
    roots = existing_dirs(input_dir, candidates)
    if not roots:
        raise FileNotFoundError(f"None of the input candidates exist under {input_dir}: {candidates}")

    found: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*.nii.gz")):
            if is_likely_label(path):
                continue
            if reject_suffixes and any(path.name.endswith(suffix) for suffix in reject_suffixes):
                continue
            if require_suffix and not path.name.endswith(require_suffix):
                continue
            found.append(path)
        if found:
            break

    if not found and require_suffix:
        return discover_nifti_cases(input_dir, candidates, case_id_fn, require_suffix=None)
    if not found:
        raise RuntimeError(f"No NIfTI images found under candidate roots: {[p.as_posix() for p in roots]}")

    cases: list[CaseItem] = []
    seen: set[str] = set()
    for path in found:
        case_id = case_id_fn(path)
        if case_id in seen:
            raise RuntimeError(f"Duplicate case id after normalization: {case_id}")
        seen.add(case_id)
        cases.append(CaseItem(case_id=case_id, image_path=path, nnunet_id=case_id))
    return cases


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_nnunet_images(cases: Sequence[CaseItem], images_ts: Path) -> None:
    images_ts.mkdir(parents=True, exist_ok=True)
    for item in cases:
        link_or_copy(item.image_path, images_ts / f"{item.nnunet_id}_0000.nii.gz")


def write_task_json(cases: Sequence[CaseItem], output_task_dir: Path, task_json_name: str) -> None:
    records = [
        {
            "case_id": item.case_id,
            "segmentation": f"{item.case_id}-pred.nii.gz",
        }
        for item in cases
    ]
    (output_task_dir / task_json_name).write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")


def convert_nnunet_output(cases: Sequence[CaseItem], pred_dir: Path, output_task_dir: Path, task_json_name: str) -> None:
    output_task_dir.mkdir(parents=True, exist_ok=True)
    for item in cases:
        src = pred_dir / f"{item.nnunet_id}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction for {item.case_id}: {src}")
        shutil.copy2(src, output_task_dir / f"{item.case_id}-pred.nii.gz")
    write_task_json(cases, output_task_dir, task_json_name)


def run_subprocess(cmd: Sequence[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_nnunet_predict(
    *,
    nnunet_predict_bin: str,
    input_images: Path,
    output_pred: Path,
    dataset: str,
    results_root: Path,
    work_root: Path,
) -> None:
    require_path(results_root / dataset / f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIGURATION}", f"{dataset} nnU-Net results")
    env = os.environ.copy()
    env["nnUNet_raw"] = str(work_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(work_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(results_root)
    output_pred.mkdir(parents=True, exist_ok=True)
    cmd = [
        nnunet_predict_bin,
        "-i",
        str(input_images),
        "-o",
        str(output_pred),
        "-d",
        dataset,
        "-c",
        NNUNET_CONFIGURATION,
        "-tr",
        NNUNET_TRAINER,
        "-p",
        NNUNET_PLANS,
        "-f",
        *NNUNET_FOLDS,
        "-chk",
        NNUNET_CHECKPOINT,
    ]
    run_subprocess(cmd, env=env)


def run_task1(args: argparse.Namespace, work_root: Path) -> None:
    log("Task1: discovering CT inputs")
    cases = discover_nifti_cases(
        args.input_dir,
        candidates=("t1_ct/images", "t1_ct", "task1/images", "task1", "ct/images", "ct", ""),
        case_id_fn=task1_case_id,
        reject_suffixes=("-US.nii.gz",),
    )
    log(f"Task1: {len(cases)} cases")
    images_ts = work_root / "task1" / "nnUNet_raw" / TASK1_DATASET / "imagesTs"
    pred_dir = work_root / "task1_pred"
    prepare_nnunet_images(cases, images_ts)
    results_root = resolve_existing(TASK1_RESULTS_CANDIDATES, "Task1 v25 nnU-Net results")
    run_nnunet_predict(
        nnunet_predict_bin=args.nnunet_predict_bin,
        input_images=images_ts,
        output_pred=pred_dir,
        dataset=TASK1_DATASET,
        results_root=results_root,
        work_root=work_root / "task1",
    )
    convert_nnunet_output(cases, pred_dir, args.output_dir / "t1_ct", "task1_predictions.json")
    log("Task1: done")


def run_task2(args: argparse.Namespace, work_root: Path) -> None:
    log("Task2: discovering TEE inputs")
    cases = discover_nifti_cases(
        args.input_dir,
        candidates=("t2_tee/images", "t2_tee", "task2/images", "task2", "tee/images", "tee", ""),
        case_id_fn=task2_case_id,
        require_suffix="-US.nii.gz",
    )
    log(f"Task2: {len(cases)} cases")
    images_ts = work_root / "task2" / "nnUNet_raw" / TASK2_DATASET / "imagesTs"
    pred_dir = work_root / "task2_pred"
    prepare_nnunet_images(cases, images_ts)
    results_root = resolve_existing(TASK2_RESULTS_CANDIDATES, "Task2 v19 nnU-Net results")
    run_nnunet_predict(
        nnunet_predict_bin=args.nnunet_predict_bin,
        input_images=images_ts,
        output_pred=pred_dir,
        dataset=TASK2_DATASET,
        results_root=results_root,
        work_root=work_root / "task2",
    )
    convert_nnunet_output(cases, pred_dir, args.output_dir / "t2_tee", "task2_predictions.json")
    log("Task2: done")


def is_task3_label(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("_label_bin.png") or "_label" in name or "_mask" in name or "_pred" in name


def any_image(root: Path) -> bool:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not is_task3_label(path):
            return True
    return False


def discover_task3_root(input_dir: Path) -> Path:
    candidates = existing_dirs(
        input_dir,
        ("t3_vid/images", "t3_vid", "task3/images", "task3", "video/images", "video", "images"),
    )
    for root in candidates:
        if any_image(root):
            return root
    if any_image(input_dir):
        return input_dir
    raise RuntimeError(f"No Task3 image files found under {input_dir}")


def run_task3(args: argparse.Namespace) -> None:
    ckpts = resolve_task3_ckpts()
    presence_gate_ckpt = resolve_task3_presence_gate_ckpt()
    task3_input = discover_task3_root(args.input_dir)
    task3_out = args.output_dir / "t3_vid"
    task3_out.mkdir(parents=True, exist_ok=True)
    log(f"Task3: input root {task3_input}")
    env = os.environ.copy()
    env["MVAA_NO_PRETRAINED_INIT"] = "1"
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "task3" / "generate_task3_multi_ensemble_predictions.py"),
        "--ckpts",
        *(str(p) for p in ckpts),
        "--weights",
        *TASK3_WEIGHTS,
        "--threshold",
        f"{TASK3_THRESHOLD:.6f}",
        "--presence-gate-ckpt",
        str(presence_gate_ckpt),
        "--presence-gate-model-indices",
        "5",
        "--data-dir",
        str(task3_input),
        "--submission-task-dir",
        str(task3_out),
        "--tta",
        "--amp",
        "--device",
        args.task3_device,
    ]
    run_subprocess(cmd, env=env, cwd=REPO_ROOT / "task3")
    log("Task3: done")


def validate_outputs(output_dir: Path, skipped: Iterable[str]) -> None:
    skipped_set = set(skipped)
    expected = {
        "t1_ct": "task1_predictions.json",
        "t2_tee": "task2_predictions.json",
        "t3_vid": "task3_predictions.json",
    }
    for task_dir, json_name in expected.items():
        if task_dir in skipped_set:
            continue
        json_path = output_dir / task_dir / json_name
        require_path(json_path, f"{task_dir} prediction JSON")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", [])
        if not cases:
            raise RuntimeError(f"{json_path} contains no cases")
        log(f"{task_dir}: wrote {len(cases)} cases")


def main() -> int:
    args = parse_args()
    require_path(args.input_dir, "input directory")
    if not args.skip_task1:
        resolve_existing(TASK1_RESULTS_CANDIDATES, "Task1 v25 weights")
    if not args.skip_task2:
        resolve_existing(TASK2_RESULTS_CANDIDATES, "Task2 v19 weights")
    if not args.skip_task3:
        resolve_task3_ckpts()
        resolve_task3_presence_gate_ckpt()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.work_dir / "mvaa_v39_safe"
    clean_work_dir(work_root)

    skipped: list[str] = []
    if args.skip_task1:
        skipped.append("t1_ct")
    else:
        run_task1(args, work_root)
    if args.skip_task2:
        skipped.append("t2_tee")
    else:
        run_task2(args, work_root)
    if args.skip_task3:
        skipped.append("t3_vid")
    else:
        run_task3(args)

    validate_outputs(args.output_dir, skipped)
    log("All requested tasks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
