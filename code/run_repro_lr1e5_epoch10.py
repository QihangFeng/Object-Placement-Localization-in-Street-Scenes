#!/usr/bin/env python3
"""
Teacher-facing entrypoint for the lr=1e-5, epoch=10 reproduction run.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PACKAGE_ROOT / "code"


def require_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Required Python package '{module_name}' is not available.\n"
            "Install dependencies first with:\n"
            "  pip install -r requirements-repro.txt"
        ) from exc


def precheck_resnet18():
    require_module("torch")
    require_module("torchvision")
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the ResNet-18 pretrained weights.\n"
            "This commonly happens on a machine without internet access during the first run.\n"
            f"Cache directory: {cache_dir}\n"
            "Try pre-caching the weights with:\n"
            "  python -c \"from torchvision.models import resnet18, ResNet18_Weights; "
            "resnet18(weights=ResNet18_Weights.DEFAULT)\"\n"
            f"Original error: {exc}"
        ) from exc


def verify_cityscapes_root(cityscapes_root):
    required_subdirs = [
        "leftImg8bit/train",
        "leftImg8bit/val",
        "gtFine/train",
        "gtFine/val",
    ]
    missing = [str(cityscapes_root / subdir) for subdir in required_subdirs if not (cityscapes_root / subdir).exists()]
    if missing:
        raise FileNotFoundError(
            "Cityscapes root is missing required directories:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def verify_built_dataset(data_root):
    required_files = [
        data_root / "train" / "annotations_train.jsonl",
        data_root / "val" / "annotations_val.jsonl",
    ]
    for required in required_files:
        if not required.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {required}")

    sample_globs = [
        data_root / "train" / "backgrounds",
        data_root / "train" / "location",
        data_root / "train" / "objects",
        data_root / "val" / "backgrounds",
        data_root / "val" / "location",
        data_root / "val" / "objects",
    ]
    for sample_dir in sample_globs:
        if not sample_dir.exists():
            raise FileNotFoundError(f"Expected dataset directory not found: {sample_dir}")
        try:
            next(path for path in sample_dir.rglob("*") if path.is_file())
        except StopIteration as exc:
            raise FileNotFoundError(f"Dataset directory is empty: {sample_dir}") from exc


def ensure_clean_output_dir(path, resume):
    if path.exists() and any(path.iterdir()) and not resume:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {path}\n"
            "Please provide a new workspace directory, or pass --resume if you want to continue the same run."
        )


def print_layout(cityscapes_root, data_root, checkpoints_dir, results_dir, logs_dir, workspace_dir):
    print(f"package_root   = {PACKAGE_ROOT}")
    print(f"cityscapes_root = {cityscapes_root}")
    print(f"workspace_dir  = {workspace_dir}")
    print(f"data_root      = {data_root}")
    print("will create/use:")
    print(f"  checkpoints: {checkpoints_dir}")
    print(f"  results:     {results_dir}")
    print(f"  logs:        {logs_dir}")


def run_and_tee(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(command)}")
    print(f"    log -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser(description="Run the lr=1e-5, epoch=10 reproduction pipeline")
    parser.add_argument("--cityscapes_root", required=True, help="Path to raw Cityscapes data root")
    parser.add_argument("--workspace_dir", required=True, help="Where new outputs should be written")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional existing bootplace_like_data directory. Defaults to <workspace_dir>/bootplace_like_data",
    )
    parser.add_argument("--skip_build_data", action="store_true", help="Skip dataset construction and reuse an existing data_root")
    parser.add_argument("--resume", action="store_true", help="Allow reusing existing checkpoints in the workspace")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--build_num_workers", type=int, default=1)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_candidates", type=int, default=256)
    args = parser.parse_args()

    cityscapes_root = Path(args.cityscapes_root).resolve()
    workspace_dir = Path(args.workspace_dir).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else workspace_dir / "bootplace_like_data"
    checkpoints_dir = workspace_dir / "checkpoints"
    results_dir = workspace_dir / "results"
    logs_dir = workspace_dir / "logs"

    print_layout(
        cityscapes_root=cityscapes_root,
        data_root=data_root,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        workspace_dir=workspace_dir,
    )

    require_module("numpy")
    require_module("PIL")
    require_module("torch")
    require_module("torchvision")
    if not args.skip_build_data:
        require_module("cv2")
    precheck_resnet18()
    verify_cityscapes_root(cityscapes_root)

    ensure_clean_output_dir(checkpoints_dir, resume=args.resume)
    ensure_clean_output_dir(results_dir, resume=args.resume)
    ensure_clean_output_dir(logs_dir, resume=args.resume)

    workspace_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_build_data:
        verify_built_dataset(data_root)
    else:
        build_cmd = [
            sys.executable,
            str(CODE_DIR / "build_dataset.py"),
            "--cityscapes_root",
            str(cityscapes_root),
            "--out_root",
            str(data_root),
            "--num_workers",
            str(args.build_num_workers),
        ]
        run_and_tee(build_cmd, logs_dir / "build_dataset.log")

    if not args.skip_build_data:
        verify_built_dataset(data_root)

    shared_train_args = [
        "--data_root",
        str(data_root),
        "--cityscapes_root",
        str(cityscapes_root),
        "--save_dir",
        str(checkpoints_dir),
        "--num_epochs",
        "10",
        "--lr",
        "1e-5",
        "--batch_size",
        str(args.batch_size),
        "--num_candidates",
        str(args.num_candidates),
        "--num_workers",
        str(args.train_num_workers),
        "--device",
        args.device,
    ]
    if args.resume:
        shared_train_args.append("--resume")

    run_and_tee(
        [sys.executable, str(CODE_DIR / "train_candidate.py"), *shared_train_args],
        logs_dir / "train_candidate.log",
    )
    run_and_tee(
        [sys.executable, str(CODE_DIR / "train_supportsurface.py"), *shared_train_args],
        logs_dir / "train_supportsurface.log",
    )
    run_and_tee(
        [sys.executable, str(CODE_DIR / "train_hardconstraint.py"), *shared_train_args],
        logs_dir / "train_hardconstraint.log",
    )

    metrics_output = results_dir / "metrics_all.json"
    run_and_tee(
        [
            sys.executable,
            str(CODE_DIR / "extract_metrics.py"),
            "--models_dir",
            str(checkpoints_dir),
            "--output",
            str(metrics_output),
        ],
        logs_dir / "extract_metrics.log",
    )

    print("\nReproduction run completed successfully.")
    print(f"checkpoints_dir = {checkpoints_dir}")
    print(f"metrics_file    = {metrics_output}")
    print(f"logs_dir        = {logs_dir}")


if __name__ == "__main__":
    main()
