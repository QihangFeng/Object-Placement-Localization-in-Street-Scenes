#!/usr/bin/env python3
"""
Zero-data verification for bundled reference checkpoints and metrics.
"""

import argparse
import json
import math
from pathlib import Path


CHECKPOINTS = {
    "candidate": "01_candidate_last.pt",
    "supportsurface": "02_supportsurface_last.pt",
    "hardconstraint": "03_ss_hard_last.pt",
}


def load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for checkpoint verification.\n"
            "Install dependencies first with:\n"
            "  pip install -r requirements-verify.txt"
        ) from exc
    return torch


def summarize_checkpoint(torch_mod, ckpt_path):
    ckpt = torch_mod.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    return {
        "epoch": ckpt.get("epoch"),
        "history_len": len(history),
        "best_val_top1_iou": ckpt.get("best_val_top1_iou"),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify bundled checkpoints and metrics for consistency")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--package_root", default=None)
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve() if args.package_root else Path(__file__).resolve().parent.parent
    checkpoints_dir = package_root / "checkpoints"
    metrics_path = package_root / "results" / "metrics_all.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Reference metrics file not found: {metrics_path}\n"
            "Expected a bundled 10-epoch metrics_all.json."
        )

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    torch_mod = load_torch()
    summaries = {}
    missing = []
    for name, filename in CHECKPOINTS.items():
        ckpt_path = checkpoints_dir / filename
        if not ckpt_path.exists():
            missing.append(str(ckpt_path))
            continue
        summaries[name] = summarize_checkpoint(torch_mod, ckpt_path)

    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Reference checkpoint(s) are missing from the bundle:\n"
            f"{missing_str}\n"
            "For this experiment package, copy the lr=1e-5 / epoch-10 last checkpoints into checkpoints/ "
            "before distributing the final zip."
        )

    for name, summary in summaries.items():
        if name not in metrics:
            raise KeyError(f"metrics_all.json is missing entry '{name}'")
        metrics_entry = metrics[name]
        expected = {
            "epoch": metrics_entry.get("epoch"),
            "history_len": len(metrics_entry.get("history", [])),
            "best_val_top1_iou": metrics_entry.get("best_val_top1_iou"),
        }
        if summary["epoch"] != expected["epoch"]:
            raise AssertionError(f"{name}: checkpoint epoch {summary['epoch']} != metrics epoch {expected['epoch']}")
        if summary["history_len"] != expected["history_len"]:
            raise AssertionError(
                f"{name}: checkpoint history_len {summary['history_len']} != metrics history_len {expected['history_len']}"
            )
        if not math.isclose(
            float(summary["best_val_top1_iou"]),
            float(expected["best_val_top1_iou"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"{name}: checkpoint best_val_top1_iou {summary['best_val_top1_iou']} "
                f"!= metrics value {expected['best_val_top1_iou']}"
            )

    print("Reference bundle verification passed.")
    for name, summary in summaries.items():
        print(
            f"{name:16s} epoch={summary['epoch']:>2}  "
            f"history_len={summary['history_len']:>2}  "
            f"best_val_top1_iou={summary['best_val_top1_iou']:.10f}"
        )
        if args.verbose:
            print(f"  checkpoint = {checkpoints_dir / CHECKPOINTS[name]}")
    print(f"metrics_file = {metrics_path}")


if __name__ == "__main__":
    main()
