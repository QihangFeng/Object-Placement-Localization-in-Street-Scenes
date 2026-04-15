#!/usr/bin/env python3
"""
extract_metrics.py  --  Extract training history from checkpoints to JSON.

Reads 01_candidate_last.pt, 02_supportsurface_last.pt, 03_ss_hard_last.pt
and outputs a lightweight metrics_all.json for local analysis.
"""

import argparse
import json
from pathlib import Path

import torch


CHECKPOINT_MAP = {
    "candidate": "01_candidate_last.pt",
    "supportsurface": "02_supportsurface_last.pt",
    "hardconstraint": "03_ss_hard_last.pt",
}


def extract_one(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    best_val_top1_iou = ckpt.get("best_val_top1_iou", None)
    epoch = ckpt.get("epoch", None)
    return {
        "epoch": epoch,
        "best_val_top1_iou": best_val_top1_iou,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from checkpoints")
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Directory containing the three expected last.pt checkpoints")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for metrics JSON (default: <models_dir>/metrics_all.json)")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Allow missing checkpoints and extract only those that exist")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else models_dir / "metrics_all.json"
    results = {}
    missing = []

    for name, filename in CHECKPOINT_MAP.items():
        ckpt_path = models_dir / filename
        if not ckpt_path.exists():
            missing.append(str(ckpt_path))
            if args.allow_missing:
                print(f"[WARN] Not found: {ckpt_path}, skipping {name}")
                continue
            continue
        print(f"Loading {name} from {ckpt_path} ...")
        results[name] = extract_one(ckpt_path)
        print(f"  epoch={results[name]['epoch']}, "
              f"best_val_top1_iou={results[name]['best_val_top1_iou']}, "
              f"history_len={len(results[name]['history'])}")

    if missing and not args.allow_missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required checkpoints:\n"
            f"{missing_str}\n"
            "Pass --allow_missing only for ad-hoc debugging."
        )

    def _json_default(obj):
        """Handle numpy/torch types that json.dump can't serialize."""
        if hasattr(obj, 'item'):  # torch.Tensor scalar or numpy scalar
            return obj.item()
        if hasattr(obj, 'tolist'):  # numpy array or torch.Tensor
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nSaved to {output_path}")
    print(f"Models extracted: {list(results.keys())}")


if __name__ == "__main__":
    main()
