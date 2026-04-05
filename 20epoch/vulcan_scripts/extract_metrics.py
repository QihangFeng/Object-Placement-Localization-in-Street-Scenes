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
    parser.add_argument("--models_dir", type=str,
                        default="/scratch/zchai3/ece740_placement/models")
    parser.add_argument("--output", type=str,
                        default="/scratch/zchai3/ece740_placement/models/metrics_all.json")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    results = {}

    for name, filename in CHECKPOINT_MAP.items():
        ckpt_path = models_dir / filename
        if not ckpt_path.exists():
            print(f"[WARN] Not found: {ckpt_path}, skipping {name}")
            continue
        print(f"Loading {name} from {ckpt_path} ...")
        results[name] = extract_one(ckpt_path)
        print(f"  epoch={results[name]['epoch']}, "
              f"best_val_top1_iou={results[name]['best_val_top1_iou']}, "
              f"history_len={len(results[name]['history'])}")

    def _json_default(obj):
        """Handle numpy/torch types that json.dump can't serialize."""
        if hasattr(obj, 'item'):  # torch.Tensor scalar or numpy scalar
            return obj.item()
        if hasattr(obj, 'tolist'):  # numpy array or torch.Tensor
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nSaved to {args.output}")
    print(f"Models extracted: {list(results.keys())}")


if __name__ == "__main__":
    main()
