#!/usr/bin/env python3
"""
build_dataset.py -- Generate bootplace_like_data from raw Cityscapes inputs.

Pipeline summary:
  1. Scan instance maps, extract object bounding boxes, and save RGBA object crops.
  2. For each target object, inpaint the source image and save the background plus annotations.
"""

import argparse
import json
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV is required but not installed.\n"
        "Install it with either:\n"
        "  pip install opencv-python\n"
        "or:\n"
        "  pip install opencv-python-headless"
    )
import numpy as np
from PIL import Image
from tqdm import tqdm

# Constants

ALLOWED_CLASS_ID_TO_NAME = {
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}

# Utility functions

def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_instance_map(path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel instance map, got shape {arr.shape} from {path}")
    return arr


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def make_prompt_text(class_name):
    article = "an" if class_name[0].lower() in ("a", "e", "i", "o", "u") else "a"
    return f"place {article} {class_name}"


def inpaint_remove_instance(rgb, target_mask, dilate_ksize=9, inpaint_radius=3):
    mask_u8 = (target_mask.astype(np.uint8) * 255)
    if dilate_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bg_bgr = cv2.inpaint(bgr, mask_u8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    return bg_rgb, mask_u8


# Step 1: extract object crops

def save_rgba_object_patch(rgb, mask, bbox, out_path, context_pad=4):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1p, y1p, x2p, y2p = clamp_bbox(
        x1 - context_pad, y1 - context_pad,
        x2 + context_pad, y2 + context_pad, w, h,
    )
    crop_rgb = rgb[y1p:y2p + 1, x1p:x2p + 1].copy()
    crop_mask = mask[y1p:y2p + 1, x1p:x2p + 1].copy()
    alpha = (crop_mask.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(out_path)
    return {
        "crop_bbox_xyxy_abs": [x1p, y1p, x2p, y2p],
        "crop_size_wh": [int(x2p - x1p + 1), int(y2p - y1p + 1)],
    }


def process_one_image(rgb_path, instance_path, out_objects_dir, split, min_area, context_pad):
    rgb = load_rgb(rgb_path)
    inst = load_instance_map(instance_path)
    h, w = inst.shape
    records = []
    unique_ids = np.unique(inst)

    for instance_id in unique_ids:
        instance_id = int(instance_id)
        if instance_id < 1000:
            continue
        class_id = instance_id // 1000
        if class_id not in ALLOWED_CLASS_ID_TO_NAME:
            continue
        class_name = ALLOWED_CLASS_ID_TO_NAME[class_id]
        mask = (inst == instance_id)
        area = int(mask.sum())
        if area < min_area:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw = int(x2 - x1 + 1)
        bh = int(y2 - y1 + 1)

        stem = rgb_path.stem.replace("_leftImg8bit", "")
        obj_name = f"{stem}_{class_name}_{instance_id}"
        obj_rel_path = Path(split) / "objects" / class_name / f"{obj_name}.png"
        obj_abs_path = out_objects_dir / class_name / f"{obj_name}.png"

        crop_meta = save_rgba_object_patch(
            rgb=rgb, mask=mask, bbox=bbox,
            out_path=obj_abs_path, context_pad=context_pad,
        )

        record = {
            "sample_id": obj_name,
            "split": split,
            "city": rgb_path.parent.name,
            "source_image_name": rgb_path.name,
            "source_image_rel": str(Path("leftImg8bit") / split / rgb_path.parent.name / rgb_path.name),
            "instance_map_rel": str(Path("gtFine") / split / instance_path.parent.name / instance_path.name),
            "class_name": class_name,
            "class_id": class_id,
            "instance_id": instance_id,
            "image_size_wh": [w, h],
            "bbox_xyxy_abs": [x1, y1, x2, y2],
            "bbox_wh_abs": [bw, bh],
            "mask_area": area,
            "object_patch_rel": str(obj_rel_path),
            "crop_bbox_xyxy_abs": crop_meta["crop_bbox_xyxy_abs"],
            "crop_size_wh": crop_meta["crop_size_wh"],
        }
        records.append(record)
    return records


def _step1_wrapper(args):
    """Wrapper for process_one_image to work with imap_unordered.
    Defined at module level for multiprocessing pickle compatibility.
    """
    return process_one_image(*args)


def build_cityscapes_objects(cityscapes_root, out_root, split, min_area, context_pad, max_images, num_workers=1):
    city_root = Path(cityscapes_root)
    out_root = Path(out_root)
    left_root = city_root / "leftImg8bit" / split
    fine_root = city_root / "gtFine" / split

    if not left_root.exists():
        raise FileNotFoundError(f"Not found: {left_root}")
    if not fine_root.exists():
        raise FileNotFoundError(f"Not found: {fine_root}")

    out_split_root = out_root / split
    out_objects_dir = out_split_root / "objects"
    out_split_root.mkdir(parents=True, exist_ok=True)

    instance_paths = sorted(fine_root.glob("*/*_gtFine_instanceIds.png"))
    if max_images > 0:
        instance_paths = instance_paths[:max_images]

    print(f"[Step 1] split={split}, num instance maps={len(instance_paths)}, workers={num_workers}")

    # Build args list, filtering missing RGB images
    args_list = []
    for inst_path in instance_paths:
        city = inst_path.parent.name
        stem = inst_path.name.replace("_gtFine_instanceIds.png", "")
        rgb_name = f"{stem}_leftImg8bit.png"
        rgb_path = left_root / city / rgb_name
        if not rgb_path.exists():
            print(f"[WARN] Missing RGB image, skip: {rgb_path}")
            continue
        args_list.append((rgb_path, inst_path, out_objects_dir, split, min_area, context_pad))

    # Parallel execution
    all_records = []
    class_counter = {name: 0 for name in ALLOWED_CLASS_ID_TO_NAME.values()}

    if num_workers > 1 and len(args_list) > 1:
        with mp.Pool(num_workers) as pool:
            for records in tqdm(
                pool.imap_unordered(_step1_wrapper, args_list),
                total=len(args_list), desc=f"objects-{split}",
            ):
                for r in records:
                    class_counter[r["class_name"]] += 1
                all_records.extend(records)
    else:
        for args in tqdm(args_list, desc=f"objects-{split}"):
            records = process_one_image(*args)
            for r in records:
                class_counter[r["class_name"]] += 1
            all_records.extend(records)

    all_records = sorted(all_records, key=lambda record: record["sample_id"])

    index_path = out_split_root / f"index_{split}.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "split": split,
        "num_source_images": len(instance_paths),
        "num_objects": len(all_records),
        "class_counter": class_counter,
        "min_area": min_area,
        "context_pad": context_pad,
    }
    summary_path = out_split_root / f"summary_{split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  index saved to: {index_path}")
    print(f"  num objects: {len(all_records)}")
    for k, v in class_counter.items():
        print(f"    {k:12s} {v}")

    return {"index_path": index_path, "summary_path": summary_path}


# Step 2: generate inpainted backgrounds

def _process_one_source(args):
    """Process one source image: load, inpaint targets, save backgrounds.
    Defined at module level for multiprocessing pickle compatibility.
    """
    (source_rel, group_records, selected_targets,
     city_root, bg_dir, loc_dir, split, dilate_ksize, inpaint_radius) = args

    city_root = Path(city_root)
    bg_dir = Path(bg_dir)
    loc_dir = Path(loc_dir)

    rgb = load_rgb(city_root / source_rel)
    inst = load_instance_map(city_root / group_records[0]["instance_map_rel"])

    all_boxes_same_image = []
    for r in group_records:
        all_boxes_same_image.append({
            "sample_id": r["sample_id"],
            "class_name": r["class_name"],
            "instance_id": r["instance_id"],
            "bbox_xyxy_abs": r["bbox_xyxy_abs"],
        })

    results = []
    for target in selected_targets:
        target_instance_id = int(target["instance_id"])
        target_bbox = target["bbox_xyxy_abs"]
        target_class = target["class_name"]
        sample_id = target["sample_id"]

        target_mask = (inst == target_instance_id)
        if target_mask.sum() == 0:
            continue

        bg_rgb, used_mask_u8 = inpaint_remove_instance(
            rgb=rgb, target_mask=target_mask,
            dilate_ksize=dilate_ksize, inpaint_radius=inpaint_radius,
        )

        bg_rel = Path(split) / "backgrounds" / f"{sample_id}.png"
        bg_abs = bg_dir / f"{sample_id}.png"
        Image.fromarray(bg_rgb).save(bg_abs)

        scene_boxes = []
        for obj in all_boxes_same_image:
            if int(obj["instance_id"]) == target_instance_id:
                continue
            scene_boxes.append({
                "class_name": obj["class_name"],
                "instance_id": obj["instance_id"],
                "bbox_xyxy_abs": obj["bbox_xyxy_abs"],
            })

        loc_data = {
            "sample_id": sample_id,
            "prompt": make_prompt_text(target_class),
            "target_class": target_class,
            "target_instance_id": target_instance_id,
            "target_bbox_xyxy_abs": target_bbox,
            "scene_boxes_xyxy_abs": scene_boxes,
            "image_size_wh": target["image_size_wh"],
            "background_rel": str(bg_rel),
            "object_patch_rel": target["object_patch_rel"],
            "source_image_rel": target["source_image_rel"],
            "instance_map_rel": target["instance_map_rel"],
        }

        loc_path = loc_dir / f"{sample_id}.json"
        with open(loc_path, "w", encoding="utf-8") as f:
            json.dump(loc_data, f, ensure_ascii=False, indent=2)

        results.append(loc_data)

    return results


def build_cityscapes_backgrounds(
    cityscapes_root, out_root, split, index_path,
    dilate_ksize, inpaint_radius, max_targets_per_image, random_seed,
    num_workers=1,
):
    city_root = Path(cityscapes_root)
    out_root = Path(out_root)
    split_root = out_root / split
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Not found: {index_path}")

    records = read_jsonl(index_path)
    print(f"[Step 2] split={split}, loaded object records={len(records)}, workers={num_workers}")

    bg_dir = split_root / "backgrounds"
    loc_dir = split_root / "location"
    bg_dir.mkdir(parents=True, exist_ok=True)
    loc_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for r in records:
        grouped[r["source_image_rel"]].append(r)

    print(f"  num source images = {len(grouped)}")

    # Pre-select targets in main process (preserves RNG order)
    rng = random.Random(random_seed)
    work_items = []
    for source_rel, group_records in grouped.items():
        selected_targets = group_records
        if max_targets_per_image is not None and len(group_records) > max_targets_per_image:
            selected_targets = rng.sample(group_records, k=max_targets_per_image)
        work_items.append((
            source_rel, group_records, selected_targets,
            str(city_root), str(bg_dir), str(loc_dir), split,
            dilate_ksize, inpaint_radius,
        ))

    # Parallel execution
    final_records = []
    if num_workers > 1 and len(work_items) > 1:
        with mp.Pool(num_workers) as pool:
            for batch in tqdm(
                pool.imap_unordered(_process_one_source, work_items),
                total=len(work_items), desc=f"bg-{split}",
            ):
                final_records.extend(batch)
    else:
        for item in tqdm(work_items, desc=f"bg-{split}"):
            final_records.extend(_process_one_source(item))

    final_records = sorted(final_records, key=lambda record: record["sample_id"])

    annotations_path = split_root / f"annotations_{split}.jsonl"
    with open(annotations_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  annotations saved to: {annotations_path}")
    print(f"  num samples: {len(final_records)}")

    return {"annotations_path": annotations_path, "num_samples": len(final_records)}


# Main entrypoint

def main():
    parser = argparse.ArgumentParser(description="Build bootplace_like_data from Cityscapes")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--min_area", type=int, default=256)
    parser.add_argument("--context_pad", type=int, default=4)
    parser.add_argument("--dilate_ksize", type=int, default=9)
    parser.add_argument("--inpaint_radius", type=int, default=3)
    parser.add_argument("--train_max_images", type=int, default=-1,
                        help="Max train images to process (-1 = all)")
    parser.add_argument("--val_max_images", type=int, default=-1,
                        help="Max val images to process (-1 = all)")
    parser.add_argument("--train_max_targets", type=int, default=3,
                        help="Max targets per image for train")
    parser.add_argument("--val_max_targets", type=int, default=2,
                        help="Max targets per image for val")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers for data processing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output directory")
    args = parser.parse_args()

    # Verify cityscapes data exists
    cs = Path(args.cityscapes_root)
    for subdir in ["leftImg8bit/train", "leftImg8bit/val", "gtFine/train", "gtFine/val"]:
        p = cs / subdir
        if not p.exists():
            raise FileNotFoundError(f"Cityscapes directory not found: {p}")
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {out_root}\n"
            "Pass --overwrite to reuse this directory, or choose a new output path."
        )

    print(f"Cityscapes root: {args.cityscapes_root}")
    print(f"Output root: {out_root}")
    print()
    # Train split
    print("=" * 60)
    print("TRAIN SPLIT")
    print("=" * 60)
    train_obj = build_cityscapes_objects(
        cityscapes_root=args.cityscapes_root,
        out_root=args.out_root,
        split="train",
        min_area=args.min_area,
        context_pad=args.context_pad,
        max_images=args.train_max_images,
        num_workers=args.num_workers,
    )
    train_bg = build_cityscapes_backgrounds(
        cityscapes_root=args.cityscapes_root,
        out_root=args.out_root,
        split="train",
        index_path=train_obj["index_path"],
        dilate_ksize=args.dilate_ksize,
        inpaint_radius=args.inpaint_radius,
        max_targets_per_image=args.train_max_targets,
        random_seed=42,
        num_workers=args.num_workers,
    )
    print()
    # Validation split
    print("=" * 60)
    print("VAL SPLIT")
    print("=" * 60)
    val_obj = build_cityscapes_objects(
        cityscapes_root=args.cityscapes_root,
        out_root=args.out_root,
        split="val",
        min_area=args.min_area,
        context_pad=args.context_pad,
        max_images=args.val_max_images,
        num_workers=args.num_workers,
    )
    val_bg = build_cityscapes_backgrounds(
        cityscapes_root=args.cityscapes_root,
        out_root=args.out_root,
        split="val",
        index_path=val_obj["index_path"],
        dilate_ksize=args.dilate_ksize,
        inpaint_radius=args.inpaint_radius,
        max_targets_per_image=args.val_max_targets,
        random_seed=0,
        num_workers=args.num_workers,
    )

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Train samples: {train_bg['num_samples']}")
    print(f"  Val samples:   {val_bg['num_samples']}")
    print(f"  Output dir:    {args.out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
