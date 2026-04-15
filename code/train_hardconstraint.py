#!/usr/bin/env python3
"""
train_hardconstraint.py  --  Stage 3: HardConstraint model
  - support_ratio = 0.95  (candidates from support surface)
  - semantic features computed normally
  - candidate_valid_mask computed via build_candidate_valid_mask()
  - best_candidate_idx = argmax(valid_ious) with valid mask
  - scores masked_fill with -1e4 for invalid candidates
Saves: 03_ss_hard_last.pt / 03_ss_hard_best.pt
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm import tqdm

# Constants

ROAD_LABEL_ID = 7
SIDEWALK_LABEL_ID = 8
PARKING_LABEL_ID = 9
PERSON_LABEL_IDS = {24, 25}
VEHICLE_LABEL_IDS = {26, 27, 28, 31, 32, 33}
ALLOWED_CLASS_ID_TO_NAME = {
    24: "person", 25: "rider", 26: "car", 27: "truck",
    28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle",
}
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Utility functions

def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_label_map(path: Path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel label map, got shape {arr.shape} from {path}")
    return arr


def source_rel_to_label_path(cityscapes_root, source_image_rel):
    # Normalize backslashes to forward slashes for cross-platform compatibility
    source_image_rel = source_image_rel.replace("\\", "/")
    p = Path(source_image_rel)
    split = p.parts[1]
    city = p.parts[2]
    filename = p.name
    stem = filename.replace("_leftImg8bit.png", "")
    label_path = Path(cityscapes_root) / "gtFine" / split / city / f"{stem}_gtFine_labelIds.png"
    return label_path


def normalize_box(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def box_area(boxes):
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def pairwise_iou(boxes1, boxes2):
    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


# Tokenizer and vocabulary

class SimplePromptTokenizer:
    def __init__(self, texts, max_len=8):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.max_len = max_len
        vocab = []
        for t in texts:
            vocab.extend(t.lower().strip().split())
        uniq = [self.pad_token, self.unk_token] + sorted(set(vocab))
        self.stoi = {w: i for i, w in enumerate(uniq)}
        self.itos = {i: w for w, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        tokens = text.lower().strip().split()
        ids = [self.stoi.get(tok, self.stoi[self.unk_token]) for tok in tokens]
        ids = ids[: self.max_len]
        attn = [1] * len(ids)
        while len(ids) < self.max_len:
            ids.append(self.stoi[self.pad_token])
            attn.append(0)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def build_scene_class_vocab(train_ann_path):
    train_records = read_jsonl(train_ann_path)
    class_names = set()
    for r in train_records:
        class_names.add(r["target_class"])
        for s in r["scene_boxes_xyxy_abs"]:
            class_names.add(s["class_name"])
    class_names = sorted(class_names)
    stoi = {"<pad>": 0, "<unk>": 1}
    for c in class_names:
        stoi[c] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return {"stoi": stoi, "itos": itos}


# Support-surface helpers

def get_support_mask(label_map, prompt):
    prompt = prompt.lower()
    if "person" in prompt or "rider" in prompt:
        support_mask = (label_map == SIDEWALK_LABEL_ID)
        if support_mask.sum() < 50:
            support_mask = (label_map == SIDEWALK_LABEL_ID) | (label_map == ROAD_LABEL_ID)
    elif "car" in prompt or "truck" in prompt or "bus" in prompt or "train" in prompt:
        support_mask = (label_map == ROAD_LABEL_ID) | (label_map == PARKING_LABEL_ID)
    elif "bicycle" in prompt or "motorcycle" in prompt:
        support_mask = (
            (label_map == ROAD_LABEL_ID) |
            (label_map == SIDEWALK_LABEL_ID) |
            (label_map == PARKING_LABEL_ID)
        )
    else:
        support_mask = (label_map == ROAD_LABEL_ID) | (label_map == SIDEWALK_LABEL_ID)
    return support_mask.astype(np.uint8)


def generate_global_fallback_candidates(
    grid_size=5,
    scales=(0.10, 0.16, 0.24),
    aspect_ratios=(0.5, 1.0, 2.0),
):
    centers = torch.linspace(0.08, 0.92, steps=grid_size)
    boxes = []
    for cy in centers:
        for cx in centers:
            for s in scales:
                for ar in aspect_ratios:
                    w = s * math.sqrt(ar)
                    h = s / math.sqrt(ar)
                    x1 = max(0.0, float(cx - w / 2))
                    y1 = max(0.0, float(cy - h / 2))
                    x2 = min(1.0, float(cx + w / 2))
                    y2 = min(1.0, float(cy + h / 2))
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)


GLOBAL_FALLBACK_CANDIDATES = generate_global_fallback_candidates()


def get_class_shape_priors(prompt):
    prompt = prompt.lower()
    if "person" in prompt or "rider" in prompt:
        scales = (0.10, 0.14, 0.18)
        aspect_ratios = (0.35, 0.45, 0.60)
    elif "car" in prompt or "truck" in prompt or "bus" in prompt or "train" in prompt:
        scales = (0.16, 0.22, 0.30)
        aspect_ratios = (1.2, 1.6, 2.0)
    elif "bicycle" in prompt or "motorcycle" in prompt:
        scales = (0.12, 0.16, 0.20)
        aspect_ratios = (0.8, 1.1, 1.5)
    else:
        scales = (0.12, 0.18, 0.24)
        aspect_ratios = (0.7, 1.0, 1.4)
    return scales, aspect_ratios


def sample_support_surface_candidates(
    label_map, prompt, num_total=256, support_ratio=0.95, seed=42,
):
    rng = np.random.default_rng(seed)
    H, W = label_map.shape
    support_mask = get_support_mask(label_map, prompt)
    ys, xs = np.where(support_mask > 0)
    scales, aspect_ratios = get_class_shape_priors(prompt)
    num_support = int(num_total * support_ratio)
    num_global = num_total - num_support
    boxes = []
    if len(xs) > 0 and num_support > 0:
        replace_flag = len(xs) < num_support
        sel = rng.choice(len(xs), size=num_support, replace=replace_flag)
        combo_list = [(s, ar) for s in scales for ar in aspect_ratios]
        for idx in sel:
            cx = (xs[idx] + 0.5) / W
            by = (ys[idx] + 0.5) / H
            s, ar = combo_list[rng.integers(0, len(combo_list))]
            bw = s * math.sqrt(ar)
            bh = s / math.sqrt(ar)
            x1 = float(cx - bw / 2)
            x2 = float(cx + bw / 2)
            y2 = float(by)
            y1 = float(by - bh)
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(1.0, x2)
            y2 = min(1.0, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    if num_global > 0:
        global_cands = GLOBAL_FALLBACK_CANDIDATES
        if len(global_cands) <= num_global:
            extra = global_cands
        else:
            perm = torch.randperm(len(global_cands))[:num_global]
            extra = global_cands[perm]
        boxes.extend(extra.tolist())
    while len(boxes) < num_total:
        extra = GLOBAL_FALLBACK_CANDIDATES[torch.randint(0, len(GLOBAL_FALLBACK_CANDIDATES), (1,))]
        boxes.extend(extra.tolist())
    boxes = torch.tensor(boxes[:num_total], dtype=torch.float32)
    return boxes, support_mask


def resize_label_map_to_image_size(label_map, size_hw=(224, 224)):
    img = Image.fromarray(label_map.astype(np.uint8))
    img = img.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    return np.array(img)


def compute_candidate_semantic_features(label_map_resized, candidate_boxes):
    H, W = label_map_resized.shape
    feats = []
    road_mask = ((label_map_resized == ROAD_LABEL_ID) | (label_map_resized == PARKING_LABEL_ID)).astype(np.float32)
    sidewalk_mask = (label_map_resized == SIDEWALK_LABEL_ID).astype(np.float32)
    person_mask = np.isin(label_map_resized, list(PERSON_LABEL_IDS)).astype(np.float32)
    vehicle_mask = np.isin(label_map_resized, list(VEHICLE_LABEL_IDS)).astype(np.float32)
    for box in candidate_boxes:
        x1, y1, x2, y2 = box.tolist()
        ix1 = max(0, min(int(x1 * W), W - 1))
        iy1 = max(0, min(int(y1 * H), H - 1))
        ix2 = max(0, min(int(np.ceil(x2 * W)), W))
        iy2 = max(0, min(int(np.ceil(y2 * H)), H))
        if ix2 <= ix1:
            ix2 = min(W, ix1 + 1)
        if iy2 <= iy1:
            iy2 = min(H, iy1 + 1)
        road_ratio = float(road_mask[iy1:iy2, ix1:ix2].mean())
        sidewalk_ratio = float(sidewalk_mask[iy1:iy2, ix1:ix2].mean())
        person_ratio = float(person_mask[iy1:iy2, ix1:ix2].mean())
        vehicle_ratio = float(vehicle_mask[iy1:iy2, ix1:ix2].mean())
        feats.append([road_ratio, sidewalk_ratio, person_ratio, vehicle_ratio])
    return torch.tensor(feats, dtype=torch.float32)


def build_candidate_valid_mask(label_map_resized, candidate_boxes, prompt):
    H, W = label_map_resized.shape
    support_mask = get_support_mask(label_map_resized, prompt).astype(np.uint8)
    sidewalk_mask = (label_map_resized == SIDEWALK_LABEL_ID).astype(np.float32)
    road_mask = ((label_map_resized == ROAD_LABEL_ID) | (label_map_resized == PARKING_LABEL_ID)).astype(np.float32)
    person_mask = np.isin(label_map_resized, list(PERSON_LABEL_IDS)).astype(np.float32)
    vehicle_mask = np.isin(label_map_resized, list(VEHICLE_LABEL_IDS)).astype(np.float32)
    valid = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box.tolist()
        ix1 = max(0, min(int(x1 * W), W - 1))
        iy1 = max(0, min(int(y1 * H), H - 1))
        ix2 = max(0, min(int(np.ceil(x2 * W)), W))
        iy2 = max(0, min(int(np.ceil(y2 * H)), H))
        if ix2 <= ix1:
            ix2 = min(W, ix1 + 1)
        if iy2 <= iy1:
            iy2 = min(H, iy1 + 1)
        bcx = max(0, min(int(((x1 + x2) / 2) * W), W - 1))
        bcy = max(0, min(int(y2 * H) - 1, H - 1))
        bottom_center_on_support = 1.0 if support_mask[bcy, bcx] > 0 else 0.0
        box_h = max(1, iy2 - iy1)
        strip_h = max(2, int(box_h * 0.15))
        sy1 = max(0, iy2 - strip_h)
        sy2 = iy2
        bottom_strip_support_ratio = float(support_mask[sy1:sy2, ix1:ix2].mean()) if sy2 > sy1 else 0.0
        sidewalk_ratio = float(sidewalk_mask[iy1:iy2, ix1:ix2].mean())
        road_ratio_val = float(road_mask[iy1:iy2, ix1:ix2].mean())
        person_ratio = float(person_mask[iy1:iy2, ix1:ix2].mean())
        vehicle_ratio = float(vehicle_mask[iy1:iy2, ix1:ix2].mean())
        p = prompt.lower()
        if "person" in p or "rider" in p:
            ok = (
                bottom_center_on_support > 0.5 and
                bottom_strip_support_ratio >= 0.35 and
                vehicle_ratio <= 0.20
            )
        elif "car" in p or "truck" in p or "bus" in p or "train" in p:
            ok = (
                bottom_center_on_support > 0.5 and
                bottom_strip_support_ratio >= 0.45 and
                sidewalk_ratio <= 0.20 and
                person_ratio <= 0.10 and
                road_ratio_val >= 0.20
            )
        elif "bicycle" in p or "motorcycle" in p:
            ok = (
                bottom_center_on_support > 0.5 and
                bottom_strip_support_ratio >= 0.30
            )
        else:
            ok = (
                bottom_center_on_support > 0.5 and
                bottom_strip_support_ratio >= 0.30
            )
        valid.append(1.0 if ok else 0.0)
    valid = torch.tensor(valid, dtype=torch.float32)
    if valid.sum() == 0:
        fallback_scores = []
        support_mask = get_support_mask(label_map_resized, prompt).astype(np.uint8)
        for box in candidate_boxes:
            x1, y1, x2, y2 = box.tolist()
            ix1 = max(0, min(int(x1 * W), W - 1))
            iy1 = max(0, min(int(y1 * H), H - 1))
            ix2 = max(0, min(int(np.ceil(x2 * W)), W))
            iy2 = max(0, min(int(np.ceil(y2 * H)), H))
            if ix2 <= ix1:
                ix2 = min(W, ix1 + 1)
            if iy2 <= iy1:
                iy2 = min(H, iy1 + 1)
            box_h = max(1, iy2 - iy1)
            strip_h = max(2, int(box_h * 0.15))
            sy1 = max(0, iy2 - strip_h)
            sy2 = iy2
            score = float(support_mask[sy1:sy2, ix1:ix2].mean()) if sy2 > sy1 else 0.0
            fallback_scores.append(score)
        best_idx = int(np.argmax(fallback_scores))
        valid[best_idx] = 1.0
    return valid


# Model definition

class BootplaceSupportSurfaceScoringModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_scene_classes,
        text_embed_dim=128,
        hidden_dim=256,
        scene_dim=128,
        scene_class_embed_dim=32,
        sem_feat_dim=4,
        freeze_image_encoder=True,
    ):
        super().__init__()
        self.freeze_image_encoder = freeze_image_encoder

        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)

        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_feat_dim = 512

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.scene_box_encoder = nn.Sequential(
            nn.Linear(4, scene_dim),
            nn.ReLU(inplace=True),
            nn.Linear(scene_dim, scene_dim),
            nn.ReLU(inplace=True),
        )
        self.scene_class_embedding = nn.Embedding(
            num_scene_classes, scene_class_embed_dim, padding_idx=0,
        )
        self.scene_fuse = nn.Sequential(
            nn.Linear(scene_dim + scene_class_embed_dim, scene_dim),
            nn.ReLU(inplace=True),
            nn.Linear(scene_dim, scene_dim),
            nn.ReLU(inplace=True),
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.candidate_box_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.roi_proj = nn.Sequential(
            nn.Linear(self.image_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.candidate_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.semantic_encoder = nn.Sequential(
            nn.Linear(sem_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_encoder.eval()
        return self

    def encode_image(self, images):
        fmap = self.image_encoder(images)
        pooled = self.global_pool(fmap).flatten(1)
        global_feat = self.image_proj(pooled)
        return fmap, global_feat

    def encode_text(self, input_ids, attention_mask):
        emb = self.text_embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        txt_feat = self.text_proj(pooled)
        return txt_feat

    def encode_scene(self, scene_boxes, scene_class_ids, scene_box_masks):
        if scene_boxes.shape[1] == 0:
            B = scene_boxes.shape[0]
            device = scene_boxes.device
            hidden_dim = self.scene_proj[0].out_features
            return torch.zeros((B, hidden_dim), device=device)
        box_feat = self.scene_box_encoder(scene_boxes)
        cls_feat = self.scene_class_embedding(scene_class_ids)
        obj_feat = torch.cat([box_feat, cls_feat], dim=-1)
        obj_feat = self.scene_fuse(obj_feat)
        mask = scene_box_masks.unsqueeze(-1).float()
        summed = (obj_feat * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        scn_feat = self.scene_proj(pooled)
        return scn_feat

    def _build_rois(self, candidate_boxes, image_h, image_w, device):
        B, K, _ = candidate_boxes.shape
        boxes_px = candidate_boxes.clone()
        boxes_px[..., 0] *= image_w
        boxes_px[..., 2] *= image_w
        boxes_px[..., 1] *= image_h
        boxes_px[..., 3] *= image_h
        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, 1).float()
        rois = torch.cat([batch_ids, boxes_px], dim=-1).reshape(B * K, 5)
        return rois

    def encode_candidates(self, fmap, candidate_boxes, image_h, image_w):
        B, K, _ = candidate_boxes.shape
        device = candidate_boxes.device
        rois = self._build_rois(candidate_boxes, image_h, image_w, device)
        spatial_scale = fmap.shape[-1] / float(image_w)
        roi_feat = roi_align(
            fmap, rois, output_size=(1, 1),
            spatial_scale=spatial_scale, aligned=True,
        )
        roi_feat = roi_feat.flatten(1)
        roi_feat = self.roi_proj(roi_feat)
        roi_feat = roi_feat.view(B, K, -1)
        box_feat = self.candidate_box_encoder(candidate_boxes.view(B * K, 4))
        box_feat = box_feat.view(B, K, -1)
        cand_feat = torch.cat([roi_feat, box_feat], dim=-1)
        cand_feat = self.candidate_fuse(cand_feat)
        return cand_feat

    def encode_candidate_semantics(self, candidate_sem_feats):
        B, K, D = candidate_sem_feats.shape
        sem_feat = self.semantic_encoder(candidate_sem_feats.view(B * K, D))
        sem_feat = sem_feat.view(B, K, -1)
        return sem_feat

    def forward(
        self, images, input_ids, attention_mask,
        scene_boxes, scene_class_ids, scene_box_masks,
        candidate_boxes, candidate_sem_feats,
    ):
        B, _, H, W = images.shape
        fmap, global_feat = self.encode_image(images)
        txt_feat = self.encode_text(input_ids, attention_mask)
        scn_feat = self.encode_scene(scene_boxes, scene_class_ids, scene_box_masks)
        query_feat = self.query_proj(torch.cat([global_feat, txt_feat, scn_feat], dim=-1))
        cand_feat = self.encode_candidates(fmap, candidate_boxes, H, W)
        sem_feat = self.encode_candidate_semantics(candidate_sem_feats)
        K = candidate_boxes.shape[1]
        query_expand = query_feat.unsqueeze(1).expand(B, K, -1)
        score_in = torch.cat([
            cand_feat, sem_feat, query_expand,
            cand_feat * query_expand, sem_feat * query_expand,
        ], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)
        best_idx = torch.argmax(scores, dim=1)
        pred_boxes = candidate_boxes[torch.arange(B, device=images.device), best_idx]
        return {"scores": scores, "best_idx": best_idx, "pred_boxes": pred_boxes}


# Loss functions

def build_soft_targets(candidate_ious, best_candidate_idx, candidate_valid_mask, gamma=2.0):
    B, K = candidate_ious.shape
    soft_targets = []
    for b in range(B):
        ious = candidate_ious[b].clamp(min=0) * candidate_valid_mask[b]
        weights = ious ** gamma
        s = weights.sum()
        if float(s) < 1e-8:
            target = torch.zeros_like(weights)
            target[best_candidate_idx[b]] = 1.0
        else:
            target = weights / s
        soft_targets.append(target)
    return torch.stack(soft_targets, dim=0)


def topk_iou_from_scores(scores, candidate_ious, k=1):
    topk_idx = torch.topk(scores, k=k, dim=1).indices
    gathered = torch.gather(candidate_ious, dim=1, index=topk_idx)
    return gathered.max(dim=1).values


def compute_candidate_scoring_loss(
    scores, candidate_ious, best_candidate_idx, candidate_valid_mask,
    soft_weight=1.0, hard_weight=0.5, gamma=2.0,
):
    log_probs = F.log_softmax(scores, dim=1)
    soft_targets = build_soft_targets(
        candidate_ious=candidate_ious,
        best_candidate_idx=best_candidate_idx,
        candidate_valid_mask=candidate_valid_mask,
        gamma=gamma,
    )
    loss_soft = -(soft_targets * log_probs).sum(dim=1).mean()
    loss_hard = F.cross_entropy(scores, best_candidate_idx)
    total = soft_weight * loss_soft + hard_weight * loss_hard
    top1_iou = topk_iou_from_scores(scores, candidate_ious, k=1).mean()
    top5_iou = topk_iou_from_scores(scores, candidate_ious, k=5).mean()
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_soft": float(loss_soft.detach().cpu()),
        "loss_hard": float(loss_hard.detach().cpu()),
        "top1_iou": float(top1_iou.detach().cpu()),
        "top5_iou": float(top5_iou.detach().cpu()),
    }


# Dataset definition (hardconstraint variant)

class BootplaceSupportSurfaceDataset(Dataset):
    def __init__(
        self, data_root, cityscapes_root, split="train",
        tokenizer=None, scene_class_vocab=None,
        image_size=224, num_candidates=256,
        support_ratio=0.95,         # <<< KEY DIFF: support surface guided
        base_seed=42,
    ):
        self.data_root = Path(data_root)
        self.cityscapes_root = Path(cityscapes_root)
        self.split = split
        self.records = read_jsonl(self.data_root / split / f"annotations_{split}.jsonl")
        self.tokenizer = tokenizer
        self.scene_class_vocab = scene_class_vocab
        self.image_size = image_size
        self.num_candidates = num_candidates
        self.support_ratio = support_ratio
        self.base_seed = base_seed
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    def __len__(self):
        return len(self.records)

    def _normalize_box_tensor(self, box, w, h):
        x1, y1, x2, y2 = box
        return torch.tensor([x1 / w, y1 / h, x2 / w, y2 / h], dtype=torch.float32)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # 1. image
        bg_path = self.data_root / rec["background_rel"].replace("\\", "/")
        image = Image.open(bg_path).convert("RGB")
        image_tensor = self.transform(image)

        # 2. image size
        w, h = rec["image_size_wh"]

        # 3. target bbox
        target_bbox = self._normalize_box_tensor(rec["target_bbox_xyxy_abs"], w, h)

        # 4. prompt
        prompt = rec["prompt"]
        input_ids, attention_mask = self.tokenizer.encode(prompt)

        # 5. scene boxes + scene class ids
        scene_boxes = []
        scene_class_ids = []
        for item in rec["scene_boxes_xyxy_abs"]:
            scene_boxes.append(self._normalize_box_tensor(item["bbox_xyxy_abs"], w, h))
            cls_id = self.scene_class_vocab["stoi"].get(
                item["class_name"], self.scene_class_vocab["stoi"]["<unk>"]
            )
            scene_class_ids.append(cls_id)
        if len(scene_boxes) == 0:
            scene_boxes = torch.zeros((0, 4), dtype=torch.float32)
            scene_class_ids = torch.zeros((0,), dtype=torch.long)
        else:
            scene_boxes = torch.stack(scene_boxes, dim=0)
            scene_class_ids = torch.tensor(scene_class_ids, dtype=torch.long)

        # 6. label map
        label_path = source_rel_to_label_path(self.cityscapes_root, rec["source_image_rel"])
        label_map = load_label_map(label_path)

        # 7. candidates  (support_ratio=0.95 => support surface guided)
        candidate_boxes, support_mask = sample_support_surface_candidates(
            label_map=label_map, prompt=prompt,
            num_total=self.num_candidates,
            support_ratio=self.support_ratio,     # 0.95
            seed=self.base_seed + idx,
        )

        # 8. resized label map for semantic features + valid mask
        label_map_resized = resize_label_map_to_image_size(
            label_map, size_hw=(self.image_size, self.image_size)
        )

        # <<< KEY DIFF: compute real semantic features
        candidate_sem_feats = compute_candidate_semantic_features(
            label_map_resized, candidate_boxes
        )

        # <<< KEY DIFF: compute real valid mask (hard constraint)
        candidate_valid_mask = build_candidate_valid_mask(
            label_map_resized=label_map_resized,
            candidate_boxes=candidate_boxes,
            prompt=prompt,
        )

        # 9. IoU supervision
        candidate_ious = pairwise_iou(candidate_boxes, target_bbox.unsqueeze(0)).squeeze(1)

        # <<< KEY DIFF: mask ious with valid_mask, then argmax
        valid_ious = candidate_ious.clone()
        valid_ious[candidate_valid_mask < 0.5] = -1.0
        best_candidate_idx = torch.argmax(valid_ious)

        return {
            "image": image_tensor,
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_bbox": target_bbox,
            "scene_boxes": scene_boxes,
            "scene_class_ids": scene_class_ids,
            "candidate_boxes": candidate_boxes,
            "candidate_sem_feats": candidate_sem_feats,
            "candidate_valid_mask": candidate_valid_mask,
            "candidate_ious": candidate_ious,
            "best_candidate_idx": best_candidate_idx,
            "target_class": rec["target_class"],
            "background_rel": rec["background_rel"],
        }


# Collation helpers

def support_surface_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    target_bboxes = torch.stack([x["target_bbox"] for x in batch], dim=0)
    candidate_boxes = torch.stack([x["candidate_boxes"] for x in batch], dim=0)
    candidate_sem_feats = torch.stack([x["candidate_sem_feats"] for x in batch], dim=0)
    candidate_valid_mask = torch.stack([x["candidate_valid_mask"] for x in batch], dim=0)
    candidate_ious = torch.stack([x["candidate_ious"] for x in batch], dim=0)
    best_candidate_idx = torch.stack([x["best_candidate_idx"] for x in batch], dim=0)
    max_num_scene = max(x["scene_boxes"].shape[0] for x in batch)
    padded_scene_boxes = []
    padded_scene_class_ids = []
    scene_box_masks = []
    for x in batch:
        n = x["scene_boxes"].shape[0]
        if n < max_num_scene:
            pad_boxes = torch.zeros((max_num_scene - n, 4), dtype=torch.float32)
            boxes = torch.cat([x["scene_boxes"], pad_boxes], dim=0)
            pad_cls = torch.zeros((max_num_scene - n,), dtype=torch.long)
            cls_ids = torch.cat([x["scene_class_ids"], pad_cls], dim=0)
            mask = torch.cat([
                torch.ones(n, dtype=torch.float32),
                torch.zeros(max_num_scene - n, dtype=torch.float32),
            ], dim=0)
        else:
            boxes = x["scene_boxes"]
            cls_ids = x["scene_class_ids"]
            mask = torch.ones(n, dtype=torch.float32)
        padded_scene_boxes.append(boxes)
        padded_scene_class_ids.append(cls_ids)
        scene_box_masks.append(mask)
    padded_scene_boxes = torch.stack(padded_scene_boxes, dim=0)
    padded_scene_class_ids = torch.stack(padded_scene_class_ids, dim=0)
    scene_box_masks = torch.stack(scene_box_masks, dim=0)
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_bboxes": target_bboxes,
        "scene_boxes": padded_scene_boxes,
        "scene_class_ids": padded_scene_class_ids,
        "scene_box_masks": scene_box_masks,
        "candidate_boxes": candidate_boxes,
        "candidate_sem_feats": candidate_sem_feats,
        "candidate_valid_mask": candidate_valid_mask,
        "candidate_ious": candidate_ious,
        "best_candidate_idx": best_candidate_idx,
        "prompts": [x["prompt"] for x in batch],
        "target_classes": [x["target_class"] for x in batch],
        "background_rels": [x["background_rel"] for x in batch],
    }


# Training loop (hardconstraint variant)

def run_one_epoch_support_surface(model, loader, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_total_sum = 0.0
    loss_soft_sum = 0.0
    loss_hard_sum = 0.0
    top1_iou_sum = 0.0
    top5_iou_sum = 0.0
    num_batches = 0

    desc = "train" if is_train else "val"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        scene_boxes = batch["scene_boxes"].to(device, non_blocking=True)
        scene_class_ids = batch["scene_class_ids"].to(device, non_blocking=True)
        scene_box_masks = batch["scene_box_masks"].to(device, non_blocking=True)
        candidate_boxes = batch["candidate_boxes"].to(device, non_blocking=True)
        candidate_sem_feats = batch["candidate_sem_feats"].to(device, non_blocking=True)
        candidate_valid_mask = batch["candidate_valid_mask"].to(device, non_blocking=True)
        candidate_ious = batch["candidate_ious"].to(device, non_blocking=True)
        best_candidate_idx = batch["best_candidate_idx"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model(
                images=images, input_ids=input_ids, attention_mask=attention_mask,
                scene_boxes=scene_boxes, scene_class_ids=scene_class_ids,
                scene_box_masks=scene_box_masks,
                candidate_boxes=candidate_boxes,
                candidate_sem_feats=candidate_sem_feats,
            )
            scores = out["scores"]

            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print("Found non-finite scores, skip batch")
                continue

            # <<< KEY DIFF: apply hard constraint mask
            scores = scores.masked_fill(candidate_valid_mask < 0.5, -1e4)

            loss, stats = compute_candidate_scoring_loss(
                scores=scores,
                candidate_ious=candidate_ious,
                best_candidate_idx=best_candidate_idx,
                candidate_valid_mask=candidate_valid_mask,
                soft_weight=1.0, hard_weight=0.5, gamma=2.0,
            )

            if not torch.isfinite(loss):
                print("Found non-finite loss, skip batch")
                continue

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        loss_total_sum += stats["loss_total"]
        loss_soft_sum += stats["loss_soft"]
        loss_hard_sum += stats["loss_hard"]
        top1_iou_sum += stats["top1_iou"]
        top5_iou_sum += stats["top5_iou"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_total_sum / num_batches:.4f}",
            "top1": f"{top1_iou_sum / num_batches:.4f}",
            "top5": f"{top5_iou_sum / num_batches:.4f}",
        })

    return {
        "loss_total": loss_total_sum / max(1, num_batches),
        "loss_soft": loss_soft_sum / max(1, num_batches),
        "loss_hard": loss_hard_sum / max(1, num_batches),
        "top1_iou": top1_iou_sum / max(1, num_batches),
        "top5_iou": top5_iou_sum / max(1, num_batches),
    }


# Main entrypoint

def main():
    parser = argparse.ArgumentParser(description="Train HardConstraint model")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_candidates", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Pre-cache ResNet-18 weights before training begins.
    import os
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        resnet18(weights=ResNet18_Weights.DEFAULT)
        print("ResNet-18 weights: OK (cached)")
    except Exception as e:
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        raise RuntimeError(
            "Failed to load the ResNet-18 pretrained weights.\n"
            "This commonly happens on a machine without internet access during the first run.\n"
            f"Cache directory: {cache_dir}\n"
            "Try pre-caching the weights with:\n"
            "  python -c \"from torchvision.models import resnet18, ResNet18_Weights; "
            "resnet18(weights=ResNet18_Weights.DEFAULT)\"\n"
            f"Original error: {e}"
        )

    torch.manual_seed(0)
    data_root = Path(args.data_root).resolve()
    cityscapes_root = Path(args.cityscapes_root).resolve()
    save_dir = Path(args.save_dir).resolve()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    pin_memory = device == "cuda"
    print(f"data_root       = {data_root}")
    print(f"cityscapes_root = {cityscapes_root}")
    print(f"save_dir        = {save_dir}")
    print(f"device          = {device}")

    # Build tokenizer and vocab.
    train_ann_path = data_root / "train" / "annotations_train.jsonl"
    val_ann_path = data_root / "val" / "annotations_val.jsonl"
    if not train_ann_path.exists():
        raise FileNotFoundError(f"Expected training annotations file not found: {train_ann_path}")
    if not val_ann_path.exists():
        raise FileNotFoundError(f"Expected validation annotations file not found: {val_ann_path}")
    train_records = read_jsonl(train_ann_path)
    train_prompts = [r["prompt"] for r in train_records]
    prompt_tokenizer = SimplePromptTokenizer(train_prompts, max_len=8)
    scene_class_vocab = build_scene_class_vocab(train_ann_path)

    print(f"prompt vocab size = {prompt_tokenizer.vocab_size}")
    print(f"scene class vocab size = {len(scene_class_vocab['stoi'])}")

    # datasets
    train_dataset = BootplaceSupportSurfaceDataset(
        data_root=data_root,
        cityscapes_root=cityscapes_root,
        split="train",
        tokenizer=prompt_tokenizer,
        scene_class_vocab=scene_class_vocab,
        image_size=args.image_size,
        num_candidates=args.num_candidates,
        support_ratio=0.95,
        base_seed=42,
    )
    val_dataset = BootplaceSupportSurfaceDataset(
        data_root=data_root,
        cityscapes_root=cityscapes_root,
        split="val",
        tokenizer=prompt_tokenizer,
        scene_class_vocab=scene_class_vocab,
        image_size=args.image_size,
        num_candidates=args.num_candidates,
        support_ratio=0.95,
        base_seed=4242,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=support_surface_collate_fn, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=support_surface_collate_fn, pin_memory=pin_memory,
    )

    # model
    model = BootplaceSupportSurfaceScoringModel(
        vocab_size=prompt_tokenizer.vocab_size,
        num_scene_classes=len(scene_class_vocab["stoi"]),
        text_embed_dim=128, hidden_dim=256,
        scene_dim=128, scene_class_embed_dim=32,
        sem_feat_dim=4, freeze_image_encoder=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    save_dir = save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = save_dir / "03_ss_hard_last.pt"
    best_ckpt = save_dir / "03_ss_hard_best.pt"

    if last_ckpt.exists() and not args.resume:
        raise RuntimeError(
            f"Checkpoint already exists: {last_ckpt}\n"
            "Pass --resume to continue this run, or choose a different --save_dir."
        )

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt["history"]
        best_val_top1 = ckpt.get("best_val_top1_iou", -1.0)
        print(f"Resumed from epoch {start_epoch}, best_val_top1={best_val_top1:.4f}")
    else:
        start_epoch = 0
        history = []
        best_val_top1 = -1.0

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.num_epochs} ===")

        train_stats = run_one_epoch_support_surface(model, train_loader, optimizer=optimizer, device=device)
        val_stats = run_one_epoch_support_surface(model, val_loader, optimizer=None, device=device)

        print(f"  train | total={train_stats['loss_total']:.4f}  soft={train_stats['loss_soft']:.4f}  hard={train_stats['loss_hard']:.4f}  top1={train_stats['top1_iou']:.4f}  top5={train_stats['top5_iou']:.4f}")
        print(f"  val   | total={val_stats['loss_total']:.4f}  soft={val_stats['loss_soft']:.4f}  hard={val_stats['loss_hard']:.4f}  top1={val_stats['top1_iou']:.4f}  top5={val_stats['top5_iou']:.4f}")

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        is_new_best = val_stats["top1_iou"] > best_val_top1
        if is_new_best:
            best_val_top1 = val_stats["top1_iou"]

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_top1_iou": best_val_top1,
        }

        torch.save(ckpt_payload, last_ckpt)
        print(f"  saved last checkpoint -> {last_ckpt}")

        if is_new_best:
            torch.save(ckpt_payload, best_ckpt)
            print(f"  saved BEST checkpoint -> {best_ckpt}  (top1={best_val_top1:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
