# -*- coding: utf-8 -*-
"""
7-class merged dataset loader.

目录结构 (与 dataset_summary.md 对齐):
    merged_dataset/
    ├── train/{sample}/
    │       frame_0000.jpg ... frame_0015.jpg
    │       label.txt          (单行单整数 0~6)
    └── val/{sample}/  ...

读取规则:
    - 帧数固定为 16；不足 16 的样本用零张量填充。
    - 标签优先读取 label.txt；缺失时回退到文件夹名关键词（兼容历史目录）。
"""

import os
import math
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


DEFAULT_CLASS_NAMES = [
    "other",       # 0
    "eat",         # 1
    "drink",       # 2
    "convulsion",  # 3
    "limp",        # 4
    "sneeze",      # 5
    "vomit",       # 6
]


def _read_label_file(label_path):
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read().strip().split()
        if not text:
            return -1
        return int(text[0])
    except Exception:
        return -1


def _fallback_label_from_name(name, class_names):
    name_lower = name.lower()
    for idx, kw in enumerate(class_names):
        if kw and kw.lower() in name_lower:
            return idx
    return -1


class MergedVideoDataset(Dataset):
    """16-frame clip dataset for the 7-class merged_dataset."""

    def __init__(self, root, split="train", cfg=None, is_train=True):
        self.root = os.path.join(root, split) if split else root
        self.cfg = cfg if cfg is not None else {}
        self.is_train = is_train

        self.class_names = self.cfg.get("class_names", DEFAULT_CLASS_NAMES)
        self.num_classes = self.cfg.get("num_classes", len(self.class_names))
        self.seq_len = self.cfg.get("seq_len",
                                    self.cfg.get("temporal_window", 16))
        self.img_size = self.cfg.get("imgsz", 224)

        self.transform_eval = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        aug_cfg = self.cfg.get("augment", {})
        self.scale_min = aug_cfg.get("scale_min", 0.9)
        self.scale_max = aug_cfg.get("scale_max", 1.1)
        self.rotate_deg = aug_cfg.get("rotate", 15)
        self.translate = aug_cfg.get("translate", 0.1)
        self.fliplr = aug_cfg.get("fliplr", 0.5)
        self.brightness = aug_cfg.get("brightness", 0.1)
        self.contrast = aug_cfg.get("contrast", 0.1)
        self.random_erase_p = aug_cfg.get("random_erase_p", 0.0)
        self.cj_sat = aug_cfg.get("color_jitter_sat", 0.0)
        self.cj_hue = aug_cfg.get("color_jitter_hue", 0.0)

        self.samples = []   # [(folder_path, label)]
        self.targets = []
        self._scan()

        if len(self.samples) == 0:
            print(f"❌ [Dataset] No sample loaded from {self.root}")
        else:
            dist = {i: 0 for i in range(self.num_classes)}
            for t in self.targets:
                dist[t] = dist.get(t, 0) + 1
            named = {self.class_names[i] if i < len(self.class_names) else str(i): c
                     for i, c in dist.items()}
            print(f"✅ [Dataset:{split}] 共 {len(self.samples)} 个样本")
            print(f"📊 [Dataset:{split}] 分布: {named}")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------
    def _scan(self):
        if not os.path.exists(self.root):
            print(f"❌ [Dataset] 路径不存在: {self.root}")
            return

        for entry in sorted(os.listdir(self.root)):
            if entry.startswith("."):
                continue
            folder = os.path.join(self.root, entry)
            if not os.path.isdir(folder):
                continue

            label_path = os.path.join(folder, "label.txt")
            if os.path.isfile(label_path):
                label = _read_label_file(label_path)
            else:
                label = _fallback_label_from_name(entry, self.class_names)

            if not (0 <= label < self.num_classes):
                continue

            images = [f for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not images:
                continue

            self.samples.append((folder, label))
            self.targets.append(label)

    # ------------------------------------------------------------------
    # Per-sample augmentation (consistent across the 16 frames of a clip)
    # ------------------------------------------------------------------
    def _build_train_transform(self):
        do_flip = random.random() < self.fliplr
        scale = random.uniform(self.scale_min, self.scale_max)
        rot = random.uniform(-self.rotate_deg, self.rotate_deg)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        bright = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast = 1.0 + random.uniform(-self.contrast, self.contrast)

        ops = [transforms.Resize((self.img_size, self.img_size))]
        cj_kwargs = dict(brightness=(bright, bright),
                         contrast=(contrast, contrast))
        if self.cj_sat > 0:
            cj_kwargs["saturation"] = (max(0.0, 1.0 - self.cj_sat),
                                        1.0 + self.cj_sat)
        if self.cj_hue > 0:
            cj_kwargs["hue"] = (-self.cj_hue, self.cj_hue)
        ops.append(transforms.ColorJitter(**cj_kwargs))
        if do_flip:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        ops.append(transforms.RandomAffine(
            degrees=(rot, rot),
            translate=(abs(tx), abs(ty)) if (tx or ty) else None,
            scale=(scale, scale),
        ))
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
        if self.random_erase_p > 0:
            ops.append(transforms.RandomErasing(p=self.random_erase_p,
                                                 scale=(0.02, 0.2),
                                                 ratio=(0.3, 3.3)))
        return transforms.Compose(ops)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def _load_frames(self, folder):
        all_imgs = sorted([f for f in os.listdir(folder)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not all_imgs:
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size)

        if len(all_imgs) >= self.seq_len:
            idxs = np.linspace(0, len(all_imgs) - 1, self.seq_len).astype(int)
        else:
            idxs = list(range(len(all_imgs)))

        tfm = self._build_train_transform() if self.is_train else self.transform_eval

        frames = []
        for i in idxs:
            try:
                img = Image.open(os.path.join(folder, all_imgs[i])).convert("RGB")
                frames.append(tfm(img))
            except Exception:
                frames.append(torch.zeros(3, self.img_size, self.img_size))

        while len(frames) < self.seq_len:
            frames.append(torch.zeros(3, self.img_size, self.img_size))
        return torch.stack(frames[: self.seq_len])

    def __getitem__(self, idx):
        folder, label = self.samples[idx]
        try:
            pixel_values = self._load_frames(folder)
        except Exception as e:
            print(f"⚠️ load {folder} failed: {e}")
            pixel_values = torch.zeros(self.seq_len, 3, self.img_size, self.img_size)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Imbalance utilities
    # ------------------------------------------------------------------
    def class_counts(self):
        counts = np.bincount(np.asarray(self.targets, dtype=np.int64),
                             minlength=self.num_classes)
        return counts

    def get_class_weights(self, device="cpu", scheme="inverse"):
        """1/freq normalized to mean 1."""
        counts = self.class_counts().astype(np.float64)
        if counts.sum() == 0:
            return None
        if scheme == "effective":
            beta = 0.9999
            eff = 1.0 - np.power(beta, counts)
            w = (1.0 - beta) / np.maximum(eff, 1e-8)
        else:
            w = 1.0 / np.maximum(counts, 1.0)
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float32, device=device)

    def make_sampler(self, ingestion_ids=(1, 2), abnormal_ids=(3, 4, 5, 6),
                     other_id=0, abnormal_boost=1.2, ingestion_boost=1.0,
                     other_boost=0.8):
        """
        WeightedRandomSampler — 让分层目标采样更均衡:
            - 异常类 (T1 正类) 略微上采样
            - 吃喝类 (T2 正类) 保持
            - other 略微下采样
        """
        counts = self.class_counts().astype(np.float64)
        base = 1.0 / np.maximum(counts, 1.0)
        per_class = base.copy()
        for c in abnormal_ids:
            if c < len(per_class):
                per_class[c] *= abnormal_boost
        for c in ingestion_ids:
            if c < len(per_class):
                per_class[c] *= ingestion_boost
        if other_id < len(per_class):
            per_class[other_id] *= other_boost

        weights = np.asarray([per_class[t] for t in self.targets],
                             dtype=np.float64)
        return WeightedRandomSampler(weights=torch.as_tensor(weights,
                                                             dtype=torch.double),
                                     num_samples=len(weights),
                                     replacement=True)


def get_dataset(root, split, cfg, is_train=True):
    return MergedVideoDataset(root, split=split, cfg=cfg, is_train=is_train)
