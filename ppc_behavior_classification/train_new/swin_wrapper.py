# -*- coding: utf-8 -*-
"""
Swin3D-B teacher wrapper (与 train/swin_wrapper.py 等价, 默认 7 类).

输入: [B, T, C, H, W]
输出: (logits[B, num_classes], features[list])  -- features 顺序与 train/ 中一致
"""

import os
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SwinTeacher(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=7, dropout=0.1):
        super().__init__()
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        print(f"👨‍🏫 [SwinTeacher] Video Swin-B -> {num_classes} classes")

        try:
            weights = Swin3D_B_Weights.KINETICS400_V1
            self.model = swin3d_b(weights=weights)
        except Exception as e:
            print(f"⚠️ K400 weight download failed: {e}; random init")
            self.model = swin3d_b(weights=None)

        if num_classes != 400:
            old_head = self.model.head
            if isinstance(old_head, nn.Sequential):
                in_feat = old_head[-1].in_features
                drop_p = old_head[0].p if isinstance(old_head[0], nn.Dropout) else dropout
            else:
                in_feat = old_head.in_features
                drop_p = dropout
            print(f"🔪 [Head Swap] {in_feat} -> {num_classes}")
            self.model.head = nn.Sequential(
                nn.Dropout(p=drop_p),
                nn.Linear(in_feat, num_classes),
            )

        if checkpoint_path and isinstance(checkpoint_path, str) \
                and checkpoint_path.endswith(".pth") \
                and os.path.exists(checkpoint_path):
            print(f"📥 [SwinTeacher] loading checkpoint: {checkpoint_path}")
            try:
                sd = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                clean = {}
                for k, v in sd.items():
                    new_k = k.replace("base.model.", "").replace("module.", "")
                    clean[new_k] = v
                miss, unexp = self.model.load_state_dict(clean, strict=False)
                print(f"   loaded (missing={len(miss)}, unexpected={len(unexp)})")
            except Exception as e:
                print(f"⚠️ load failed: {e}")

        # Hooks for multi-stage features (与原 train/swin_wrapper 对齐)
        self.features = []
        def hook_fn(module, inp, out):
            if out.dim() == 5:
                out = out.permute(0, 4, 1, 2, 3)
            self.features.append(out)

        target_layers = [1, 2, 3, 4]
        num_layers = len(self.model.features)
        for idx in target_layers:
            if idx < num_layers:
                self.model.features[idx].register_forward_hook(hook_fn)

    def forward(self, x):
        # x: [B, T, C, H, W] -> swin3d expects [B, C, T, H, W]
        self.features = []
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.model(x)
        return logits, self.features
