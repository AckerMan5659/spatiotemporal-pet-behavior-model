# -*- coding: utf-8 -*-
"""
Hierarchical priority loss for 7-class merged dataset.

Targets (priority T1 > T2 > T3):
    T1 — 二分类:  Normal {0,1,2}  vs  Abnormal {3,4,5,6}
    T2 — 吃喝:    Ingestion {1,2} vs  Other {0}    (仅在 Normal 子集计算)
    T3 — 细粒度: 7-way cross-entropy

Logits 是 7 维; T1 / T2 由 7 维 softmax 的子集概率合并得到, 保证三个目标
在同一组参数上联合优化, 同时通过权重 w1 > w2 > w3 锁定优先级。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7


class HierarchicalPriorityLoss(nn.Module):
    def __init__(
        self,
        normal_ids=(0, 1, 2),
        abnormal_ids=(3, 4, 5, 6),
        ingestion_ids=(1, 2),
        other_id=0,
        w_t1=2.0,
        w_t2=1.2,
        w_t3=1.0,
        label_smoothing=0.1,
        class_weight=None,
        focal_gamma=0.0,
    ):
        super().__init__()
        self.normal_ids = list(normal_ids)
        self.abnormal_ids = list(abnormal_ids)
        self.ingestion_ids = list(ingestion_ids)
        self.other_id = int(other_id)

        self.w_t1 = float(w_t1)
        self.w_t2 = float(w_t2)
        self.w_t3 = float(w_t3)
        self.label_smoothing = label_smoothing
        self.focal_gamma = float(focal_gamma)

        if class_weight is not None and not torch.is_tensor(class_weight):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight",
                             class_weight if class_weight is not None
                             else torch.empty(0))

        # T3 uses CE with optional class weight + label smoothing
        # (we will instantiate at call-time so the device matches the logits)

    # ------------------------------------------------------------------
    @staticmethod
    def _binary_bce(p_pos, target_pos, eps=EPS):
        p_pos = p_pos.clamp(eps, 1.0 - eps)
        return -(target_pos * torch.log(p_pos) +
                 (1.0 - target_pos) * torch.log(1.0 - p_pos))

    def _focal_weight(self, p_pos, target_pos):
        if self.focal_gamma <= 0:
            return torch.ones_like(p_pos)
        pt = torch.where(target_pos > 0.5, p_pos, 1.0 - p_pos)
        return (1.0 - pt).pow(self.focal_gamma)

    # ------------------------------------------------------------------
    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 7]
            targets: [B] in [0, 6]
        Returns:
            total_loss, dict(stats) — l_t1, l_t2, l_t3, plus pred buckets.
        """
        device = logits.device
        probs = F.softmax(logits, dim=1)

        # --- T1: Normal vs Abnormal ----------------------------------
        p_abn = probs[:, self.abnormal_ids].sum(dim=1)
        bin_target = torch.zeros_like(p_abn)
        for c in self.abnormal_ids:
            bin_target = bin_target + (targets == c).float()
        bin_target = bin_target.clamp(0.0, 1.0)

        bce_t1 = self._binary_bce(p_abn, bin_target)
        w_focal = self._focal_weight(p_abn, bin_target)
        loss_t1 = (bce_t1 * w_focal).mean()

        # --- T2: Ingestion vs Other (only normal samples) ------------
        is_normal = torch.zeros_like(targets, dtype=torch.bool)
        for c in self.normal_ids:
            is_normal = is_normal | (targets == c)

        loss_t2 = torch.tensor(0.0, device=device)
        if is_normal.any():
            normal_logits = logits[is_normal]
            normal_targets = targets[is_normal]
            p_normal_subset = F.softmax(normal_logits[:, self.normal_ids], dim=1)

            # ingestion = sum of P(eat) + P(drink) within the normal head subset
            ing_local_ids = [self.normal_ids.index(c) for c in self.ingestion_ids
                             if c in self.normal_ids]
            p_ing = p_normal_subset[:, ing_local_ids].sum(dim=1)

            ing_target = torch.zeros_like(p_ing)
            for c in self.ingestion_ids:
                ing_target = ing_target + (normal_targets == c).float()
            ing_target = ing_target.clamp(0.0, 1.0)

            bce_t2 = self._binary_bce(p_ing, ing_target)
            w_focal2 = self._focal_weight(p_ing, ing_target)
            loss_t2 = (bce_t2 * w_focal2).mean()

        # --- T3: 7-way CE with class weight + label smoothing --------
        cw = self.class_weight if self.class_weight.numel() > 0 else None
        if cw is not None:
            cw = cw.to(device)
        ce = F.cross_entropy(logits, targets,
                             weight=cw,
                             label_smoothing=self.label_smoothing)
        loss_t3 = ce

        total = self.w_t1 * loss_t1 + self.w_t2 * loss_t2 + self.w_t3 * loss_t3

        stats = {
            "l_t1": float(loss_t1.detach().cpu()),
            "l_t2": float(loss_t2.detach().cpu()),
            "l_t3": float(loss_t3.detach().cpu()),
            "p_abn_mean": float(p_abn.detach().mean().cpu()),
        }
        return total, stats


# ----------------------------------------------------------------------
# Convenience: convert 7-way logits -> 2-way / 2-way / 7-way predictions
# ----------------------------------------------------------------------
def hierarchical_predictions(logits,
                             normal_ids=(0, 1, 2),
                             abnormal_ids=(3, 4, 5, 6),
                             ingestion_ids=(1, 2)):
    probs = F.softmax(logits, dim=1)
    p_abn = probs[:, list(abnormal_ids)].sum(dim=1)
    p_norm = probs[:, list(normal_ids)].sum(dim=1)

    t1_pred = (p_abn > p_norm).long()   # 0=normal, 1=abnormal

    # T2 valid only when predicted normal
    p_ing = probs[:, list(ingestion_ids)].sum(dim=1)
    p_other = probs[:, 0]  # other id 默认 0
    t2_pred = (p_ing > p_other).long()  # 0=other, 1=ingestion

    t3_pred = probs.argmax(dim=1)
    return t1_pred, t2_pred, t3_pred, probs
