# -*- coding: utf-8 -*-
"""
共享训练组件 — 用于 7-class 流水线防欠拟合/过拟合:

    1. WarmupCosineLR:  线性 warmup + 余弦衰减到 min_lr。
                        类别变多后 head 全新, 前几个 epoch 大梯度容易毁掉 K400 权重,
                        warmup 给 backbone 一个缓冲期。

    2. MixUpCutMix:     视频级 MixUp / CutMix。对 16 帧序列在 batch 内做混合,
                        给 7 类 (尤其是 limp / sneeze) 一个软目标分布, 显著缓解过拟合。

    3. soft_hierarchical_loss(): 在 MixUp 启用时, 用线性混合的 one-hot 取代硬标签,
                                 计算分层 loss (T1 / T2 / T3 全部支持软标签)。
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


# ----------------------------------------------------------------------
# 1. Warmup + Cosine LR
# ----------------------------------------------------------------------
class WarmupCosineLR:
    """按 step (而不是 epoch) 调用 step(), 更平滑。"""

    def __init__(self, optimizer, warmup_steps, total_steps, base_lrs=None,
                 min_lr=1e-6):
        self.optim = optimizer
        self.warmup = max(1, int(warmup_steps))
        self.total = max(self.warmup + 1, int(total_steps))
        self.min_lr = min_lr
        if base_lrs is None:
            base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.base_lrs = list(base_lrs)
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, g in enumerate(self.optim.param_groups):
            base = self.base_lrs[i]
            if s <= self.warmup:
                lr = base * s / float(self.warmup)
            else:
                progress = (s - self.warmup) / max(1, self.total - self.warmup)
                progress = min(1.0, progress)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * \
                    (1.0 + math.cos(math.pi * progress))
            g["lr"] = lr

    def get_lr(self):
        return [g["lr"] for g in self.optim.param_groups]


# ----------------------------------------------------------------------
# 2. Video-level MixUp / CutMix
# ----------------------------------------------------------------------
class MixUpCutMix:
    """
    Args:
        alpha:        Beta(α,α) for MixUp
        cutmix_alpha: Beta(α,α) for CutMix
        prob:         整个 mix-augment 触发的概率
        switch_prob:  触发后, CutMix vs MixUp 的概率
        num_classes:  分类数, 用于生成 one-hot 软标签
    """

    def __init__(self, alpha=0.2, cutmix_alpha=1.0, prob=0.5,
                 switch_prob=0.5, num_classes=7):
        self.alpha = float(alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.num_classes = int(num_classes)

    def _one_hot(self, y, device):
        return F.one_hot(y, num_classes=self.num_classes).float().to(device)

    def __call__(self, x, y):
        """
        x: [B, T, C, H, W]
        y: [B] LongTensor
        Returns:
            x_mixed: [B, T, C, H, W]
            y_soft : [B, num_classes]   (始终是 soft target)
            applied: bool               (是否真的混合了, 供日志使用)
        """
        device = x.device
        y_soft = self._one_hot(y, device)
        if np.random.rand() >= self.prob or x.size(0) < 2:
            return x, y_soft, False

        B = x.size(0)
        perm = torch.randperm(B, device=device)
        x2 = x[perm]
        y2_soft = y_soft[perm]

        use_cutmix = np.random.rand() < self.switch_prob and self.cutmix_alpha > 0
        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            _, _, _, H, W = x.shape
            cut_rat = math.sqrt(1.0 - lam)
            cw, ch = int(W * cut_rat), int(H * cut_rat)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            x1_ = max(cx - cw // 2, 0); x2_ = min(cx + cw // 2, W)
            y1_ = max(cy - ch // 2, 0); y2_ = min(cy + ch // 2, H)
            x_out = x.clone()
            x_out[:, :, :, y1_:y2_, x1_:x2_] = x2[:, :, :, y1_:y2_, x1_:x2_]
            lam = 1.0 - ((x2_ - x1_) * (y2_ - y1_) / float(H * W))
        else:
            lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
            x_out = lam * x + (1.0 - lam) * x2

        y_mixed = lam * y_soft + (1.0 - lam) * y2_soft
        return x_out, y_mixed, True


# ----------------------------------------------------------------------
# 3. Soft-label 版本的分层 loss (供 MixUp 启用时使用)
# ----------------------------------------------------------------------
def hierarchical_soft_loss(logits, y_soft, hier_loss_module,
                           normal_ids, abnormal_ids, ingestion_ids,
                           w_t1, w_t2, w_t3,
                           class_weight=None, label_smoothing=0.0):
    """
    与 HierarchicalPriorityLoss 等价, 但接收 soft target (B, num_classes)。
    用于 MixUp / CutMix 启用时的训练; 不混合时 (y_soft 仍为 one-hot) 结果与硬标签等价。
    """
    eps = 1e-7
    device = logits.device
    probs = F.softmax(logits, dim=1)

    abn_idx = torch.as_tensor(list(abnormal_ids), device=device, dtype=torch.long)
    norm_idx = torch.as_tensor(list(normal_ids), device=device, dtype=torch.long)
    ing_idx = torch.as_tensor(list(ingestion_ids), device=device, dtype=torch.long)

    # T1
    p_abn = probs.index_select(1, abn_idx).sum(dim=1).clamp(eps, 1 - eps)
    t1_target = y_soft.index_select(1, abn_idx).sum(dim=1).clamp(0.0, 1.0)
    loss_t1 = -(t1_target * torch.log(p_abn) +
                (1.0 - t1_target) * torch.log(1.0 - p_abn)).mean()

    # T2 (only normal mass)
    normal_mass = y_soft.index_select(1, norm_idx).sum(dim=1)
    if (normal_mass > 0).any():
        p_norm_subset = F.softmax(logits.index_select(1, norm_idx), dim=1)
        ing_local = [list(normal_ids).index(c) for c in ingestion_ids
                     if c in list(normal_ids)]
        ing_local_t = torch.as_tensor(ing_local, device=device, dtype=torch.long)
        p_ing = p_norm_subset.index_select(1, ing_local_t).sum(dim=1).clamp(eps, 1 - eps)
        t2_target = y_soft.index_select(1, ing_idx).sum(dim=1)
        # 只在 normal 子集上计算 (用 normal_mass 加权)
        bce_t2 = -(t2_target * torch.log(p_ing) +
                   (normal_mass - t2_target).clamp(min=0) * torch.log(1.0 - p_ing))
        denom = normal_mass.sum().clamp(min=eps)
        loss_t2 = bce_t2.sum() / denom
    else:
        loss_t2 = torch.tensor(0.0, device=device)

    # T3: KL(y_soft, softmax(logits)) — 等价于带 label smoothing 的 soft-CE
    if label_smoothing > 0:
        y_soft = (1.0 - label_smoothing) * y_soft + \
                 label_smoothing / y_soft.size(1)
    log_p = F.log_softmax(logits, dim=1)
    if class_weight is not None:
        cw = class_weight.to(device).view(1, -1)
        loss_t3 = -(y_soft * cw * log_p).sum(dim=1).mean()
    else:
        loss_t3 = -(y_soft * log_p).sum(dim=1).mean()

    total = w_t1 * loss_t1 + w_t2 * loss_t2 + w_t3 * loss_t3
    stats = {
        "l_t1": float(loss_t1.detach().cpu()),
        "l_t2": float(loss_t2.detach().cpu()),
        "l_t3": float(loss_t3.detach().cpu()),
    }
    return total, stats
