# -*- coding: utf-8 -*-
"""
共享训练组件 — 7-class 流水线防欠拟合/过拟合:

    1. WarmupCosineLR       线性 warmup + 余弦衰减到 min_lr（step 粒度）
    2. MixUpCutMix          视频级 MixUp / CutMix
    3. hierarchical_soft_loss   MixUp 启用时的软标签分层 loss
                                支持 T1 / T2 / T3(sneeze-vomit) / T4 全链路
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Warmup + Cosine LR
# ──────────────────────────────────────────────────────────────────────
class WarmupCosineLR:
    """按 step 调用 step()，比 epoch-level scheduler 更平滑。"""

    def __init__(self, optimizer, warmup_steps, total_steps,
                 base_lrs=None, min_lr=1e-6):
        self.optim    = optimizer
        self.warmup   = max(1, int(warmup_steps))
        self.total    = max(self.warmup + 1, int(total_steps))
        self.min_lr   = float(min_lr)
        self.base_lrs = (list(base_lrs) if base_lrs is not None
                         else [g["lr"] for g in optimizer.param_groups])
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, g in enumerate(self.optim.param_groups):
            base = self.base_lrs[i]
            if s <= self.warmup:
                lr = base * s / float(self.warmup)
            else:
                progress = min(1.0, (s - self.warmup) /
                               max(1, self.total - self.warmup))
                lr = self.min_lr + 0.5 * (base - self.min_lr) * \
                     (1.0 + math.cos(math.pi * progress))
            g["lr"] = lr

    def get_lr(self):
        return [g["lr"] for g in self.optim.param_groups]


# ──────────────────────────────────────────────────────────────────────
# 2. Video-level MixUp / CutMix
# ──────────────────────────────────────────────────────────────────────
class MixUpCutMix:
    def __init__(self, alpha=0.2, cutmix_alpha=1.0,
                 prob=0.5, switch_prob=0.5, num_classes=7):
        self.alpha        = float(alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob         = float(prob)
        self.switch_prob  = float(switch_prob)
        self.num_classes  = int(num_classes)

    def _one_hot(self, y, device):
        return F.one_hot(y, num_classes=self.num_classes).float().to(device)

    def __call__(self, x, y):
        """
        x: [B, T, C, H, W]   y: [B] LongTensor
        Returns: x_mixed, y_soft [B, C], applied bool
        """
        device = x.device
        y_soft = self._one_hot(y, device)
        if np.random.rand() >= self.prob or x.size(0) < 2:
            return x, y_soft, False

        B    = x.size(0)
        perm = torch.randperm(B, device=device)
        x2   = x[perm]
        y2s  = y_soft[perm]

        if np.random.rand() < self.switch_prob and self.cutmix_alpha > 0:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            _, _, _, H, W = x.shape
            cut_rat = math.sqrt(1.0 - lam)
            cw, ch  = int(W * cut_rat), int(H * cut_rat)
            cx, cy  = np.random.randint(W), np.random.randint(H)
            x1_ = max(cx - cw // 2, 0); x2_ = min(cx + cw // 2, W)
            y1_ = max(cy - ch // 2, 0); y2_ = min(cy + ch // 2, H)
            x_out = x.clone()
            x_out[:, :, :, y1_:y2_, x1_:x2_] = x2[:, :, :, y1_:y2_, x1_:x2_]
            lam = 1.0 - (x2_ - x1_) * (y2_ - y1_) / float(H * W)
        else:
            lam   = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
            x_out = lam * x + (1.0 - lam) * x2

        return x_out, lam * y_soft + (1.0 - lam) * y2s, True


# ──────────────────────────────────────────────────────────────────────
# 3. Soft-label hierarchical loss  (T1 / T2 / T3 / T4)
# ──────────────────────────────────────────────────────────────────────
def hierarchical_soft_loss(logits, y_soft,
                           hier_loss_module,        # unused, kept for compat
                           normal_ids, abnormal_ids,
                           ingestion_ids, sneeze_vomit_ids,
                           w_t1, w_t2, w_t3, w_t4,
                           class_weight=None, label_smoothing=0.0):
    """
    与 HierarchicalPriorityLoss.forward() 等价但接收软标签 [B, 7]。
    MixUp 未触发时 y_soft 为 one-hot，结果与硬标签完全一致。

    T3 (sneeze vs vomit): 使用 y_soft 中 class 5/6 的概率质量比值作为软目标。
    """
    eps    = 1e-7
    device = logits.device
    probs  = F.softmax(logits, dim=1)

    abn_idx = torch.as_tensor(list(abnormal_ids), device=device, dtype=torch.long)
    norm_idx = torch.as_tensor(list(normal_ids),  device=device, dtype=torch.long)
    ing_idx  = torch.as_tensor(list(ingestion_ids), device=device, dtype=torch.long)
    sv_idx   = torch.as_tensor(list(sneeze_vomit_ids), device=device, dtype=torch.long)

    # ── T1 ───────────────────────────────────────────────────────────
    p_abn    = probs.index_select(1, abn_idx).sum(dim=1).clamp(eps, 1 - eps)
    t1_tgt   = y_soft.index_select(1, abn_idx).sum(dim=1).clamp(0.0, 1.0)
    loss_t1  = -(t1_tgt * torch.log(p_abn) +
                 (1.0 - t1_tgt) * torch.log(1.0 - p_abn)).mean()

    # ── T2 ───────────────────────────────────────────────────────────
    normal_mass = y_soft.index_select(1, norm_idx).sum(dim=1)
    if (normal_mass > 0).any():
        p_norm_sub = F.softmax(logits.index_select(1, norm_idx), dim=1)
        ing_local  = [list(normal_ids).index(c) for c in ingestion_ids
                      if c in list(normal_ids)]
        ing_local_t = torch.as_tensor(ing_local, device=device, dtype=torch.long)
        p_ing   = p_norm_sub.index_select(1, ing_local_t).sum(dim=1).clamp(eps, 1 - eps)
        t2_tgt  = y_soft.index_select(1, ing_idx).sum(dim=1)
        bce_t2  = -(t2_tgt * torch.log(p_ing) +
                    (normal_mass - t2_tgt).clamp(min=0) * torch.log(1.0 - p_ing))
        loss_t2 = bce_t2.sum() / normal_mass.sum().clamp(min=eps)
    else:
        loss_t2 = torch.tensor(0.0, device=device)

    # ── T3: sneeze vs vomit (soft mass ratio as target) ──────────────
    sv_mass = y_soft.index_select(1, sv_idx).sum(dim=1)   # [B]
    if (sv_mass > 0).any():
        # renormalized soft prob within sneeze/vomit
        p_sv     = probs.index_select(1, sv_idx)          # [B, 2]
        p_sv_n   = p_sv / (p_sv.sum(dim=1, keepdim=True) + eps)   # [B, 2]
        # soft target: fraction of sv_mass belonging to vomit (idx 1)
        vomit_id = sneeze_vomit_ids[1]
        t3_soft  = (y_soft[:, vomit_id] / sv_mass.clamp(min=eps)).clamp(0.0, 1.0)
        # BCE on P(vomit | sneeze_or_vomit)
        p_vomit  = p_sv_n[:, 1].clamp(eps, 1 - eps)
        bce_t3   = -(t3_soft * torch.log(p_vomit) +
                     (1.0 - t3_soft) * torch.log(1.0 - p_vomit))
        loss_t3  = (bce_t3 * sv_mass).sum() / sv_mass.sum().clamp(min=eps)
    else:
        loss_t3 = torch.tensor(0.0, device=device)

    # ── T4: soft CE (label_smoothing applied to y_soft) ──────────────
    y_s = y_soft
    if label_smoothing > 0:
        y_s = (1.0 - label_smoothing) * y_s + label_smoothing / y_s.size(1)
    log_p = F.log_softmax(logits, dim=1)
    if class_weight is not None:
        cw     = class_weight.to(device).view(1, -1)
        loss_t4 = -(y_s * cw * log_p).sum(dim=1).mean()
    else:
        loss_t4 = -(y_s * log_p).sum(dim=1).mean()

    total  = w_t1 * loss_t1 + w_t2 * loss_t2 + w_t3 * loss_t3 + w_t4 * loss_t4
    stats  = {
        "l_t1": float(loss_t1.detach().cpu()),
        "l_t2": float(loss_t2.detach().cpu()),
        "l_t3": float(loss_t3.detach().cpu()),
        "l_t4": float(loss_t4.detach().cpu()),
    }
    return total, stats
