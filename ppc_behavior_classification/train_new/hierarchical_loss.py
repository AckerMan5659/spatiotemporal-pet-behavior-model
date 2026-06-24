# -*- coding: utf-8 -*-
"""
Hierarchical priority loss for 7-class merged dataset.

Targets (priority T1 > T2 = T3 > T4):
    T1 — 二分类:        Normal {0,1,2}    vs  Abnormal {3,4,5,6}     w_t1 (最高)
    T2 — 吃喝:          Ingestion {1,2}  vs  Other {0}               w_t2 (中)
    T3 — 打喷嚏/呕吐:  Sneeze {5}       vs  Vomit {6}               w_t3 (中, 与T2同级)
    T4 — 细粒度:        7-way cross-entropy                           w_t4 (最低)

T2 仅在 Normal 子集 (GT∈{0,1,2}) 计算。
T3 仅在 sneeze/vomit 子集 (GT∈{5,6}) 计算；两者视觉相似，需专项区分。
T4 在全部样本上用带 class_weight + focal + label_smoothing 的 CE。
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
        sneeze_vomit_ids=(5, 6),
        other_id=0,
        w_t1=1.8,
        w_t2=1.2,
        w_t3=1.2,
        w_t4=1.0,
        label_smoothing=0.1,
        class_weight=None,
        focal_gamma=0.0,
    ):
        super().__init__()
        self.normal_ids = list(normal_ids)
        self.abnormal_ids = list(abnormal_ids)
        self.ingestion_ids = list(ingestion_ids)
        self.sneeze_vomit_ids = list(sneeze_vomit_ids)
        self.other_id = int(other_id)

        self.w_t1 = float(w_t1)
        self.w_t2 = float(w_t2)
        self.w_t3 = float(w_t3)
        self.w_t4 = float(w_t4)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)

        if class_weight is not None and not torch.is_tensor(class_weight):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight",
                             class_weight if class_weight is not None
                             else torch.empty(0))

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

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 7]
            targets: [B] int in [0, 6]
        Returns:
            total_loss, stats dict (l_t1 / l_t2 / l_t3 / l_t4 / p_abn_mean)
        """
        device = logits.device
        probs = F.softmax(logits, dim=1)

        # ── T1: Normal vs Abnormal ────────────────────────────────────
        abn_idx = torch.as_tensor(self.abnormal_ids, device=device, dtype=torch.long)
        p_abn = probs.index_select(1, abn_idx).sum(dim=1)
        bin_target = sum((targets == c).float() for c in self.abnormal_ids).clamp(0.0, 1.0)

        bce_t1 = self._binary_bce(p_abn, bin_target)
        loss_t1 = (bce_t1 * self._focal_weight(p_abn, bin_target)).mean()

        # ── T2: Ingestion vs Other  (Normal GT only) ──────────────────
        is_normal = sum((targets == c) for c in self.normal_ids).bool()
        loss_t2 = torch.tensor(0.0, device=device)
        if is_normal.any():
            norm_idx = torch.as_tensor(self.normal_ids, device=device, dtype=torch.long)
            n_logits  = logits[is_normal]
            n_targets = targets[is_normal]
            p_norm_sub = F.softmax(n_logits.index_select(1, norm_idx), dim=1)

            ing_local   = [self.normal_ids.index(c) for c in self.ingestion_ids
                           if c in self.normal_ids]
            ing_local_t = torch.as_tensor(ing_local, device=device, dtype=torch.long)
            p_ing       = p_norm_sub.index_select(1, ing_local_t).sum(dim=1)

            ing_target = sum((n_targets == c).float() for c in self.ingestion_ids).clamp(0.0, 1.0)
            bce_t2     = self._binary_bce(p_ing, ing_target)
            loss_t2    = (bce_t2 * self._focal_weight(p_ing, ing_target)).mean()

        # ── T3: Sneeze vs Vomit  (sneeze/vomit GT only) ───────────────
        is_sv = sum((targets == c) for c in self.sneeze_vomit_ids).bool()
        loss_t3 = torch.tensor(0.0, device=device)
        if is_sv.any():
            sv_idx    = torch.as_tensor(self.sneeze_vomit_ids, device=device, dtype=torch.long)
            sv_logits = logits[is_sv].index_select(1, sv_idx)   # [n_sv, 2]
            sv_raw    = targets[is_sv]
            sv_targets = torch.zeros_like(sv_raw)
            for local_i, global_c in enumerate(self.sneeze_vomit_ids):
                sv_targets[sv_raw == global_c] = local_i
            loss_t3 = F.cross_entropy(sv_logits, sv_targets,
                                      label_smoothing=self.label_smoothing)

        # ── T4: 7-way CE  (all samples) ───────────────────────────────
        cw = self.class_weight if self.class_weight.numel() > 0 else None
        if cw is not None:
            cw = cw.to(device)
        if self.focal_gamma > 0:
            with torch.no_grad():
                p_t4 = F.softmax(logits, dim=1).gather(
                    1, targets.unsqueeze(1)).squeeze(1)
                fl_w = (1.0 - p_t4).pow(self.focal_gamma)
            nll_t4 = F.cross_entropy(logits, targets, weight=cw,
                                      label_smoothing=self.label_smoothing,
                                      reduction="none")
            loss_t4 = (fl_w * nll_t4).mean()
        else:
            loss_t4 = F.cross_entropy(logits, targets, weight=cw,
                                       label_smoothing=self.label_smoothing)

        total = (self.w_t1 * loss_t1
                 + self.w_t2 * loss_t2
                 + self.w_t3 * loss_t3
                 + self.w_t4 * loss_t4)

        stats = {
            "l_t1": float(loss_t1.detach().cpu()),
            "l_t2": float(loss_t2.detach().cpu()),
            "l_t3": float(loss_t3.detach().cpu()),
            "l_t4": float(loss_t4.detach().cpu()),
            "p_abn_mean": float(p_abn.detach().mean().cpu()),
        }
        return total, stats


# ──────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────
def hierarchical_predictions(logits,
                             normal_ids=(0, 1, 2),
                             abnormal_ids=(3, 4, 5, 6),
                             ingestion_ids=(1, 2),
                             sneeze_vomit_ids=(5, 6)):
    """
    Returns:
        t1_pred [B]: 0=normal,  1=abnormal
        t2_pred [B]: 0=other,   1=ingestion   (有效范围: t1=0)
        t3_pred [B]: 0=sneeze,  1=vomit        (有效范围: t1=1, class in {5,6})
        t4_pred [B]: 0-6 argmax fine-grained
        probs   [B, 7]
    """
    probs = F.softmax(logits, dim=1)

    p_abn  = probs[:, list(abnormal_ids)].sum(dim=1)
    p_norm = probs[:, list(normal_ids)].sum(dim=1)
    t1_pred = (p_abn > p_norm).long()

    p_ing   = probs[:, list(ingestion_ids)].sum(dim=1)
    p_other = probs[:, list(normal_ids)[0]]
    t2_pred = (p_ing > p_other).long()

    sv = list(sneeze_vomit_ids)
    p_sv      = probs[:, sv]
    p_sv_norm = p_sv / (p_sv.sum(dim=1, keepdim=True) + 1e-7)
    t3_pred   = p_sv_norm.argmax(dim=1)   # 0=sneeze, 1=vomit

    t4_pred = probs.argmax(dim=1)

    return t1_pred, t2_pred, t3_pred, t4_pred, probs
