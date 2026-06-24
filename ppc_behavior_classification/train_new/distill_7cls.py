# -*- coding: utf-8 -*-
"""
Phase 2 — Knowledge Distillation: Swin3D-B (Teacher) -> RepViT-GRU (Student)

7-class hierarchical objective (priority T1 > T2 > T3).

Loss = γ * L_GT_hier + β * KL(T||S) + α(t) * FeatureCosine
       α(t) = α0 * (1 - 0.5 * epoch_ratio)   逐步降低 feat 项
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from timm.utils import ModelEmaV2

torch.backends.cudnn.benchmark = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")
for p in (HERE, ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_7cls import get_dataset                                  # noqa: E402
from hierarchical_loss import HierarchicalPriorityLoss, hierarchical_predictions  # noqa: E402
from swin_wrapper import SwinTeacher                                  # noqa: E402
from train_utils import WarmupCosineLR, MixUpCutMix, hierarchical_soft_loss  # noqa: E402
from recognizers.gru_model import RepViT_GRU                          # noqa: E402


# ----------------------------------------------------------------------
class CompositeDistillLoss(nn.Module):
    def __init__(self, hcfg, num_classes, alpha=1.0, beta=8.0, gamma=1.0,
                 temperature=4.0, class_weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temp = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.hier_gt = HierarchicalPriorityLoss(
            normal_ids=hcfg["normal_ids"],
            abnormal_ids=hcfg["abnormal_ids"],
            ingestion_ids=hcfg["ingestion_ids"],
            sneeze_vomit_ids=hcfg["sneeze_vomit_ids"],
            other_id=hcfg["other_id"],
            w_t1=hcfg["w_t1_binary"],
            w_t2=hcfg["w_t2_ingestion"],
            w_t3=hcfg["w_t3_sneeze_vomit"],
            w_t4=hcfg["w_t4_fine"],
            label_smoothing=hcfg.get("label_smoothing", 0.1),
            class_weight=class_weight,
            focal_gamma=hcfg.get("focal_gamma", 0.0),
        )

    def forward(self, s_logits, t_logits, s_feats, t_feats, labels, B, T, epoch_ratio):
        loss_gt, stats = self.hier_gt(s_logits, labels)

        # KL
        p_s = F.log_softmax(s_logits / self.temp, dim=1)
        p_t = F.softmax(t_logits / self.temp, dim=1)
        loss_kd = self.kl_div(p_s, p_t) * (self.temp ** 2)

        # Feature similarity
        loss_feat = torch.tensor(0.0, device=s_logits.device)
        for i, s_f in enumerate(s_feats):
            t_f = t_feats[i].detach()
            if s_f.dim() == 4 and t_f.dim() == 5:
                C, H, W = s_f.shape[1], s_f.shape[2], s_f.shape[3]
                s_f = s_f.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            if s_f.shape[-3:] != t_f.shape[-3:]:
                s_f = F.interpolate(s_f, size=t_f.shape[-3:],
                                    mode="trilinear", align_corners=False)
            sim_map = F.cosine_similarity(s_f, t_f, dim=1)
            loss_feat = loss_feat + (1.0 - sim_map.mean())

        current_alpha = self.alpha * (1.0 - 0.5 * epoch_ratio)
        total = self.gamma * loss_gt + self.beta * loss_kd + current_alpha * loss_feat
        stats.update({
            "l_kd": float(loss_kd.detach().cpu()),
            "l_feat": float(loss_feat.detach().cpu()),
        })
        return total, stats


class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self._fn)
        self.features = None

    def _fn(self, module, inp, out):
        self.features = out[0] if isinstance(out, (list, tuple)) else out

    def remove(self):
        self.hook.remove()


class FeatureProjector(nn.Module):
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(s, t, 1, bias=False),
                          nn.BatchNorm2d(t), nn.ReLU())
            for s, t in zip(s_ch, t_ch)
        ])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]


# ----------------------------------------------------------------------
def _student_logits_and_feats(student, hooks, x):
    """Compute student logits via forward_seq and collect hooked feats."""
    logits = student.forward_seq(x)
    feats = [h.features for h in hooks]
    return logits, feats


def evaluate_student(student_eval, loader, hcfg, num_classes, device):
    student_eval.eval()
    preds, tgts, t1_p, t1_t, t2_p, t2_t, t3_p, t3_t = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            v = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            logits = student_eval.forward_seq(v)
            tp1, tp2, tp3, tp4, _ = hierarchical_predictions(
                logits,
                normal_ids=hcfg["normal_ids"],
                abnormal_ids=hcfg["abnormal_ids"],
                ingestion_ids=hcfg["ingestion_ids"],
                sneeze_vomit_ids=hcfg["sneeze_vomit_ids"],
            )
            preds.extend(tp4.cpu().numpy())
            tgts.extend(y.cpu().numpy())
            t1_p.extend(tp1.cpu().numpy())
            t1_t.extend([1 if int(v) in hcfg["abnormal_ids"] else 0
                         for v in y.cpu().numpy()])
            for yp, yt in zip(tp2.cpu().numpy(), y.cpu().numpy()):
                if int(yt) in hcfg["normal_ids"]:
                    t2_p.append(int(yp))
                    t2_t.append(1 if int(yt) in hcfg["ingestion_ids"] else 0)
            sv_ids = hcfg["sneeze_vomit_ids"]
            for yp3, yt3 in zip(tp3.cpu().numpy(), y.cpu().numpy()):
                if int(yt3) in sv_ids:
                    t3_p.append(int(yp3))
                    t3_t.append(sv_ids.index(int(yt3)))
    acc_t4 = sum(int(p == t) for p, t in zip(preds, tgts)) / max(1, len(preds))
    acc_t1 = sum(int(p == t) for p, t in zip(t1_p, t1_t)) / max(1, len(t1_p))
    acc_t2 = (sum(int(p == t) for p, t in zip(t2_p, t2_t)) / max(1, len(t2_p))
              if t2_p else 0.0)
    acc_t3 = (sum(int(p == t) for p, t in zip(t3_p, t3_t)) / max(1, len(t3_p))
              if t3_p else 0.0)
    return {"acc_t1": acc_t1, "acc_t2": acc_t2, "acc_t3": acc_t3, "acc_t4": acc_t4,
            "preds": preds, "targets": tgts}


# ----------------------------------------------------------------------
def train_distill(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = cfg["dataset"]
    d_cfg = cfg["distillation"]
    hcfg = cfg["hierarchical"]
    num_classes = ds_cfg["num_classes"]

    os.makedirs(d_cfg["output_dir"], exist_ok=True)

    print("\n🚀 [Phase 2] Distillation: Swin3D-B -> RepViT-GRU (7-class)")

    rec_cfg = {
        "imgsz": ds_cfg["imgsz"],
        "seq_len": ds_cfg["seq_len"],
        "num_classes": num_classes,
        "class_names": ds_cfg["class_names"],
        "augment": ds_cfg.get("augment", {}),
    }

    train_ds = get_dataset(ds_cfg["root"], "train", rec_cfg, is_train=True)
    val_ds = get_dataset(ds_cfg["root"], "val", rec_cfg, is_train=False)

    sampler = train_ds.make_sampler(
        ingestion_ids=hcfg["ingestion_ids"],
        abnormal_ids=hcfg["abnormal_ids"],
        other_id=hcfg["other_id"],
    )
    train_loader = DataLoader(train_ds, batch_size=d_cfg["batch_size"],
                              sampler=sampler, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=d_cfg["batch_size"],
                            shuffle=False, num_workers=4)

    # ---- Teacher (frozen) ------------------------------------------
    teacher_ckpt = d_cfg["teacher_checkpoint"]
    if not os.path.exists(teacher_ckpt):
        print(f"⚠️ teacher checkpoint not found: {teacher_ckpt}; using fresh K400 init")
        teacher_ckpt = None
    teacher = SwinTeacher(checkpoint_path=teacher_ckpt,
                          num_classes=num_classes).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Student ----------------------------------------------------
    student = RepViT_GRU(num_classes=num_classes,
                         hidden_dim=d_cfg["student_hidden_dim"],
                         pretrained=d_cfg.get("student_pretrained", True)).to(device)
    student_ema = ModelEmaV2(student, decay=d_cfg["ema_decay"])

    # ---- Hooks --------------------------------------------------------
    if hasattr(student.backbone, "stages"):
        s_stages = student.backbone.stages
        hooks = [FeatureHook(s_stages[i]) for i in [-3, -2, -1]]
    else:
        hooks = [FeatureHook(student.backbone)]

    dummy = torch.randn(2, ds_cfg["seq_len"], 3, ds_cfg["imgsz"], ds_cfg["imgsz"]).to(device)
    with torch.no_grad():
        _, t_all = teacher(dummy)
        student.forward_seq(dummy)
        t_feats_target = t_all[-len(hooks):]

    t_channels = [f.shape[1] for f in t_feats_target]
    s_channels = []
    for h in hooks:
        feat = h.features
        s_channels.append(feat[0].shape[1] if isinstance(feat, (list, tuple))
                          else feat.shape[1])
    projector = FeatureProjector(s_channels, t_channels).to(device)

    # ---- Loss / Optim -----------------------------------------------
    class_weight = train_ds.get_class_weights(device=device,
                                              scheme="inverse") if hcfg.get(
        "use_class_weight", True) else None

    criterion = CompositeDistillLoss(
        hcfg=hcfg, num_classes=num_classes,
        alpha=d_cfg["alpha_feat"], beta=d_cfg["beta_kd"],
        gamma=d_cfg["gamma_gt"], temperature=d_cfg["temperature"],
        class_weight=class_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=d_cfg["lr"], weight_decay=d_cfg["weight_decay"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * d_cfg["epochs"]
    warmup_steps = steps_per_epoch * int(d_cfg.get("warmup_epochs", 0))
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps,
                               total_steps=total_steps,
                               min_lr=float(d_cfg.get("min_lr", 1e-6)))
    scaler = GradScaler()
    grad_clip = float(d_cfg.get("grad_clip", 0.0))

    # MixUp / CutMix (蒸馏阶段)
    mix_cfg = ds_cfg.get("mixup", {})
    mixer = (MixUpCutMix(alpha=mix_cfg.get("alpha", 0.2),
                         cutmix_alpha=mix_cfg.get("cutmix_alpha", 1.0),
                         prob=mix_cfg.get("prob", 0.5),
                         switch_prob=mix_cfg.get("switch_prob", 0.5),
                         num_classes=num_classes)
             if (d_cfg.get("use_mixup", False) and mix_cfg.get("enabled", False))
             else None)
    if mixer:
        print(f"🥣 [Distill] MixUp/CutMix enabled")

    best_score = -1.0
    no_improve = 0
    patience = int(d_cfg.get("patience", 10**9))
    out_pth = os.path.join(d_cfg["output_dir"], d_cfg["output_name"])

    for epoch in range(d_cfg["epochs"]):
        student.train(); projector.train()
        epoch_ratio = epoch / max(1, d_cfg["epochs"])

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{d_cfg['epochs']}")
        ema_kd = ema_t1 = ema_t3 = ema_ft = 0.0

        for i, batch in enumerate(pbar):
            v = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            B, T = v.shape[:2]

            # MixUp/CutMix 仅作用于 GT 路径; Teacher 用未混合 v 算 logits (KD 更稳)
            v_mix, y_soft, applied = (mixer(v, y) if mixer is not None
                                       else (v, None, False))

            with autocast("cuda"):
                with torch.no_grad():
                    t_logits, t_all = teacher(v)
                    t_feats_target = t_all[-len(hooks):]

                s_logits, s_feats_raw = _student_logits_and_feats(student, hooks, v_mix)
                s_feats_proj = projector([
                    f[0] if isinstance(f, (list, tuple)) else f
                    for f in s_feats_raw
                ])

                if applied:
                    # MixUp 启用: 用 soft GT 算 hier_gt; KL/feature 仍按 criterion 公式
                    loss_gt_soft, stats_soft = hierarchical_soft_loss(
                        s_logits, y_soft, criterion.hier_gt,
                        normal_ids=hcfg["normal_ids"],
                        abnormal_ids=hcfg["abnormal_ids"],
                        ingestion_ids=hcfg["ingestion_ids"],
                        sneeze_vomit_ids=hcfg["sneeze_vomit_ids"],
                        w_t1=hcfg["w_t1_binary"],
                        w_t2=hcfg["w_t2_ingestion"],
                        w_t3=hcfg["w_t3_sneeze_vomit"],
                        w_t4=hcfg["w_t4_fine"],
                        class_weight=class_weight,
                        label_smoothing=hcfg.get("label_smoothing", 0.1),
                    )
                    p_s = F.log_softmax(s_logits / criterion.temp, dim=1)
                    p_t = F.softmax(t_logits / criterion.temp, dim=1)
                    loss_kd = criterion.kl_div(p_s, p_t) * (criterion.temp ** 2)
                    # Feature loss (与 criterion 内部一致)
                    loss_feat = torch.tensor(0.0, device=s_logits.device)
                    for j, s_f in enumerate(s_feats_proj):
                        t_f = t_feats_target[j].detach()
                        if s_f.dim() == 4 and t_f.dim() == 5:
                            C, H, W = s_f.shape[1], s_f.shape[2], s_f.shape[3]
                            s_f = s_f.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
                        if s_f.shape[-3:] != t_f.shape[-3:]:
                            s_f = F.interpolate(s_f, size=t_f.shape[-3:],
                                                mode="trilinear", align_corners=False)
                        sim = F.cosine_similarity(s_f, t_f, dim=1)
                        loss_feat = loss_feat + (1.0 - sim.mean())
                    cur_alpha = criterion.alpha * (1.0 - 0.5 * epoch_ratio)
                    loss = (criterion.gamma * loss_gt_soft
                            + criterion.beta * loss_kd
                            + cur_alpha * loss_feat)
                    stats = {
                        "l_t1": stats_soft["l_t1"],
                        "l_t3": stats_soft["l_t3"],
                        "l_t4": stats_soft["l_t4"],
                        "l_kd": float(loss_kd.detach().cpu()),
                        "l_feat": float(loss_feat.detach().cpu()),
                    }
                else:
                    loss, stats = criterion(s_logits, t_logits,
                                            s_feats_proj, t_feats_target,
                                            y, B, T, epoch_ratio)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projector.parameters()),
                    max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            student_ema.update(student)
            scheduler.step()

            ema_kd = (ema_kd * i + stats["l_kd"]) / (i + 1)
            ema_t1 = (ema_t1 * i + stats["l_t1"]) / (i + 1)
            ema_t3 = (ema_t3 * i + stats["l_t3"]) / (i + 1)
            ema_ft = (ema_ft * i + stats["l_feat"]) / (i + 1)
            pbar.set_postfix(T1=f"{ema_t1:.3f}", T3=f"{ema_t3:.3f}",
                             KD=f"{ema_kd:.2f}", Ft=f"{ema_ft:.2f}",
                             lr=f"{scheduler.get_lr()[0]:.2e}")

        metrics = evaluate_student(student_ema.module, val_loader, hcfg,
                                   num_classes, device)
        print(f"\n📊 Val(EMA) | T1={metrics['acc_t1']:.2%} "
              f"| T2={metrics['acc_t2']:.2%} | T3={metrics['acc_t3']:.2%} "
              f"| T4={metrics['acc_t4']:.2%}")
        try:
            cm = confusion_matrix(metrics["targets"], metrics["preds"],
                                  labels=list(range(num_classes)))
            print("CM:")
            for r in cm:
                print(" ", r.tolist())
            print(classification_report(
                metrics["targets"], metrics["preds"],
                labels=list(range(num_classes)),
                target_names=ds_cfg["class_names"],
                zero_division=0,
            ))
        except Exception:
            pass

        score = (10.0 * metrics["acc_t1"]
                 + 3.0 * metrics["acc_t2"]
                 + 3.0 * metrics["acc_t3"]
                 + 1.0 * metrics["acc_t4"])
        if score > best_score:
            best_score = score
            torch.save(student_ema.module.state_dict(), out_pth)
            torch.save(student.state_dict(),
                       out_pth.replace("_ema.pth", ".pth")
                       if out_pth.endswith("_ema.pth") else out_pth + ".raw.pth")
            print(f"🏆 saved best student EMA -> {out_pth} (score={score:.3f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"⚠️ no improve {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"🛑 [Distill] early stop at epoch {epoch+1}")
                break
        print("-" * 60)

    # Cleanup hooks
    for h in hooks:
        h.remove()
    print(f"\n✅ Phase 2 done. Best weight @ {out_pth}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.path.join(HERE, "config.yaml"))
    args = parser.parse_args()
    train_distill(args.config)
