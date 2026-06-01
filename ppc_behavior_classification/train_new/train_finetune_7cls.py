# -*- coding: utf-8 -*-
"""
Phase 1 — Teacher Fine-tuning (Swin3D-B) on the 7-class merged dataset.

- Hierarchical loss (T1 binary > T2 ingestion > T3 fine 7-class CE)
- EMA + Gradient Clipping + StochasticDepth + Early Stopping
- 自动 class_weight (1/freq), 可选 WeightedRandomSampler

Usage:
    python train_finetune_7cls.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from timm.utils import ModelEmaV2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset_7cls import get_dataset
from hierarchical_loss import HierarchicalPriorityLoss, hierarchical_predictions
from swin_wrapper import SwinTeacher
from train_utils import WarmupCosineLR, MixUpCutMix, hierarchical_soft_loss


# ----------------------------------------------------------------------
def accum_steps_safe(t_cfg):
    return max(1, int(t_cfg.get("accum_steps", 1)))


def apply_stochastic_depth(model, drop_prob=0.2):
    count = 0
    for m in model.modules():
        if isinstance(m, torchvision.ops.StochasticDepth):
            m.p = drop_prob
            count += 1
    print(f"🌿 StochasticDepth -> {drop_prob} on {count} layers")


class TrainableSwin(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None, drop_path=0.2):
        super().__init__()
        self.base = SwinTeacher(checkpoint_path=checkpoint_path,
                                num_classes=num_classes, dropout=0.1)
        apply_stochastic_depth(self.base.model, drop_prob=drop_path)
        for p in self.base.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.base(x)[0]


# ----------------------------------------------------------------------
def evaluate(model_eval, loader, criterion, hcfg, device):
    model_eval.eval()
    preds, tgts = [], []
    t1_p, t1_t = [], []
    t2_p, t2_t = [], []  # only over normal subset
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with autocast("cuda"):
                logits = model_eval(x)
                loss, _ = criterion(logits, y)
            val_loss += float(loss)
            tp1, tp2, tp3, _ = hierarchical_predictions(
                logits,
                normal_ids=hcfg["normal_ids"],
                abnormal_ids=hcfg["abnormal_ids"],
                ingestion_ids=hcfg["ingestion_ids"],
            )
            preds.extend(tp3.cpu().numpy())
            tgts.extend(y.cpu().numpy())
            t1_p.extend(tp1.cpu().numpy())
            t1_t.extend([1 if int(v) in hcfg["abnormal_ids"] else 0
                         for v in y.cpu().numpy()])
            # T2 valid only when GT is normal
            for yp, yt, raw_logit in zip(tp2.cpu().numpy(), y.cpu().numpy(), logits):
                if int(yt) in hcfg["normal_ids"]:
                    t2_p.append(int(yp))
                    t2_t.append(1 if int(yt) in hcfg["ingestion_ids"] else 0)

    val_loss /= max(1, len(loader))
    n = max(1, len(preds))
    acc7 = sum(int(p == t) for p, t in zip(preds, tgts)) / n
    acc_t1 = (sum(int(p == t) for p, t in zip(t1_p, t1_t)) / max(1, len(t1_p)))
    acc_t2 = (sum(int(p == t) for p, t in zip(t2_p, t2_t)) / max(1, len(t2_p))
              if t2_p else 0.0)

    return {
        "val_loss": val_loss,
        "acc_t1": acc_t1,
        "acc_t2": acc_t2,
        "acc_t3": acc7,
        "preds": preds,
        "targets": tgts,
        "t1_pred": t1_p,
        "t1_true": t1_t,
    }


# ----------------------------------------------------------------------
def train(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = cfg["dataset"]
    t_cfg = cfg["teacher"]
    hcfg = cfg["hierarchical"]
    num_classes = ds_cfg["num_classes"]

    os.makedirs(t_cfg["output_dir"], exist_ok=True)

    print("\n🦄 [Phase 1] Teacher Fine-tuning Swin3D-B on 7-class merged_dataset")

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
    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"],
                              sampler=sampler,
                              num_workers=t_cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"],
                            shuffle=False,
                            num_workers=t_cfg["num_workers"])

    model = TrainableSwin(num_classes=num_classes,
                          checkpoint_path=t_cfg.get("checkpoint_path"),
                          drop_path=t_cfg["drop_path"]).to(device)

    print(f"🛡️ EMA decay = {t_cfg['ema_decay']}")
    model_ema = ModelEmaV2(model, decay=t_cfg["ema_decay"])

    # class weight from training data
    class_weight = train_ds.get_class_weights(device=device,
                                              scheme="inverse") if hcfg.get(
        "use_class_weight", True) else None

    criterion = HierarchicalPriorityLoss(
        normal_ids=hcfg["normal_ids"],
        abnormal_ids=hcfg["abnormal_ids"],
        ingestion_ids=hcfg["ingestion_ids"],
        other_id=hcfg["other_id"],
        w_t1=hcfg["w_t1_binary"],
        w_t2=hcfg["w_t2_ingestion"],
        w_t3=hcfg["w_t3_fine"],
        label_smoothing=hcfg.get("label_smoothing", 0.1),
        class_weight=class_weight,
        focal_gamma=hcfg.get("focal_gamma", 0.0),
    ).to(device)

    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "head" in n else backbone_params).append(p)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": t_cfg["lr"]},
        {"params": head_params, "lr": t_cfg["lr"]},
    ], weight_decay=t_cfg["weight_decay"])

    # Warmup + Cosine on step granularity
    steps_per_epoch = max(1, len(train_loader) // accum_steps_safe(t_cfg))
    total_steps = steps_per_epoch * t_cfg["epochs"]
    warmup_steps = steps_per_epoch * int(t_cfg.get("warmup_epochs", 0))
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps,
                               total_steps=total_steps,
                               min_lr=float(t_cfg.get("min_lr", 1e-6)))
    scaler = GradScaler()

    # MixUp / CutMix
    mix_cfg = ds_cfg.get("mixup", {})
    mixer = (MixUpCutMix(alpha=mix_cfg.get("alpha", 0.2),
                         cutmix_alpha=mix_cfg.get("cutmix_alpha", 1.0),
                         prob=mix_cfg.get("prob", 0.5),
                         switch_prob=mix_cfg.get("switch_prob", 0.5),
                         num_classes=num_classes)
             if (t_cfg.get("use_mixup", False) and mix_cfg.get("enabled", False))
             else None)
    if mixer:
        print(f"🥣 MixUp/CutMix enabled (prob={mixer.prob}, switch={mixer.switch_prob})")

    best_acc_t1 = 0.0
    best_score = -1.0
    no_improve = 0
    accum = t_cfg["accum_steps"]
    patience = t_cfg["patience"]
    freeze_bb_eps = int(t_cfg.get("freeze_backbone_epochs", 0))

    out_pth = os.path.join(t_cfg["output_dir"], t_cfg["output_name"])

    for epoch in range(t_cfg["epochs"]):
        # 前几个 epoch 冻结 backbone, 防 head 大梯度污染 K400 权重
        if freeze_bb_eps > 0:
            for p in backbone_params:
                p.requires_grad = (epoch >= freeze_bb_eps)
            if epoch < freeze_bb_eps:
                if epoch == 0:
                    print(f"❄️  Backbone frozen for first {freeze_bb_eps} epoch(s)")
            elif epoch == freeze_bb_eps:
                print("🔥 Backbone unfrozen")

        model.train()
        ema_l1, ema_l2, ema_l3 = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{t_cfg['epochs']}")
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with autocast("cuda"):
                if mixer is not None:
                    x_mix, y_soft, applied = mixer(x, y)
                    logits = model(x_mix)
                    loss, stats = hierarchical_soft_loss(
                        logits, y_soft, criterion,
                        normal_ids=hcfg["normal_ids"],
                        abnormal_ids=hcfg["abnormal_ids"],
                        ingestion_ids=hcfg["ingestion_ids"],
                        w_t1=hcfg["w_t1_binary"],
                        w_t2=hcfg["w_t2_ingestion"],
                        w_t3=hcfg["w_t3_fine"],
                        class_weight=class_weight,
                        label_smoothing=hcfg.get("label_smoothing", 0.1),
                    )
                else:
                    logits = model(x)
                    loss, stats = criterion(logits, y)
                loss = loss / accum

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=t_cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                model_ema.update(model)
                scheduler.step()

            ema_l1 = 0.9 * ema_l1 + 0.1 * stats["l_t1"] if i else stats["l_t1"]
            ema_l2 = 0.9 * ema_l2 + 0.1 * stats["l_t2"] if i else stats["l_t2"]
            ema_l3 = 0.9 * ema_l3 + 0.1 * stats["l_t3"] if i else stats["l_t3"]
            pbar.set_postfix(T1=f"{ema_l1:.3f}", T2=f"{ema_l2:.3f}",
                             T3=f"{ema_l3:.3f}",
                             lr=f"{scheduler.get_lr()[0]:.2e}")

        # ---------------- Evaluate (EMA model) -----------------------
        metrics = evaluate(model_ema.module, val_loader, criterion, hcfg, device)
        print(f"\n📊 Val | T1(bin)={metrics['acc_t1']:.2%} "
              f"| T2(ing)={metrics['acc_t2']:.2%} "
              f"| T3(7-cls)={metrics['acc_t3']:.2%} "
              f"| Loss={metrics['val_loss']:.4f}")

        try:
            cm = confusion_matrix(metrics["targets"], metrics["preds"],
                                  labels=list(range(num_classes)))
            print("CM (rows=GT 0..6, cols=Pred 0..6):")
            for r in cm:
                print(" ", r.tolist())
        except Exception:
            pass

        # Priority score: T1 dominates, T2 next, T3 last
        score = (10.0 * metrics["acc_t1"]
                 + 3.0 * metrics["acc_t2"]
                 + 1.0 * metrics["acc_t3"])

        is_best = score > best_score
        if is_best:
            best_score = score
            best_acc_t1 = metrics["acc_t1"]
            torch.save(model_ema.module.state_dict(), out_pth)
            print(f"🏆 saved best EMA -> {out_pth} (score={score:.3f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"⚠️ no improve {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"🛑 early stop at epoch {epoch+1}")
                break
        print("-" * 60)

    print(f"\n✅ Phase 1 done. Best T1 = {best_acc_t1:.2%}, weight @ {out_pth}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.path.join(HERE, "config.yaml"))
    args = parser.parse_args()
    train(args.config)
