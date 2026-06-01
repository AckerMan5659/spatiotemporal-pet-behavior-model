# -*- coding: utf-8 -*-
"""
Phase 3 — Quantization-Aware Training (RepViT-GRU) with self-distillation.

- FP32 ema 权重既作为 Student 的初始化, 也作为同结构 Teacher 进行同构蒸馏。
- 7-class hierarchical loss + KL distillation。
- 导出 batch*time parallel 形式 ONNX (评估用)。
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from torch.ao.quantization import (
    QConfig, FakeQuantize,
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)
from torch.onnx import register_custom_op_symbolic


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")
for p in (HERE, ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_7cls import get_dataset                                  # noqa: E402
from hierarchical_loss import HierarchicalPriorityLoss, hierarchical_predictions  # noqa: E402
from train_utils import WarmupCosineLR                                # noqa: E402
from recognizers.gru_model import RepViT_GRU                          # noqa: E402


# Magic patch — fix exporter 'aten::copy' lowering on some PyTorch versions
def _symbolic_copy(g, self, src, non_blocking):
    return src
register_custom_op_symbolic("aten::copy", _symbolic_copy, 13)


# ----------------------------------------------------------------------
class RepViT_QAT_Parallel_Wrapper(nn.Module):
    """B*T parallel wrapper. Quant on the conv body, FP32 on GRU/FC."""
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.gap = base_model.gap
        self.fc = base_model.fc
        self.gru = base_model.gru
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.quant(x)
        feats = self.backbone(x)[0]
        feats = self.dequant(feats)
        v = self.gap(feats).flatten(1).view(b, t, -1)
        out, _ = self.gru(v, None)
        return self.fc(out[:, -1, :])


class DistillKL(nn.Module):
    def __init__(self, temp=4.0):
        super().__init__()
        self.temp = temp

    def forward(self, s, t):
        return F.kl_div(F.log_softmax(s / self.temp, dim=1),
                        F.softmax(t / self.temp, dim=1),
                        reduction="batchmean") * (self.temp ** 2)


# ----------------------------------------------------------------------
def _teacher_forward(teacher, frames):
    b, t, c, h, w = frames.shape
    t_in = frames.view(b * t, c, h, w)
    t_feats = teacher.backbone(t_in)[0]
    t_feats = teacher.gap(t_feats).flatten(1).view(b, t, -1)
    t_out, _ = teacher.gru(t_feats, None)
    return teacher.fc(t_out[:, -1, :])


def validate(qat_model, loader, hcfg, num_classes, device):
    qat_model.eval()
    preds, tgts, t1_p, t1_t, t2_p, t2_t = [], [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            v = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            logits = qat_model(v)
            tp1, tp2, tp3, _ = hierarchical_predictions(
                logits,
                normal_ids=hcfg["normal_ids"],
                abnormal_ids=hcfg["abnormal_ids"],
                ingestion_ids=hcfg["ingestion_ids"],
            )
            preds.extend(tp3.cpu().numpy())
            tgts.extend(y.cpu().numpy())
            t1_p.extend(tp1.cpu().numpy())
            t1_t.extend([1 if int(vv) in hcfg["abnormal_ids"] else 0
                         for vv in y.cpu().numpy()])
            for yp, yt in zip(tp2.cpu().numpy(), y.cpu().numpy()):
                if int(yt) in hcfg["normal_ids"]:
                    t2_p.append(int(yp))
                    t2_t.append(1 if int(yt) in hcfg["ingestion_ids"] else 0)
    acc7 = sum(int(p == t) for p, t in zip(preds, tgts)) / max(1, len(preds))
    acc_t1 = sum(int(p == t) for p, t in zip(t1_p, t1_t)) / max(1, len(t1_p))
    acc_t2 = (sum(int(p == t) for p, t in zip(t2_p, t2_t)) / max(1, len(t2_p))
              if t2_p else 0.0)
    return acc_t1, acc_t2, acc7, preds, tgts


# ----------------------------------------------------------------------
def train_qat(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = cfg["dataset"]
    q_cfg = cfg["qat"]
    hcfg = cfg["hierarchical"]
    num_classes = ds_cfg["num_classes"]
    hidden = cfg["distillation"]["student_hidden_dim"]

    os.makedirs(q_cfg["output_dir"], exist_ok=True)

    print("\n🚀 [Phase 3] QAT + Self-distillation (7-class)")

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

    train_loader = DataLoader(train_ds, batch_size=q_cfg["batch_size"],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=q_cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    # -------- Load FP32 teacher / float init -----------------------
    pre_path = q_cfg["pretrained_path"]
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"FP32 checkpoint not found: {pre_path}")

    sd = torch.load(pre_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    teacher = RepViT_GRU(num_classes=num_classes, hidden_dim=hidden, pretrained=False)
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if hasattr(torch, "compile"):
        try:
            teacher = torch.compile(teacher, mode="default")
            print("✅ teacher compiled")
        except Exception as e:
            print(f"⚠️ teacher compile failed: {e}")

    float_model = RepViT_GRU(num_classes=num_classes, hidden_dim=hidden, pretrained=False)
    float_model.load_state_dict(sd, strict=False)
    qat_model = RepViT_QAT_Parallel_Wrapper(float_model).to(device)

    # -------- QConfig --------------------------------------------------
    act_q = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=255, dtype=torch.quint8,
        qscheme=torch.per_tensor_affine, reduce_range=True)
    weight_q = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0)
    qat_model.qconfig = QConfig(activation=act_q, weight=weight_q)

    # Skip quantization on shallow conv layers and recurrent / fc heads
    if hasattr(qat_model.backbone, "stem"):
        qat_model.backbone.stem.qconfig = None
    elif hasattr(qat_model.backbone, "stages"):
        qat_model.backbone.stages[0].qconfig = None
        qat_model.backbone.stages[1].qconfig = None
    qat_model.gru.qconfig = None
    qat_model.fc.qconfig = None
    qat_model.gap.qconfig = None

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    # -------- Loss / Optim --------------------------------------------
    class_weight = train_ds.get_class_weights(device=device,
                                              scheme="inverse") if hcfg.get(
        "use_class_weight", True) else None
    hier_loss = HierarchicalPriorityLoss(
        normal_ids=hcfg["normal_ids"],
        abnormal_ids=hcfg["abnormal_ids"],
        ingestion_ids=hcfg["ingestion_ids"],
        other_id=hcfg["other_id"],
        w_t1=hcfg["w_t1_binary"],
        w_t2=hcfg["w_t2_ingestion"],
        w_t3=hcfg["w_t3_fine"],
        label_smoothing=hcfg.get("label_smoothing", 0.1),
        class_weight=class_weight,
    ).to(device)
    kd_loss = DistillKL(temp=q_cfg["temperature"]).to(device)

    optimizer = optim.AdamW(qat_model.parameters(), lr=q_cfg["lr"],
                            weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * q_cfg["epochs"]
    warmup_steps = steps_per_epoch * int(q_cfg.get("warmup_epochs", 0))
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps,
                               total_steps=total_steps,
                               min_lr=float(q_cfg.get("min_lr", 1e-7)))
    scaler = GradScaler()

    best_score = -1.0
    save_pth = os.path.join(q_cfg["output_dir"], q_cfg["output_pth"])
    save_onnx = os.path.join(q_cfg["output_dir"], q_cfg["output_onnx_parallel"])
    freeze_bn_ep = int(q_cfg["epochs"] * q_cfg["freeze_bn_ratio"])

    for epoch in range(q_cfg["epochs"]):
        qat_model.train()
        if epoch >= freeze_bn_ep:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if epoch == freeze_bn_ep:
                print(f"🧊 freezing BN at epoch {epoch+1}")

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{q_cfg['epochs']}")
        ema_t1 = ema_t3 = ema_kd = 0.0
        for i, batch in enumerate(pbar):
            v = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.no_grad():
                t_logits = _teacher_forward(teacher, v)

            with autocast("cuda"):
                s_logits = qat_model(v)
                loss_hier, stats = hier_loss(s_logits, y)
                loss_kd = kd_loss(s_logits, t_logits)
                loss = (1.0 - q_cfg["alpha_kd"]) * loss_hier \
                       + q_cfg["alpha_kd"] * loss_kd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            ema_t1 = (ema_t1 * i + stats["l_t1"]) / (i + 1)
            ema_t3 = (ema_t3 * i + stats["l_t3"]) / (i + 1)
            ema_kd = (ema_kd * i + float(loss_kd.detach().cpu())) / (i + 1)
            pbar.set_postfix(T1=f"{ema_t1:.3f}", T3=f"{ema_t3:.3f}",
                             KD=f"{ema_kd:.3f}",
                             lr=f"{scheduler.get_lr()[0]:.2e}")
        acc_t1, acc_t2, acc7, preds, tgts = validate(qat_model, val_loader,
                                                     hcfg, num_classes, device)
        print(f"\n📊 QAT Val | T1={acc_t1:.2%} | T2={acc_t2:.2%} | T3={acc7:.2%}")
        try:
            cm = confusion_matrix(tgts, preds, labels=list(range(num_classes)))
            for r in cm:
                print(" ", r.tolist())
        except Exception:
            pass

        score = 10.0 * acc_t1 + 3.0 * acc_t2 + 1.0 * acc7
        if score > best_score:
            best_score = score
            torch.save(qat_model.state_dict(), save_pth)
            print(f"🏆 QAT saved -> {save_pth} (score={score:.3f})")

    # -------- Export parallel ONNX for evaluation ---------------------
    print("\n📦 Exporting parallel ONNX (B*T) for evaluation ...")
    if os.path.exists(save_pth):
        qat_model.load_state_dict(torch.load(save_pth, map_location="cpu"))
    qat_model.cpu().eval()
    try:
        dummy = torch.randn(1, ds_cfg["seq_len"], 3, ds_cfg["imgsz"], ds_cfg["imgsz"])
        torch.onnx.export(
            qat_model, (dummy,), save_onnx,
            opset_version=13,
            input_names=["input_seq"], output_names=["output_logits"],
            dynamic_axes={"input_seq": {0: "batch"},
                          "output_logits": {0: "batch"}},
        )
        print(f"✅ exported -> {save_onnx}")
    except Exception as e:
        print(f"❌ export failed: {e}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.path.join(HERE, "config.yaml"))
    args = parser.parse_args()
    train_qat(args.config)
