# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import sys
import torch.nn.functional as F
# ⚡ 修复警告: 使用新版 AMP API
from torch.cuda.amp import GradScaler
from collections import defaultdict

# 引入项目组件
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from dataset import get_dataset
from recognizers.gru_model import RepViT_GRU

from torch.ao.quantization import (
    QConfig, FakeQuantize,
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
)

# [Magic Patch] 修复导出报错
from torch.onnx import register_custom_op_symbolic
def symbolic_copy(g, self, src, non_blocking):
    return src
register_custom_op_symbolic('aten::copy', symbolic_copy, 13)

# ================= 🚀 配置 =================
CFG = {
    'imgsz': 224,
    'seq_len': 16,
    'data_dir': 'D:/Desktop/PET1/cropped_dataset/abnormal_3_3_16',
    'split': 'train',
    'batch_size': 8,
    'lr': 4e-5,
    'epochs': 20,
    'temp': 4.0,
    'alpha': 0.85,
    'num_workers': 8,
    'balance_ratio': 1.5  # 🆕 新增：Other类不超过 eat/drink 最大值的 1.5 倍
}

PRETRAINED_PATH = "../outputs/best_model_abnormal/0313/repvit_best_distilled_ema_0316_ab.pth"
SAVE_PATH = "../outputs/best_model_abnormal/0313/repvit_qat_distill_0316_3cls_ab.pth"
ONNX_SAVE_PATH = "../outputs/best_model_abnormal/0313/repvit_qat_distill_0316_3cls_ab.onnx"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ================= ⚡ 数据集平衡函数 =================
def balance_dataset(dataset, max_ratio=1.5, other_class_idx=0):
    """
    通过从后往前截断的方式，限制 other 类的样本数量，使其不超过 max(eat, drink) * max_ratio
    假定类别索引: 0=eat, 1=drink, 2=other (根据你的数据集动态调整)
    """
    print(f"\n⚖️  正在执行类别平衡 (Max Ratio: {max_ratio})...")
    indices_by_class = defaultdict(list)

    # 尝试快速获取标签(如果 dataset 暴露了属性)
    labels = None
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'samples'):
        labels = [s[1] for s in dataset.samples]

    # 如果没有暴露属性，则遍历获取(可能会花几秒钟到一两分钟)
    if labels is None:
        print("   ⏳ 未检测到快速标签属性，扫描数据集标签中...")
        labels = []
        for i in tqdm(range(len(dataset)), desc="   扫描标签", leave=False):
            # 获取 label
            labels.append(int(dataset[i]['labels'].item()))

    # 按类别归类索引
    for i, label in enumerate(labels):
        indices_by_class[label].append(i)

    # 获取少数类(0和1)的数量
    minority_counts = [len(indices_by_class.get(0, [])), len(indices_by_class.get(1, []))]
    if not minority_counts or max(minority_counts) == 0:
        print("   ⚠️ 警告: 未找到有效的小数类样本，跳过平衡。")
        return dataset

    max_minority = max(minority_counts)
    limit = int(max_minority * max_ratio)

    final_indices = []
    for cls_idx, indices in indices_by_class.items():
        if cls_idx == other_class_idx:
            # 💡 核心逻辑：从后往前截断 (直接取前 limit 个即可，尾部的数据被抛弃)
            kept_indices = indices[:limit]
            print(f"   📉 类别 {cls_idx} (Other): {len(indices)} -> {len(kept_indices)} (截断尾部)")
            final_indices.extend(kept_indices)
        else:
            print(f"   ✅ 类别 {cls_idx} (Eat/Drink): 保持 {len(indices)} 个样本")
            final_indices.extend(indices)

    # 使用 Subset 包装
    balanced_subset = Subset(dataset, final_indices)
    print(f"   🎯 平衡后数据集总大小: {len(balanced_subset)}\n")
    return balanced_subset


# ================= ⚡ 并行化 QAT 模型包装 =================
class RepViT_QAT_Parallel_Wrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone
        self.gap = original_model.gap
        self.fc = original_model.fc
        self.gru = original_model.gru
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.quant(x)
        features = self.backbone(x)[0]
        features = self.dequant(features)
        x = self.gap(features)
        x = x.flatten(1)
        x = x.view(b, t, -1)
        out, _ = self.gru(x, None)
        return self.fc(out[:, -1, :])

# ================= 蒸馏 Loss =================
class DistillLoss(nn.Module):
    def __init__(self, temp=4.0):
        super().__init__()
        self.temp = temp
    def forward(self, s, t):
        return F.kl_div(
            F.log_softmax(s / self.temp, dim=1),
            F.softmax(t / self.temp, dim=1),
            reduction='batchmean'
        ) * (self.temp ** 2)

# ================= ⚡ 验证函数 =================
def validate_fast(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validating", leave=False):
            frames = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            logits = model(frames)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    return 100 * correct / total

def main_qat_compile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动 QAT (Parallel + AMP + Torch.Compile)...")

    # 1. Teacher (FP32)
    print("👨‍🏫 加载 Teacher...")
    teacher_model = RepViT_GRU(num_classes=3, hidden_dim=256, pretrained=False)
    ckpt = torch.load(PRETRAINED_PATH, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    teacher_model.load_state_dict(sd, strict=False)
    teacher_model.to(device).eval()
    for p in teacher_model.parameters(): p.requires_grad = False

    if hasattr(torch, 'compile'):
        try:
            print("🚀 正在编译 Teacher 模型 (torch.compile)...")
            teacher_model = torch.compile(teacher_model, mode='default')
            print("✅ Teacher 编译成功")
        except Exception as e:
            print(f"⚠️ 编译失败，回退到普通模式: {e}")

    # 2. Student (QAT)
    print("🧑‍🎓 准备 Student (QAT)...")
    float_model = RepViT_GRU(num_classes=3, hidden_dim=256, pretrained=False)
    float_model.load_state_dict(sd, strict=False)
    qat_model = RepViT_QAT_Parallel_Wrapper(float_model)
    qat_model.to(device)

    # QConfig
    act_qconfig = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=255, dtype=torch.quint8,
        qscheme=torch.per_tensor_affine, reduce_range=True
    )
    weight_qconfig = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0
    )
    qat_model.qconfig = QConfig(activation=act_qconfig, weight=weight_qconfig)

    if hasattr(qat_model.backbone, 'stem'): qat_model.backbone.stem.qconfig = None
    elif hasattr(qat_model.backbone, 'stages'):
        qat_model.backbone.stages[0].qconfig = None
        qat_model.backbone.stages[1].qconfig = None
    qat_model.gru.qconfig = None
    qat_model.fc.qconfig = None
    qat_model.gap.qconfig = None

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    # 3. 数据集 (带类别平衡)
    print(f"📂 读取数据集 (Workers={CFG['num_workers']})...")
    train_dataset = get_dataset(CFG['data_dir'], CFG['split'], CFG, is_train=True)

    # 🌟 对训练集执行平衡截断 🌟
    train_dataset = balance_dataset(train_dataset, max_ratio=CFG['balance_ratio'])

    train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=CFG['num_workers'], pin_memory=True)

    val_dataset = None
    try:
        val_dataset = get_dataset(CFG['data_dir'], 'test', CFG, is_train=False)
        if len(val_dataset) == 0: raise Exception("Empty Test Set")
        print(f"✅ 使用 Test 集验证")
        # 🌟 对验证集也执行平衡截断 🌟
        val_dataset = balance_dataset(val_dataset, max_ratio=CFG['balance_ratio'])
    except:
        try:
            val_dataset = get_dataset(CFG['data_dir'], 'val', CFG, is_train=False)
            if len(val_dataset) == 0: raise Exception("Empty Val Set")
            print(f"✅ 使用 Val 集验证")
            # 🌟 对验证集也执行平衡截断 🌟
            val_dataset = balance_dataset(val_dataset, max_ratio=CFG['balance_ratio'])
        except:
            print("⚠️ 未找到有效验证集，划分训练集 20%...")
            subset_len = int(len(train_dataset) * 0.2)
            if subset_len == 0: subset_len = 1
            val_dataset, _ = torch.utils.data.random_split(train_dataset, [subset_len, len(train_dataset)-subset_len])
            # 注意：如果从已平衡的 train_dataset 切分，无需再平衡一次
            print(f"✅ 使用训练集子集验证 (数量: {len(val_dataset)})")

    val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False,
                            num_workers=CFG['num_workers'], pin_memory=True)

    optimizer = optim.AdamW(qat_model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = DistillLoss(temp=CFG['temp'])
    scaler = GradScaler()

    best_acc = 0.0
    freeze_bn_epoch = int(CFG['epochs'] * 0.7)

    print(f"🏎️  开始极速训练...")

    for epoch in range(CFG['epochs']):
        qat_model.train()
        if epoch >= freeze_bn_epoch:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if epoch == freeze_bn_epoch: print(f"🧊 [Epoch {epoch+1}] BN 冻结")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
        total_loss = 0

        for i, batch in enumerate(pbar):
            frames = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.no_grad():
                b, t, c, h, w = frames.shape
                t_in = frames.view(b * t, c, h, w)
                t_feats = teacher_model.backbone(t_in)[0]
                t_feats = teacher_model.gap(t_feats).flatten(1).view(b, t, -1)
                t_out, _ = teacher_model.gru(t_feats, None)
                t_logits = teacher_model.fc(t_out[:, -1, :])

            with torch.amp.autocast('cuda'):
                s_logits = qat_model(frames)
                loss = (1 - CFG['alpha']) * criterion_ce(s_logits, labels) + \
                       CFG['alpha'] * criterion_kd(s_logits, t_logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'best': f"{best_acc:.2f}%"})

        scheduler.step()
        val_acc = validate_fast(qat_model, val_loader, device)
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} | Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(qat_model.state_dict(), SAVE_PATH)
            print(f"   🏆 Saved Best: {SAVE_PATH}")

    print(f"\n🎉 训练结束! Best: {best_acc:.2f}%")
    if os.path.exists(SAVE_PATH):
        qat_model.load_state_dict(torch.load(SAVE_PATH))
    qat_model.cpu().eval()

    try:
        dummy_seq = torch.randn(1, 16, 3, 224, 224)
        torch.onnx.export(
            qat_model, (dummy_seq,), ONNX_SAVE_PATH,
            opset_version=13,
            input_names=['input_seq'], output_names=['output_logits'],
            dynamo=False,
            dynamic_axes={'input_seq': {0: 'batch'}, 'output_logits': {0: 'batch'}}
        )
        print(f"✅ 导出成功: {ONNX_SAVE_PATH}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

if __name__ == "__main__":
    main_qat_compile()