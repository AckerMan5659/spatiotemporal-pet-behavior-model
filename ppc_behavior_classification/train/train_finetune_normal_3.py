import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler
from torch.amp import autocast 
from sklearn.metrics import confusion_matrix 
import torchvision # 用于查找 StochasticDepth
from timm.utils import ModelEmaV2 # 🔥 引入 EMA

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import get_dataset

try:
    from swin_wrapper import SwinTeacher
except ImportError:
    print("⚠️ 警告: 未找到 SwinTeacher，請確保 swin_wrapper.py 在路徑中")
    SwinTeacher = None

# ==============================================================================
# 🔥 [1. 改進版分層 Loss] 
# ==============================================================================
class HierarchicalPriorityLoss(nn.Module):
    def __init__(self, other_idx=2, binary_weight=1.1, fine_weight=1.0):
        super().__init__()
        self.other_idx = other_idx
        self.w_binary = binary_weight
        self.w_fine = fine_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        binary_targets = (targets != self.other_idx).float()
        probs = F.softmax(logits, dim=1)
        prob_other = probs[:, self.other_idx]
        prob_action = 1.0 - prob_other + 1e-7
        
        loss_binary = -(binary_targets * torch.log(prob_action) + (1 - binary_targets) * torch.log(prob_other)).mean()

        mask_action = (targets != self.other_idx)
        loss_fine = torch.tensor(0.0, device=logits.device)
        
        if mask_action.sum() > 0:
            action_logits = logits[mask_action][:, :2] 
            action_targets = targets[mask_action]      
            loss_fine = self.ce(action_logits, action_targets)

        total_loss = self.w_binary * loss_binary + self.w_fine * loss_fine
        return total_loss, loss_binary.item(), loss_fine.item()

# ==============================================================================
# 🔒 [配置區]
# ==============================================================================
TEACHER_SETTINGS = {
    "dataset_path": "/userhome/cs/u3650650/cropped_dataset/normal_3", 
    "checkpoint_path": None,
    "output_dir": "outputs/teacher_finetuned_3cls_0312", 
    "num_classes": 3,
    "batch_size": 4,      
    "accum_steps": 8,     
    "epochs": 35,         # 加入早停后，epochs 可以稍微设大一点
    "lr": 4e-5,           
    "num_workers": 4,
}

# 🔥 [防过拟合 1] 动态修改 DropPath (Stochastic Depth) 的辅助函数
def apply_stochastic_depth(model, drop_prob=0.2):
    count = 0
    for module in model.modules():
        if isinstance(module, torchvision.ops.StochasticDepth):
            module.p = drop_prob
            count += 1
    print(f"🌿 [防过拟合] 已将 {count} 个 StochasticDepth 层的丢弃率修改为 {drop_prob}")

class TrainableSwin(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None):
        super().__init__()
        print(f"🏗️ 初始化 Swin-B (全解冻 + 高级防过拟合)...")
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"⚠️ 未找到指定權重 {checkpoint_path}，將使用 K400 預訓練或隨機初始化")
            checkpoint_path = None 
            
        self.base = SwinTeacher(checkpoint_path=checkpoint_path)
        
        # 激活 DropPath
        apply_stochastic_depth(self.base.model, drop_prob=0.2)
        
        # -----------------------------------------------------------
        # 全解冻
        # -----------------------------------------------------------
        unfrozen_count = 0
        for name, param in self.base.model.named_parameters():
            param.requires_grad = True 
            unfrozen_count += 1

        # 重置分类头
        old_head = self.base.model.head
        in_features = old_head.in_features if not isinstance(old_head, nn.Sequential) else old_head[-1].in_features
        
        self.base.model.head = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.base.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base(x)[0]

def train():
    print(f"\n🦄 [TEACHER FINAL] Swin-B | EMA + GradClip + DropPath + EarlyStop")
    
    cfg = TEACHER_SETTINGS
    
    rec_cfg = {
        'imgsz': 224, 
        'temporal_window': 16,
        'class_names': ['eat', 'drink', 'other'] 
    }
    
    print("--- Loading Train Set ---")
    train_ds = get_dataset(cfg["dataset_path"], 'train', rec_cfg, is_train=True)
    print("--- Loading Val Set ---")
    val_ds = get_dataset(cfg["dataset_path"], 'val', rec_cfg, is_train=False) 
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    model = TrainableSwin(num_classes=cfg["num_classes"], checkpoint_path=cfg["checkpoint_path"]).cuda()
    
    # 🔥 [防过拟合 2] 引入 Model EMA
    # decay 0.999 适合大 batch 大 dataset，我们数据少，用 0.99 或 0.995 更好
    print("🛡️ [防过拟合] 启用 Model EMA (decay=0.995)")
    model_ema = ModelEmaV2(model, decay=0.995)

    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    backbone_lr = cfg['lr'] * 1
    head_lr = cfg['lr']
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr}, 
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=0.1)
    
    print(f"🔧 優化器組: Backbone LR={backbone_lr:.2e}, Head LR={head_lr:.2e}")

    criterion = HierarchicalPriorityLoss(other_idx=2, binary_weight=1.1, fine_weight=1.0)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    best_acc = 0.0
    best_val_loss = float('inf')
    patience = 7  # 容忍 7 个 Epoch 没有提升
    no_improve_epochs = 0
    
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"🚀 開始微調 (Epochs: {cfg['epochs']})...")

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0; avg_bin = 0; avg_fine = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
        
        for i, batch in enumerate(pbar):
            x = batch['pixel_values'].cuda(non_blocking=True)
            y = batch['labels'].cuda(non_blocking=True)
            
            with autocast('cuda'):
                logits = model(x)
                loss, l_bin, l_fine = criterion(logits, y)
                loss = loss / cfg['accum_steps']
            
            scaler.scale(loss).backward()
            
            if (i+1) % cfg['accum_steps'] == 0 or (i+1) == len(train_loader):
                # 🔥 [防过拟合 3] Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 更新 EMA 模型 (仅在优化器 step 后更新)
                model_ema.update(model)
            
            total_loss += loss.item() * cfg['accum_steps']
            avg_bin = 0.9 * avg_bin + 0.1 * l_bin if i > 0 else l_bin
            avg_fine = 0.9 * avg_fine + 0.1 * l_fine if i > 0 else l_fine
            
            pbar.set_postfix(L=f"{loss.item()*cfg['accum_steps']:.3f}", Bin=f"{avg_bin:.3f}", Fine=f"{avg_fine:.3f}")

        scheduler.step()
        
        # ==========================================
        # Validation (🔥 使用 EMA 模型进行评估)
        # ==========================================
        model_ema.module.eval() # 注意：评估 EMA 模型
        all_preds = []; all_targets = []
        val_loss_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['pixel_values'].cuda()
                y = batch['labels'].cuda()
                with autocast('cuda'):
                    # 用 EMA 预测
                    logits = model_ema.module(x)
                    # 计算 Val Loss 用于 Early Stopping
                    v_loss, _, _ = criterion(logits, y)
                    val_loss_total += v_loss.item()
                    
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss_avg = val_loss_total / len(val_loader)
        correct = sum([p == t for p, t in zip(all_preds, all_targets)])
        val_acc = correct / len(all_preds) if len(all_preds) > 0 else 0
        
        print(f"\n📊 EMA Val Acc: {val_acc:.2%} | Val Loss: {val_loss_avg:.4f}")
        
        try:
            cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
            print(f"       [Pred Eat] [Pred Drink] [Pred Other]")
            print(f"True Eat   : {cm[0]}")
            print(f"True Drink : {cm[1]}")
            print(f"True Other : {cm[2]}")
            
            drink_as_eat = cm[1,0]
            eat_as_other = cm[0,2]
            print(f"Diagnose -> Drink->Eat Err: {drink_as_eat} | Eat->Other Err: {eat_as_other}")
        except: pass
        
        # 🔥 [防过拟合 4] Early Stopping & 存优逻辑
        is_best = False
        if val_acc > best_acc:
            best_acc = val_acc
            is_best = True
        elif val_acc == best_acc and val_loss_avg < best_val_loss:
            # 准确率相同，但 Loss 更低也算 Best
            best_val_loss = val_loss_avg
            is_best = True
            
        if is_best:
            best_val_loss = min(best_val_loss, val_loss_avg)
            # 保存 EMA 模型，它的泛化性最好
            torch.save(model_ema.module.state_dict(), os.path.join(cfg["output_dir"], "teacher_best_0223_ema.pth"))
            print("🏆 New Best EMA Teacher Saved!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"⚠️ 连续 {no_improve_epochs} 轮未提升最佳 Acc。")
            
        if no_improve_epochs >= patience:
            print(f"\n🛑 触发早停机制 (Early Stopping)! 已连续 {patience} 轮无明显提升。")
            break
            
        print("-" * 50)

if __name__ == "__main__":
    train()