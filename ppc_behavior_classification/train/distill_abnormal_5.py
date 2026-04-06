import os
import sys
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV2
from sklearn.metrics import confusion_matrix 

# 🔴 [加速優化]
torch.backends.cudnn.benchmark = True 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recognizers.gru_model import RepViT_GRU 
from dataset import get_dataset

try:
    from swin_wrapper import SwinTeacher
except ImportError:
    try:
        from distill.swin_wrapper import SwinTeacher
    except ImportError:
        print("❌ Error: 找不到 swin_wrapper.py")
        raise

# ==============================================================================
# 🔥 [1. 分層優先 Loss] 5分类适配
# ==============================================================================
class HierarchicalPriorityLoss(nn.Module):
    def __init__(self, normal_idx=0, binary_weight=1.0, fine_weight=1.0, boost_idx=1, boost_weight=2.0):
        super().__init__()
        self.normal_idx = normal_idx
        self.boost_idx = boost_idx      
        self.boost_weight = boost_weight
        self.w_binary = binary_weight
        self.w_fine = fine_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # --- Binary Loss ---
        binary_targets = (targets != self.normal_idx).float() 
        probs = F.softmax(logits, dim=1)
        prob_normal = probs[:, self.normal_idx]
        prob_abnormal = 1.0 - prob_normal + 1e-7
        
        bce_per_sample = -(binary_targets * torch.log(prob_abnormal) + (1 - binary_targets) * torch.log(prob_normal))
        
        sample_weights = torch.ones_like(binary_targets)
        sample_weights[targets == self.boost_idx] = self.boost_weight
        
        loss_binary = (bce_per_sample * sample_weights).mean()

        # --- Fine Loss ---
        mask_abnormal = (targets != self.normal_idx)
        loss_fine = torch.tensor(0.0, device=logits.device)
        
        if mask_abnormal.sum() > 0:
            abnormal_logits = logits[mask_abnormal][:, 1:5]
            abnormal_targets = targets[mask_abnormal] - 1
            loss_fine = self.ce(abnormal_logits, abnormal_targets)

        total = self.w_binary * loss_binary + self.w_fine * loss_fine
        return total, loss_binary.item(), loss_fine.item()

# ==============================================================================
# 🌟 [2. 蒸餾 Loss 組合] 
# ==============================================================================
class CompositeDistillLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=8.0, gamma=1.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha   
        self.beta = beta     
        self.gamma = gamma   
        self.temp = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        
        # 🔥 GT Loss 使用增強版的分層 Loss (0:normal, 1:conv, 2:limp, 3:sneeze, 4:vomit)
        self.gt_loss_fn = HierarchicalPriorityLoss(
            normal_idx=0, 
            boost_idx=1, # 设定convulsion为重点关注
            binary_weight=1.5, 
            fine_weight=1.0, 
            boost_weight=1.0
        )

    def forward(self, s_logits, t_logits, s_feats, t_feats, labels, B, T, epoch_ratio):
        loss_gt, l_bin, l_fine = self.gt_loss_fn(s_logits, labels)
        
        p_s = F.log_softmax(s_logits / self.temp, dim=1)
        p_t = F.softmax(t_logits / self.temp, dim=1)
        loss_kd = self.kl_div(p_s, p_t) * (self.temp ** 2)

        loss_feat = 0.0
        for i, s_f in enumerate(s_feats):
            t_f = t_feats[i].detach()
            if s_f.dim() == 4 and t_f.dim() == 5:
                C, H, W = s_f.shape[1], s_f.shape[2], s_f.shape[3]
                s_f = s_f.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            if s_f.shape[-3:] != t_f.shape[-3:]:
                s_f = F.interpolate(s_f, size=t_f.shape[-3:], mode='trilinear', align_corners=False)
            sim_map = F.cosine_similarity(s_f, t_f, dim=1)
            loss_feat += (1.0 - sim_map.mean())

        current_alpha = self.alpha * (1.0 - 0.5 * epoch_ratio)
        total_loss = self.gamma * loss_gt + self.beta * loss_kd + current_alpha * loss_feat
        return total_loss, l_bin, l_fine, loss_kd.item(), loss_feat.item()

class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        if isinstance(output, (list, tuple)): self.features = output[0]
        else: self.features = output
    def remove(self): self.hook.remove()

class FeatureProjector(nn.Module):
    def __init__(self, s_ch, t_ch):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(s, t, 1, bias=False), nn.BatchNorm2d(t), nn.ReLU()) 
            for s, t in zip(s_ch, t_ch)
        ])
    def forward(self, feats): return [p(f) for p, f in zip(self.proj, feats)]

def train_distill():
    cfg_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(cfg_path, 'r', encoding='utf-8') as f: cfg = yaml.safe_load(f)
    
    device = torch.device("cuda")
    
    mode = "normal"
    task_cfg = cfg['task_configs'][mode]
    # 🔥 5分类类别配置
    task_cfg['recognizer']['class_names'] = ['normal', 'convulsion', 'limp', 'sneeze', 'vomit'] 
    task_cfg['recognizer']['num_classes'] = 5
    
    print(f"\n🚀 Start Distillation | Classes: {task_cfg['recognizer']['class_names']}")
    
    train_ds = get_dataset(task_cfg['dataset_path'], 'train', task_cfg['recognizer'], is_train=True)
    val_ds = get_dataset(task_cfg['dataset_path'], 'val', task_cfg['recognizer'], is_train=False) 

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    teacher_ckpt = ROOT_DIR + cfg['distillation'].get('teacher_checkpoint', '/teacher/teacher_best_5cls.pth')
    if not os.path.exists(teacher_ckpt):
        teacher_ckpt = ROOT_DIR + "/teacher/teacher_best_5cls.pth"
    
    print(f"👨‍🏫 Loading Teacher from: {teacher_ckpt}")
    teacher = SwinTeacher(checkpoint_path=teacher_ckpt, num_classes=5).to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False

    print("👶 Building Student (RepViT-GRU)...")
    student = RepViT_GRU(num_classes=5, hidden_dim=256, pretrained=True).to(device)
    model_ema = ModelEmaV2(student, decay=0.995)

    s_hooks = []
    if hasattr(student.backbone, 'stages'):
        s_stages = student.backbone.stages
        s_hooks = [FeatureHook(s_stages[i]) for i in [-3, -2, -1]]
    else:
        s_hooks = [FeatureHook(student.backbone)]
        
    dummy = torch.randn(2, 16, 3, 224, 224).to(device)
    with torch.no_grad():
        _, t_all = teacher(dummy)
        t_feats = t_all[-len(s_hooks):] 
        student.forward_seq(dummy)
    
    t_channels = [f.shape[1] for f in t_feats]
    s_channels = [h.features[0].shape[1] if isinstance(h.features, (list,tuple)) else h.features.shape[1] for h in s_hooks]
    
    projector = FeatureProjector(s_channels, t_channels).to(device)

    lr = 3e-4 
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()), lr=lr, weight_decay=0.05)
    criterion = CompositeDistillLoss(alpha=1.0, beta=8.0, gamma=1.0, temperature=4.0)
    
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    out_dir = "outputs/distilled_5cls_final"
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        student.train()
        projector.train()
        epoch_ratio = epoch / epochs
        avg_bin = 0; avg_fine = 0; avg_kd = 0; avg_ft = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            v = batch['pixel_values'].to(device, non_blocking=True)
            l = batch['labels'].to(device, non_blocking=True)
            B, T = v.shape[:2]
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    t_logits, t_all = teacher(v)
                    t_feats_target = t_all[-len(s_hooks):]
                
                s_logits = student.forward_seq(v)
                s_feats_raw = [h.features[0] if isinstance(h.features, (list,tuple)) else h.features for h in s_hooks]
                s_feats_proj = projector(s_feats_raw)
                
                loss, l_bin, l_fine, l_kd, l_feat = criterion(
                    s_logits, t_logits, s_feats_proj, t_feats_target, l, B, T, epoch_ratio
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model_ema.update(student)
            
            avg_bin = (avg_bin * i + l_bin) / (i + 1)
            avg_fine = (avg_fine * i + l_fine) / (i + 1)
            avg_kd = (avg_kd * i + l_kd) / (i + 1)
            avg_ft = (avg_ft * i + l_feat) / (i + 1)
            pbar.set_postfix(Bin=f"{avg_bin:.3f}", Fine=f"{avg_fine:.3f}", KD=f"{avg_kd:.2f}", Ft=f"{avg_ft:.2f}")

        scheduler.step()
        
        student.eval()
        all_preds = []; all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                v, l = batch['pixel_values'].to(device), batch['labels'].to(device)
                out = model_ema.module.forward_seq(v)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_targets.extend(l.cpu().numpy())
        
        acc = sum([p == t for p, t in zip(all_preds, all_targets)]) / len(all_preds) if len(all_preds)>0 else 0
        print(f"\n📊 Val Acc (EMA): {acc:.2%}")
        
        try:
            # 🔥 5分类混淆矩阵
            cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])
            print(f"           [Norm] [Conv] [Limp] [Snee] [Vomit]")
            print(f"True Norm : {cm[0]}")
            print(f"True Conv : {cm[1]}")
            print(f"True Limp : {cm[2]}")
            print(f"True Snee : {cm[3]}")
            print(f"True Vomit: {cm[4]}")
            
            norm_recall = cm[0,0] / (cm[0].sum() + 1e-6)
            conv_recall = cm[1,1] / (cm[1].sum() + 1e-6)
            limp_recall = cm[2,2] / (cm[2].sum() + 1e-6)
            snee_recall = cm[3,3] / (cm[3].sum() + 1e-6)
            vomit_recall = cm[4,4] / (cm[4].sum() + 1e-6)
            
            print(f"Recall -> Norm: {norm_recall:.2f} | Conv: {conv_recall:.2f} | Limp: {limp_recall:.2f} | Sneeze: {snee_recall:.2f} | Vomit: {vomit_recall:.2f}")
        except: pass

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), os.path.join(out_dir, "repvit_best_distilled_5cls.pth"))
            torch.save(model_ema.module.state_dict(), os.path.join(out_dir, "repvit_best_distilled_5cls_ema.pth"))
            print("🏆 Student Model Saved!")
        print("-" * 50)
    
    for h in s_hooks: h.remove()

if __name__ == "__main__":
    train_distill()