import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, root, split='train', cfg=None, is_train=True):
        self.root = os.path.join(root, split) if split else root
        self.cfg = cfg if cfg is not None else {}
        self.is_train = is_train
        self.samples = [] 
        self.targets = [] 

        # -----------------------------------------------------------
        # 1. 确定类别与关键词 (5分类)
        # -----------------------------------------------------------
        if 'class_names' in self.cfg and self.cfg['class_names']:
            self.keywords = self.cfg['class_names']
        else:
            default_classes = ["normal", "convulsion", "limp", "sneeze", "vomit"]
            print(f"⚠️ [Dataset] Config 未指定 class_names，使用默认值: {default_classes}")
            self.keywords = default_classes

        # 图像预处理
        img_size = self.cfg.get('imgsz', 224)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"🔍 [Dataset] 扫描路径: {self.root}")
        print(f"🎯 [Dataset] 目标类别: {self.keywords}")

        if not os.path.exists(self.root):
            print(f"❌ [Error] 路径不存在: {self.root}")
            return

        # -----------------------------------------------------------
        # 2. 扫描数据
        # -----------------------------------------------------------
        raw_samples = [] 
        
        top_level_entries = sorted(os.listdir(self.root))
        for entry_name in top_level_entries:
            if entry_name.startswith('.'): continue
            entry_path = os.path.join(self.root, entry_name)

            # 匹配关键词
            label = -1
            name_lower = entry_name.lower()
            for idx, keyword in enumerate(self.keywords):
                if keyword.lower() in name_lower:
                    label = idx
                    break 
            
            if label == -1: continue

            if os.path.isdir(entry_path):
                contents = [f for f in os.listdir(entry_path) if not f.startswith('.')]
                has_images = any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in contents)
                has_subdirs = any(os.path.isdir(os.path.join(entry_path, f)) for f in contents)
                
                if has_images and not has_subdirs:
                    raw_samples.append((entry_path, label))
                else:
                    for file_name in contents:
                        file_path = os.path.join(entry_path, file_name)
                        raw_samples.append((file_path, label))
            else:
                if entry_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    raw_samples.append((entry_path, label))

        # -----------------------------------------------------------
        # 3. 🔥 硬平衡逻辑 (动态适应 5 分类)
        # -----------------------------------------------------------
        if len(raw_samples) > 0:
            print("⚖️ [Dataset] 正在执行类别硬平衡 (异常样本 1.5倍策略)...")
            
            samples_by_idx = {i: [] for i in range(len(self.keywords))}
            for s in raw_samples:
                samples_by_idx[s[1]].append(s)
            
            idx_normal = -1
            abnormal_indices = []
            
            for idx, kw in enumerate(self.keywords):
                if 'normal' in kw.lower(): 
                    idx_normal = idx
                else: 
                    abnormal_indices.append(idx)
            
            if idx_normal != -1 and len(abnormal_indices) > 0:
                # 找到数量最多的异常类别
                abnormal_counts = [len(samples_by_idx[i]) for i in abnormal_indices]
                target_base = max(abnormal_counts)
                if target_base == 0: target_base = 100 
                
                target_normal = int(target_base * 1.5)
                current_normal = samples_by_idx[idx_normal]
                
                if len(current_normal) > target_normal:
                    def get_sample_id(sample_tuple):
                        path = sample_tuple[0]
                        name = os.path.basename(path).split('.')[0] 
                        parts = name.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            return int(parts[1])
                        if name.isdigit():
                            return int(name)
                        return 0
                        
                    current_normal.sort(key=get_sample_id)
                    samples_by_idx[idx_normal] = current_normal[-target_normal:]
                    print(f"   ✂️ 截断 Normal (保留大编号): {len(current_normal)} -> {target_normal}")
                else:
                    print(f"   ✅ Normal 数量 ({len(current_normal)}) 未超过限制 ({target_normal})，无需截断")
                
                self.samples = []
                for idx in samples_by_idx:
                    self.samples.extend(samples_by_idx[idx])
            else:
                print("   ⚠️ 无法识别明确的 normal 及异常类别索引，跳过硬平衡。")
                self.samples = raw_samples
        else:
            self.samples = raw_samples

        self.targets = [s[1] for s in self.samples]
        print(f"✅ 加载完成: 共 {len(self.samples)} 个样本")
        
        if len(self.samples) > 0:
            counts = {}
            for t in self.targets:
                name = self.keywords[t]
                counts[name] = counts.get(name, 0) + 1
            print(f"📊 最终分布: {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            frames = self._load_frames(video_path)
            pixel_values = torch.stack(frames)
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            img_size = self.cfg.get('imgsz', 224)
            seq_len = self.cfg.get('temporal_window', 16)
            return {
                'pixel_values': torch.zeros(seq_len, 3, img_size, img_size),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def _load_frames(self, path):
        frames = []
        seq_len = self.cfg.get('temporal_window', self.cfg.get('seq_len', 16))
        img_size = self.cfg.get('imgsz', 224)
        
        if os.path.isdir(path):
            all_images = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])
            if len(all_images) > 0:
                indices = np.linspace(0, len(all_images)-1, seq_len).astype(int)
                for i in indices:
                    try:
                        img = Image.open(all_images[i]).convert('RGB')
                        frames.append(self.transform(img))
                    except: frames.append(torch.zeros(3, img_size, img_size))
        else:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                indices = np.linspace(0, total-1, seq_len).astype(int)
                for i in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    if i in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(self.transform(Image.fromarray(frame)))
            cap.release()

        while len(frames) < seq_len:
            frames.append(torch.zeros(3, img_size, img_size))
        return frames[:seq_len]

    def get_class_weights(self, device='cpu'):
        if len(self.targets) == 0: return None
        counts = np.bincount(self.targets, minlength=len(self.keywords))
        weights = 1. / (counts + 1e-6)
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float, device=device)

def get_dataset(data_dir, split, cfg, is_train=True):
    return VideoDataset(data_dir, split=split, cfg=cfg, is_train=is_train)