import os
import cv2
import torch
import numpy as np
import random
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
        # 1. 确定类别与关键词
        # -----------------------------------------------------------
        if 'class_names' in self.cfg and self.cfg['class_names']:
            self.keywords = self.cfg['class_names']
        else:
            default_classes = ["eat", "drink", "other"]
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
        raw_samples = [] # 暂存扫描结果，稍后进行平衡处理
        
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

            # 判断是文件夹样本还是类别容器
            if os.path.isdir(entry_path):
                contents = [f for f in os.listdir(entry_path) if not f.startswith('.')]
                has_images = any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in contents)
                has_subdirs = any(os.path.isdir(os.path.join(entry_path, f)) for f in contents)
                
                if has_images and not has_subdirs:
                    # 文件夹本身是一个样本
                    raw_samples.append((entry_path, label))
                else:
                    # 类别容器，遍历子项
                    for file_name in contents:
                        file_path = os.path.join(entry_path, file_name)
                        raw_samples.append((file_path, label))
            else:
                # 文件样本 (视频)
                if entry_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    raw_samples.append((entry_path, label))

        # -----------------------------------------------------------
        # 3. 🔥 硬平衡逻辑 (Hard Balancing) - 尾部截断优先保留新数据
        # -----------------------------------------------------------
        if len(raw_samples) > 0:
            print("⚖️ [Dataset] 正在执行类别硬平衡 (Eat/Drink 1.5倍策略)...")
            
            # 分组
            samples_by_idx = {i: [] for i in range(len(self.keywords))}
            for s in raw_samples:
                samples_by_idx[s[1]].append(s)
            
            # 识别 Eat, Drink, Other 的索引
            idx_eat = -1
            idx_drink = -1
            idx_other = -1
            
            for idx, kw in enumerate(self.keywords):
                if 'eat' in kw.lower(): idx_eat = idx
                elif 'drink' in kw.lower(): idx_drink = idx
                elif 'other' in kw.lower(): idx_other = idx
            
            if idx_eat != -1 and idx_drink != -1 and idx_other != -1:
                n_eat = len(samples_by_idx[idx_eat])
                n_drink = len(samples_by_idx[idx_drink])
                
                # 设定 Other 的目标数量 = 1.5 * max(eat, drink)
                target_base = max(n_eat, n_drink)
                if target_base == 0: target_base = 100 # Fallback
                
                target_other = int(target_base * 1.5)
                
                # 对 Other 进行截断
                current_other = samples_by_idx[idx_other]
                if len(current_other) > target_other:
                    # 提取路径中的编号用于排序
                    def get_sample_id(sample_tuple):
                        path = sample_tuple[0]
                        name = os.path.basename(path).split('.')[0] # 移除后缀(如果是文件)
                        # 从右侧切分提取最后一段数字，如 other_active_123 -> 123
                        parts = name.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            return int(parts[1])
                        # 如果没有下划线但本身就是数字
                        if name.isdigit():
                            return int(name)
                        return 0
                        
                    # 1. 按编号从小到大升序排列
                    current_other.sort(key=get_sample_id)
                    # 2. 丢弃前面的（编号小的），只保留列表末尾的 target_other 个（编号大的）
                    samples_by_idx[idx_other] = current_other[-target_other:]
                    print(f"   ✂️ 截断 Other (保留大编号): {len(current_other)} -> {target_other}")
                else:
                    print(f"   ✅ Other 数量 ({len(current_other)}) 未超过限制 ({target_other})，无需截断")
                
                # 重组
                self.samples = []
                for idx in samples_by_idx:
                    self.samples.extend(samples_by_idx[idx])
            else:
                print("   ⚠️ 无法识别明确的 Eat/Drink/Other 类别索引，跳过硬平衡。")
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
            print(f"Error loading {video_path}: {e}")
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