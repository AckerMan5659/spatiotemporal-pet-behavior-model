# modules/reid_manager.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# 尝试导入 torchreid，如果环境没装需要提示用户
try:
    import torchreid
except ImportError:
    print("⚠️ Warning: 'torchreid' module not found. ReID features will be disabled.")
    torchreid = None

class ReIDManager:
    def __init__(self, model_path, device='cuda', gallery_path=None):
        self.device = device
        self.model = None
        self.enabled = False
        
        if torchreid is None:
            return

        print(f"🔄 Loading OSNet from {model_path}...")
        try:
            # 构建模型 (使用 osnet_x0_5 轻量版，适合 Jetson)
            self.model = torchreid.models.build_model(
                name='osnet_x0_5', 
                num_classes=1000, 
                pretrained=False
            )
            # 加载权重
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            self.enabled = True
        except Exception as e:
            print(f"❌ ReID Model Load Failed: {e}")
            self.enabled = False
            return

        # 预处理管道
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 加载特征库 (可选)
        self.gallery = {}
        if gallery_path and os.path.exists(gallery_path):
            self._load_gallery(gallery_path)

    def _load_gallery(self, path):
        """加载预存的宠物特征库 (代码复用自原脚本)"""
        print(f"📂 Loading gallery from {path}...")
        for pet_folder in os.listdir(path):
            pet_path = os.path.join(path, pet_folder)
            if not os.path.isdir(pet_path): continue
            
            feats = []
            for img_name in os.listdir(pet_path):
                img = cv2.imread(os.path.join(pet_path, img_name))
                if img is not None:
                    h, w = img.shape[:2]
                    f = self.extract(img, [0,0,w,h])
                    if f is not None: feats.append(f)
            
            if feats:
                avg_feat = np.mean(feats, axis=0)
                self.gallery[pet_folder] = avg_feat / (np.linalg.norm(avg_feat) + 1e-8)
                print(f"  - Loaded {pet_folder}")

    def extract(self, frame, box):
        """提取特征向量"""
        if not self.enabled: return None
        
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return None
        
        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop)
        
        try:
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img_tensor)
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Ext Err: {e}")
            return None

    def compute_similarity(self, feat1, feat2):
        if feat1 is None or feat2 is None: return 0.0
        return np.dot(feat1, feat2)