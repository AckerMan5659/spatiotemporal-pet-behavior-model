import torch
import torch.nn as nn
import cv2
import yaml
import numpy as np
import os
import sys
import timm

# 尝试导入 ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# =========================================================================
# 1. 模型定义 (来自 pipeline_gru.py)
# =========================================================================
class RepViT_GRU(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        """
        RepViT Backbone + GRU Head (Stateful Model)
        """
        super().__init__()

        # 1. 加载 RepViT Backbone
        # 使用 timm 加载标准模型，features_only=True 表示只取特征
        print(f"🏗️ [RepViT_GRU] Initializing Backbone (Pretrained={pretrained})...")
        self.backbone = timm.create_model(
            'repvit_m1',
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]
        )

        # 自动推断通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)[0]
            in_channels = feat.shape[1]
            
        # 2. 空间特征压缩
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 3. GRU 层
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)

        # 4. 分类头
        self.fc = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden_state=None):
        """
        x: [Batch, 3, 224, 224] -> 单帧图像
        hidden_state: [1, Batch, Hidden] -> 上一帧的记忆
        """
        # Backbone
        features = self.backbone(x)[0] # [B, 384, 7, 7]

        # 空间压缩
        x = self.gap(features)
        x = self.flatten(x) # [B, 384]

        # 调整维度适配 GRU: [B, 1, 384]
        x = x.unsqueeze(1)

        # GRU 推理
        out, next_hidden = self.gru(x, hidden_state)

        # 分类
        logits = self.fc(out[:, -1, :]) # [B, Classes]

        return logits, next_hidden

# =========================================================================
# 2. 适配器类 (替换原有的 ActionRecognizer)
# =========================================================================
class ActionRecognizer:
    def __init__(self, cfg_path, device='auto'):
        print(f"🧠 初始化动作识别模块 (融合 RepViT_GRU)...")
        
        # 1. 配置加载
        self.cfg = self._load_config(cfg_path)
        self.model_path = self.cfg['output_model_name']
        self.class_names = self.cfg['class_names']
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"⚖️ 加载模型权重: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")

        # 2. 初始化引擎 (ONNX 或 PyTorch)
        self.use_onnx = self.model_path.endswith('.onnx')
        self.hidden_dim = 256 # 默认值
        self.onnx_state_rank = 3 # 默认为3维 (1, B, H)

        # 预处理参数 (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.imgsz = (224, 224)
        self.context_expansion = 0.25

        if self.use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()
            
        print(f"✅ 动作识别模块就绪 ({'ONNX' if self.use_onnx else 'PyTorch'})")

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            base = yaml.safe_load(f)
        mode = base.get('mode', 'normal')
        task_cfg = base['task_configs'][mode]
        
        class_names = [name.lower() for name in task_cfg['class_names']]
        config = {
            'recognizer': dict(task_cfg['recognizer']),
            'class_names': class_names,
            'output_model_name': task_cfg['output_model_name']
        }
        config['recognizer']['num_classes'] = len(class_names)
        return config

    def _init_onnx(self):
        if not HAS_ONNX:
            raise ImportError("需要安装 onnxruntime 才能使用 .onnx 模型。")
        
        print(f"🚀 初始化 ONNX 引擎...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 根据设备选择 Provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        except Exception as e:
            print(f"⚠️ CUDA Provider 初始化失败，回退到 CPU: {e}")
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=['CPUExecutionProvider'])

        inputs = self.session.get_inputs()
        self.input_names = [i.name for i in inputs]
        self.img_input_name = inputs[0].name
        
        # 检查是否为 GRU 状态输入
        if len(inputs) >= 2:
            self.state_input_name = inputs[1].name
            shape = inputs[1].shape
            self.onnx_state_rank = len(shape)
            self.hidden_dim = shape[-1]
            print(f"🧠 ONNX State Info: Rank={self.onnx_state_rank}D, Hidden={self.hidden_dim}")

    def _init_pytorch(self):
        print(f"🧬 初始化 PyTorch 引擎...")
        self.model = RepViT_GRU(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim, # 注意：如果权重里的 hidden_dim 不一样，这里可能会有问题，通常 256 是默认
            pretrained=False
        )
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 清洗 key (去除 module. 前缀等)
        new_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 尝试加载
        try:
            self.model.load_state_dict(new_sd, strict=False)
        except Exception as e:
            print(f"⚠️ 权重加载警告: {e}")
            
        self.model.to(self.device)
        self.model.eval()

    def get_class_index(self, class_name):
        return self.class_to_idx.get(class_name.lower())

    def get_class_prob(self, probs, class_name, default=0.0):
        idx = self.get_class_index(class_name)
        if probs is None or idx is None or idx >= len(probs):
            return default
        return float(probs[idx])

    def preprocess(self, frame, box):
        """
        FusionAgent 调用此方法预处理
        注意：FusionAgent 传递单帧，但 RepViT_GRU 期望 Batch 维度
        """
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # Context Expansion (与 pipeline_gru 一致)
        w_box, h_box = x2 - x1, y2 - y1
        pad_w = int(w_box * self.context_expansion / 2)
        pad_h = int(h_box * self.context_expansion / 2)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w_img, x2 + pad_w)
        y2 = min(h_img, y2 + pad_h)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        # Resize & Transpose
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.imgsz)
        
        if self.use_onnx:
            # ONNX 需要 numpy: (1, 3, 224, 224)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std # Broadcasting
            return img.astype(np.float32)
        else:
            # PyTorch 需要 Tensor: (1, 3, 224, 224)
            tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            # Normalize
            tensor = (tensor - torch.from_numpy(self.mean[0,:,0,0]).to(self.device).view(3,1,1)) / \
                     torch.from_numpy(self.std[0,:,0,0]).to(self.device).view(3,1,1)
            return tensor.unsqueeze(0).to(self.device)

    def predict(self, frame, box, prev_state=None, seq_idx=0):
        """
        统一推理接口
        输入:
            prev_state: 上一帧的 GRU 状态 (Tensor or Numpy)
            seq_idx: (在此 GRU 模型中被忽略，保留是为了兼容接口)
        输出:
            probs: (Numpy array) 类别概率
            new_state: (Tensor or Numpy) 更新后的状态
        """
        inp = self.preprocess(frame, box)
        if inp is None:
            return None, prev_state

        # ==========================
        # ONNX 推理分支
        # ==========================
        if self.use_onnx:
            # 初始化状态
            if prev_state is None:
                shape = (1, 1, self.hidden_dim) if self.onnx_state_rank == 3 else (1, self.hidden_dim)
                prev_state = np.zeros(shape, dtype=np.float32)
            
            # 准备输入字典
            inputs = {self.img_input_name: inp, self.state_input_name: prev_state}
            
            try:
                logits, next_state = self.session.run(None, inputs)
                # Softmax
                probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs[0], next_state
            except Exception as e:
                print(f"❌ ONNX Inference Error: {e}")
                return None, prev_state

        # ==========================
        # PyTorch 推理分支
        # ==========================
        else:
            # 初始化状态
            if prev_state is None:
                # PyTorch GRU 期望 (num_layers, batch, hidden) -> (1, 1, hidden)
                prev_state = torch.zeros(1, 1, self.hidden_dim).to(self.device)
            elif not isinstance(prev_state, torch.Tensor):
                # 防止传入了 numpy (如果之前跑的是 onnx)
                prev_state = torch.tensor(prev_state).to(self.device)

            with torch.no_grad():
                logits, next_state = self.model(inp, prev_state)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                return probs, next_state