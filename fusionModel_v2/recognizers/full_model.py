import torch
import torch.nn as nn
from .backbones import create_timm_backbone
# ✅ 导入新 Head: TSMConsensusHead
from .behavior_head import LSTMHead, TransformerHead, MambaHead, TSMConsensusHead
# ✅ 导入 VMamba Head
from .vmamba_head import MobileViT_VMamba_Head

class EndToEndRecognizer(nn.Module):
    """
    一个端到端的模型，组合 Backbone 和 Head。
    """
    def __init__(self, cfg_recognizer: dict):
        super().__init__()
        rec = cfg_recognizer
        head_type = rec.get("head", "lstm")

        # 🔥 关键参数提取
        # 1. 是否启用 TSM (仅当配置中明确写了 use_tsm: True 时启用)
        use_tsm = rec.get("use_tsm", False)
        # 2. 必须获取时间窗口大小，TSM 需要此参数来正确 Reshape
        temporal_window = rec.get("temporal_window", 16)

        # 3. 判定是否为 VMamba (Backbone 需要返回 4D 空间特征)
        is_vmamba = (head_type == "vmamba")

        # -----------------------------------------------------------
        # 1. 初始化 Backbone
        # -----------------------------------------------------------
        # create_timm_backbone 需要我们在 backbones.py 中修改过接口
        use_pretrained = rec.get('pretrained', True)

        self.backbone, self.feat_dim = create_timm_backbone(
            rec['timm_model'],
            pretrained=use_pretrained,
            out_dim=rec.get('feature_dim', 0),
            return_spatial=is_vmamba,  # VMamba 需要 (B, C, H, W)
            use_tsm=use_tsm,           # ✅ 传入 TSM 开关
            num_frames=temporal_window # ✅ 传入 帧数
        )

        # -----------------------------------------------------------
        # 2. 初始化 Head
        # -----------------------------------------------------------
        num_classes = rec["num_classes"]
        print(f"🏗️ Initializing Head: Type='{head_type}' | TSM={use_tsm} | Feat Dim={self.feat_dim} | Classes={num_classes}")

        if head_type == "tsm_head":
            # 🚀 [新增] TSM 专用简单 Head
            self.head = TSMConsensusHead(
                feat_dim=self.feat_dim,
                num_classes=num_classes,
                dropout=rec.get("dropout", 0.0)
            )

        elif head_type == "mamba":
            # 原始 1D Mamba
            d_state = rec.get("mamba_d_state", 16)
            self.head = MambaHead(
                feat_dim=self.feat_dim,
                d_state=d_state,
                num_classes=num_classes
            )

        elif head_type == "vmamba":
            # VMamba 分支
            d_state = rec.get("mamba_d_state", 16)
            print(f"🐍 Using VMambaHead (2D-Scan, d_state={d_state})")
            self.head = MobileViT_VMamba_Head(
                in_channels=self.feat_dim,
                num_classes=num_classes,
                d_state=d_state
            )

        elif head_type == "lstm":
            hidden = rec.get("lstm_hidden", 256)
            layers = rec.get("lstm_layers", 2)
            self.head = LSTMHead(
                feat_dim=self.feat_dim,
                hidden=hidden,
                layers=layers,
                num_classes=num_classes
            )

        elif head_type == "transformer":
            depth = rec.get("transformer_depth", 2)
            num_heads = rec.get("transformer_heads", 4)
            self.head = TransformerHead(
                feat_dim=self.feat_dim,
                num_classes=num_classes,
                depth=depth,
                num_heads=num_heads
            )

        else:
            raise ValueError(f"❌ Unknown head type: {head_type}")

        print(f"✅ EndToEndRecognizer created successfully.")

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 1. Reshape for Backbone (B*T, C, H, W)
        # 对于 TSM 架构，我们在 backbones.py 里会将其 reshape 回 (Batch, Frames, ...) 处理后再 flatten 回来
        x = x.view(B * T, C, H, W)

        # 2. Extract Features
        # - 如果是 TSM + tsm_head: 返回 (B*T, F) (已经经过了 Shift 和 Conv)
        # - 如果是 VMamba: 返回 (B*T, C, H', W')
        # - 如果是 LSTM/Trans: 返回 (B*T, F) (无 Shift)
        features = self.backbone(x)

        # 3. Head Processing
        if isinstance(self.head, MobileViT_VMamba_Head):
            # VMamba 需要完整 5D 张量：(B, T, C, H, W)
            _, C_feat, H_feat, W_feat = features.shape
            features = features.view(B, T, C_feat, H_feat, W_feat)
            logits = self.head(features)
        else:
            # 普通 Head (包含 TSMConsensusHead) 需要 (B, T, F)
            # 此时 features 是 (B*T, F)，我们需要拆开 B 和 T
            features = features.view(B, T, self.feat_dim)
            logits = self.head(features)

        return logits