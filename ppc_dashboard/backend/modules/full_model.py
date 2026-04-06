import torch
import torch.nn as nn
from .backbones import create_timm_backbone
from .behavior_head import LSTMHead, TransformerHead, MambaHead
# ✅ 导入新 Head
from .vmamba_head import MobileViT_VMamba_Head

class EndToEndRecognizer(nn.Module):
    """
    一个端到端的模型，组合 Backbone 和 Head。
    """
    def __init__(self, cfg_recognizer: dict):
        super().__init__()
        rec = cfg_recognizer
        head_type = rec.get("head", "lstm")

        # 🔥 关键判定：如果是 vmamba，backbone 必须返回空间特征 (H, W)
        is_vmamba = (head_type == "vmamba")

        # 1. 初始化 Backbone
        # create_timm_backbone 返回 (model, feat_dim)
        self.backbone, self.feat_dim = create_timm_backbone(
            rec['timm_model'],
            pretrained=True,
            out_dim=rec.get('feature_dim', 0),
            return_spatial=is_vmamba  # ✅ 传入此参数
        )

        # 2. 初始化 Head
        num_classes = rec["num_classes"]
        print(f"🏗️ Initializing Head: Type='{head_type}' | Feature Dim={self.feat_dim} | Classes={num_classes}")

        if head_type == "mamba":
            # 原始 1D Mamba
            d_state = rec.get("mamba_d_state", 16)
            self.head = MambaHead(
                feat_dim=self.feat_dim,
                d_state=d_state,
                num_classes=num_classes
            )

        elif head_type == "vmamba":
            # 🚀 新增 VMamba 分支
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
        x = x.view(B * T, C, H, W)

        # 2. Extract Features
        # 如果是 VMamba，backbone 返回 (B*T, C, H', W')
        # 如果是 普通Head，backbone 返回 (B*T, C)
        features = self.backbone(x)

        # 3. Head Processing
        if isinstance(self.head, MobileViT_VMamba_Head):
            # VMamba 需要完整 5D 张量：(B, T, C, H, W)
            # 需要恢复 T 维度
            _, C_feat, H_feat, W_feat = features.shape
            features = features.view(B, T, C_feat, H_feat, W_feat)
            logits = self.head(features)
        else:
            # 普通 Head 需要 (B, T, F)
            features = features.view(B, T, self.feat_dim)
            logits = self.head(features)

        return logits