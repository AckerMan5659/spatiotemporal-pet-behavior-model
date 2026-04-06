import torch
import torch.nn as nn

# --- 尝试导入 transformers ---
try:
    from transformers import MambaConfig, MambaModel
except ImportError:
    MambaModel = None
    # 只有当确实使用 MambaHead 时才警告，避免 TSM 模式下刷屏
    pass

class LSTMHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 2, num_classes: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        y, _ = self.lstm(x)
        logits = self.fc(y[:, -1, :])
        return logits


class TransformerHead(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int = 4, depth: int = 2, num_classes: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        y = self.encoder(x)
        return self.cls(y[:, -1, :])


class MambaHead(nn.Module):
    def __init__(self, feat_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_classes: int = 8):
        super().__init__()
        if MambaModel is None:
            raise ImportError("MambaHead requires 'transformers'. Run: pip install transformers")

        # 1. 配置 Mamba
        self.config = MambaConfig(
            hidden_size=feat_dim,
            state_size=d_state,
            conv_kernel=d_conv,
            expand=expand,
            num_hidden_layers=1,
            vocab_size=1,
            use_cache=True       # ✅ 始终开启 Cache 支持
        )

        # 2. 创建模型
        self.mamba = MambaModel(self.config)

        # 3. 分类层
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, cache_params=None, store_state=False, seq_idx=0):
        """
        x: (Batch, Seq_Len, Dim)
        """
        cache_position = None
        if store_state or cache_params is not None:
            cache_position = torch.tensor([seq_idx], device=x.device, dtype=torch.long)

        outputs = self.mamba(
            inputs_embeds=x,
            use_cache=True,
            cache_params=cache_params,
            cache_position=cache_position
        )

        last_hidden_state = outputs.last_hidden_state
        logits = self.fc(last_hidden_state[:, -1, :])

        if cache_params is not None or store_state:
            return logits, outputs.cache_params
        else:
            return logits


class TSMConsensusHead(nn.Module):
    """
    ✅ [新增] TSM 专用 Head
    Backbone (RepViT+TSM) 已经通过 Shift 操作融合了时序信息。
    这里的任务非常简单：
    1. 对时间维度取平均 (Consensus)，让每个片段产出一个特征向量。
    2. 线性分类。
    """
    def __init__(self, feat_dim: int, num_classes: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x input: (B, T, F) -> 来自 Backbone 的 pooled features

        # 1. Temporal Consensus (对时间维度取平均)
        # 相比只取最后一帧 (x[:, -1, :])，平均池化利用了所有帧的信息，更鲁棒
        x_mean = x.mean(dim=1) # [B, F]

        x_mean = self.dropout(x_mean)
        logits = self.fc(x_mean) # [B, NumClasses]

        return logits