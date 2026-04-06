import torch
import torch.nn as nn

# --- 尝试导入 transformers ---
try:
    from transformers import MambaConfig, MambaModel
except ImportError:
    MambaModel = None
    print("⚠️ Warning: 'transformers' library not found. MambaHead will fail if used.")


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
        cache_params: 上一帧状态
        store_state: (bool) 强制返回状态，用于推理时的第一帧初始化
        seq_idx: (int) 当前是序列的第几帧 (用于构建 cache_position)
        """

        # 🔥 关键修复：构建 cache_position
        # 在推理模式下 (store_state=True 或 cache_params!=None)，必须指定 cache_position
        cache_position = None
        if store_state or cache_params is not None:
            # x 的 seq_len 为 1，cache_position 应该是一个包含 [seq_idx] 的 Tensor
            cache_position = torch.tensor([seq_idx], device=x.device, dtype=torch.long)

        # 调用 Mamba
        outputs = self.mamba(
            inputs_embeds=x,
            use_cache=True,
            cache_params=cache_params,
            cache_position=cache_position # ✅ 传入位置索引
        )

        last_hidden_state = outputs.last_hidden_state

        # 取最后一个时间步
        logits = self.fc(last_hidden_state[:, -1, :])

        # 如果传入了 cache_params (非空) 或者 store_state=True (强制要求)，则返回元组
        # 否则 (训练模式)，只返回 logits
        if cache_params is not None or store_state:
            return logits, outputs.cache_params
        else:
            return logits