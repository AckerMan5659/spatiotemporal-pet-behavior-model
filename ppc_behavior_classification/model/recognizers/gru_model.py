import torch
import torch.nn as nn
import timm

class RepViT_GRU(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, pretrained=True):
        """
        RepViT Backbone + GRU Head (Stateful Model)
        """
        super().__init__()

        # 1. 加载 RepViT Backbone (不带 TSM!)
        # 使用 timm 加载标准模型，features_only=True 表示只取特征，不取分类头
        # out_indices=[-1] 表示只取最后一层的特征
        print(f"🏗️ Initializing RepViT-M1 Backbone (Pretrained={pretrained})...")
        self.backbone = timm.create_model(
            'repvit_m1',
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]
        )

        # 获取 Backbone 输出通道数 (RepViT-M1 通常是 384)
        # 这里的 dummy input 用来自动推断通道数，防止写死出错
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)[0]
            in_channels = feat.shape[1]
            print(f"ℹ️ Backbone Output Channels: {in_channels}")

        # 2. 空间特征压缩
        # 将 [Batch, 384, 7, 7] -> [Batch, 384, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 3. GRU 层 (核心时间模块)
        # input_size: 384
        # hidden_size: 256 (记忆容量)
        # batch_first=True: 输入形状为 [Batch, Seq, Feature]
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)

        # 4. 分类头
        self.fc = nn.Linear(hidden_dim, num_classes)

        # 5. 权重初始化
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
        hidden_state: [1, Batch, Hidden] -> 上一帧的记忆 (Tensor)
        """
        # --- 1. Backbone 提取特征 ---
        # output is a list, take the last one
        features = self.backbone(x)[0] # [B, 384, 7, 7]

        # --- 2. 空间压缩 ---
        x = self.gap(features)
        x = self.flatten(x) # [B, 384]

        # --- 3. 调整维度适配 GRU ---
        # GRU 需要 sequence 维度: [Batch, Seq_Len, Features]
        # 这里 Seq_Len = 1 (因为我们是逐帧推理)
        x = x.unsqueeze(1) # [B, 1, 384]

        # --- 4. GRU 推理 ---
        # out: [B, 1, Hidden]
        # next_hidden: [1, B, Hidden]
        out, next_hidden = self.gru(x, hidden_state)

        # --- 5. 分类 ---
        # 取序列最后一个时间步的结果 (其实一共就1步)
        logits = self.fc(out[:, -1, :]) # [B, Classes]

        return logits, next_hidden

    def forward_seq(self, x):
        """
        训练时使用：一次性处理一个序列
        x: [Batch, T, 3, 224, 224]
        """
        b, t, c, h, w = x.shape
        # 1. Merge Batch and Time
        x = x.view(b * t, c, h, w)

        # 2. Backbone
        features = self.backbone(x)[0]
        x = self.gap(features)
        x = self.flatten(x) # [B*T, 384]

        # 3. Unmerge to feed GRU
        x = x.view(b, t, -1) # [B, T, 384]

        # 4. GRU (自动处理整个序列)
        out, _ = self.gru(x) # [B, T, Hidden]

        # 5. FC (对每一帧都分类)
        # 我们通常只需要最后一帧用于 Loss，或者对所有帧算 Loss
        # 这里返回最后一帧用于分类
        logits = self.fc(out[:, -1, :])

        return logits