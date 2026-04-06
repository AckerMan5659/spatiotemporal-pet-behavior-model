import torch
import torch.nn as nn

try:
    from transformers import MambaConfig, MambaModel
except ImportError:
    print("❌ 错误: 未安装 transformers 库。")
    MambaModel = None

class VMambaBlock(nn.Module):
    """
    纯空间 Mamba 块：处理图像特征 (H, W)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if MambaModel is None:
            raise ImportError("无法初始化 VMamba，因为缺少 transformers 库。")

        config = MambaConfig(
            hidden_size=dim,
            state_size=d_state,
            conv_kernel=d_conv,
            expand=expand,
            num_hidden_layers=1,
            vocab_size=1,
            use_cache=False # 空间扫描不需要缓存
        )
        self.mamba_core = MambaModel(config)
        self.norm = nn.LayerNorm(dim)

    def forward_scan(self, x):
        outputs = self.mamba_core(inputs_embeds=x)
        return outputs.last_hidden_state

    def forward(self, x):
        # x: (Batch * Time, H, W, C)
        BT, H, W, C = x.shape

        # 1. Top-Left -> Bottom-Right
        v1 = x.view(BT, -1, C)
        out1 = self.forward_scan(v1).view(BT, H, W, C)

        # 2. Top-Right -> Bottom-Left
        v2 = torch.flip(x, dims=[2]).view(BT, -1, C)
        out2 = self.forward_scan(v2).view(BT, H, W, C)
        out2 = torch.flip(out2, dims=[2])

        # 3. Bottom-Left -> Top-Right
        v3 = torch.flip(x, dims=[1]).view(BT, -1, C)
        out3 = self.forward_scan(v3).view(BT, H, W, C)
        out3 = torch.flip(out3, dims=[1])

        # 4. Bottom-Right -> Top-Left
        v4 = torch.flip(x, dims=[1, 2]).view(BT, -1, C)
        out4 = self.forward_scan(v4).view(BT, H, W, C)
        out4 = torch.flip(out4, dims=[1, 2])

        out = (out1 + out2 + out3 + out4) / 4.0
        return self.norm(out)

class MobileViT_VMamba_Head(nn.Module):
    def __init__(self, in_channels, num_classes, d_state=16):
        super().__init__()
        print(f"🚀 初始化 ST-VMamba Head (Spatial-SS2D + Temporal-Mamba) | In: {in_channels}")

        # 1. 空间建模 (处理单帧内的像素关系)
        # 输入: (BT, H, W, C) -> 输出: (BT, H, W, C)
        self.spatial_vmamba = VMambaBlock(dim=in_channels, d_state=d_state)

        # 空间池化：把 (H,W) 变成 1 个特征向量
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. 时序建模 (处理帧与帧之间的关系) [新增]
        # 输入: (B, T, C) -> 输出: (B, T, C)
        self.temporal_config = MambaConfig(
            hidden_size=in_channels,
            state_size=d_state,
            num_hidden_layers=1,
            vocab_size=1,
            use_cache=True # ✅ 开启缓存，支持流式推理
        )
        self.temporal_mamba = MambaModel(self.temporal_config)

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x, cache_params=None, store_state=False, seq_idx=0, **kwargs):
        """
        x: (B, T, C, H, W)
        cache_params: 上一时刻的时序 Mamba 状态
        """
        B, T, C, H, W = x.shape

        # =================================================
        # 第一步：空间建模 (Spatial VMamba)
        # 此时我们将 B 和 T 合并，因为空间扫描是帧独立的
        # =================================================
        # (B, T, C, H, W) -> (B*T, C, H, W) -> (B*T, H, W, C)
        x_spatial = x.view(B * T, C, H, W).permute(0, 2, 3, 1).contiguous()

        # 运行 SS2D 扫描
        x_spatial = self.spatial_vmamba(x_spatial)

        # 还原并池化: (B*T, H, W, C) -> (B*T, C, H, W) -> (B*T, C, 1, 1)
        x_spatial = x_spatial.permute(0, 3, 1, 2)
        x_feat = self.avg_pool(x_spatial).flatten(1) # (B*T, C)

        # 恢复时序维度: (B, T, C)
        x_seq = x_feat.view(B, T, C)

        # =================================================
        # 第二步：时序建模 (Temporal Mamba)
        # 利用 Mamba 记忆动作的历史信息
        # =================================================

        # 构建 cache_position (Fix for newer transformers)
        cache_position = None
        if (store_state or cache_params is not None) and T == 1:
            cache_position = torch.tensor([seq_idx], device=x.device, dtype=torch.long)

        mamba_out = self.temporal_mamba(
            inputs_embeds=x_seq,
            cache_params=cache_params,
            use_cache=True, # 始终开启，方便推理和训练统一
            cache_position=cache_position
        )

        # 取最后一个时间步的特征
        last_hidden = mamba_out.last_hidden_state[:, -1, :] # (B, C)

        logits = self.classifier(last_hidden)

        # 返回逻辑
        if store_state:
            return logits, mamba_out.cache_params
        return logits