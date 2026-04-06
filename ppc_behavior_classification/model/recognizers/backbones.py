import timm
import torch
import torch.nn as nn

# =================================================================
# 1. TSM 模块定义 (保持不变)
# =================================================================
class TSM(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        # x: [B*T, C, H, W]
        nt, c, h, w = x.size()
        # n_batch = nt // self.n_segment

        # 强制 reshape，不依赖 -1 推断
        x = x.reshape(nt // self.n_segment, self.n_segment, c, h, w)

        out = torch.zeros_like(x)
        fold = c // self.fold_div

        # TSM Shift
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)

class TSMBlockWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super().__init__()
        self.tsm = TSM(n_segment=n_segment)
        self.block = block

    def forward(self, x):
        x = self.tsm(x)
        return self.block(x)

# =================================================================
# 2. 核心修正: inject_tsm
# =================================================================
def inject_tsm(model, n_segment=8):
    """
    遍历 timm 模型，将深层 Stage 的 Block 替换为 TSMBlockWrapper
    ✅ [Fix]: 增加了对 RepVitStage 等自定义容器的兼容性处理
    """
    # 1. 尝试获取 stages 或 features
    stages = getattr(model, 'stages', None)
    if stages is None:
        stages = getattr(model, 'blocks', None) # 兼容其他架构
        if stages is None:
            # 最后的尝试：如果是 Sequential 模型 (如 MobileNetV2 某些变体)
            if isinstance(model, nn.Sequential):
                stages = model
            else:
                features = getattr(model, 'features', None)
                if features is not None:
                    stages = features

    if stages is None:
        print("⚠️ Warning: 无法找到模型的 stages/blocks/features，TSM 注入失败。")
        return model

    # 2. 遍历 Stage
    # 注意：Timm 的 stages 无论是不是 Sequential，通常本身是可迭代的（List 或 ModuleList）
    # 但里面的元素（Stage）可能是自定义对象
    for idx, stage in enumerate(stages):
        # 策略：跳过浅层 (Stage 0, 1)
        if idx < 2:
            continue

        # --- 🔥 关键修复开始: 获取真正的 block 容器 ---
        target_blocks = stage # 默认假设 stage 本身就是 blocks 列表

        # 如果 stage 不可迭代 (比如 RepVitStage 对象)，则找它的内部属性
        if not isinstance(stage, (nn.Sequential, nn.ModuleList, list)):
            if hasattr(stage, 'blocks'):
                target_blocks = stage.blocks
            elif hasattr(stage, 'layers'):
                target_blocks = stage.layers
            else:
                # 如果找不到显式的 blocks 属性，尝试通过 named_children 暴力查找卷积块
                # 这种情况下无法通过索引替换，只能 setattr
                print(f"  -> [Info] Stage {idx} is custom object, trying named_children replacement...")
                for name, child in stage.named_children():
                    if hasattr(child, 'token_mixer') or hasattr(child, 'conv'):
                        print(f"     Injecting TSM into {name}")
                        setattr(stage, name, TSMBlockWrapper(child, n_segment))
                continue # 处理完这个特殊 stage，跳到下一个
        # --- 关键修复结束 ---

        # 3. 遍历 Block 进行替换 (针对 Sequential/ModuleList)
        # 只有当 target_blocks 是可索引的容器时才执行
        if isinstance(target_blocks, (nn.Sequential, nn.ModuleList)):
            for i, block in enumerate(target_blocks):
                # 判定是否是包含卷积/TokenMixer 的层 (避开 Identity 或 Pooling)
                if hasattr(block, 'token_mixer') or hasattr(block, 'conv') or hasattr(block, 'block') or "Block" in str(type(block)):
                    # print(f"  -> Injecting TSM at Stage {idx} Block {i}")
                    target_blocks[i] = TSMBlockWrapper(block, n_segment)

    print(f"✅ TSM (Frames={n_segment}) 已注入模型深层 (Stage 2+)。")
    return model

# =================================================================
# 3. create_timm_backbone (保持接口兼容)
# =================================================================
def create_timm_backbone(name: str, pretrained: bool = True, out_dim: int = 0, return_spatial: bool = False,
                         use_tsm: bool = False, num_frames: int = 8):

    pool_type = '' if return_spatial else 'avg'

    # 创建模型
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        features_only=False,
        global_pool=pool_type
    )

    # 注入 TSM
    if use_tsm:
        print(f"🏗️ 正在为 {name} 注入 TSM 模块 (Frames={num_frames})...")
        inject_tsm(model, n_segment=num_frames)

    # 推断维度
    try:
        with torch.no_grad():
            dummy_input = torch.randn(2 * num_frames, 3, 224, 224)
            y = model(dummy_input)

            if return_spatial:
                feat_dim = y.shape[1]
            else:
                feat_dim = y.shape[-1]
    except Exception as e:
        print(f"⚠️ 警告: 自动推断 feat_dim 失败: {e}")
        feat_dim = getattr(model, 'num_features', None)
        if isinstance(feat_dim, (list, tuple)): feat_dim = feat_dim[-1]

    if out_dim and out_dim != feat_dim and not return_spatial:
        projector = nn.Linear(feat_dim, out_dim)
        return nn.Sequential(model, projector), out_dim

    return model, feat_dim