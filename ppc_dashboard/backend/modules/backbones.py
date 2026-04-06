import timm
import torch
import torch.nn as nn

def create_timm_backbone(name: str, pretrained: bool = True, out_dim: int = 0, return_spatial: bool = False):
    """
    Args:
        return_spatial (bool): 如果为 True，返回 (B, C, H, W)；否则返回 (B, C)
    """
    # 1. 根据需求决定是否进行全局池化
    # 如果是用 VMamba，我们需要空间信息，所以 global_pool 设为空
    pool_type = '' if return_spatial else 'avg'

    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        features_only=False,
        global_pool=pool_type
    )

    # 2. 自动推断特征维度
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            y = model(dummy_input)

            # 🔥 关键修改：根据输出形状正确获取 Channel 维度
            if return_spatial:
                # y shape: (1, C, H, W) -> feat_dim = C = y.shape[1]
                feat_dim = y.shape[1]
            else:
                # y shape: (1, C) -> feat_dim = C = y.shape[1] (或者 -1)
                feat_dim = y.shape[-1]

    except Exception as e:
        print(f"⚠️ 警告: 自动推断 feat_dim 失败: {e}")
        # 回退逻辑 (略)
        feat_dim = getattr(model, 'num_features', None) or getattr(model, 'num_classes', None)
        if isinstance(feat_dim, (list, tuple)):
            feat_dim = feat_dim[-1]

    # 3. 如果需要投影 (通常用于 spatial 时不建议在这里投影，留给 Head 做)
    if out_dim and out_dim != feat_dim and not return_spatial:
        projector = nn.Linear(feat_dim, out_dim)
        return nn.Sequential(model, projector), out_dim

    return model, feat_dim