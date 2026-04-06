import torch
import torch.nn as nn
import os
import torchvision
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

class SwinTeacher(nn.Module):
    # 🔴 [修改] 增加 num_classes 参数，默认为 400
    def __init__(self, checkpoint_path=None, num_classes=400):
        super().__init__()
        print(f"👨‍🏫 初始化 Teacher: Video Swin-B (via Torchvision) -> Classes: {num_classes}")
        
        # 1. 加载标准模型 (默认 400 类)
        print("📥 正在加载/下载 Torchvision 官方 K400 权重...")
        try:
            weights = Swin3D_B_Weights.KINETICS400_V1
            self.model = swin3d_b(weights=weights)
        except Exception as e:
            print(f"❌ 下载失败，使用随机初始化: {e}")
            self.model = swin3d_b(weights=None)

        # 2. 🔴 [新增] 如果目标类别不是 400，进行“换头手术”
        # 必须在加载微调权重之前做这一步，否则形状不匹配会加载失败
        if num_classes != 400:
            old_head = self.model.head
            if isinstance(old_head, nn.Sequential):
                in_features = old_head[-1].in_features
                dropout_p = old_head[0].p
            else:
                in_features = old_head.in_features
                dropout_p = 0.5
            
            print(f"🔪 [自动换头] 替换分类头: {in_features} -> {num_classes}")
            self.model.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )

        # 3. 加载微调权重
        if checkpoint_path and isinstance(checkpoint_path, str) and checkpoint_path.endswith('.pth'):
            print(f"📥 加载微调权重: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                 print(f"⚠️ 警告: 权重文件不存在 {checkpoint_path}，跳过加载！")
            else:
                try:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = k.replace("base.model.", "").replace("module.", "")
                        new_state_dict[new_k] = v
                    
                    # 现在架构匹配了，Missing 和 Unexpected 应该都是 0 (或者很少)
                    missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"   权重加载完毕 (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
                except Exception as e:
                    print(f"⚠️ 权重加载出错: {e}")

        self.model.eval()
        
        # 4. 注册 Hooks (保持不变)
        self.features = []
        def hook_fn(module, input, output):
            if output.dim() == 5:
                output = output.permute(0, 4, 1, 2, 3)
            self.features.append(output)

        target_layers = [1, 2, 3, 4]
        num_layers = len(self.model.features)
        for idx in target_layers:
            if idx < num_layers:
                self.model.features[idx].register_forward_hook(hook_fn)

    def forward(self, x):
        self.features = []
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.model(x)
        return logits, self.features