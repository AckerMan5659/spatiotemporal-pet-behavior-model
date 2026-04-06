# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import sys
import warnings

# 忽略一些无关紧要的导出警告，保持控制台清爽
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 自动设置路径，确保能导入项目中的 recognizers
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recognizers.gru_model import RepViT_GRU

# -------------------------------------------------------------------
# 2. 导出专用的包装类 (2D 状态版 - 适配流式推理)
# -------------------------------------------------------------------
class RepViT_GRU_2D(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 提取原模型的各个组件
        self.backbone = original_model.backbone
        self.gap = original_model.gap
        self.fc = original_model.fc

        # 关键：将 nn.GRU (序列处理) 移植到 nn.GRUCell (单步处理)
        # 这样导出的 ONNX 输入是 [Batch, C, H, W] 和 [Batch, Hidden]
        input_size = original_model.gru.input_size
        hidden_size = original_model.gru.hidden_size
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        print(f"💉 正在移植 GRU 权重 (Hidden={hidden_size})...")
        # 权重赋值
        self.gru_cell.weight_ih.data = original_model.gru.weight_ih_l0.data
        self.gru_cell.weight_hh.data = original_model.gru.weight_hh_l0.data
        self.gru_cell.bias_ih.data = original_model.gru.bias_ih_l0.data
        self.gru_cell.bias_hh.data = original_model.gru.bias_hh_l0.data

    def forward(self, input_img, in_state):
        # --- A. Backbone (将被量化为 INT8) ---
        features = self.backbone(input_img)

        # 稳健性检查：处理 Timm 可能返回列表的情况
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # --- B. Head (将被保护为 FP32) ---
        x = self.gap(features)
        x = x.flatten(1)

        # 显式调用 gru_cell，ONNX 节点名通常会包含 "gru_cell"
        # 这对于后续混合精度量化脚本的 "Name Filter" 至关重要
        h_out = self.gru_cell(x, in_state)

        logits = self.fc(h_out)

        return logits, h_out

# -------------------------------------------------------------------
# 3. 辅助函数：算子融合 (Deploy Mode)
# -------------------------------------------------------------------
def fuse_repvit(module):
    """
    将 RepViT 切换到部署模式，融合卷积和 BatchNorm。
    这能显著提升速度并确保与训练时（如果是用 EMA 或 Deploy 权重）的结构对齐。
    """
    if hasattr(module, 'switch_to_deploy'):
        try:
            module.switch_to_deploy()
        except Exception as e:
            pass # 某些层可能已经融合过或不需要融合，忽略错误

    for child in module.children():
        fuse_repvit(child)

# -------------------------------------------------------------------
# 4. 主导出函数
# -------------------------------------------------------------------
def export_model():
    # 路径配置
    pth_path = "../outputs/best_model_abnormal/repvit_best_distilled_5cls_0316.pth"
    output_dir = "../outputs/best_model_abnormal/0313"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "repvit_best_distilled_5cls_0316.onnx")

    print(f"🚀 [Step 1] 初始化模型并加载权重...")
    # 注意：num_classes 需与训练时保持一致
    model_orig = RepViT_GRU(num_classes=5, hidden_dim=256, pretrained=False)

    if os.path.exists(pth_path):
        print(f"📥 正在加载权重: {pth_path}")
        checkpoint = torch.load(pth_path, map_location="cpu")
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # 自动清洗分布式训练产生的 module. 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_orig.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ 警告: 找不到权重文件 {pth_path}，将导出随机权重模型用于测试！")

    model_orig.eval()

    # 执行算子融合 (CN + BN -> CN)
    print("✨ 执行 RepViT 结构重参数化融合...")
    fuse_repvit(model_orig)

    # 包装模型
    model_to_export = RepViT_GRU_2D(model_orig)
    model_to_export.eval()

    # 准备虚拟输入
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_state = torch.zeros(1, 256)

    print(f"📦 [Step 2] 正在导出 ONNX 模型 (Opset 18)...")

    # 🔥 核心修改：使用 Opset 18 并禁用 dynamo 以获得最佳稳定性
    try:
        torch.onnx.export(
            model_to_export,
            (dummy_img, dummy_state),
            onnx_path,
            export_params=True,
            opset_version=18,       # 👈 修复点：使用高版本 Opset 避免降级失败
            do_constant_folding=True,
            input_names=['input_img', 'in_state'],
            output_names=['output_logits', 'out_state'],
            dynamo=False,           # 👈 修复点：禁用 dynamo 以支持 dynamic_axes
            dynamic_axes={
                'input_img': {0: 'batch'},
                'in_state': {0: 'batch'},
                'output_logits': {0: 'batch'},
                'out_state': {0: 'batch'}
            }
        )
        print(f"✅ 导出成功: {onnx_path}")
        print(f"💡 接下来请运行 quantize_mixed.py (混合精度量化脚本) 生成最终模型。")

    except Exception as e:
        print(f"\n❌ 导出遭遇严重错误:\n{e}")
        print("\n👉 建议排查:")
        print("   1. 确保 PyTorch 版本 >= 2.0")
        print("   2. 检查 pth 权重是否损坏")

if __name__ == "__main__":
    export_model()