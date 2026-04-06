# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import sys

# ================= 1. 環境設置 =================
# 確保能找到項目根目錄下的模塊
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recognizers.gru_model import RepViT_GRU
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import QConfig, FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

# ================= 🔥 [Magic Patch] 核心修復 =================
# 解決 "aten::copy ... not supported" 在 Opset 13 下的報錯
from torch.onnx import register_custom_op_symbolic

def symbolic_copy(g, self, src, non_blocking):
    return src

register_custom_op_symbolic('aten::copy', symbolic_copy, 13)
# ===========================================================

# ================= 2. 定義單幀導出 Wrapper =================
class RepViT_QAT_SingleFrame_Export_Wrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # ⚠️ 這裡的組件命名必須和 train_qat.py 中的 Parallel Wrapper 完全一致
        # 這樣 load_state_dict 時才能無縫對接權重
        self.backbone = original_model.backbone
        self.gap = original_model.gap
        self.fc = original_model.fc
        self.gru = original_model.gru

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x, hidden_state=None):
        # --- 輸入格式 ---
        # x: [Batch, 3, 224, 224] (單張圖片)
        # hidden_state: [1, Batch, 256] (歷史狀態，可為 None)

        x = self.quant(x)                 # Float -> Int8
        features = self.backbone(x)[0]    # CNN 提取特徵
        features = self.dequant(features) # Int8 -> Float

        x = self.gap(features)            # [B, C, 1, 1]
        x = x.flatten(1)                  # [B, C]
        x = x.unsqueeze(1)                # [B, 1, C] 適配 GRU

        # GRU 單步推理
        out, next_hidden = self.gru(x, hidden_state)

        logits = self.fc(out[:, -1, :])   # [B, Num_Classes]

        # 返回：當前幀分類結果 + 新的隱狀態 (供下一幀使用)
        return logits, next_hidden

# ================= 3. 主導出邏輯 =================
def export_onnx_single_frame():
    NUM_CLASSES = 3
    HIDDEN_DIM = 256

    # ⚠️ 指向你最新訓練出來的 QAT 權重
    PTH_PATH = os.path.join(ROOT_DIR, "outputs/best_model_normal/0213/repvit_qat_distill_3_0213.pth")
    # ⚠️ 導出專供 Pipeline 使用的單幀 ONNX 模型
    ONNX_PATH = os.path.join(ROOT_DIR, "outputs/best_model_normal/0213/repvit_qat_single_0213.onnx")

    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    print(f"🚀 開始導出單幀流式 QAT 模型...")
    print(f"📂 讀取權重: {PTH_PATH}")

    # 1. 創建基礎模型
    base_model = RepViT_GRU(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM, pretrained=False)

    # 2. 套上導出專用殼
    qat_model = RepViT_QAT_SingleFrame_Export_Wrapper(base_model)

    # 3. 配置 QConfig (必須與訓練代碼完全一致)
    act_qconfig = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=255, dtype=torch.quint8,
        qscheme=torch.per_tensor_affine, reduce_range=True
    )
    weight_qconfig = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0
    )
    qat_model.qconfig = QConfig(activation=act_qconfig, weight=weight_qconfig)

    # 4. 應用保護策略 (禁用敏感層量化，必須與訓練時一模一樣)
    if hasattr(qat_model.backbone, 'stem'):
        qat_model.backbone.stem.qconfig = None
        print("🛡️ 保護: Backbone Stem 不量化")
    if hasattr(qat_model.backbone, 'stages'):
        qat_model.backbone.stages[0].qconfig = None
        qat_model.backbone.stages[1].qconfig = None
        print("🛡️ 保護: Stages 0 & 1 不量化")

    qat_model.gru.qconfig = None
    qat_model.fc.qconfig = None
    qat_model.gap.qconfig = None
    print("🛡️ 保護: GRU, FC, GAP 不量化")

    # 5. 準備 QAT 結構
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    # 6. 加載權重
    if not os.path.exists(PTH_PATH):
        print(f"❌ 錯誤: 找不到文件 {PTH_PATH}")
        return

    checkpoint = torch.load(PTH_PATH, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 清洗 'module.' 或 '_orig_mod.' 前綴 (相容 torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '').replace('module.', '')
        new_state_dict[new_k] = v

    try:
        qat_model.load_state_dict(new_state_dict, strict=True)
        print("✅ 權重加載成功 (Strict Mode)")
    except Exception as e:
        print(f"❌ 權重加載失敗: {e}")
        return

    # 7. 切換到評估模式 (凍結 Observer，應用量化參數)
    qat_model.eval()
    qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # 8. 構造 Dummy Input (單幀 + 狀態 格式)
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_state = torch.zeros(1, 1, HIDDEN_DIM)

    # 9. 導出 ONNX
    print(f"⚡ 正在導出到: {ONNX_PATH}")
    try:
        torch.onnx.export(
            qat_model,
            (dummy_img, dummy_state),
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input_img', 'in_state'],
            output_names=['output_logits', 'out_state'],

            # 🔥 必須設為 False (使用 Legacy Exporter，兼容 QAT 圖)
            dynamo=False,

            dynamic_axes={
                'input_img': {0: 'batch'},
                'output_logits': {0: 'batch'},
                'in_state': {1: 'batch'},
                'out_state': {1: 'batch'}
            }
        )
        print(f"✅ 導出成功！")
        print(f"   部署時請使用此 ONNX 模型。")
        print(f"   👉 輸入 1: input_img 形狀 [Batch, 3, 224, 224]")
        print(f"   👉 輸入 2: in_state  形狀 [1, Batch, {HIDDEN_DIM}]")
    except Exception as e:
        print(f"❌ 導出失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_onnx_single_frame()