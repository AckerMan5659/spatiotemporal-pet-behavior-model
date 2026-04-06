# -*- coding: utf-8 -*-
import torch
import onnxruntime as ort
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入项目组件
from dataset import get_dataset
from recognizers.gru_model import RepViT_GRU

# ================= 🚀 配置区域 =================
CFG = {
    'imgsz': 224,
    'seq_len': 16,
    'data_dir': 'D:/Desktop/PET1/cropped_dataset/normal_3',
    'split': 'val'
}

# ⚡️ 速度控制开关
RUN_CONFIG = {
    'RUN_PTH': False,        # ✅ PyTorch (现在可以正常跑了)
    'RUN_FIXED': True,     # ✅ FP32 基准
    'RUN_DYNAMIC': False,   # ✅ 动态 Int8
    'RUN_MIXED': False,     # ✅ 混合精度 PTQ
    'RUN_QAT': True,       # ✅ QAT Int8
    'MAX_SAMPLES': None     # 设为 None 跑全量
}

# 📂 模型路径
PTH_PATH = "../outputs/distilled/repvit_m1_gru_best_3.pth"
ONNX_FIXED = "../outputs/best_model_normal/0313/repvit_best_distilled_0313.onnx"
ONNX_DYNAMIC = "../outputs/quantized/repvit_m1_gru_int8_3.onnx"
ONNX_MIXED = "../outputs/repvit_m1_gru_mixed_int8.onnx"
ONNX_QAT = "../outputs/best_model_normal/0313/repvit_qat_distill_single_0313_3cls.onnx"

OUTPUT_PLOT_PATH = "outputs/deploy/confusion_matrix_compare_distill_3.png"
CLASS_NAMES = ["eat", "drink", "other"] #["normal", "convulsion", "vomit"] #["eat", "drink", "other"]
NUM_CLASSES = 3
HIDDEN_DIM = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===========================================

class SequenceEvaluator:
    def __init__(self):
        self.model = None
        # 1. 初始化 PyTorch
        if RUN_CONFIG['RUN_PTH']:
            print("🧬 初始化 PyTorch 引擎...")
            self.model = RepViT_GRU(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)
            if os.path.exists(PTH_PATH):
                print(f"   👉 加载 PyTorch 权重: {PTH_PATH}")
                ckpt = torch.load(PTH_PATH, map_location='cpu')
                sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                self.model.load_state_dict(sd)
            else:
                print(f"   ⚠️ 未找到 PyTorch 权重: {PTH_PATH}")
            self.model.to(DEVICE).eval()

        print("🚀 初始化 ONNX 引擎...")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        # 2. Fixed FP32
        self.sess_fixed = None
        if RUN_CONFIG['RUN_FIXED']:
            if os.path.exists(ONNX_FIXED):
                print(f"   👉 加载 Fixed FP32 模型: {ONNX_FIXED}")
                self.sess_fixed = ort.InferenceSession(ONNX_FIXED, opts, providers=['CPUExecutionProvider'])
            else:
                print(f"   ⚠️ 未找到 FP32 模型: {ONNX_FIXED}")

        # 3. Dynamic Int8
        self.sess_dynamic = None
        if RUN_CONFIG['RUN_DYNAMIC']:
            if os.path.exists(ONNX_DYNAMIC):
                print(f"   👉 加载 Dynamic Int8 模型: {ONNX_DYNAMIC}")
                self.sess_dynamic = ort.InferenceSession(ONNX_DYNAMIC, opts, providers=['CPUExecutionProvider'])
            else:
                print(f"   ⚠️ 未找到 Dynamic 模型: {ONNX_DYNAMIC}")

        # 4. Mixed PTQ
        self.sess_mixed = None
        if RUN_CONFIG['RUN_MIXED']:
            if os.path.exists(ONNX_MIXED):
                print(f"   👉 加载 Mixed PTQ 模型: {ONNX_MIXED}")
                self.sess_mixed = ort.InferenceSession(ONNX_MIXED, opts, providers=['CPUExecutionProvider'])
            else:
                print(f"   ⚠️ 未找到 Mixed 模型: {ONNX_MIXED}")

        # 5. QAT Int8
        self.sess_qat = None
        if RUN_CONFIG['RUN_QAT']:
            if os.path.exists(ONNX_QAT):
                print(f"   👉 加载 QAT Int8 模型: {ONNX_QAT}")
                self.sess_qat = ort.InferenceSession(ONNX_QAT, opts, providers=['CPUExecutionProvider'])
            else:
                print(f"   ⚠️ 未找到 QAT 模型: {ONNX_QAT}")

    def evaluate(self):
        raw_dataset = get_dataset(CFG['data_dir'], CFG['split'], CFG, is_train=False)
        data_loader = DataLoader(
            raw_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        total_len = len(raw_dataset)
        max_samples = RUN_CONFIG['MAX_SAMPLES']
        if max_samples is None: max_samples = total_len

        print(f"🚀 全面对比评估 | 计划测试: {max_samples} 样本")

        metrics = {
            'pth': 0, 'fixed': 0, 'dynamic': 0, 'mixed': 0, 'qat': 0, 'total': 0
        }

        plot_data = {
            'y_true': [],
            'pred_pth': [], 'pred_fixed': [], 'pred_dynamic': [], 'pred_mixed': [], 'pred_qat': []
        }

        pbar = tqdm(total=max_samples, desc="🎬 推理中")

        for i, batch in enumerate(data_loader):
            if i >= max_samples: break

            imgs_tensor = batch['pixel_values'].to(DEVICE)
            frames_numpy = batch['pixel_values'].squeeze(0).numpy()
            label = int(batch['labels'].item())

            metrics['total'] += 1
            plot_data['y_true'].append(label)

            # --- 0. PyTorch Inference (已修复: 逐帧循环) ---
            if self.model:
                with torch.no_grad():
                    h = None # 初始化隐状态
                    logits_pth = None
                    # 循环 16 帧
                    for f in range(CFG['seq_len']):
                        # 切片获取单帧 [1, 3, 224, 224]
                        frame = imgs_tensor[:, f, :, :, :]
                        # 前向传播 (img, h) -> (logits, h_new)
                        out = self.model(frame, h)

                        # 拆包
                        if isinstance(out, tuple):
                            logits_pth, h = out
                        else:
                            logits_pth = out # 仅防守性编程

                    # 取最后一帧的预测结果
                    if logits_pth is not None:
                        pred_pth = torch.argmax(logits_pth, dim=1).item()
                        plot_data['pred_pth'].append(pred_pth)
                        if pred_pth == label:
                            metrics['pth'] += 1

            # --- 1. Fixed ONNX ---
            if self.sess_fixed:
                h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
                for f in range(CFG['seq_len']):
                    inputs = {'input_img': frames_numpy[f:f+1], 'in_state': h}
                    logits, h = self.sess_fixed.run(None, inputs)
                pred = np.argmax(logits)
                plot_data['pred_fixed'].append(pred)
                if pred == label: metrics['fixed'] += 1

            # --- 2. Dynamic Int8 ---
            if self.sess_dynamic:
                h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
                for f in range(CFG['seq_len']):
                    inputs = {'input_img': frames_numpy[f:f+1], 'in_state': h}
                    logits, h = self.sess_dynamic.run(None, inputs)
                pred = np.argmax(logits)
                plot_data['pred_dynamic'].append(pred)
                if pred == label: metrics['dynamic'] += 1

            # --- 3. Mixed PTQ ---
            if self.sess_mixed:
                h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
                for f in range(CFG['seq_len']):
                    inputs = {'input_img': frames_numpy[f:f+1], 'in_state': h}
                    logits, h = self.sess_mixed.run(None, inputs)
                pred = np.argmax(logits)
                plot_data['pred_mixed'].append(pred)
                if pred == label: metrics['mixed'] += 1

            # --- 4. QAT Int8 ---
            if self.sess_qat:
                h = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
                for f in range(CFG['seq_len']):
                    inputs = {'input_img': frames_numpy[f:f+1], 'in_state': h}
                    logits, h = self.sess_qat.run(None, inputs)
                pred = np.argmax(logits)
                plot_data['pred_qat'].append(pred)
                if pred == label: metrics['qat'] += 1

            pbar.update(1)

        pbar.close()
        self.report(metrics)
        self.plot_matrices(plot_data)

    def report(self, m):
        total = m['total']
        if total == 0:
            print("❌ 没有测试样本")
            return

        print("\n" + "═"*70)
        print(f"📊 最终评估报告 (N={total})")
        print("═"*70)

        # 修复：打印 PyTorch 结果
        if RUN_CONFIG['RUN_PTH']:
            print(f"🔥 PyTorch (Orig)   准确率: {m['pth']/total*100:6.2f}%")

        if RUN_CONFIG['RUN_FIXED']:
            print(f"✅ Fixed (FP32)     准确率: {m['fixed']/total*100:6.2f}%")

        if RUN_CONFIG['RUN_DYNAMIC']:
            print(f"🔹 Dynamic (Int8)   准确率: {m['dynamic']/total*100:6.2f}%")

        if RUN_CONFIG['RUN_MIXED']:
            print(f"📉 Mixed (PTQ)      准确率: {m['mixed']/total*100:6.2f}%")

        if RUN_CONFIG['RUN_QAT']:
            print(f"🚀 QAT (Int8)       准确率: {m['qat']/total*100:6.2f}%")
        print("-" * 70)

    def plot_matrices(self, data):
        predictions = {}
        # 修复：加入 PyTorch 数据
        if RUN_CONFIG['RUN_PTH']: predictions['PyTorch'] = data['pred_pth']

        if RUN_CONFIG['RUN_FIXED']: predictions['Fixed FP32'] = data['pred_fixed']
        if RUN_CONFIG['RUN_DYNAMIC']: predictions['Dynamic Int8'] = data['pred_dynamic']
        if RUN_CONFIG['RUN_MIXED']: predictions['Mixed PTQ'] = data['pred_mixed']
        if RUN_CONFIG['RUN_QAT']: predictions['QAT Int8'] = data['pred_qat']

        if not predictions: return

        print("🎨 正在绘制混淆矩阵...")
        num_plots = len(predictions)
        # 自动调整画布
        fig, axes = plt.subplots(1, num_plots, figsize=(4.5 * num_plots, 4.5))

        # 确保 axes 是列表/数组，方便统一处理
        if num_plots == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        class_names = CLASS_NAMES if CLASS_NAMES else [str(i) for i in range(NUM_CLASSES)]
        y_true = data['y_true']

        idx = 0
        for name, y_pred in predictions.items():
            if not y_pred: continue

            ax = axes[idx]
            cm = confusion_matrix(y_true, y_pred, normalize='true')
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=ax, square=True, cbar=False)
            ax.set_title(name)
            idx += 1

        plt.tight_layout()
        os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
        plt.savefig(OUTPUT_PLOT_PATH)
        print(f"✅ 图表已保存: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    evaluator = SequenceEvaluator()
    evaluator.evaluate()