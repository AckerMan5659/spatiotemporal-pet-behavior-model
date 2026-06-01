# Train_New — 7-Class Hierarchical Pipeline

基于新版 `merged_dataset` (7 类, 每样本 16 帧 + `label.txt`) 重新搭建的
**端到端训练 → 蒸馏 → 量化 → 单帧导出** 流水线。

## 标签与分层目标

| Label | Class | Bucket |
|-------|-------|--------|
| 0 | other | Normal |
| 1 | eat | Normal / Ingestion |
| 2 | drink | Normal / Ingestion |
| 3 | convulsion | Abnormal |
| 4 | limp | Abnormal |
| 5 | sneeze | Abnormal |
| 6 | vomit | Abnormal |

**Target priority** (loss 与最佳模型 score 都按此加权):
- **Target 1** Normal {0,1,2} vs Abnormal {3,4,5,6} —— 最高优先级 (w=2.0)
- **Target 2** Ingestion {1,2} vs Other {0} (仅 Normal 子集) —— 次优先级 (w=1.2)
- **Target 3** 7-way fine cross-entropy —— 兜底 (w=1.0)

Best-model 选择采用 `score = 10·Acc_T1 + 3·Acc_T2 + 1·Acc_T3`, 保证 T1 始终主导。

## 文件清单

| 文件 | 阶段 | 说明 |
|------|------|------|
| `config.yaml` | — | 全流程统一配置 (数据集 / 分层权重 / 各阶段超参) |
| `dataset_7cls.py` | 数据 | 读取 `merged_dataset/{split}/{sample}/label.txt`, 16 帧固定, WeightedRandomSampler |
| `hierarchical_loss.py` | 数据 | `HierarchicalPriorityLoss` + `hierarchical_predictions` |
| `swin_wrapper.py` | Teacher | Swin3D-B 包装器 (默认 7 类) |
| `train_finetune_7cls.py` | Phase 1 | Teacher 微调 (Swin3D-B + EMA + 分层 Loss + Early Stop) |
| `distill_7cls.py` | Phase 2 | Swin3D-B → RepViT-GRU 蒸馏 (特征 + KL + 分层 GT) |
| `train_qat_7cls.py` | Phase 3 | QAT + 同构自蒸馏, 导出 B*T 并行 ONNX (评估用) |
| `export_onnx_7cls.py` | Phase 4 | 单帧 + GRUCell 状态接口的最终部署 ONNX |

## 链路 (与 `train_weight_README.md` 对齐)

```
config.yaml
   │
   ▼
train_finetune_7cls.py  ──►  outputs_new/teacher_finetuned_7cls/teacher_best_7cls_ema.pth
   │
   ▼
distill_7cls.py         ──►  outputs_new/distilled_7cls/repvit_best_distilled_7cls_ema.pth
   │
   ▼
train_qat_7cls.py       ──►  outputs_new/qat_7cls/repvit_qat_distill_7cls.pth (+ parallel onnx)
   │
   ▼
export_onnx_7cls.py     ──►  outputs_new/export_7cls/repvit_qat_distill_single_7cls.onnx 🌟
```

## 使用

```bash
cd train_new

# Phase 1: Teacher finetune
python train_finetune_7cls.py --config config.yaml

# Phase 2: Distill to RepViT-GRU
python distill_7cls.py --config config.yaml

# Phase 3: QAT
python train_qat_7cls.py --config config.yaml

# Phase 4: streaming export
python export_onnx_7cls.py --config config.yaml --source qat
# 或者从 FP32 蒸馏权重导出 (不带 INT8 假量化节点)
python export_onnx_7cls.py --config config.yaml --source fp32
```

## 与旧版 `train/` 的差异

| 维度 | 旧版 `train/` | 新版 `train_new/` |
|------|---------------|--------------------|
| 数据集结构 | 旧的 normal_3 / abnormal_5 各自一套 | 合并 `merged_dataset`, 单一 7 类 |
| 类别识别 | 依赖文件夹名关键词 | 直接读 `label.txt`, 关键词作为兜底 |
| 训练目标 | 两个独立模型 (3-cls / 5-cls) | 单一 7-cls 模型 + 分层优先级 Loss |
| Sampler | 静态硬平衡截断 | `WeightedRandomSampler` + class_weight |
| Best 模型选择 | 单纯 Acc | 分层 score (T1 主导) |
| Loss | 2-level | 3-level (Normal/Abnormal → Ingestion/Other → 7-way) |
| 部署目标 | 3 类 / 5 类 ONNX | 7 类单一流式 ONNX |

注: 旧脚本 (`train/*.py`) 全部保留, 未做任何修改。
