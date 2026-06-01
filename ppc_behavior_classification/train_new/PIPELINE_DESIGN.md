# 7-Class 分层训练管道设计与参数优化总结

## 一、背景：从 3/5 类迁移到 7 类的挑战

| 维度 | 旧版 | 新版 |
|------|------|------|
| 训练数据规模 | normal_3 + abnormal_5 两套独立数据集 | 合并 `merged_dataset` 4435 train / 1098 val |
| 模型数量 | 2 个独立模型 (3-cls + 5-cls) | 1 个 7-cls 统一模型 |
| 最小类 | ~400+ | **limp 293**（仅 other 的 18.9%） |
| 类间相似性 | 正常 vs 异常差异明显 | 7 类中 eat/drink、sneeze/vomit 视觉相近 |
| 核心风险 | 欠拟合（少 epoch）、过拟合（少数类背诵）同时存在 |

---

## 二、分层目标设计（核心创新）

```
T1 (最高优先级, w=1.8):  Normal {0,1,2}  vs  Abnormal {3,4,5,6}
T2 (次优先级,   w=1.2):  Ingestion {1,2} vs  Other {0}          ← 仅 Normal 子集
T3 (兜底,       w=1.2):  7-way Fine-grained CE
```

**设计动机**：7 维 logits 上通过 softmax 子集合并出 P(异常) / P(吃喝|正常)，
让一个模型同时优化三个目标，且高优先目标主导梯度方向。

**最佳模型评分**：`score = 10·Acc_T1 + 3·Acc_T2 + 1·Acc_T3`
→ 保证选模型时 T1（安全关键的正常/异常识别）始终主导。

---

## 三、防欠拟合措施

### 3.1 训练时长调整

| 阶段 | 旧 epoch | 新 epoch | 原因 |
|------|---------|---------|------|
| Teacher 微调 | 35 | **50** | 7 类决策边界更复杂 |
| 知识蒸馏 | 50 | **70** | Student 容量小，学习更慢 |
| QAT | 20 | **30** | FakeQuant Observer 收敛更慢 |

### 3.2 LR Warmup + Cosine 调度（WarmupCosineLR）

```
Phase 1 (Teacher):  warmup 3 epochs → cosine → min_lr 5e-7
Phase 2 (Distill):  warmup 3 epochs → cosine → min_lr 1e-6
Phase 3 (QAT):      warmup 2 epochs → cosine → min_lr 1e-7
```

- 按 **step** 粒度调度（非 epoch），更平滑
- warmup 期间 backbone 冻结 1 epoch（teacher 微调），防 head 大梯度污染 K400 权重
- 余弦末端更低 min_lr 保证充分收敛

### 3.3 KD 软化程度增强

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| KD temperature | 4.0 | **5.0** | 7 类软标签分布更宽，需更软 |
| Distill `beta_kd` | 8.0 | **6.0** | 给 hier GT 更大优化空间 |
| Distill `alpha_feat` | 1.0 | **0.6** | RepViT 难精确模仿 Swin 7-cls 特征 |
| `w_t3_fine` | 1.0 | **1.2** | 7-way CE 本身更难，增加梯度比重 |
| QAT `alpha_kd` | 0.85 | **0.7** | 同上，给 hier GT 更多权重 |

### 3.4 Patience 与 Early Stop 扩展

| 阶段 | 旧 patience | 新 patience |
|------|-------------|-------------|
| Teacher | 7 | **12** |
| Distill | 无 | **15**（新增） |
| QAT | — | — |

7 类初期 val 波动大，patience 过小会导致提前截断未充分收敛的训练。

---

## 四、防过拟合措施

### 4.1 数据增强增强

| 参数 | 旧值 | 新值 |
|------|------|------|
| scale 范围 | 0.9–1.1 | **0.85–1.15** |
| 旋转角度 | 15° | **20°** |
| 亮度/对比度抖动 | 0.10 | **0.18** |
| 饱和度 | 无 | **0.20**（新增） |
| 色相 | 无 | **±0.05**（新增） |
| RandomErasing | 无 | **p=0.25, scale 2%-20%**（新增） |

→ 增强的意义：类间视觉相似度越高，需要更强扰动迫使模型学习语义而非纹理。

### 4.2 MixUp / CutMix（视频级）

```python
# 配置
prob = 0.5          # 50% 的 batch 做混合
switch_prob = 0.5   # MixUp 与 CutMix 各一半
alpha = 0.2         # Beta(0.2, 0.2) — 轻混合
cutmix_alpha = 1.0  # Beta(1.0, 1.0) — 中等混合
```

- **仅在 Phase 1 (Teacher) 和 Phase 2 (Distill) 启用**
- Phase 3 (QAT) 不开：FakeQuant 与软标签互相干扰
- Distill 中 **Teacher 仍看原图**，Student 看混合图 → Teacher 给出更高质量软标签
- 配合 `hierarchical_soft_loss` 支持 T1/T2/T3 全链路软标签计算

### 4.3 正则化增强

| 参数 | 旧值 | 新值 |
|------|------|------|
| `label_smoothing` | 0.10 | **0.15** |
| `focal_gamma` | 0.0（关闭） | **1.0**（启用） |
| Teacher `drop_path` | 0.2 | **0.3** |
| Distill `weight_decay` | 0.05 | **0.08** |
| Teacher `weight_decay` | 0.10 | **0.05**（配合 DropPath 已够） |
| Teacher EMA decay | 0.995 | **0.9995** |
| Distill EMA decay | 0.995 | **0.999** |

**Focal Loss** (`gamma=1.0`) 的作用：对 limp (293 samples) 等难分类自动增加 loss 权重，
比固定 class_weight 更动态，不依赖先验统计。

### 4.4 WeightedRandomSampler + class_weight 双重平衡

```
class_weight 策略:  w_i = 1 / count_i,  归一化到均值为 1
Sampler 加权:       abnormal 类 ×1.2，ingestion 类 ×1.0，other 类 ×0.8
```

- `class_weight` 影响 **loss 计算**（每个样本的 gradient 贡献）
- `Sampler` 影响 **batch 组成**（少数类更频繁出现）
- 两者协同：Sampler 保证 limp 每个 epoch 被多看，class_weight 保证其 loss 更大

---

## 五、QAT 阶段专项优化

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `freeze_bn_ratio` | 0.7 | **0.6** | BN 统计更早冻结，防 7 类 QAT 后期漂移 |
| 量化跳过层 | backbone stem/stage 0-1, gru, fc, gap | 同上 | 浅层和时序模块 FP32 |

**QAT 自蒸馏设计**：FP32 ema 权重既是 Student 的浮点初始化，也作为同构 Teacher 进行 KL 蒸馏
→ 防止 INT8 量化降点（FP32 → INT8 精度损失可控制在 1-2% 以内）。

---

## 六、各阶段输出产物链路

```
merged_dataset (train 4435 / val 1098, 16 frames + label.txt)
        │
        ▼ Phase 1: train_finetune_7cls.py
outputs_new/teacher_finetuned_7cls/teacher_best_7cls_ema.pth    (~371MB, Swin3D-B)
        │
        ▼ Phase 2: distill_7cls.py
outputs_new/distilled_7cls/repvit_best_distilled_7cls_ema.pth   (~20MB, RepViT-GRU FP32)
        │
        ▼ Phase 3: train_qat_7cls.py
outputs_new/qat_7cls/repvit_qat_distill_7cls.pth                (~22MB, QAT with FakeQuant)
outputs_new/qat_7cls/repvit_qat_distill_7cls.onnx               (评估用, B×T 并行)
        │
        ▼ Phase 4: export_onnx_7cls.py
outputs_new/export_7cls/repvit_qat_distill_single_7cls.onnx     🌟 最终部署物
        输入: input_img[1,3,224,224] + in_state[1,256]
        输出: output_logits[1,7]    + out_state[1,256]
```

---

## 七、超参对比速查表

| 参数 | Phase 1 Teacher | Phase 2 Distill | Phase 3 QAT |
|------|----------------|-----------------|-------------|
| epochs | 50 | 70 | 30 |
| batch_size | 4 (accum×8=32) | 16 | 8 |
| base LR | 3e-5 | 2.5e-4 | 3e-5 |
| min LR | 5e-7 | 1e-6 | 1e-7 |
| warmup epochs | 3 | 3 | 2 |
| weight_decay | 0.05 | 0.08 | 1e-4 |
| EMA decay | 0.9995 | 0.999 | — |
| patience | 12 | 15 | — |
| KD temp | — | 5.0 | 5.0 |
| MixUp | ✅ | ✅ | ❌ |
| drop_path | 0.3 | — | — |
| label_smoothing | 0.15 | 0.15 | 0.15 |
| focal_gamma | 1.0 | 1.0 | 1.0 |

---

## 八、已知限制与建议

1. **无独立测试集**：当前只有 train/val；上线前建议从 val 再切 20% 作为 holdout test。
2. **limp 样本最少 (293)**：若最终 T3 召回率仍低，可单独对 limp 做离线数据增强（时间插值扩充）。
3. **Swin3D-B Teacher 体积大 (~371MB)**：训练服务器显存 ≥16GB 建议；batch=4+accum=8 可在 12GB 上运行。
4. **QAT ONNX 兼容性**：最终 streaming ONNX 使用 opset 18，部署端 OnnxRuntime ≥ 1.15。
5. **MixUp 对 T2 的影响**：混合后 eat/drink 边界变模糊，T2 准确率可能在训练期略低于无 MixUp，
   但泛化性更好；若 val T2 明显下降可将 `mixup.prob` 调至 0.3。
