# 合并数据集结构总览

## 目录结构

```
merged_dataset/
├── train/                      # 4435 个样本文件夹
│   ├── {sample_name}/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   ├── ...
│   │   ├── frame_0015.jpg      # 固定 16 帧，顺序连续
│   │   └── label.txt           # 单行，单个整数 0-6
│   └── ...
└── val/                        # 1098 个样本文件夹
    └── {sample_name}/
        └── ...（同上）
```

---

## 标签体系（7 类）

| label | 类别 | 语义 | 类型 |
|---|---|---|---|
| 0 | other | 其他正常行为（休息 / 活跃 / 跳跃 / 纯正常片段） | 正常 |
| 1 | eat | 进食 | 正常 |
| 2 | drink | 饮水 | 正常 |
| 3 | convulsion | 抽搐 / 痉挛 | 异常 |
| 4 | limp | 跛行 | 异常 |
| 5 | sneeze | 打喷嚏 | 异常 |
| 6 | vomit | 呕吐 | 异常 |

---

## 数据量与分布

| label | 类别 | train | val | train% |
|---|---|---|---|---|
| 0 | other | 1552 | 384 | 35.0% |
| 1 | eat | 585 | 145 | 13.2% |
| 2 | drink | 469 | 118 | 10.6% |
| 3 | convulsion | 548 | 136 | 12.4% |
| 4 | limp | 293 | 73 | 6.6% |
| 5 | sneeze | 454 | 112 | 10.2% |
| 6 | vomit | 534 | 130 | 12.0% |
| — | **合计** | **4435** | **1098** | — |

train : val ≈ 80 : 20，各类比例基本一致。

---

## 单样本输入格式

| 属性 | 值 |
|---|---|
| 序列长度 | 16 帧（固定，frame_0000 ~ frame_0015） |
| 图像格式 | JPG，RGB |
| 标签格式 | `label.txt`：单行单整数，`int ∈ [0, 6]` |
| 时间含义 | 一段连续视频片段的等间隔抽帧 |

---

## 训练管道关键参数

```python
NUM_CLASSES    = 7
SEQ_LEN        = 16          # 每个样本的帧数
LABEL_FILE     = "label.txt"
SPLITS         = ["train", "val"]

# 各类样本数（train），可用于计算 class_weight
CLASS_COUNTS_TRAIN = {
    0: 1552,  # other
    1:  585,  # eat
    2:  469,  # drink
    3:  548,  # convulsion
    4:  293,  # limp
    5:  454,  # sneeze
    6:  534,  # vomit
}
```

---

## 注意事项

1. **类别不平衡**：`limp`（293）仅为 `other`（1552）的 19%，建议训练时加 `class_weight` 或使用 Focal Loss。
2. **other 来源混合**：该类包含 5_cls 正常片段、SSB 采集数据、active / rest / jump 行为，特征分布较宽，模型需有足够表达力区分与异常类的边界。
3. **样本命名不统一**：`convulsion_cat_1.mp4_0`、`eat_0` 等命名混存，加载时只需按文件夹遍历，无需解析文件夹名。
4. **无独立测试集**：当前只有 train / val，若需 test 集需从 val 中再划分或另行采集。
