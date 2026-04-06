import os
import cv2
import math
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================= 配置参数 =================
INPUT_DIR = r"D:\Desktop\PET1\cropped_dataset\abnormal_3"     # 您的原始資料路徑
OUTPUT_BASE_DIR = r"D:\Desktop\PET1\cropped_dataset\abnormal_3_cropped"    # 输出基础路径 (代码会自动创建 train 和 val)
YOLO_MODEL_PATH = "yolo/yolo12x.pt"
EXPAND_RATIO = 1.25                              # 邊框擴大係數 (1 + 25%)
FRAMES_PER_CLIP = 16                             # 每個序列保留的影格數
MAX_CLIP_DURATION = 4.9                          # 為了保證嚴格「小於5秒」，設定最大切分時長為 4.9 秒
TARGET_CLASSES = [15, 16]                          # 限定檢測的類別 ID (1和2代表您的貓狗類別)

# 标签映射字典：将文件夹名称映射为对应的数字标签
LABEL_MAP = {
    "convulsion": 1,
    "vomit": 2
}
# ============================================

def check_gpu():
    """检查并打印 GPU 使用情况"""
    print("-" * 40)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ 成功检测到 GPU: {device_name}")
        return 'cuda:0'
    else:
        print("❌ 未检测到可用的 GPU，将使用 CPU 进行推理。")
        return 'cpu'
    print("-" * 40)

def get_clip_segments(duration):
    """根据视频总时长计算切分段的起始和结束时间（秒）"""
    if duration < 5.0:
        return [(0.0, duration)]

    num_clips = math.ceil(duration / MAX_CLIP_DURATION)
    if num_clips <= 4:
        clip_duration = duration / num_clips
        return [(i * clip_duration, (i + 1) * clip_duration) for i in range(num_clips)]
    else:
        num_clips = 4
        clip_duration = MAX_CLIP_DURATION
        interval = (duration - clip_duration) / (num_clips - 1)
        return [(i * interval, i * interval + clip_duration) for i in range(num_clips)]

def process_video(video_path, class_name, class_counters, model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return

    duration = total_frames / fps
    segments = get_clip_segments(duration)

    # 内存缓冲区：存储当前视频所有片段的信息和图片数据
    local_seqs = {}
    target_frames_dict = {}

    for i, (start_time, end_time) in enumerate(segments):
        start_frame = min(int(start_time * fps), total_frames - 1)
        end_frame = min(int(end_time * fps), total_frames - 1)

        if end_frame <= start_frame:
            continue

        frame_indices = np.linspace(start_frame, end_frame - 1, FRAMES_PER_CLIP, dtype=int)

        # 初始化这个片段的数据结构
        local_seqs[i] = {
            'frames': [None] * FRAMES_PER_CLIP,
            'discard': False,   # 是否因为出现多个目标被丢弃
            'filled': 0         # 已存入的有效图片数
        }

        for idx, f_idx in enumerate(frame_indices):
            if f_idx not in target_frames_dict:
                target_frames_dict[f_idx] = []
            target_frames_dict[f_idx].append((i, idx))

    if not target_frames_dict:
        cap.release()
        return

    max_target_frame = max(target_frames_dict.keys())
    current_frame_idx = 0
    last_valid_box = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in target_frames_dict:
            # 过滤掉已经被判定丢弃的片段，不再浪费时间裁剪
            active_tasks = [(seq_i, idx) for seq_i, idx in target_frames_dict[current_frame_idx] if not local_seqs[seq_i]['discard']]

            if active_tasks:
                img_h, img_w = frame.shape[:2]
                results = model(frame, classes=TARGET_CLASSES, verbose=False)
                boxes = results[0].boxes

                # ================= 核心判定逻辑 =================
                if len(boxes) > 1:
                    # 发现多目标！把需要用到这一帧的所有片段全部打上“废弃”标记
                    for seq_i, _ in active_tasks:
                        local_seqs[seq_i]['discard'] = True
                else:
                    # 正常情况：1个目标或没检测到目标
                    crop_box = None
                    if len(boxes) == 1:
                        best_box = boxes[0].xyxy[0].cpu().numpy()
                        crop_box = best_box
                        last_valid_box = best_box
                    elif last_valid_box is not None:
                        crop_box = last_valid_box

                    if crop_box is not None:
                        x1, y1, x2, y2 = crop_box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w, h = x2 - x1, y2 - y1

                        new_w = w * EXPAND_RATIO
                        new_h = h * EXPAND_RATIO

                        nx1 = max(0, int(cx - new_w / 2))
                        ny1 = max(0, int(cy - new_h / 2))
                        nx2 = min(img_w, int(cx + new_w / 2))
                        ny2 = min(img_h, int(cy + new_h / 2))

                        cropped_frame = frame[ny1:ny2, nx1:nx2]
                    else:
                        cropped_frame = frame

                    # 存入缓冲区
                    for seq_i, idx in active_tasks:
                        if not local_seqs[seq_i]['discard']:
                            local_seqs[seq_i]['frames'][idx] = cropped_frame
                            local_seqs[seq_i]['filled'] += 1

        current_frame_idx += 1
        if current_frame_idx > max_target_frame:
            break

    cap.release()

    # ================= 延迟保存逻辑 =================
    label_val = LABEL_MAP.get(class_name.lower(), -1)

    # 遍历缓冲区，只保存完美无暇的片段
    for i, seq_data in local_seqs.items():
        if not seq_data['discard'] and seq_data['filled'] == FRAMES_PER_CLIP:

            # 分配全局序号，确定 Train/Val 分组 (保证严格的 8:2)
            seq_idx = class_counters.get(class_name, 0)
            subset_name = "train" if (seq_idx % 10) < 8 else "val"

            seq_dir = Path(OUTPUT_BASE_DIR) / subset_name / f"{class_name}_{seq_idx}"
            seq_dir.mkdir(parents=True, exist_ok=True)

            # 把内存里的 16 张图写入硬盘
            for idx, img in enumerate(seq_data['frames']):
                img_filename = seq_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(img_filename), img)

            # 写入数字标签
            with open(seq_dir / "label.txt", "w", encoding="utf-8") as f:
                f.write(str(label_val))

            # 只有真正写入硬盘了，序号才会 +1
            class_counters[class_name] = seq_idx + 1


def main():
    device = check_gpu()
    print("正在加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    print("-" * 40)

    input_path = Path(INPUT_DIR)
    class_counters = {}

    print("开始处理视频...")
    for file_path in input_path.rglob('*'):
        if file_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            relative_parts = file_path.relative_to(input_path).parts
            if len(relative_parts) > 0:
                class_name = relative_parts[0]
                print(f"处理中: {file_path.name}")
                process_video(file_path, class_name, class_counters, model)

    print("=" * 40)
    print("🎉 处理完成！最终得到清洗后的纯净数据集:")
    for cls_name, total_seqs in class_counters.items():
        train_count = sum(1 for i in range(total_seqs) if (i % 10) < 8)
        val_count = total_seqs - train_count
        print(f"类别 [{cls_name}]: 共 {total_seqs} 个纯净序列 (Train: {train_count}, Val: {val_count})")

if __name__ == "__main__":
    main()