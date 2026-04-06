import cv2
import numpy as np
import os
import random
from ultralytics import YOLO
import albumentations as A

class ReIDDataSynthesizer:
    def __init__(self, yolo_seg_model_path, bg_dir, output_dir):
        # 加载分割模型
        self.model = YOLO(yolo_seg_model_path)
        self.bg_dir = bg_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

        # 仅保留水平翻转以保证背景真实感
        self.bg_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])

        # 类别映射字典 (45:碗, 15:猫, 16:狗)
        self.class_names = {45: 'bowl', 15: 'cat', 16: 'dog'}

    def compute_iou(self, boxA, boxB):
        """计算两个边界框的交并比，用于防遮挡检测"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def extract_assets_from_image(self, source_img_path, target_classes):
        """定向提取特定类别的前景物料"""
        img = cv2.imread(source_img_path)
        if img is None: return []

        results = self.model(img, verbose=False)
        assets = []

        for result in results:
            if result.masks is None: continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for mask, box in zip(masks, boxes):
                cls_id = int(box[5])
                if cls_id not in target_classes: continue

                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                x1, y1, x2, y2 = map(int, box[:4])

                rgba_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                rgba_img[:, :, 3] = mask * 255

                cropped_asset = rgba_img[y1:y2, x1:x2]
                assets.append({'image': cropped_asset, 'class_id': cls_id})

        return assets

    def paste_soft_blend(self, background, asset, x_offset, y_offset):
        """边缘羽化与柔和透明度混合"""
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = asset.shape[:2]

        if x_offset + fg_w > bg_w or y_offset + fg_h > bg_h:
            return background, False

        src_rgb = asset[:, :, :3]
        alpha = asset[:, :, 3]

        # 边缘羽化
        alpha_blur = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha_fg = alpha_blur.astype(np.float32) / 255.0
        alpha_fg = np.expand_dims(alpha_fg, axis=2)
        alpha_bg = 1.0 - alpha_fg

        roi_bg = background[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w].astype(np.float32)
        blended = (alpha_fg * src_rgb + alpha_bg * roi_bg).astype(np.uint8)
        background[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] = blended

        return background, True

    def process_asset_relative(self, asset_img, bg_w, bg_h, relative_width_range, y_start_ratio):
        """基于背景宽度的相对缩放"""
        orig_h, orig_w = asset_img.shape[:2]
        aspect_ratio = orig_h / orig_w

        target_width_ratio = random.uniform(relative_width_range[0], relative_width_range[1])
        new_w = int(bg_w * target_width_ratio)
        new_h = int(new_w * aspect_ratio)

        if new_h > bg_h * 0.8:
            new_h = int(bg_h * 0.8)
            new_w = int(new_h / aspect_ratio)

        asset_img_resized = cv2.resize(asset_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fg_h, fg_w = asset_img_resized.shape[:2]

        max_x = bg_w - fg_w
        min_y = int(bg_h * y_start_ratio)
        max_y = bg_h - fg_h

        if min_y > max_y: min_y = max(0, max_y)

        return asset_img_resized, fg_w, fg_h, max_x, min_y, max_y

    def _get_balanced_sequence(self, items, length):
        """
        【全新工具函数：洗牌循环分配】
        确保输入列表中的每一个元素都被尽可能均匀、满额地使用，且顺序随机，保证组合多样性。
        """
        sequence = []
        while len(sequence) < length:
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            sequence.extend(shuffled_items)
        # 截取所需长度返回
        return sequence[:length]

    def synthesize_new_data(self, catdog_assets, bowl_assets, num_images_to_generate=600):
        """核心双通道合成流水线"""
        bg_files = [os.path.join(self.bg_dir, f) for f in os.listdir(self.bg_dir) if f.endswith(('.jpg', '.png'))]
        if not bg_files:
            raise ValueError("背景池目录为空！")

        print(f"\n>>> 样本分布策略检查:")
        print(f"  - 背景数: {len(bg_files)}，每张背景预计将被使用 ~{num_images_to_generate/len(bg_files):.1f} 次")
        print(f"  - 碗素材数: {len(bowl_assets)}，每个碗预计将被使用 ~{num_images_to_generate/len(bowl_assets):.1f} 次")
        print(f"  - 猫狗素材数: {len(catdog_assets)}，每只猫狗预计将被使用 ~{num_images_to_generate/len(catdog_assets):.1f} 次")

        # 预先生成均衡且打乱的分配序列，完美保证全样本覆盖与多样性
        bg_seq = self._get_balanced_sequence(bg_files, num_images_to_generate)
        bowl_seq = self._get_balanced_sequence(bowl_assets, num_images_to_generate)
        catdog_seq = self._get_balanced_sequence(catdog_assets, num_images_to_generate)

        for i in range(num_images_to_generate):
            # 取出当前轮次的对应素材
            bg_path = bg_seq[i]
            bowl = bowl_seq[i]
            catdog = catdog_seq[i]

            bg_img = cv2.imread(bg_path)
            bg_img = self.bg_transform(image=bg_img)['image']
            bg_h, bg_w = bg_img.shape[:2]

            yolo_labels = []

            # ================= 第1步：处理并粘贴【碗】 =================
            bowl_img, b_w, b_h, b_maxx, b_miny, b_maxy = self.process_asset_relative(
                bowl['image'].copy(), bg_w, bg_h, relative_width_range=(0.08, 0.12), y_start_ratio=0.8
            )

            if b_maxx > 0 and b_maxy > 0:
                bowl_x = random.randint(0, b_maxx)
                bowl_y = random.randint(b_miny, b_maxy)

                bg_img, success_b = self.paste_soft_blend(bg_img, bowl_img, bowl_x, bowl_y)
                if success_b:
                    x_center = (bowl_x + b_w / 2.0) / bg_w
                    y_center = (bowl_y + b_h / 2.0) / bg_h
                    yolo_labels.append(f"{bowl['class_id']} {x_center:.6f} {y_center:.6f} {b_w/bg_w:.6f} {b_h/bg_h:.6f}")
                    bowl_box = [bowl_x, bowl_y, bowl_x + b_w, bowl_y + b_h]
            else:
                success_b = False

            # ================= 第2步：处理并粘贴【猫狗】 =================
            catdog_img, c_w, c_h, c_maxx, c_miny, c_maxy = self.process_asset_relative(
                catdog['image'].copy(), bg_w, bg_h, relative_width_range=(0.15, 0.18), y_start_ratio=0.68
            )

            if c_maxx > 0 and c_maxy > 0:
                best_cx, best_cy = -1, -1

                for _ in range(50):
                    cx = random.randint(0, c_maxx)
                    cy = random.randint(c_miny, c_maxy)

                    if success_b:
                        catdog_box = [cx, cy, cx + c_w, cy + c_h]
                        iou = self.compute_iou(catdog_box, bowl_box)
                        if iou < 0.3:
                            best_cx, best_cy = cx, cy
                            break
                    else:
                        best_cx, best_cy = cx, cy
                        break

                if best_cx != -1:
                    bg_img, success_c = self.paste_soft_blend(bg_img, catdog_img, best_cx, best_cy)
                    if success_c:
                        x_center = (best_cx + c_w / 2.0) / bg_w
                        y_center = (best_cy + c_h / 2.0) / bg_h
                        yolo_labels.append(f"{catdog['class_id']} {x_center:.6f} {y_center:.6f} {c_w/bg_w:.6f} {c_h/bg_h:.6f}")

            # 保存结果
            if yolo_labels:
                out_name = f"synth_data_{i:06d}"
                cv2.imwrite(os.path.join(self.output_dir, 'images', f"{out_name}.jpg"), bg_img)

                with open(os.path.join(self.output_dir, 'labels', f"{out_name}.txt"), 'w') as f:
                    f.write('\n'.join(yolo_labels))

                if (i + 1) % 100 == 0:
                    print(f"[进度] 已生成 {i + 1} 张...")

# ================= 运行示例 =================
if __name__ == "__main__":
    synthesizer = ReIDDataSynthesizer(
        yolo_seg_model_path='./yolo26x-seg.pt',
        bg_dir='./background_pool',
        output_dir='./synthesized_dataset'
    )

    dir_catdog = './source_catdog'
    dir_bowl = './source_bowl'

    catdog_assets = []
    bowl_assets = []

    print(">>> 正在提取 [猫/狗] 物料...")
    if os.path.exists(dir_catdog):
        for img_name in os.listdir(dir_catdog):
            if img_name.endswith(('.jpg', '.png')):
                assets = synthesizer.extract_assets_from_image(os.path.join(dir_catdog, img_name), target_classes=[15, 16])
                catdog_assets.extend(assets)
        print(f"✅ 成功提取 {len(catdog_assets)} 只 猫/狗。")

    print(">>> 正在提取 [碗] 物料...")
    if os.path.exists(dir_bowl):
        for img_name in os.listdir(dir_bowl):
            if img_name.endswith(('.jpg', '.png')):
                assets = synthesizer.extract_assets_from_image(os.path.join(dir_bowl, img_name), target_classes=[45])
                bowl_assets.extend(assets)
        print(f"✅ 成功提取 {len(bowl_assets)} 个 碗。")

    if catdog_assets and bowl_assets:
        print("\n>>> 开始执行双通道混合拼贴流水线...")
        # 【修改】严格设定生成数量为 600 张
        synthesizer.synthesize_new_data(catdog_assets, bowl_assets, num_images_to_generate=600)
        print("\n🎉 全部 600 张流水线执行完毕！")
    else:
        print("\n❌ 错误：猫狗或碗的物料提取为空！")