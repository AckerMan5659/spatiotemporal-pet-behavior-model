import os
import cv2
import time
import torch
import numpy as np
import yaml
from collections import defaultdict, deque
from tqdm import tqdm

# 导入项目组件
from detectors.yolo_detector import YOLODetector
from recognizers.gru_model import RepViT_GRU

# ✅ 尝试导入 ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# ✅ 尝试导入 thop 用于计算 FLOPs 和 参数量
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    print("⚠️ Warning: 'thop' library not found. FLOPs and Params calculation will be skipped. Install via 'pip install thop'")
    HAS_THOP = False

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class BehaviorPipeline:
    def __init__(self, cfg_path: str, model_override: str = None):
        """
        全能行为识别管道 (支持 .onnx 和 .pth，含业务逻辑修正 & 平滑处理 & 性能统计 & 计算量/参数量动态统计)
        """
        print(f"🔧 Loading configuration from: {cfg_path}")
        self.cfg = load_config(cfg_path)

        # 1. 基础配置
        self.mode = self.cfg.get("mode", "normal")
        task_cfg = self.cfg["task_configs"][self.mode]
        pipeline_cfg = self.cfg.get("pipeline", {})

        self.class_names = [c.lower() for c in task_cfg["class_names"]]
        self.frame_skip = pipeline_cfg.get("frame_skip", 1)
        self.min_box = pipeline_cfg.get("min_box_size", 10)
        self.num_classes = len(self.class_names)

        # 🔥 业务逻辑配置：查找关键类别的索引
        self.idx_map = {name: i for i, name in enumerate(self.class_names)}
        self.jump_idx = self.idx_map.get('jump', -1)
        self.active_idx = self.idx_map.get('active', -1)
        self.rest_idx = self.idx_map.get('rest', -1)

        # 平滑缓冲
        self.smooth_window = 5
        self.result_buffers = defaultdict(lambda: deque(maxlen=self.smooth_window))

        # 2. 预处理参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        # 3. 初始化 YOLO
        det_cfg = self.cfg["detector"]
        self.detector = YOLODetector(
            model_path=det_cfg["model"],
            conf=det_cfg["conf"],
            iou=det_cfg["iou"],
            imgsz=det_cfg["imgsz"],
            device=det_cfg["device"],
            tracker=det_cfg["tracker"],
            classes=det_cfg["classes"]
        )

        # 4. 确定模型路径
        if model_override:
            self.model_path = model_override
        else:
            out_dir = pipeline_cfg.get("output_dir", "outputs")
            name = task_cfg.get("output_model_name", "repvit_m1_gru.onnx")
            self.model_path = os.path.join(out_dir, name)
            if not os.path.exists(self.model_path):
                self.model_path = "outputs/distilled/repvit_m1_gru_best.pth"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found: {self.model_path}")

        # 5. 初始化引擎
        self.use_onnx = self.model_path.endswith('.onnx')
        self.gru_states = {}
        self.hidden_dim = 256

        if self.use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()

        self.context_expansion = 0.25
        self.timers = {"yolo": 0.0, "infer": 0.0, "calls": 0}

        # ✅ 统计计数器
        self.stats = {
            "yolo_runs": 0,
            "cls_inferences": 0,
            "yolo_gflops": 0.0,
            "cls_gflops": 0.0,
            "yolo_params": 0.0,
            "cls_params": 0.0
        }

        # 计算单次推理的计算量和参数量
        self._calc_model_complexity()
        self.last_labels = {}

    def _calc_model_complexity(self):
        """
        ✅✅✅ 计算模型理论计算量 (GFLOPs) 和参数量 (Params)
        注意: thop 返回的是 MACs，这里统一乘以 2 转换为业界标准的 FLOPs 宣传值
        """
        if not HAS_THOP: return
        print("🧮 Calculating model complexity (FLOPs & Params)...")

        # --- 1. 计算 Classifier 的 FLOPs & Params ---
        try:
            tmp_model = RepViT_GRU(num_classes=self.num_classes, hidden_dim=self.hidden_dim, pretrained=False)
            tmp_model.eval()

            dummy_img = torch.randn(1, 3, 224, 224)
            dummy_state = torch.randn(1, 1, self.hidden_dim)

            macs_cls, params_cls = profile(tmp_model, inputs=(dummy_img, dummy_state), verbose=False)

            self.stats["cls_gflops"] = (macs_cls * 2) / 1e9  # MACs * 2 = FLOPs
            self.stats["cls_params"] = params_cls / 1e6      # 转换为百万 (M)
            print(f"   🔹 Classifier Complexity: {self.stats['cls_gflops']:.4f} GFLOPs, Params: {self.stats['cls_params']:.2f} M")
        except Exception as e:
            print(f"   ⚠️ Failed to calc Classifier FLOPs/Params: {e}")

        # --- 2. 计算 YOLO 的 FLOPs & Params (动态分辨率) ---
        try:
            # 动态获取 config.yaml 里配置的 yolo imgsz
            det_imgsz = self.cfg.get("detector", {}).get("imgsz", 640)
            if isinstance(det_imgsz, (list, tuple)):
                h_imgsz, w_imgsz = det_imgsz[0], det_imgsz[1]
            else:
                h_imgsz, w_imgsz = det_imgsz, det_imgsz

            if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'model'):
                yolo_pt_model = self.detector.model.model
                device = next(yolo_pt_model.parameters()).device

                # 使用动态尺寸生成 dummy 数据，反映真实的算力消耗
                dummy_yolo = torch.randn(1, 3, h_imgsz, w_imgsz).to(device)
                macs_yolo, params_yolo = profile(yolo_pt_model, inputs=(dummy_yolo,), verbose=False)

                self.stats["yolo_gflops"] = (macs_yolo * 2) / 1e9  # MACs * 2 = FLOPs
                self.stats["yolo_params"] = params_yolo / 1e6
                print(f"   🔹 YOLO Complexity:      {self.stats['yolo_gflops']:.4f} GFLOPs (Input: {w_imgsz}x{h_imgsz}), Params: {self.stats['yolo_params']:.2f} M")
            else:
                self.stats["yolo_gflops"] = 6.5
                self.stats["yolo_params"] = 2.6
                print(f"   ⚠️ Cannot profile YOLO directly, assuming {self.stats['yolo_gflops']} GFLOPs, Params: {self.stats['yolo_params']} M (Estimated)")
        except Exception as e:
            self.stats["yolo_gflops"] = 6.5
            self.stats["yolo_params"] = 2.6
            print(f"   🔹 YOLO Complexity:      {self.stats['yolo_gflops']:.4f} GFLOPs, Params: {self.stats['yolo_params']:.2f} M (Estimated due to error: {e})")

    def _init_onnx(self):
        if not HAS_ONNX: raise ImportError("Need onnxruntime for .onnx models")
        print(f"🚀 Initializing ONNX Engine: {self.model_path}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(self.model_path, sess_options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        self.input_names = [i.name for i in inputs]
        self.is_gru = len(inputs) >= 2
        if self.is_gru:
            shape = inputs[1].shape
            self.onnx_state_rank = len(shape)
            self.hidden_dim = shape[-1]
            print(f"🧠 ONNX State Rank: {self.onnx_state_rank}D, Hidden: {self.hidden_dim}")
        self.img_input_name = inputs[0].name
        if self.is_gru: self.state_input_name = inputs[1].name

    def _init_pytorch(self):
        print(f"🧬 Initializing PyTorch Engine: {self.model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_model = RepViT_GRU(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            pretrained=False
        )
        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.torch_model.load_state_dict(new_sd, strict=False)
        self.torch_model.to(self.device).eval()
        self.is_gru = True

    def preprocess_batch(self, crop_list):
        if not crop_list: return None
        processed = []
        for crop in crop_list:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.transpose(2, 0, 1)
            processed.append(img)
        batch_np = np.stack(processed).astype(np.float32) / 255.0
        batch_np = (batch_np - self.mean) / self.std
        return batch_np.astype(np.float32)

    def process_frame(self, frame):
        annotated = frame.copy()
        h_img, w_img = frame.shape[:2]

        t0 = time.time()
        detections = self.detector(frame)
        self.timers["yolo"] += (time.time() - t0)
        self.stats["yolo_runs"] += 1

        valid_ids = []
        crops = []
        bboxes = []
        current_ids = set()

        for det in detections:
            tid = det["track_id"]
            if tid is None: continue
            x1, y1, x2, y2 = map(int, det["xyxy"])
            if (x2-x1) < self.min_box or (y2-y1) < self.min_box: continue
            current_ids.add(tid)
            w_box, h_box = x2-x1, y2-y1
            pad_w, pad_h = int(w_box * self.context_expansion / 2), int(h_box * self.context_expansion / 2)
            ex_x1, ex_y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            ex_x2, ex_y2 = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
            crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
            if crop.size == 0: continue
            valid_ids.append(tid)
            crops.append(crop)
            bboxes.append((x1, y1, x2, y2))

        if valid_ids:
            t_infer = time.time()
            img_batch_np = self.preprocess_batch(crops)
            self.stats["cls_inferences"] += len(valid_ids)

            if self.use_onnx:
                state_list = []
                for tid in valid_ids:
                    if tid not in self.gru_states:
                        shape = (1, 1, self.hidden_dim) if self.onnx_state_rank == 3 else (1, self.hidden_dim)
                        self.gru_states[tid] = np.zeros(shape, dtype=np.float32)
                    state_list.append(self.gru_states[tid])
                axis = 1 if self.onnx_state_rank == 3 else 0
                state_batch = np.concatenate(state_list, axis=axis)
                inputs = {self.img_input_name: img_batch_np, self.state_input_name: state_batch}
                logits, next_states = self.session.run(None, inputs)
                for i, tid in enumerate(valid_ids):
                    if self.onnx_state_rank == 3:
                        self.gru_states[tid] = next_states[:, i:i+1, :]
                    else:
                        self.gru_states[tid] = next_states[i:i+1, :]
            else:
                img_tensor = torch.from_numpy(img_batch_np).to(self.device)
                state_list = []
                for tid in valid_ids:
                    if tid not in self.gru_states:
                        self.gru_states[tid] = torch.zeros(1, 1, self.hidden_dim).to(self.device)
                    state_list.append(self.gru_states[tid])
                state_tensor = torch.cat(state_list, dim=1)
                with torch.no_grad():
                    out_logits, out_states = self.torch_model(img_tensor, state_tensor)
                logits = out_logits.cpu().numpy()
                for i, tid in enumerate(valid_ids):
                    self.gru_states[tid] = out_states[:, i:i+1, :].clone()

            self.timers["infer"] += (time.time() - t_infer)
            self.timers["calls"] += 1

            probs = self.softmax(logits)
            for i, tid in enumerate(valid_ids):
                final_cls_id, final_conf = self._apply_business_logic(tid, probs[i])
                self._draw_result(annotated, tid, bboxes[i], final_cls_id, final_conf)

        self._cleanup(current_ids)
        return annotated

    def _apply_business_logic(self, tid, prob_arr):
        raw_cls_id = np.argmax(prob_arr)
        raw_conf = prob_arr[raw_cls_id]
        if raw_cls_id == self.jump_idx and self.jump_idx != -1:
            if raw_conf < 0.85:
                score_active = prob_arr[self.active_idx] if self.active_idx != -1 else 0
                score_rest = prob_arr[self.rest_idx] if self.rest_idx != -1 else 0
                if score_active > score_rest:
                    raw_cls_id = self.active_idx
                    raw_conf = score_active
                elif self.rest_idx != -1:
                    raw_cls_id = self.rest_idx
                    raw_conf = score_rest

        self.result_buffers[tid].append(raw_cls_id)
        final_cls_id = raw_cls_id
        final_conf = raw_conf

        if len(self.result_buffers[tid]) > 0:
            counts = np.bincount(list(self.result_buffers[tid]))
            voted_cls_id = np.argmax(counts)
            if voted_cls_id != raw_cls_id:
                final_cls_id = voted_cls_id
                final_conf = prob_arr[final_cls_id]

        return final_cls_id, final_conf

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _draw_result(self, img, tid, bbox, cls_id, conf):
        if cls_id < len(self.class_names):
            label = self.class_names[cls_id]
        else:
            label = f"Class_{cls_id}"
        color = (0, 255, 0)
        if label.lower() in ["eat", "drink", "vomit", "convulsion"]: color = (0, 0, 255)
        x1, y1, x2, y2 = bbox
        self.last_labels[tid] = (f"{label} {conf:.2f}", color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID:{tid} {label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_frame_track_only(self, frame):
        t0 = time.time()
        detections = self.detector(frame)
        self.timers["yolo"] += (time.time() - t0)
        self.stats["yolo_runs"] += 1

        annotated = frame.copy()
        current_ids = set()
        for det in detections:
            tid = det["track_id"]
            if tid is None: continue
            current_ids.add(tid)
            x1, y1, x2, y2 = map(int, det["xyxy"])

            if tid in self.last_labels:
                text, color = self.last_labels[tid]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID:{tid} {text}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 2)
        return annotated

    def _cleanup(self, current_ids):
        if self.timers["calls"] % 60 == 0:
            for tid in list(self.gru_states.keys()):
                if tid not in current_ids: del self.gru_states[tid]
            for tid in list(self.result_buffers.keys()):
                if tid not in current_ids: del self.result_buffers[tid]
            for tid in list(self.last_labels.keys()):
                if tid not in current_ids: del self.last_labels[tid]

def run(cfg_path, video_path, model_path=None, save_path=None):
    try:
        pipe = BehaviorPipeline(cfg_path, model_override=model_path)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    width = int(cap.get(3)); height = int(cap.get(4)); fps = cap.get(5)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if save_path else None

    pbar = tqdm(total=int(cap.get(7)))
    fps_history = deque(maxlen=30)
    frame_idx = 0; frame_skip = pipe.frame_skip

    # ✅ ✅ ✅ 修复核心代码：允许 OpenCV 窗口缩放
    cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
    # 可选：如果你希望窗口初始弹出来的时候有个默认尺寸，可以取消下面这行的注释
    # cv2.resizeWindow("Monitor", 1280, 720)
    # ✅ ✅ ✅

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            t_start = time.time()

            if frame_idx % frame_skip == 0:
                out = pipe.process_frame(frame); mode="Full"
            else:
                out = pipe.process_frame_track_only(frame); mode="Track"

            t_cost = time.time() - t_start
            fps_history.append(1.0/max(t_cost, 1e-5))

            info = f"FPS: {sum(fps_history)/len(fps_history):.1f} ({mode}) | {'ONNX' if pipe.use_onnx else 'PyTorch'}"
            cv2.putText(out, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if writer: writer.write(out)
            cv2.imshow("Monitor", out)
            if cv2.waitKey(1)==27: break
            frame_idx+=1; pbar.update(1)
    finally:
        cap.release();
        if writer: writer.release()
        cv2.destroyAllWindows()

        # 🔥🔥🔥 性能、计算量与参数量报告 🔥🔥🔥
        total_yolo_ops = pipe.stats['yolo_runs'] * pipe.stats['yolo_gflops']
        total_cls_ops = pipe.stats['cls_inferences'] * pipe.stats['cls_gflops']
        total_gflops = total_yolo_ops + total_cls_ops

        print("\n" + "="*50)
        print("📊 PERFORMANCE & COMPLEXITY REPORT")
        print("="*50)
        print(f"🔹 Time (YOLO):         {pipe.timers['yolo']:.2f} s")
        print(f"🔹 Time (Infer):        {pipe.timers['infer']:.2f} s")
        print("-" * 50)
        # YOLO 统计
        print(f"🔹 YOLO Counts:         {pipe.stats['yolo_runs']} frames")
        print(f"🔹 YOLO Unit Cost:      {pipe.stats['yolo_gflops']:.3f} GFLOPs")
        print(f"🔹 YOLO Params:         {pipe.stats['yolo_params']:.2f} M")
        print("-" * 50)
        # Classifier 统计
        print(f"🔹 Classifier Counts:   {pipe.stats['cls_inferences']} crops")
        print(f"🔹 Classifier Unit Cost:{pipe.stats['cls_gflops']:.3f} GFLOPs")
        print(f"🔹 Classifier Params:   {pipe.stats['cls_params']:.2f} M")
        print("-" * 50)
        # 整体耗算力汇总
        print(f"🟢 TOTAL COMPUTATION:   {total_gflops:.4f} GFLOPs")
        print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(root, "config.yaml")
    if not os.path.exists(cfg): cfg = "config.yaml"

    run(cfg, args.video, args.model, args.save)