import os
import torch
from ultralytics import YOLO

def _normalize_device(dev):
    """
    將 "auto", 0, "0", "cuda" 等輸入標準化為 "cpu" 或 "cuda:0"。
    """
    if dev in (None, "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(dev, int):
        if torch.cuda.is_available() and dev < torch.cuda.device_count():
            return f"cuda:{dev}"
        else:
            return "cpu"
    if isinstance(dev, str):
        s = dev.strip().lower()
        if s == "cuda":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if s.isdigit():
            if torch.cuda.is_available() and int(s) < torch.cuda.device_count():
                return f"cuda:{s}"
            else:
                return "cpu"
        if "cuda" in s and not torch.cuda.is_available():
            print(f"⚠️ 警告: 请求 {s} 但 CUDA 不可用, 已回退到 'cpu'")
            return "cpu"
        return s
    return "cpu"

class YOLODetector:
    """
    (PyTorch 版本) 封裝 YOLO v11/v13
    """
    def __init__(self, model_path="yolov12n.pt", conf=0.3, iou=0.5, imgsz=640, device="auto", tracker=None, classes=None):
        self.device = _normalize_device(device) # e.g., "cpu" or "cuda:0"

        # 檢查 .pt 文件是否存在
        is_incompatible = False
        # if "11" in model_path or "13" in model_path:
        #     is_incompatible = True

        file_exists = os.path.exists(model_path) and model_path.endswith(".pt")

        if not file_exists or is_incompatible:

            if is_incompatible and file_exists:
                print(f"⚠️ 警告: {model_path} 與您安裝的 'ultralytics' 庫不兼容。")
                print(f"   (根本原因: 缺少 'DSC3k2' 等自定義模塊)。")
                print(f"   -> 將強制回退到兼容的 yolov8n.pt。")
                auto_model = "yolov8n.pt"
            else:
                print(f"⚠️ 模型文件 {model_path} 不是有效的 .pt 文件，將回退到 yolov8n.pt。")
                auto_model = "yolov8n.pt"

            print(f"⬇ 正在加載: {auto_model}")
            self.model = YOLO(auto_model) # 加載兼容的 v8 模型

        else:
            self.model = YOLO(model_path)
            print(f"✅ 成功加载本地模型: {model_path}")

        try:
            self.model.to(self.device)
            print(f"✅ PyTorch 模型已成功移動到: {self.device}")
        except Exception as e:
            print(f"⚠️ 警告：將 PyTorch 模型移動到 {self.device} 時出錯: {e}")

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.tracker_cfg = tracker

        if self.tracker_cfg:
            print(f"✅ Tracker '{self.tracker_cfg}' 已啟用。 (正在嘗試在 {self.device} 上運行)")
        else:
            print("✅ Tracker 未配置 (將使用 .predict())。")

        # --- [關鍵修改] 統計 YOLO 總參數量 ---
        try:
            # 移除了 'if p.requires_grad'
            self.params = sum(p.numel() for p in self.model.model.parameters())
        except Exception as e:
            print(f"⚠️ 無法計算 YOLO 參數量: {e}")
            self.params = 0
        # --- [修改結束] ---


    def __call__(self, frame):
        if self.tracker_cfg:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                persist=True,
                tracker=self.tracker_cfg,
                classes=self.classes,
                verbose=False
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False
            )

        dets = []
        if not results or results[0].boxes is None:
            return dets

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else []
        classes = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else []
        ids = None
        try:
            ids = res.boxes.id.cpu().numpy()
        except Exception:
            pass

        for i, b in enumerate(boxes):
            dets.append({
                "xyxy": b.tolist(),
                "conf": float(confs[i]) if len(confs) > i else 1.0,
                "cls": int(classes[i]) if len(classes) > i else 0,
                "track_id": int(ids[i]) if ids is not None and ids[i] is not None else None
            })
        return dets