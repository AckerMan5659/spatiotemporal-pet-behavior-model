import asyncio
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import httpx
import numpy as np
import yaml
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from core.bowl_manager import BowlManager
from core.fusion_agent import FusionAgent
from core.rule_engine import RuleEngine
from core.utils import get_box_center


# ==================== 配置 ====================
# 使用正斜杠 / 避免 Linux 环境下的路径解析问题
YOLO_PATH = "checkpoints/yolo26n_petroom_cctv_bgfix_v1.pt"
ACTION_CONFIG = "config.yaml"
ACTION_MODEL_PATH = "checkpoints/repvit_best_distilled_0313.onnx"
REID_DISTANCE_THRESHOLD = 200
REID_TIME_THRESHOLD = 2.0
MAX_WORKERS = max(2, (os.cpu_count() or 4) // 4)
MAX_CONCURRENT_TASKS = 10  # 背压机制
DETECTOR_DEVICE = "cuda:0"
ACTION_PROVIDERS = ["CUDAExecutionProvider"]


# ==================== 动作识别模型 ====================
class RepViTGRURecognizer:
    def __init__(self, cfg_path: str, model_path: Optional[str] = None):
        if ort is None:
            raise ImportError("Need onnxruntime to load RepViT_GRU ONNX model")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        mode = cfg.get("mode", "normal")
        task_cfg = cfg["task_configs"][mode]
        output_model_name = task_cfg.get("output_model_name", ACTION_MODEL_PATH)

        self.class_names = [name.lower() for name in task_cfg["class_names"]]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.model_path = model_path or output_model_name
        self.hidden_dim = 256
        self.context_expansion = 0.25
        self.imgsz = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._init_onnx()

    def _init_onnx(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=ACTION_PROVIDERS,
        )
        inputs = self.session.get_inputs()
        self.input_names = [item.name for item in inputs]
        self.img_input_name = inputs[0].name
        self.is_gru = len(inputs) >= 2
        if self.is_gru:
            self.state_input_name = inputs[1].name
            shape = inputs[1].shape
            self.onnx_state_rank = len(shape)
            self.hidden_dim = shape[-1]
        else:
            self.onnx_state_rank = 0

    def preprocess(self, frame, box):
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        w_box, h_box = x2 - x1, y2 - y1
        pad_w = int(w_box * self.context_expansion / 2)
        pad_h = int(h_box * self.context_expansion / 2)

        ex_x1 = max(0, x1 - pad_w)
        ex_y1 = max(0, y1 - pad_h)
        ex_x2 = min(w_img, x2 + pad_w)
        ex_y2 = min(h_img, y2 + pad_h)
        crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
        if crop.size == 0:
            return None

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.imgsz)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = (img - self.mean) / self.std
        return img.astype(np.float32)

    def get_class_index(self, class_name: str):
        return self.class_to_idx.get(class_name.lower())

    def get_class_prob(self, probs, class_name: str, default: float = 0.0):
        idx = self.get_class_index(class_name)
        if probs is None or idx is None or idx >= len(probs):
            return default
        return float(probs[idx])

    def predict(self, frame, box, prev_state=None, seq_idx=0):
        del seq_idx
        inp = self.preprocess(frame, box)
        if inp is None:
            return None, prev_state

        inputs = {self.img_input_name: inp}
        if self.is_gru:
            if prev_state is None:
                shape = (1, 1, self.hidden_dim) if self.onnx_state_rank == 3 else (1, self.hidden_dim)
                prev_state = np.zeros(shape, dtype=np.float32)
            inputs[self.state_input_name] = prev_state

        outputs = self.session.run(None, inputs)
        if self.is_gru:
            logits, next_state = outputs
        else:
            logits = outputs[0]
            next_state = prev_state

        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs[0], next_state


# ==================== 全局模型实例 ====================
class ModelManager:
    def __init__(self):
        self.recognizer = None
        self.rule_engine = None

    def load(self):
        self.recognizer = RepViTGRURecognizer(
            cfg_path=ACTION_CONFIG,
            model_path=ACTION_MODEL_PATH,
        )
        self.rule_engine = RuleEngine()


model_manager = ModelManager()
task_results = TTLCache(maxsize=100, ttl=3600)  # 最多100个任务，1小时TTL
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    model_manager.load()
    yield
    # 关闭时清理
    executor.shutdown(wait=True)


app = FastAPI(title="Pet Behavior Inference API", lifespan=lifespan)


# ==================== 数据模型 ====================
class InferRequest(BaseModel):
    video_url: HttpUrl


class TaskResponse(BaseModel):
    task_id: str
    status: str


class AgentState(BaseModel):
    agent_id: int
    cls_id: int
    state: str
    eating_duration: float
    drinking_duration: float
    probs: Optional[Dict[str, float]] = None
    log_msg: Optional[str] = None


class FrameResult(BaseModel):
    frame_idx: int
    timestamp: float
    agents: List[AgentState]


class InferResult(BaseModel):
    task_id: str
    status: str
    total_frames: int
    fps: float
    results: List[FrameResult]
    error: Optional[str] = None


# ==================== 推理核心 ====================
def process_video(video_path: str, task_id: str):
    """视频推理核心逻辑"""
    cap = None
    try:
        # [方案 1] 在任务内部实例化 YOLO，彻底避免跨任务的追踪器状态污染
        local_detector = YOLO(YOLO_PATH)
        local_detector.to(DETECTOR_DEVICE)

        # [方案 2] 双重保险：明确重置追踪器内部状态，防止 GCMC 尝试引用前一个视频的特征
        if hasattr(local_detector, 'predictor') and local_detector.predictor:
            local_detector.predictor.trackers = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        bowl_manager = BowlManager(update_interval=30)
        agents = {}
        id_map = {}
        frame_idx = 0
        frame_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # YOLO 检测
            results = local_detector.track(
                frame, persist=True, verbose=False, iou=0.5, conf=0.35, device=DETECTOR_DEVICE
            )[0]

            # 更新碗管理器
            bowl_manager.update(frame, frame_idx, results)
            bowl_boxes, bowl_types = bowl_manager.get_info()

            active_real_ids = []

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                assigned_real_ids = set()

                for box, raw_id, cls_id in zip(boxes, track_ids, clss):
                    raw_id = int(raw_id)
                    cls_id = int(cls_id)
                    if cls_id == 0:
                        continue

                    # Re-ID
                    real_id = raw_id
                    if raw_id in id_map:
                        real_id = id_map[raw_id]
                    else:
                        center = get_box_center(box)
                        best_match = None
                        min_dist = float('inf')
                        for existing_id, agent in agents.items():
                            if agent.cls_id != cls_id:
                                continue
                            if existing_id in assigned_real_ids:
                                continue
                            if (current_time - agent.last_update_time) > REID_TIME_THRESHOLD:
                                continue
                            if agent.last_box is not None:
                                last_center = get_box_center(agent.last_box)
                                dist = np.linalg.norm(center - last_center)
                                if dist < REID_DISTANCE_THRESHOLD and dist < min_dist:
                                    min_dist = dist
                                    best_match = existing_id
                        if best_match is not None:
                            real_id = best_match
                            id_map[raw_id] = real_id
                        else:
                            id_map[raw_id] = raw_id

                    assigned_real_ids.add(real_id)
                    active_real_ids.append(real_id)

                    # Agent 更新
                    if real_id not in agents:
                        agents[real_id] = FusionAgent(real_id, cls_id)

                    agents[real_id].update(
                        frame, box,
                        bowl_boxes, bowl_types,
                        model_manager.rule_engine, model_manager.recognizer,
                        current_time
                    )

            # 收集当前帧结果
            frame_agents = []
            for real_id in active_real_ids:
                agent = agents[real_id]
                # 适配修改后的 get_info (返回5个值)
                state, timers, _, probs, log_msg = agent.get_info()
                
                # 转换概率分布为可读字典
                prob_dict = None
                if probs is not None:
                    prob_dict = {
                        name: float(probs[idx]) 
                        for name, idx in model_manager.recognizer.class_to_idx.items()
                        if idx < len(probs)
                    }

                frame_agents.append(AgentState(
                    agent_id=real_id,
                    cls_id=agent.cls_id,
                    state=state,
                    eating_duration=timers.get('EATING', 0.0),
                    drinking_duration=timers.get('DRINKING', 0.0),
                    probs=prob_dict,
                    log_msg=log_msg if log_msg else None
                ))

            frame_results.append(FrameResult(
                frame_idx=frame_idx,
                timestamp=current_time,
                agents=frame_agents
            ))

            frame_idx += 1

        # 保存结果
        task_results[task_id] = InferResult(
            task_id=task_id,
            status="completed",
            total_frames=frame_idx,
            fps=fps,
            results=frame_results
        ).model_dump()

    except Exception as e:
        task_results[task_id] = InferResult(
            task_id=task_id,
            status="failed",
            total_frames=0,
            fps=0.0,
            results=[],
            error=str(e)
        ).model_dump()
    finally:
        if cap is not None:
            cap.release()


async def download_video(url: str) -> str:
    """异步下载视频到临时文件"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", str(url)) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                temp_file.write(chunk)
    temp_file.close()
    return temp_file.name


# ==================== API 端点 ====================
@app.post("/infer", response_model=TaskResponse)
async def create_inference_task(request: InferRequest):
    """创建推理任务"""
    async with task_semaphore:
        task_id = str(uuid.uuid4())
        task_results[task_id] = {"status": "downloading"}
        asyncio.create_task(download_and_process(task_id, str(request.video_url)))
        return TaskResponse(task_id=task_id, status="downloading")


async def download_and_process(task_id: str, video_url: str):
    """后台下载并推理"""
    video_path = None
    try:
        video_path = await download_video(video_url)
        task_results[task_id] = {"status": "processing"}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, process_video, video_path, task_id)
    except Exception as e:
        task_results[task_id] = InferResult(
            task_id=task_id,
            status="failed",
            total_frames=0,
            fps=0.0,
            results=[],
            error=f"下载失败: {str(e)}"
        ).model_dump()
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass


@app.get("/result/{task_id}", response_model=InferResult)
async def get_inference_result(task_id: str):
    """获取推理结果"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="任务不存在")

    result = task_results[task_id]
    status = result.get("status", "processing")

    if status in ["downloading", "processing"]:
        return InferResult(
            task_id=task_id,
            status=status,
            total_frames=0,
            fps=0.0,
            results=[]
        )

    return InferResult(**result)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.recognizer is not None,
        "active_tasks": len(task_results),
        "max_workers": MAX_WORKERS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090, workers=1)