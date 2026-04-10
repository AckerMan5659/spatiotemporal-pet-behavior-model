import os
import sys
import platform
import urllib.parse

# 限制底層 C++ 庫的併發執行緒數，防止多核搶佔導致死鎖
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENVINO_NUM_THREADS"] = "1"

import cv2
import time
import datetime
import sqlite3
import numpy as np
import threading
import multiprocessing as mp
import logging
import argparse
import gc

# ====================================================================
# 🛡️ 階段 0: 開機自檢 (POST) 與 故障自癒 (Self-Healing)
# ====================================================================
def system_pre_flight_check():
    print("🔍 [系統自檢] 正在驗證環境完整性...")
    missing_pkgs = []

    try: import cv2
    except ImportError: missing_pkgs.append("opencv-python")
    try: import psutil
    except ImportError: missing_pkgs.append("psutil")
    try: import flask
    except ImportError: missing_pkgs.append("flask")

    if missing_pkgs:
        print(f"\n❌ [致命錯誤] 缺少核心運行庫: {', '.join(missing_pkgs)}")
        sys.exit(1)

    cpu_cores = os.cpu_count() or 4
    if cpu_cores <= 4:
        try:
            import site
            for p in site.getsitepackages():
                ov_libs = os.path.join(p, "openvino", "libs")
                gpu_dll = os.path.join(ov_libs, "openvino_intel_gpu_plugin.dll")
                if os.path.exists(gpu_dll):
                    os.rename(gpu_dll, gpu_dll + ".bak")
                    print("🔧 [故障自癒] 偵測到低配硬體，已物理隔離 OpenVINO GPU 插件，規避閃退風險。")
        except Exception:
            pass

    print("✅ [系統自檢] 依賴完整，自癒程序執行完畢。")

system_pre_flight_check()

# ====================================================================
# 📦 階段 1: 安全匯入核心業務庫
# ====================================================================
try:
    import torch
    import psutil
    from flask import Flask, jsonify, Response, request, send_from_directory
    from flask_cors import CORS
except ImportError as e:
    print(f"❌ 匯入核心庫時發生異常: {e}")
    sys.exit(1)

HAS_PSUTIL = True

try:
    from waitress import serve
    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False

DB_PATH = "records.db"
KEEP_RECORDS_DAYS = 7

# ====================================================================
# 🧠 階段 2: 智能硬體探測與四象限策略工廠 (含架構感知)
# ====================================================================
class HardwareProfiler:
    @staticmethod
    def analyze():
        report = {
            "cpu_cores": os.cpu_count() or 4,
            "has_cuda": False,
            "is_aarch64": platform.machine().lower() in ['aarch64', 'arm64'],
            "cuda_device_name": "None",
            "tier": "UNKNOWN"
        }

        try:
            if torch.cuda.is_available():
                report["has_cuda"] = True
                report["cuda_device_name"] = torch.cuda.get_device_name(0)
        except: pass

        if report["has_cuda"] and report["is_aarch64"]:
            report["tier"] = "TIER_JETSON"
        elif report["has_cuda"]:
            report["tier"] = "TIER_CUDA"
        elif report["cpu_cores"] >= 6:
            report["tier"] = "TIER_HIGH"
        else:
            report["tier"] = "TIER_LOW"

        return report

class StrategyFactory:
    @staticmethod
    def get_strategy(hw_report):
        tier = hw_report["tier"]

        strategy = {
            "tier": tier,
            "use_qsv": False,
            "target_fps": 5.0,
            "imgsz": 480,
            "device": "cpu",
            "model_format": "openvino_fp32",
            "max_cams_suggested": 2,
            "static_refresh_interval": 20.0,
            "max_targets_per_cam": 3,
            "processing_mode": "sequential"
        }

        if tier == "TIER_CUDA":
            strategy.update({
                "use_qsv": True, "target_fps": 25.0, "imgsz": 640, "device": "cuda",
                "model_format": "cuda_pt", "max_cams_suggested": 8,
                "static_refresh_interval": 5.0, "max_targets_per_cam": 9,
                "processing_mode": "parallel"
            })
        elif tier == "TIER_JETSON":
            strategy.update({
                "use_qsv": False, "target_fps": 20.0, "imgsz": 640, "device": "cuda",
                "model_format": "tensorrt", "max_cams_suggested": 6,
                "static_refresh_interval": 15.0, "max_targets_per_cam": 6,
                "processing_mode": "sequential"
            })
        elif tier == "TIER_HIGH":
            strategy.update({
                "use_qsv": True, "target_fps": 15.0, "imgsz": 480, "device": "cpu",
                "model_format": "openvino_int8", "max_cams_suggested": 4,
                "static_refresh_interval": 10.0, "max_targets_per_cam": 6,
                "processing_mode": "parallel"
            })
        else: 
            strategy.update({
                "use_qsv": False, "target_fps": 5.0, "imgsz": 480, "device": "cpu",
                "model_format": "openvino_fp32", "max_cams_suggested": 2,
                "static_refresh_interval": 20.0, "max_targets_per_cam": 3,
                "processing_mode": "sequential"
            })

        return strategy

# ====================================================================
# 🗄️ 資料庫與清理執行緒
# ====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 異常錄影紀錄表
    c.execute('''CREATE TABLE IF NOT EXISTS video_records (
                                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                              cam_id INTEGER, filename TEXT, start_time DATETIME, end_time DATETIME,
                                                              trigger_action TEXT, max_confidence REAL, status TEXT)''')
    # 🔥 新增：每秒行為寬表 (八路攝影機整合為一條，極大節省空間)
    c.execute('''CREATE TABLE IF NOT EXISTS behavior_logs (
                                                              timestamp DATETIME PRIMARY KEY, 
                                                              cam0 TEXT, cam1 TEXT, cam2 TEXT, cam3 TEXT, 
                                                              cam4 TEXT, cam5 TEXT, cam6 TEXT, cam7 TEXT)''')
    conn.commit()
    conn.close()

def db_writer_thread(db_queue):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    while True:
        try:
            record = db_queue.get()
            if record is None: break
            c.execute('''INSERT INTO video_records
                         (cam_id, filename, start_time, end_time, trigger_action, max_confidence, status)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (record['cam_id'], record['filename'], record['start_time'],
                       record['end_time'], record['trigger_action'], record['max_confidence'], record['status']))
            conn.commit()
        except Exception: pass
    conn.close()

def db_cleanup_thread():
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=KEEP_RECORDS_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. 清理過期的錄影檔案
            c.execute("SELECT id, filename FROM video_records WHERE start_time < ?", (cutoff_date,))
            old_records = c.fetchall()
            for row in old_records:
                record_id, fname = row
                thumb_name = fname.replace('/videos/', '/thumbnails/').replace('\\videos\\', '\\thumbnails\\').replace('.webm', '.mp4', '.jpg')
                if os.path.exists(fname):
                    try: os.remove(fname)
                    except: pass
                if os.path.exists(thumb_name):
                    try: os.remove(thumb_name)
                    except: pass
            if old_records:
                c.execute("DELETE FROM video_records WHERE start_time < ?", (cutoff_date,))
            
            # 2. 🔥 清理過期的秒級行為日誌
            c.execute("DELETE FROM behavior_logs WHERE timestamp < ?", (cutoff_date,))
            
            conn.commit()
            conn.close()
        except Exception: pass
        time.sleep(3600) # 每小時檢查一次

# 🔥 新增：每秒行為記錄執行緒 (1秒1次，提取8路鏡頭主導行為寫入一條紀錄)
def behavior_logger_thread(shared_dict):
    while True:
        time.sleep(1.0)
        try:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [now_str]
            
            for cid in range(8):
                dom_behavior = "None"
                try:
                    data = shared_dict.get(cid, {})
                    if data.get("stats", {}).get("status") == "online":
                        active_states = data.get("active_states", {})
                        if active_states:
                            behavior_counts = {}
                            # 統計該畫面中所有寵物的最高機率行為
                            for tid, st in active_states.items():
                                best_b = "Rest"
                                max_p = 0
                                for k, v in st.get("probs", {}).items():
                                    if k in ["Eat", "Drink", "Rest", "Jump", "Act"] and v > max_p:
                                        max_p = v; best_b = k
                                behavior_counts[best_b] = behavior_counts.get(best_b, 0) + 1
                            
                            # 取數量最多的行為作為主導行為
                            if behavior_counts:
                                dom_behavior = max(behavior_counts, key=behavior_counts.get)
                except: pass
                row_data.append(dom_behavior)

            # 寫入資料庫
            conn = sqlite3.connect(DB_PATH, timeout=5)
            c = conn.cursor()
            c.execute('''INSERT OR IGNORE INTO behavior_logs 
                         (timestamp, cam0, cam1, cam2, cam3, cam4, cam5, cam6, cam7) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_data)
            conn.commit()
            conn.close()
        except Exception as e:
            pass

# ====================================================================
# ⚙️ 系統參數解析
# ====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SmartPet AI Backend (Dual-Architecture Edition)")
    parser.add_argument("--source", type=str, default="rtsp", choices=["rtsp", "local"])
    parser.add_argument("--hwaccel", type=str, default="auto", choices=["auto", "qsv", "none"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--rtsp_url_1", type=str, default="rtsp://127.0.0.1:8554/c400_sd1")
    parser.add_argument("--rtsp_url_2", type=str, default="rtsp://127.0.0.1:8554/c400_sd2")
    parser.add_argument("--rtsp_url_3", type=str, default="rtsp://127.0.0.1:8554/c400_sd3")
    parser.add_argument("--rtsp_url_4", type=str, default="rtsp://127.0.0.1:8554/c400_sd4")
    parser.add_argument("--rtsp_url_5", type=str, default="rtsp://127.0.0.1:8554/c400_sd5")
    parser.add_argument("--rtsp_url_6", type=str, default="rtsp://127.0.0.1:8554/c400_sd6")
    parser.add_argument("--rtsp_url_7", type=str, default="rtsp://127.0.0.1:8554/c400_sd7")
    parser.add_argument("--rtsp_url_8", type=str, default="rtsp://127.0.0.1:8554/c400_sd8")

    parser.add_argument("--cam_index_1", type=int, default=0)
    parser.add_argument("--cam_index_2", type=int, default=0)
    parser.add_argument("--cam_index_3", type=int, default=0)
    parser.add_argument("--cam_index_4", type=int, default=0)
    parser.add_argument("--cam_index_5", type=int, default=0)
    parser.add_argument("--cam_index_6", type=int, default=0)
    parser.add_argument("--cam_index_7", type=int, default=0)
    parser.add_argument("--cam_index_8", type=int, default=0)

    parser.add_argument("--yolo_480_int8", type=str, default="model/best_3_3_openvino_model_480_int8")
    parser.add_argument("--yolo_640_int8", type=str, default="model/best_3_3_openvino_model_640_int8")
    parser.add_argument("--yolo_480_fp32", type=str, default="model/best_3_3_openvino_model_480_fp32")
    parser.add_argument("--yolo_640_fp32", type=str, default="model/best_3_3_openvino_model_640_fp32")
    parser.add_argument("--yolo_pt", type=str, default="model/best.pt")
    parser.add_argument("--yolo_trt", type=str, default="model/best.engine")

    parser.add_argument("--num_cams", type=int, default=8)
    parser.add_argument("--max_fps", type=float, default=-1.0)
    return parser.parse_args()

def get_smart_cam_name(url, is_rtsp, index):
    if not is_rtsp: return f"USB Cam {index+1}"
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname and parsed.hostname not in ['127.0.0.1', 'localhost']:
            return f"IPC {parsed.hostname}"
        name = parsed.path.strip('/').split('?')[0].replace('_', ' ').title()
        return name if name else f"Camera {index+1}"
    except:
        return f"Camera {index+1}"

# ====================================================================
# 🎥 整合攝影機與推流經紀人
# ====================================================================
class IntegratedCamera:
    def __init__(self, source, is_rtsp=False, name="Cam", use_qsv=False):
        self.source = source
        self.is_rtsp = is_rtsp
        self.name = name
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.retry_delay = 3.0
        self.max_retry_delay = 60.0
        self.decode_ms = 0.0

        if self.is_rtsp:
            base_opts = "rtsp_transport;tcp|analyzeduration;0|probesize;32|stimeout;5000000"
            if use_qsv:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = base_opts + "|hwaccel;qsv"
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = base_opts

        self.start()

    def start(self):
        self.thread = threading.Thread(target=self.update, name=f"Read-{self.name}", daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_DSHOW)
            if not cap.isOpened():
                with self.lock:
                    self.ret = False
                    self.frame = None
                time.sleep(self.retry_delay)
                self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)
                continue
            self.retry_delay = 3.0
            while not self.stopped and cap.isOpened():
                t_decode_start = time.time()
                ret, frame = cap.read()
                current_decode_ms = (time.time() - t_decode_start) * 1000
                if not ret: break
                with self.lock:
                    self.frame = frame
                    self.ret = True
                    self.decode_ms = current_decode_ms

                time.sleep(0.005)

            with self.lock: self.ret = False
            cap.release()
            time.sleep(1)

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None: return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


# ====================================================================
# 📉 全局調速器 (CPU Governor)
# ====================================================================
def system_governor_thread(shared_dict, hardware_tier):
    if not HAS_PSUTIL: return

    current_penalty = 0.0
    is_low_tier = (hardware_tier == "TIER_LOW")

    while True:
        try:
            sys_cpu = psutil.cpu_percent(interval=1.0)

            if is_low_tier:
                if sys_cpu >= 98.0: current_penalty += 2.0
                elif sys_cpu >= 92.0: current_penalty += 1.0
                elif sys_cpu >= 86.0: current_penalty += 0.5
                elif sys_cpu <= 70.0: current_penalty -= 1.0
                elif sys_cpu <= 80.0: current_penalty -= 0.5
                current_penalty = max(0.0, min(current_penalty, 4.0))
            else:
                if sys_cpu >= 95.0: current_penalty += 1.0
                elif sys_cpu >= 90.0: current_penalty += 0.5
                elif sys_cpu <= 75.0: current_penalty -= 1.0
                elif sys_cpu <= 86.0: current_penalty -= 0.5
                current_penalty = max(0.0, min(current_penalty, 2.5))

            shared_dict["global_penalty"] = current_penalty
            shared_dict["sys_cpu"] = sys_cpu

        except Exception: pass
        time.sleep(2.0)

def get_model_path_by_strategy(imgsz, format, paths):
    if format == "cuda_pt": return paths["pt"]
    if format == "tensorrt": return paths["trt"]
    if format == "openvino_int8": return paths[f"{imgsz}_int8"]
    return paths[f"{imgsz}_fp32"]

# ====================================================================
# 🚀 方案 A：單鏡頭 Worker Process (並行架構專用)
# ====================================================================
def worker_process(cam_id, source, is_rtsp, system_strategy, model_paths, shared_dict, shared_frames, stop_event, cam_configs, db_queue):
    torch.set_num_threads(1)
    gc.disable()

    if HAS_PSUTIL:
        try:
            p = psutil.Process(os.getpid())
            if os.name == 'nt': p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except: pass

    from ultralytics import YOLO
    from core.rule_engine import RuleEngine
    from core.fusion_agent import FusionAgent
    from core.bowl_manager import BowlManager
    from modules.action_recognizer import ActionRecognizer

    tier = system_strategy["tier"]
    target_device = system_strategy["device"]
    use_qsv = system_strategy["use_qsv"]
    base_target_fps = system_strategy["target_fps"]
    static_refresh_interval = system_strategy["static_refresh_interval"]
    max_targets_per_cam = system_strategy["max_targets_per_cam"]
    model_format = system_strategy["model_format"]

    initial_cfg = cam_configs.get(cam_id, {})
    loaded_imgsz = initial_cfg.get("imgsz", system_strategy["imgsz"])
    current_model_path = get_model_path_by_strategy(loaded_imgsz, model_format, model_paths)

    try:
        detector = YOLO(current_model_path, task='detect')
    except Exception as e:
        print(f"❌ CAM-{cam_id+1} 模型載入失敗: {e}")
        return

    rule_engine = RuleEngine()
    config_path = "config.yaml"
    action_recognizer = ActionRecognizer(config_path) if os.path.exists(config_path) else None
    bowl_manager = BowlManager(30)
    agents, id_map, frame_idx = {}, {}, 0

    cam = IntegratedCamera(source, is_rtsp=is_rtsp, name=f"Cam-{cam_id+1}", use_qsv=use_qsv)
    video_writer, record_countdown = None, 0
    current_record_meta = None
    prev_frame_time, last_forced_yolo_time = time.time(), time.time()

    MOTION_RESIZE = (160, 120)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=35, detectShadows=False)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    cached_active_ids, cached_drawn_boxes, cached_bowl_boxes = [], [], []
    cached_frame_logs, cached_states_dict = [], {}
    cached_trigger_prob, cached_trigger_action = 0.0, ""
    cached_t_yolo, cached_t_rule, cached_t_action = 0.0, 0.0, 0.0

    while not stop_event.is_set():
        loop_start = time.time()
        if frame_idx % 100 == 0: gc.collect()

        global_penalty = shared_dict.get("global_penalty", 0.0)
        target_frame_time = 1.0 / max(1.0, base_target_fps - global_penalty)
        true_fps = 1.0 / (loop_start - prev_frame_time + 1e-6)
        prev_frame_time = loop_start

        cfg = cam_configs.get(cam_id, {})
        target_imgsz = cfg.get("imgsz", system_strategy["imgsz"])
        if target_imgsz != loaded_imgsz:
            try:
                new_model_path = get_model_path_by_strategy(target_imgsz, model_format, model_paths)
                print(f"🔄 CAM-{cam_id+1} 正在嘗試切換解析度至 {target_imgsz}p，加載模型: {new_model_path}")
                new_detector = YOLO(new_model_path, task='detect')
                detector = new_detector
                loaded_imgsz = target_imgsz
                print(f"✅ CAM-{cam_id+1} 解析度切換成功！")
            except Exception as e:
                print(f"❌ CAM-{cam_id+1} 切換 {target_imgsz}p 失敗 (可能缺少模型檔案): {e}")
                rollback_cfg = cam_configs.get(cam_id, {})
                rollback_cfg["imgsz"] = loaded_imgsz
                cam_configs[cam_id] = rollback_cfg

        ret, raw_frame = cam.read()
        if not ret or raw_frame is None:
            c_name = shared_dict.get(cam_id, {}).get("stats", {}).get("name", f"Camera {cam_id+1}")
            shared_dict[cam_id] = {"stats": {"status": "offline", "fps": 0, "camId": cam_id, "yoloMs": 0, "ruleMs": 0, "actionMs": 0, "decodeMs": 0, "name": c_name}, "logs": [], "active_states": {}}
            time.sleep(0.5)
            continue

        sim_time = frame_idx / 30.0
        should_run_yolo, is_forced_refresh, trigger_record_this_frame = True, False, False

        if tier == "TIER_LOW":
            current_gray = cv2.cvtColor(cv2.resize(raw_frame, MOTION_RESIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
            current_gray = cv2.blur(current_gray, (5, 5))
        else:
            current_gray = cv2.cvtColor(cv2.resize(raw_frame, MOTION_RESIZE), cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        fg_mask = bg_subtractor.apply(current_gray)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)

        if cv2.countNonZero(fg_mask) / (MOTION_RESIZE[0]*MOTION_RESIZE[1]) < 0.0001 and record_countdown == 0:
            should_run_yolo = False

        if not should_run_yolo and (int(time.time() / 2.0) % 8 == cam_id % 8) and (time.time() - last_forced_yolo_time > static_refresh_interval):
            should_run_yolo = True
            is_forced_refresh = True
            last_forced_yolo_time = time.time()

        if should_run_yolo:
            t_start = time.time()
            results = detector.track(raw_frame, persist=True, verbose=False, iou=0.5, conf=0.50, device=target_device, imgsz=loaded_imgsz)[0]
            cached_t_yolo = (time.time() - t_start) * 1000

            t_start = time.time()
            bowl_manager.update(raw_frame, frame_idx, results)
            bowl_boxes, bowl_types = bowl_manager.get_info()
            cached_t_rule = (time.time() - t_start) * 1000

            cached_bowl_boxes = [(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), (0, 165, 255) if bowl_types.get(i)=="Food" else (255, 191, 0) if bowl_types.get(i)=="Water" else (200, 200, 200)) for i, bb in enumerate(bowl_boxes)]

            current_frame_logs, active_states_dict_cam, drawn_pets = [], {}, []
            frame_max_trigger_prob, frame_trigger_action = 0.0, ""
            frame_action_time, did_run_action = 0.0, False

            if results.boxes.id is not None:
                dets = sorted(list(zip(results.boxes.xyxy.cpu().numpy(), results.boxes.id.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy())), key=lambda x: x[3], reverse=True)[:max_targets_per_cam]
                for box, raw_id, cls, conf in dets:
                    real_id = id_map.setdefault(int(raw_id), int(raw_id))
                    if real_id not in agents: agents[real_id] = FusionAgent(real_id, int(cls))
                    ag = agents[real_id]

                    if action_recognizer and (frame_idx % 3 == 0):
                        did_run_action = True
                        t_act_start = time.time()
                        ag.update(raw_frame, box, bowl_boxes, bowl_types, rule_engine, action_recognizer, sim_time)
                        frame_action_time += (time.time() - t_act_start) * 1000

                    state, timers, head_box, pr, last_msg = ag.get_info()
                    if pr is None: pr = {}

                    log_data = {"id": real_id, "time": time.strftime("%H:%M:%S"), "bowl": "Yes" if bowl_boxes else "No", "probs": pr, "state": ag.state}
                    current_frame_logs.append(log_data)
                    active_states_dict_cam[str(real_id)] = log_data

                    for lbl in cfg.get("record_labels", []):
                        if pr.get(lbl, 0) >= cfg.get("record_threshold", 0.70):
                            trigger_record_this_frame = True
                            if pr[lbl] > frame_max_trigger_prob: frame_max_trigger_prob, frame_trigger_action = pr[lbl], lbl

                    if ag.last_box is not None:
                        x1, y1, x2, y2 = map(int, ag.last_box)
                        color = (0, 0, 255) if ag.state == "EATING" else (255, 0, 0) if ag.state == "DRINKING" else (255, 0, 255) if ag.state == "JUMP" else (0, 255, 0)
                        drawn_pets.append((x1, y1, x2, y2, real_id, ag.state, color))

            if did_run_action: cached_t_action = frame_action_time
            cached_drawn_boxes, cached_frame_logs, cached_states_dict = drawn_pets, current_frame_logs, active_states_dict_cam
            cached_trigger_prob, cached_trigger_action = frame_max_trigger_prob, frame_trigger_action

        vis = raw_frame.copy()
        for x1, y1, x2, y2, color in cached_bowl_boxes: cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        for x1, y1, x2, y2, uid, st, color in cached_drawn_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"ID:{uid} {st}", (x1, y1-10), 0, 0.8, color, 2)

        if trigger_record_this_frame: record_countdown = 15
        if record_countdown > 0:
            cv2.circle(vis, (30, 80), 10, (0, 0, 255), -1)
            cv2.putText(vis, "REC", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if video_writer is None:
                dt_str, ts_str = datetime.datetime.now().strftime("%Y-%m-%d"), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(f"records/{dt_str}/videos", exist_ok=True); os.makedirs(f"records/{dt_str}/thumbnails", exist_ok=True)
                fn = f"records/{dt_str}/videos/CAM{cam_id+1}_{ts_str}.webm"
                video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'vp08'), max(1.0, base_target_fps - global_penalty), (vis.shape[1], vis.shape[0]))
                cv2.imwrite(f"records/{dt_str}/thumbnails/CAM{cam_id+1}_{ts_str}.jpg", vis)
                current_record_meta = {"cam_id": cam_id+1, "filename": fn, "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "trigger_actions_set": set(), "max_confidence": 0.0}
            if current_record_meta and cached_trigger_prob > 0:
                current_record_meta["trigger_actions_set"].add(cached_trigger_action)
                current_record_meta["max_confidence"] = max(current_record_meta["max_confidence"], cached_trigger_prob)
            video_writer.write(vis)
            record_countdown -= 1
            if record_countdown == 0:
                video_writer.release(); video_writer = None
                current_record_meta["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_record_meta["trigger_action"] = ",".join(current_record_meta["trigger_actions_set"]) or "Unknown"
                current_record_meta["status"] = "completed"
                db_queue.put(current_record_meta); current_record_meta = None

        cv2.rectangle(vis, (10, 10), (180, 50), (0,0,0), -1)
        mode_text = " (Force)" if is_forced_refresh else (" (Skip)" if not should_run_yolo else "")
        throttle_txt = f"[{global_penalty:.1f}x Slow]" if global_penalty > 0 else ""
        cv2.putText(vis, f"CAM {cam_id+1} | FPS {int(true_fps)}{mode_text} {throttle_txt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        stats = {
            "fps": round(true_fps, 1), "decodeMs": round(cam.decode_ms, 1),
            "yoloMs": round(cached_t_yolo, 1), "ruleMs": round(cached_t_rule, 1), "actionMs": round(cached_t_action, 1),
            "status": "online", "imgsz": loaded_imgsz, "isRecording": record_countdown > 0,
            "camId": cam_id, "name": shared_dict.get(cam_id, {}).get("stats", {}).get("name", f"Camera {cam_id+1}")
        }
        shared_dict[cam_id] = {"stats": stats, "logs": cached_frame_logs, "active_states": cached_states_dict}

        ret_img, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ret_img:
            shared_frames[cam_id] = buffer.tobytes()

        frame_idx += 1
        sleep_time = target_frame_time - (time.time() - loop_start)
        if sleep_time > 0: time.sleep(sleep_time)

    if video_writer: video_writer.release()
    cam.release()


# ====================================================================
# 🚀 方案 B：集中式引擎 (串行架構專用 - Singleton Pool)
# ====================================================================
def centralized_engine_process(configs, is_rtsp, system_strategy, model_paths, shared_dict, shared_frames, stop_event, cam_configs, db_queue, desired_cams_list):
    torch.set_num_threads(1)
    gc.disable()

    if HAS_PSUTIL:
        try:
            p = psutil.Process(os.getpid())
            if os.name == 'nt': p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception: pass

    from ultralytics import YOLO
    from core.rule_engine import RuleEngine
    from core.fusion_agent import FusionAgent
    from core.bowl_manager import BowlManager
    from modules.action_recognizer import ActionRecognizer

    tier = system_strategy["tier"]
    target_device = system_strategy["device"]
    use_qsv = system_strategy["use_qsv"]
    base_target_fps = system_strategy["target_fps"]
    static_refresh_interval = system_strategy["static_refresh_interval"]
    max_targets_per_cam = system_strategy["max_targets_per_cam"]
    model_format = system_strategy["model_format"]

    rule_engine = RuleEngine()
    config_path = "config.yaml"
    action_recognizer = ActionRecognizer(config_path) if os.path.exists(config_path) else None

    detectors, cams, loaded_imgsz_dict, bowl_managers, bg_subtractors = {}, {}, {}, {}, {}
    agents_dict, id_map_dict, frame_idx_dict = {}, {}, {}
    video_writers, record_countdowns, current_record_metas = {}, {}, {}
    prev_frame_time, last_forced_yolo_time = {}, {}

    cached_t_yolo, cached_t_rule, cached_t_action = {}, {}, {}
    cached_drawn_boxes_dict, cached_bowl_boxes_dict, cached_frame_logs_dict = {}, {}, {}
    cached_states_dict_cam, cached_trigger_prob_dict, cached_trigger_action_dict = {}, {}, {}

    MOTION_RESIZE = (160, 120)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    print(f"🚀 [推理引擎池] 中央串行運算核心啟動成功...")

    while not stop_event.is_set():
        loop_start = time.time()
        if int(loop_start) % 15 == 0: gc.collect()

        global_penalty = shared_dict.get("global_penalty", 0.0)
        target_frame_time = 1.0 / max(1.0, base_target_fps - global_penalty)
        current_desired = set(desired_cams_list)

        for cid in list(cams.keys()):
            if cid not in current_desired:
                cams[cid].release(); del cams[cid]
                if cid in detectors: del detectors[cid]
                if video_writers.get(cid): video_writers[cid].release(); del video_writers[cid]
                shared_dict[cid] = {"stats": {"status": "offline", "fps": 0, "camId": cid, "yoloMs": 0, "ruleMs": 0, "actionMs": 0, "decodeMs": 0, "name": shared_dict.get(cid, {}).get("stats", {}).get("name", f"CAM-{cid+1}")}, "logs": [], "active_states": {}}

        for cid in current_desired:
            if cid not in cams:
                cams[cid] = IntegratedCamera(configs[cid][0] if is_rtsp else configs[cid][1], is_rtsp=is_rtsp, name=get_smart_cam_name(configs[cid][0], is_rtsp, cid), use_qsv=use_qsv)
                loaded_imgsz_dict[cid] = cam_configs.get(cid, {}).get("imgsz", system_strategy["imgsz"])
                try: detectors[cid] = YOLO(get_model_path_by_strategy(loaded_imgsz_dict[cid], model_format, model_paths), task='detect')
                except Exception as e: print(f"❌ CAM-{cid+1} 模型載入失敗: {e}"); continue
                bowl_managers[cid], bg_subtractors[cid] = BowlManager(30), cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=35, detectShadows=False)
                agents_dict[cid], id_map_dict[cid], frame_idx_dict[cid] = {}, {}, 0
                video_writers[cid], record_countdowns[cid], current_record_metas[cid] = None, 0, None
                prev_frame_time[cid], last_forced_yolo_time[cid] = time.time(), time.time()
                cached_t_yolo[cid], cached_t_rule[cid], cached_t_action[cid] = 0.0, 0.0, 0.0
                cached_drawn_boxes_dict[cid], cached_bowl_boxes_dict[cid], cached_frame_logs_dict[cid] = [], [], []
                cached_states_dict_cam[cid], cached_trigger_prob_dict[cid], cached_trigger_action_dict[cid] = {}, 0.0, ""

        for cid in current_desired:
            if cid not in cams: continue

            cfg = cam_configs.get(cid, {})
            target_imgsz = cfg.get("imgsz", system_strategy["imgsz"])
            if target_imgsz != loaded_imgsz_dict[cid]:
                try:
                    new_model_path = get_model_path_by_strategy(target_imgsz, model_format, model_paths)
                    print(f"🔄 CAM-{cid+1} 正在嘗試切換解析度至 {target_imgsz}p，加載模型: {new_model_path}")
                    new_detector = YOLO(new_model_path, task='detect')
                    detectors[cid] = new_detector
                    loaded_imgsz_dict[cid] = target_imgsz
                    print(f"✅ CAM-{cid+1} 解析度切換成功！")
                except Exception as e:
                    print(f"❌ CAM-{cid+1} 切換 {target_imgsz}p 失敗: {e}")
                    rollback_cfg = cam_configs.get(cid, {})
                    rollback_cfg["imgsz"] = loaded_imgsz_dict[cid]
                    cam_configs[cid] = rollback_cfg

            ret, raw_frame = cams[cid].read()
            if not ret or raw_frame is None:
                shared_dict[cid] = {"stats": {"status": "offline", "fps": 0, "camId": cid, "yoloMs": 0, "ruleMs": 0, "actionMs": 0, "decodeMs": 0, "name": shared_dict.get(cid, {}).get("stats", {}).get("name", f"Camera {cid+1}")}, "logs": [], "active_states": {}}
                continue

            now = time.time()
            true_fps = 1.0 / (now - prev_frame_time[cid] + 1e-6)
            prev_frame_time[cid] = now
            sim_time = frame_idx_dict[cid] / 30.0

            should_run_yolo, is_forced_refresh, trigger_record_this_frame = True, False, False

            if tier == "TIER_LOW":
                current_gray = cv2.cvtColor(cv2.resize(raw_frame, MOTION_RESIZE, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
                current_gray = cv2.blur(current_gray, (5, 5))
            else:
                current_gray = cv2.cvtColor(cv2.resize(raw_frame, MOTION_RESIZE), cv2.COLOR_BGR2GRAY)
                current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

            fg_mask = cv2.morphologyEx(bg_subtractors[cid].apply(current_gray), cv2.MORPH_OPEN, morph_kernel)

            if cv2.countNonZero(fg_mask) / (MOTION_RESIZE[0] * MOTION_RESIZE[1]) < 0.0001 and record_countdowns[cid] == 0:
                should_run_yolo = False

            current_time = time.time()
            if not should_run_yolo and (int(current_time / 2.0) % 8 == cid % 8) and (current_time - last_forced_yolo_time[cid] > static_refresh_interval):
                should_run_yolo, is_forced_refresh, last_forced_yolo_time[cid] = True, True, current_time

            if should_run_yolo:
                t_start = time.time()
                results = detectors[cid].track(raw_frame, persist=True, verbose=False, iou=0.5, conf=0.50, device=target_device, imgsz=loaded_imgsz_dict[cid])[0]
                cached_t_yolo[cid] = (time.time() - t_start) * 1000

                t_start = time.time()
                bowl_managers[cid].update(raw_frame, frame_idx_dict[cid], results)
                bowl_boxes, bowl_types = bowl_managers[cid].get_info()
                cached_t_rule[cid] = (time.time() - t_start) * 1000

                cached_bowl_boxes_dict[cid] = [(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), (0, 165, 255) if bowl_types.get(i)=="Food" else (255, 191, 0) if bowl_types.get(i)=="Water" else (200, 200, 200)) for i, bb in enumerate(bowl_boxes)]

                current_frame_logs, active_states_dict_cam, drawn_pets = [], {}, []
                frame_max_trigger_prob, frame_trigger_action = 0.0, ""
                frame_action_time, did_run_action = 0.0, False

                if results.boxes.id is not None:
                    dets = sorted(list(zip(results.boxes.xyxy.cpu().numpy(), results.boxes.id.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy())), key=lambda x: x[3], reverse=True)[:max_targets_per_cam]
                    for box, raw_id, cls, conf in dets:
                        real_id = id_map_dict[cid].setdefault(int(raw_id), int(raw_id))
                        if real_id not in agents_dict[cid]: agents_dict[cid][real_id] = FusionAgent(real_id, int(cls))
                        ag = agents_dict[cid][real_id]

                        if action_recognizer and (frame_idx_dict[cid] % 3 == 0):
                            did_run_action = True
                            t_act_start = time.time()
                            ag.update(raw_frame, box, bowl_boxes, bowl_types, rule_engine, action_recognizer, sim_time)
                            frame_action_time += (time.time() - t_act_start) * 1000

                        state, timers, head_box, pr, last_msg = ag.get_info()
                        if pr is None: pr = {}

                        log_data = {"id": real_id, "time": time.strftime("%H:%M:%S"), "bowl": "Yes" if bowl_boxes else "No", "probs": pr, "state": ag.state}
                        current_frame_logs.append(log_data)
                        active_states_dict_cam[str(real_id)] = log_data

                        for lbl in cfg.get("record_labels", []):
                            if pr.get(lbl, 0) >= cfg.get("record_threshold", 0.70):
                                trigger_record_this_frame = True
                                if pr[lbl] > frame_max_trigger_prob: frame_max_trigger_prob, frame_trigger_action = pr[lbl], lbl

                        if ag.last_box is not None:
                            x1, y1, x2, y2 = map(int, ag.last_box)
                            color = (0, 0, 255) if ag.state == "EATING" else (255, 0, 0) if ag.state == "DRINKING" else (255, 0, 255) if ag.state == "JUMP" else (0, 255, 0)
                            drawn_pets.append((x1, y1, x2, y2, real_id, ag.state, color))

                if did_run_action: cached_t_action[cid] = frame_action_time
                cached_drawn_boxes_dict[cid], cached_frame_logs_dict[cid], cached_states_dict_cam[cid] = drawn_pets, current_frame_logs, active_states_dict_cam
                cached_trigger_prob_dict[cid], cached_trigger_action_dict[cid] = frame_max_trigger_prob, frame_trigger_action

            vis = raw_frame.copy()
            for x1, y1, x2, y2, color in cached_bowl_boxes_dict.get(cid, []): cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            for x1, y1, x2, y2, uid, st, color in cached_drawn_boxes_dict.get(cid, []):
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{uid} {st}", (x1, y1-10), 0, 0.8, color, 2)

            if trigger_record_this_frame: record_countdowns[cid] = 15
            is_recording = record_countdowns[cid] > 0

            if is_recording:
                cv2.circle(vis, (30, 80), 10, (0, 0, 255), -1)
                cv2.putText(vis, "REC", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if video_writers[cid] is None:
                    dt_str, ts_str = datetime.datetime.now().strftime("%Y-%m-%d"), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs(f"records/{dt_str}/videos", exist_ok=True); os.makedirs(f"records/{dt_str}/thumbnails", exist_ok=True)
                    fn = f"records/{dt_str}/videos/CAM{cid+1}_{ts_str}.webm"
                    video_writers[cid] = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'vp08'), max(1.0, base_target_fps - global_penalty), (vis.shape[1], vis.shape[0]))
                    cv2.imwrite(f"records/{dt_str}/thumbnails/CAM{cid+1}_{ts_str}.jpg", vis)
                    current_record_metas[cid] = {"cam_id": cid+1, "filename": fn, "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "trigger_actions_set": set(), "max_confidence": 0.0}
                if current_record_metas[cid] and cached_trigger_prob_dict.get(cid, 0) > 0:
                    current_record_metas[cid]["trigger_actions_set"].add(cached_trigger_action_dict[cid])
                    current_record_metas[cid]["max_confidence"] = max(current_record_metas[cid]["max_confidence"], cached_trigger_prob_dict[cid])
                video_writers[cid].write(vis)
                record_countdowns[cid] -= 1
                if record_countdowns[cid] == 0:
                    video_writers[cid].release(); video_writers[cid] = None
                    current_record_metas[cid]["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_record_metas[cid]["trigger_action"] = ",".join(current_record_metas[cid]["trigger_actions_set"]) or "Unknown"
                    current_record_metas[cid]["status"] = "completed"
                    db_queue.put(current_record_metas[cid]); current_record_metas[cid] = None

            cv2.rectangle(vis, (10, 10), (180, 50), (0,0,0), -1)
            mode_text = " (Force)" if is_forced_refresh else (" (Skip)" if not should_run_yolo else "")
            throttle_txt = f"[{global_penalty:.1f}x Slow]" if global_penalty > 0 else ""
            cv2.putText(vis, f"CAM {cid+1} | FPS {int(true_fps)}{mode_text} {throttle_txt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            stats = {
                "fps": round(true_fps, 1), "decodeMs": round(cams[cid].decode_ms, 1),
                "yoloMs": round(cached_t_yolo.get(cid, 0.0), 1), "ruleMs": round(cached_t_rule.get(cid, 0.0), 1), "actionMs": round(cached_t_action.get(cid, 0.0), 1),
                "status": "online", "imgsz": loaded_imgsz_dict[cid], "isRecording": is_recording,
                "camId": cid, "name": shared_dict.get(cid, {}).get("stats", {}).get("name", f"Camera {cid+1}")
            }
            shared_dict[cid] = {"stats": stats, "logs": cached_frame_logs_dict.get(cid, []), "active_states": cached_states_dict_cam.get(cid, {})}

            ret_img, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret_img:
                shared_frames[cid] = buffer.tobytes()

            frame_idx_dict[cid] += 1

        sleep_time = target_frame_time - (time.time() - loop_start)
        if sleep_time > 0: time.sleep(sleep_time)

# ====================================================================
# 🌐 Flask 網頁伺服器與資料消毒器
# ====================================================================
def sanitize_for_json(obj):
    if isinstance(obj, dict): return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return sanitize_for_json(obj.tolist())
    else: return obj

def create_flask_app(shared_dict, shared_frames, desired_cams, cams_lock, cam_configs):
    app = Flask(__name__)
    CORS(app)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/api/active_cams', methods=['GET', 'POST'])
    def handle_active_cams():
        if request.method == 'GET':
            with cams_lock: return jsonify({"active_cams": list(desired_cams)})
        with cams_lock:
            del desired_cams[:]
            for c in request.json.get('active_cams', []):
                if c not in desired_cams: desired_cams.append(c)
        return jsonify({"success": True, "active_cams": list(desired_cams)})

    @app.route('/stats')
    def get_stats():
        clean_stats = {}
        for k, v in shared_dict.items():
            if str(k).isdigit():
                clean_stats[str(k)] = v
        return jsonify(sanitize_for_json(clean_stats))

    @app.route('/api/config/<int:cam_id>', methods=['GET', 'POST'])
    def handle_cam_config(cam_id):
        if request.method == 'GET': return jsonify(cam_configs.get(cam_id, {}))
        data = request.json
        cfg = cam_configs.get(cam_id, {})
        if 'imgsz' in data: cfg['imgsz'] = int(data['imgsz'])
        if 'record_labels' in data: cfg['record_labels'] = data['record_labels']
        if 'record_threshold' in data: cfg['record_threshold'] = float(data['record_threshold'])
        cam_configs[cam_id] = cfg
        return jsonify({"success": True, "config": cfg})

    @app.route('/api/records', methods=['GET'])
    def get_records():
        cam_id, action, date = request.args.get('cam_id'), request.args.get('action'), request.args.get('date')
        query = "SELECT id, cam_id, filename, start_time, end_time, trigger_action, max_confidence FROM video_records WHERE status='completed'"
        params = []
        if cam_id and cam_id != 'all': query += " AND cam_id = ?"; params.append(int(cam_id))
        if action and action != 'all': query += " AND trigger_action LIKE ?"; params.append(f"%{action}%")
        if date: query += " AND start_time LIKE ?"; params.append(f"{date}%")
        query += " ORDER BY start_time DESC LIMIT 100"
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            rows = c.execute(query, params).fetchall()
            conn.close()
            return jsonify({"success": True, "records": [{"id": r[0], "cam_id": r[1], "filename": r[2].replace('\\', '/'), "start_time": r[3], "end_time": r[4], "trigger_action": r[5], "max_confidence": r[6]} for r in rows]})
        except Exception as e: return jsonify({"success": False, "error": str(e)})

    @app.route('/api/records/<int:record_id>', methods=['DELETE'])
    def delete_record(record_id):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            row = c.execute("SELECT filename FROM video_records WHERE id=?", (record_id,)).fetchone()
            if row:
                fname = row[0]
                thumb_name = fname.replace('/videos/', '/thumbnails/').replace('\\videos\\', '\\thumbnails\\').replace('.webm', '.jpg').replace('.mp4', '.jpg')
                if os.path.exists(fname): os.remove(fname)
                if os.path.exists(thumb_name): os.remove(thumb_name)
                c.execute("DELETE FROM video_records WHERE id=?", (record_id,))
                conn.commit()
                conn.close()
                return jsonify({"success": True})
            return jsonify({"success": False, "error": "Record not found"}), 404
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

    # 🔥 新增：供前端查詢歷史行為分佈的 API
    @app.route('/api/behavior_logs', methods=['GET'])
    def get_behavior_logs():
        cam_id = request.args.get('cam_id', type=int)
        start_time = request.args.get('start')
        end_time = request.args.get('end')
        
        if cam_id is None or cam_id < 0 or cam_id > 7:
            return jsonify({"success": False, "error": "Invalid cam_id"})
            
        col_name = f"cam{cam_id}"
        # 排除為 "None" 的靜止/無寵物畫面
        query = f"SELECT {col_name}, COUNT(*) FROM behavior_logs WHERE {col_name} != 'None'"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        query += f" GROUP BY {col_name}"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            rows = c.execute(query, params).fetchall()
            conn.close()
            
            stats = {r[0]: r[1] for r in rows}
            return jsonify({"success": True, "stats": stats})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    @app.route('/records/<path:filename>')
    def serve_record(filename): return send_from_directory('records', filename.replace('records/', '').replace('records\\', ''))

    def generate_frames(cam_id):
        blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "STARTING ENGINE...", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        blank_bytes = cv2.imencode(".jpg", blank_frame)[1].tobytes()
        
        last_frame = None

        try: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
        except GeneratorExit: return

        while True:
            try:
                with cams_lock: is_active = cam_id in desired_cams
                if not is_active:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
                    time.sleep(0.5)
                    continue

                frame_bytes = shared_frames.get(cam_id)
                
                if frame_bytes and frame_bytes != last_frame:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    last_frame = frame_bytes
                
                time.sleep(0.04) 
            except GeneratorExit: break

    @app.route('/video_feed/<int:cam_id>')
    def video_feed(cam_id): return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    return app

def run_flask_server(shared_dict, shared_frames, desired_cams, cams_lock, cam_configs):
    app = create_flask_app(shared_dict, shared_frames, desired_cams, cams_lock, cam_configs)
    if HAS_WAITRESS: serve(app, host='0.0.0.0', port=9527, threads=16)
    else: app.run(host='0.0.0.0', port=9527, debug=False, use_reloader=False, threaded=True)

# ====================================================================
# 🚀 主程序入口與生命週期管理 (智能路由並行或串行)
# ====================================================================
if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    ARGS = parse_args()
    init_db()

    hw_report = HardwareProfiler.analyze()

    if ARGS.device != "auto":
        hw_report["tier"] = "TIER_CUDA" if ARGS.device == "cuda" else "TIER_HIGH"

    system_strategy = StrategyFactory.get_strategy(hw_report)

    if ARGS.hwaccel != "auto":
        system_strategy["use_qsv"] = (ARGS.hwaccel == "qsv")
    if ARGS.max_fps > 0:
        system_strategy["target_fps"] = ARGS.max_fps

    model_paths = {
        "480_int8": ARGS.yolo_480_int8,
        "640_int8": ARGS.yolo_640_int8,
        "480_fp32": ARGS.yolo_480_fp32,
        "640_fp32": ARGS.yolo_640_fp32,
        "pt": ARGS.yolo_pt,
        "trt": ARGS.yolo_trt
    }

    manager = mp.Manager()
    shared_dict = manager.dict()
    shared_dict["global_penalty"] = 0.0
    cam_configs = manager.dict()
    
    shared_frames = manager.dict()
    
    db_queue = mp.Queue()
    desired_cams = manager.list([0])
    cams_lock = manager.Lock()

    configs = [
        (ARGS.rtsp_url_1, ARGS.cam_index_1), (ARGS.rtsp_url_2, ARGS.cam_index_2),
        (ARGS.rtsp_url_3, ARGS.cam_index_3), (ARGS.rtsp_url_4, ARGS.cam_index_4),
        (ARGS.rtsp_url_5, ARGS.cam_index_5), (ARGS.rtsp_url_6, ARGS.cam_index_6),
        (ARGS.rtsp_url_7, ARGS.cam_index_7), (ARGS.rtsp_url_8, ARGS.cam_index_8)
    ]
    is_rtsp = (ARGS.source == "rtsp")

    for i in range(ARGS.num_cams):
        cam_name = get_smart_cam_name(configs[i][0], is_rtsp, i)
        shared_dict[i] = {"stats": {"fps": 0, "yoloMs": 0, "ruleMs": 0, "actionMs": 0, "decodeMs": 0, "status": "offline", "camId": i, "name": cam_name}, "logs": [], "active_states": {}}
        cam_configs[i] = {"imgsz": system_strategy["imgsz"], "record_labels": [], "record_threshold": 0.70}

    print("\n" + "="*85)
    print(f"🌟 SmartPet AI Backend 啟動 - 【四象限硬體智能適配架構】")
    print(f"💻 實體硬體探測: 核心數={hw_report['cpu_cores']} | 獨立 GPU(CUDA)={hw_report['has_cuda']} | ARM={hw_report['is_aarch64']}")
    print(f"⚙️ 分配硬體梯隊: {system_strategy['tier']}")
    print("-" * 85)
    print(f"🎯 自適應策略分派與降級參數:")
    print(f"  ▶ 🚀 系統進程架構 : {'多核並行模式 (Parallel)' if system_strategy['processing_mode'] == 'parallel' else '單例串行模式 (Singleton Pool)'}")
    print(f"  ▶ 推理引擎與格式 : {system_strategy['device'].upper()} [{system_strategy['model_format']}]")
    print(f"  ▶ 基礎模型解析度 : {system_strategy['imgsz']} (前端可動態調整)")
    print(f"  ▶ 動態目標截斷   : 每路最多追蹤並分析 {system_strategy['max_targets_per_cam']} 隻目標")
    print(f"  ▶ 系統調速器模式 : {'🚨 嚴格階梯防卡死 (Max -4.0)' if system_strategy['tier'] == 'TIER_LOW' else '🟢 寬容平滑過渡 (Max -2.5)'}")
    print("="*85 + "\n")

    threading.Thread(target=run_flask_server, args=(shared_dict, shared_frames, desired_cams, cams_lock, cam_configs), daemon=True).start()
    threading.Thread(target=db_writer_thread, args=(db_queue,), daemon=True).start()
    threading.Thread(target=db_cleanup_thread, daemon=True).start()
    threading.Thread(target=system_governor_thread, args=(shared_dict, system_strategy['tier']), daemon=True).start()
    
    # 🔥 啟動每秒行為日誌寫入執行緒
    threading.Thread(target=behavior_logger_thread, args=(shared_dict,), daemon=True).start()

    if system_strategy["processing_mode"] == "sequential":
        engine_stop_event = mp.Event()
        engine_process = mp.Process(
            target=centralized_engine_process,
            args=(configs, is_rtsp, system_strategy, model_paths, shared_dict, shared_frames, engine_stop_event, cam_configs, db_queue, desired_cams),
            name="Singleton-Inference-Engine"
        )
        engine_process.daemon = True
        engine_process.start()
    else:
        active_processes = {}
        stop_events = {}

    last_report_time = time.time()

    try:
        while True:
            if system_strategy["processing_mode"] == "parallel":
                with cams_lock: current_desired = set(desired_cams)

                for cid in list(active_processes.keys()):
                    if cid not in current_desired:
                        stop_events[cid].set()
                        active_processes[cid].join(timeout=3)
                        if active_processes[cid].is_alive(): active_processes[cid].terminate()
                        del active_processes[cid]; del stop_events[cid]
                        
                        shared_frames.pop(cid, None) 
                        shared_dict[cid] = {"stats": {"status": "offline", "fps": 0, "camId": cid, "name": shared_dict.get(cid, {}).get("stats", {}).get("name")}, "logs": [], "active_states": {}}

                for cid in current_desired:
                    if cid not in active_processes and cid < ARGS.num_cams:
                        src = configs[cid][0] if is_rtsp else configs[cid][1]
                        stop_evt = mp.Event()
                        p = mp.Process(target=worker_process, args=(cid, src, is_rtsp, system_strategy, model_paths, shared_dict, shared_frames, stop_evt, cam_configs, db_queue), name=f"Worker-{cid+1}")
                        p.daemon = True
                        p.start()
                        active_processes[cid] = p
                        stop_events[cid] = stop_evt

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                last_report_time = current_time
                sys_cpu = shared_dict.get("sys_cpu", 0.0)
                global_penalty = shared_dict.get("global_penalty", 0.0)

                print("\n" + "━"*95)
                print(f"📊 [智能系統報告] {time.strftime('%H:%M:%S')} | 穩態 CPU 負載: {sys_cpu:.1f}% | 散熱調速器等級: {global_penalty:.1f}")

                total_fps_output = 0.0
                total_active_pets = 0

                for cid in range(ARGS.num_cams):
                    if cid not in list(desired_cams): continue
                    data = shared_dict.get(cid, {})
                    stats = data.get("stats", {})
                    fps, decode_ms = stats.get("fps", 0), stats.get("decodeMs", 0)
                    yolo_ms, rule_ms, action_ms = stats.get("yoloMs", 0), stats.get("ruleMs", 0), stats.get("actionMs", 0)
                    img_size = stats.get("imgsz", "-")
                    is_rec = "REC" if stats.get("isRecording", False) else "---"
                    status = stats.get("status", "offline")
                    c_name = stats.get("name", f"CAM-{cid+1}")
                    active_pets = len(data.get("active_states", {}))

                    if status == "offline":
                        print(f"   ▶ [{c_name[:15]:<15}] 狀態: offline")
                    else:
                        print(f"   ▶ [{c_name[:15]:<15}] 狀態:{status:<6} | 錄影:{is_rec} | 解碼:{decode_ms:4.1f}ms | YOLO({img_size}):{yolo_ms:4.1f}ms | 規則:{rule_ms:3.1f}ms | 動作:{action_ms:4.1f}ms | FPS:{fps:4.1f}")
                        total_fps_output += fps
                        total_active_pets += active_pets

                print(f"🌐 總下發吞吐量: {total_fps_output:.1f} 幀/秒 | 全局監測目標數: {total_active_pets} 隻")
                print("━"*95 + "\n")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("🛑 收到停止指令，正在終結所有進程...")
        if system_strategy["processing_mode"] == "sequential":
            engine_stop_event.set()
        db_queue.put(None)
    finally:
        if system_strategy["processing_mode"] == "sequential":
            engine_process.join(timeout=3)
            if engine_process.is_alive(): engine_process.terminate()
        else:
            for cid, p in active_processes.items():
                stop_events[cid].set()
                p.join(timeout=2)
                if p.is_alive(): p.terminate()
        print("👋 引擎徹底關閉。")