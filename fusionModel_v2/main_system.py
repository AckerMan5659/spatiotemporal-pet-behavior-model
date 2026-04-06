import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

# --- 引入自定义模块 ---
from core.rule_engine import RuleEngine
from core.fusion_agent import FusionAgent
from core.bowl_manager import BowlManager
from core.utils import get_box_center
from modules.action_recognizer import ActionRecognizer

# ==================================================
# 配置区域
# ==================================================
# VIDEO_PATH = r"D:\PPCdata\1223data\test\room_test_10s\20251129201320_20251129210856_range_0.5-1.0_p002.mp4"  # 👈 请 
# 修改为你的视频路径

VIDEO_PATH = r"E:\ActionRecognition\collection\temp\Videos2026-01-08_113418_830.mp4"  # 👈 请 修改为你的视频路径
YOLO_PATH = r"E:\ActionRecognition\enhanced\fusionModel\modules\best2.pt"                      # 👈 请确保路径正确 (检测狗/猫/碗) 可以换成openvino/onnx
ACTION_CONFIG = r"E:\ActionRecognition\enhanced\fusionModel\config.yaml"                    # 动作识别配置
OUTPUT_PATH = r"E:\ActionRecognition\collection\temp\test.mp4"               # 输出视频路径

# Re-ID 参数 (处理 ID 跳变)
REID_DISTANCE_THRESHOLD = 200
REID_TIME_THRESHOLD = 2.0 

# ✅ [新增] 性能监控类
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.yolo_times = []
        self.logic_action_times = [] # 包含 规则+动作模型+ReID
        self.total_frame_times = []
        
    def record(self, t_yolo, t_logic, t_total):
        self.yolo_times.append(t_yolo)
        self.logic_action_times.append(t_logic)
        self.total_frame_times.append(t_total)
        self.frame_count += 1
        
    def get_current_stats(self):
        """获取实时的平均数据（最近30帧）"""
        if not self.total_frame_times: return 0, 0, 0
        
        # 计算瞬时 FPS (基于最近一帧的总耗时)
        last_frame_time = self.total_frame_times[-1]
        fps = 1.0 / last_frame_time if last_frame_time > 0 else 0
        
        # 获取最近耗时
        curr_yolo = self.yolo_times[-1] * 1000
        curr_logic = self.logic_action_times[-1] * 1000
        
        return fps, curr_yolo, curr_logic

    def print_summary(self):
        """打印最终统计报告"""
        total_elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / total_elapsed if total_elapsed > 0 else 0
        
        avg_yolo = (sum(self.yolo_times) / len(self.yolo_times)) * 1000 if self.yolo_times else 0
        avg_logic = (sum(self.logic_action_times) / len(self.logic_action_times)) * 1000 if self.logic_action_times else 0
        
        print("\n" + "="*40)
        print("📊 性能统计报告")
        print("="*40)
        print(f"处理总帧数: {self.frame_count}")
        print(f"总运行时间: {total_elapsed:.2f}s")
        print(f"平均 FPS:   {avg_fps:.2f}")
        print("-" * 20)
        print(f"平均 YOLO 推理:  {avg_yolo:.2f} ms")
        print(f"平均 逻辑+动作:  {avg_logic:.2f} ms")
        print("="*40)

def main():
    print("🚀 系统初始化中...")
    
    detector = YOLO(YOLO_PATH) 
    action_recognizer = ActionRecognizer(cfg_path=ACTION_CONFIG, device='auto')
    rule_engine = RuleEngine()
    bowl_manager = BowlManager(update_interval=30)
    
    # ✅ [新增] 初始化监控器
    monitor = PerformanceMonitor()

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误: 找不到视频文件 {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    agents = {}
    id_map = {}
    frame_idx = 0
    
    print(f"▶ 开始处理: {width}x{height} @ {fps:.1f}fps")
    
    try:
        while True:
            # ✅ [新增] 帧开始计时
            t_frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret: break
            
            current_time = frame_idx / fps
            
            # --- [Step 1] 目标检测 (YOLO) ---
            t_yolo_start = time.time()
            results = detector.track(frame, persist=True, verbose=False, iou=0.5, conf=0.35)[0]
            t_yolo_end = time.time() # ✅ 记录 YOLO 结束时间
            
            # --- [Step 2] 更新环境 ---
            bowl_manager.update(frame, frame_idx, results)
            bowl_boxes, bowl_types = bowl_manager.get_info()
            
            # --- [Step 3] 逻辑处理 (动作识别+ReID) ---
            t_logic_start = time.time() # ✅ 逻辑开始
            
            active_real_ids = []
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                assigned_real_ids = set()
                
                for box, raw_id, cls_id in zip(boxes, track_ids, clss):
                    raw_id = int(raw_id)
                    cls_id = int(cls_id)
                    if cls_id == 0: continue 
                    
                    # === Re-ID ===
                    real_id = raw_id
                    if raw_id in id_map:
                        real_id = id_map[raw_id]
                    else:
                        center = get_box_center(box)
                        best_match = None
                        min_dist = float('inf')
                        for existing_id, agent in agents.items():
                            if agent.cls_id != cls_id: continue
                            if existing_id in assigned_real_ids: continue
                            if (current_time - agent.last_update_time) > REID_TIME_THRESHOLD: continue
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
                    
                    # === Agent Update (包含动作识别) ===
                    if real_id not in agents:
                        agents[real_id] = FusionAgent(real_id, cls_id)
                    
                    agents[real_id].update(
                        frame, box, 
                        bowl_boxes, bowl_types, 
                        rule_engine, action_recognizer, 
                        current_time
                    )
            
            t_logic_end = time.time() # ✅ 逻辑结束
            
            # --- [Step 4] 可视化 ---
            vis_frame = frame.copy()
            
            # 画碗
            for i, bb in enumerate(bowl_boxes):
                b_type = bowl_types.get(i, "Unknown")
                color = (0, 165, 255) if b_type == "Food" else (255, 191, 0)
                cv2.rectangle(vis_frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
                cv2.putText(vis_frame, b_type, (int(bb[0]), int(bb[1]-5)), 0, 0.6, color, 2) 
            
            # 画宠物
            for real_id in active_real_ids:
                agent = agents[real_id]
                state, timers, head_box = agent.get_info()
                
                if state == "EATING": color = (0, 0, 255)
                elif state == "DRINKING": color = (255, 0, 0)
                # elif state == "JUMP": color = (0, 255, 255)
                elif state == "ACTIVE": color = (0, 255, 0)
                else: color = (200, 200, 200)
                
                if agent.last_box is not None:
                    x1, y1, x2, y2 = map(int, agent.last_box)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    hx = list(map(int, head_box))
                    if (hx[2]-hx[0]) > 0: #head box valid
                        cv2.rectangle(vis_frame, (hx[0], hx[1]), (hx[2], hx[3]), (0, 255, 255), 1)
                    label = f"ID:{real_id} {state}"
                    cv2.putText(vis_frame, label, (x1, y1-10), 0, 0.8, color, 2)

            # --- [Step 5] 性能记录与显示 ---
            t_frame_end = time.time()
            
            # 记录本帧数据
            monitor.record(
                t_yolo=t_yolo_end - t_yolo_start,
                t_logic=t_logic_end - t_logic_start,
                t_total=t_frame_end - t_frame_start
            )
            
            # 获取实时统计
            curr_fps, curr_yolo_ms, curr_logic_ms = monitor.get_current_stats()
            
            # # 绘制统计面板 (左上角黑底)
            # cv2.rectangle(vis_frame, (0, 0), (280, 110), (0,0,0), -1)
            # cv2.putText(vis_frame, f"FPS: {curr_fps:.1f}", (10, 30), 0, 0.8, (0, 255, 0), 2)
            # cv2.putText(vis_frame, f"YOLO: {curr_yolo_ms:.1f}ms", (10, 60), 0, 0.6, (255, 255, 255), 1)
            # cv2.putText(vis_frame, f"Action+Logic: {curr_logic_ms:.1f}ms", (10, 85), 0, 0.6, (255, 255, 255), 1)
            # cv2.putText(vis_frame, f"Frame: {frame_idx}", (150, 30), 0, 0.6, (200, 200, 200), 1)
            
            out.write(vis_frame)
            
            if frame_idx % 3 == 0:
                cv2.imshow('Smart Pet Monitor', cv2.resize(vis_frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"进度: {frame_idx} 帧 | FPS: {curr_fps:.1f}")

    except KeyboardInterrupt:
        print("🛑 用户强制停止")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # ✅ [新增] 打印最终报告
        monitor.print_summary()
        print(f"✅ 处理完成! 结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()