import numpy as np
from collections import deque
from .utils import get_box_center

class FusionAgent:
    def __init__(self, track_id, cls_id):
        self.track_id = track_id
        self.cls_id = cls_id
        
        # ReID 相关属性
        self.reid_feature = None    
        self.gallery_name = None    
        self.last_reid_update_idx = 0

        # --- 1. FSM 配置 (已移除 JUMP) ---
        self.state = "RESTING"
        self.state_buffer = {} 
        self.last_transition_msg = "" # 用于存储最后一次状态切换的描述，供 API 调用
        
        self.STATE_THRESHOLDS = {
            "EATING": 6,
            "DRINKING": 6, 
            "ACTIVE": 4, 
            "RESTING": 15
        }
        self.VALID_TRANSITIONS = {
            "RESTING": ["ACTIVE", "EATING", "DRINKING"],
            "ACTIVE": ["RESTING", "EATING", "DRINKING"],
            "EATING": ["ACTIVE", "RESTING"],
            "DRINKING": ["ACTIVE", "RESTING"]
        }
        
        # --- 2. 物理与头部 ---
        self.center_history = deque(maxlen=6)
        self.head_box = [0,0,0,0]
        self.last_h_dir = 1 
        self.last_v_dir = 1
        
        self.speed = 0.0
        self.dx = 0.0
        self.dy = 0.0
        
        # --- 3. 统计与计时 (已移除 JUMP) ---
        self.timers = {"EATING": 0.0, "DRINKING": 0.0, "ACTIVE": 0.0, "RESTING": 0.0}
        self.last_state_time = 0.0
        self.last_update_time = 0.0
        
        # --- 4. 模型记忆 ---
        self.mamba_state = None 
        self.seq_idx = 0
        self.last_model_pred = None # 存储概率分布，供 API 暴露
        self.last_box = None

    def _update_physics(self, box):
        curr_center = get_box_center(box)
        self.center_history.append(curr_center)
        
        speed, dx, dy = 0.0, 0.0, 0.0
        if len(self.center_history) >= 2:
            first, last = self.center_history[0], self.center_history[-1]
            dt = len(self.center_history)
            dx = (last[0] - first[0]) / dt
            dy = (last[1] - first[1]) / dt
            speed = np.linalg.norm(last - first) / dt
            
        return speed, dx, dy

    def _update_head_box(self, box, speed, dx, dy, bowl_boxes):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        curr_center = get_box_center(box)

        if speed < 1.0 and len(bowl_boxes) > 0:
            if self.last_h_dir == 0: 
                min_dist = float('inf')
                target_bowl = None
                for b_box in bowl_boxes:
                    b_center = get_box_center(b_box)
                    dist = np.linalg.norm(curr_center - b_center)
                    if dist < min_dist:
                        min_dist = dist
                        target_bowl = b_center
                if target_bowl is not None:
                    self.last_h_dir = -1 if target_bowl[0] < curr_center[0] else 1
                    self.last_v_dir = -1 if target_bowl[1] < curr_center[1] else 1
        elif speed > 1.0:
            if abs(dx) > 1.0: self.last_h_dir = 1 if dx > 0 else -1
            if abs(dy) > 1.0: self.last_v_dir = 1 if dy > 0 else -1
            
        if self.last_h_dir == -1: 
            if w > h: self.head_box = [x1, y1, x1 + w * 0.45, y2]
            else: self.head_box = [x1, y2 - h * 0.45, x2, y2] if self.last_v_dir == 1 else [x1, y1, x2, y1 + h * 0.45]
        else:
            if w > h: self.head_box = [x2 - w * 0.45, y1, x2, y2]
            else: self.head_box = [x1, y2 - h * 0.45, x2, y2] if self.last_v_dir == 1 else [x1, y1, x2, y1 + h * 0.45]

    def update(self, frame, box, bowl_boxes, bowl_types, rule_engine, recognizer, current_time):
        # 1. 物理层
        self.speed, self.dx, self.dy = self._update_physics(box)
        self._update_head_box(box, self.speed, self.dx, self.dy, bowl_boxes)
        
        # 2. 规则层建议
        rule_state, interact_type = rule_engine.analyze(
            self.speed, self.dx, self.dy, self.head_box, bowl_boxes, bowl_types
        )
        
        # 3. 调度器优化
        should_run_model = False
        
        if rule_state in ["INTERACT_CANDIDATE"]:
            should_run_model = True
        elif rule_state == "ACTIVE" and self.seq_idx % 3 == 0: 
            should_run_model = True
            
        # 如果当前已经是 吃/喝 状态，强制运行模型，实现惯性
        if self.state in ["EATING", "DRINKING"]:
            should_run_model = True

        # 4. 模型推理
        model_probs = None
        if should_run_model:
            probs, new_state = recognizer.predict(
                frame, box, prev_state=self.mamba_state, seq_idx=self.seq_idx
            )
            if probs is not None:
                self.mamba_state = new_state
                self.last_model_pred = probs
                model_probs = probs
                self.seq_idx += 1
        
        # 5. 融合
        candidate = self._fuse_logic(rule_state, interact_type, model_probs, recognizer)

        # 6. FSM 滤波
        self._fsm_transition(candidate, current_time)
        
        self.last_box = box
        self.last_update_time = current_time

    def _fuse_logic(self, rule_state, interact_type, model_probs, recognizer):
        probs = model_probs if model_probs is not None else self.last_model_pred
        
        # 安全获取概率，如果字典为空则默认给 0.0
        p_eat = probs.get("eat", 0.0) if probs else 0.0
        p_drink = probs.get("drink", 0.0) if probs else 0.0
        p_active = probs.get("active", 0.0) if probs else 0.0
        p_rest = probs.get("rest", 0.0) if probs else 0.0

        if probs is not None and self.seq_idx % 10 == 0:
            print(f"🔍 [Prob Monitor] ID:{self.track_id} | Bowl:{interact_type} | Eat:{p_eat:.3f} Drink:{p_drink:.3f}")
        
        # 吃喝状态惯性 (Inertia)
        if self.state == "EATING":
            if probs is not None and p_eat > 0.2:
                return "EATING"

        if self.state == "DRINKING":
            if probs is not None and p_drink > 0.2:
                return "DRINKING"

        # 逻辑判断
        if rule_state == "INTERACT_CANDIDATE":
            if probs is not None:
                if interact_type == "Food":
                    if p_eat > 0.3 and p_eat > p_drink: return "EATING"
                    elif p_drink > 0.5: return "DRINKING"
                    else: return "ACTIVE"
                elif interact_type == "Water":
                    if p_drink > 0.3 and p_drink > p_eat: return "DRINKING"
                    elif p_eat > 0.5: return "EATING"
                    else: return "ACTIVE"
            else:
                return "EATING" if interact_type == "Food" else "DRINKING"

        if rule_state == "ACTIVE":
            if probs is not None and self.speed < 8.0:
                if p_eat > 0.6: return "EATING"
                if p_drink > 0.6: return "DRINKING"
            return "ACTIVE"

        if rule_state == "RESTING":
            if probs is not None and p_eat > 0.75: return "EATING"
            return "RESTING"

        return "ACTIVE"

    def _fsm_transition(self, candidate, t):
        if candidate == self.state:
            self.state_buffer = {} 
            dt = t - self.last_state_time if self.last_state_time > 0 else 0
            s_key = self.state
            if s_key in self.timers: self.timers[s_key] += dt
            self.last_state_time = t
            return

        valid_targets = self.VALID_TRANSITIONS.get(self.state, [])
        if candidate not in valid_targets:
            # 简化非法跳转逻辑
            if self.state == "RESTING" and candidate in ["EATING", "DRINKING"]: 
                candidate = "ACTIVE"
            elif self.state in ["EATING", "DRINKING"] and candidate == "RESTING": 
                candidate = "ACTIVE"
            else:
                return 

        self.state_buffer[candidate] = self.state_buffer.get(candidate, 0) + 1
        
        required_frames = self.STATE_THRESHOLDS.get(candidate, 5)
        if self.state_buffer[candidate] >= required_frames:
            msg = f"🔄 状态切换 ID:{self.track_id}: {self.state} -> {candidate}"
            print(msg)
            self.last_transition_msg = msg # 存储给 API
            self.state = candidate
            self.state_buffer = {}
            self.last_state_time = t

    def get_info(self):
        """
        修改后的获取信息方法
        返回: 状态, 计时器字典, 头部框, 最后一次模型概率分布, 最后一次状态切换日志
        """
        return self.state, self.timers, self.head_box, self.last_model_pred, self.last_transition_msg

    def update_reid_feature(self, new_feat, frame_idx):
        if new_feat is None: return
        if self.reid_feature is None:
            self.reid_feature = new_feat
        else:
            self.reid_feature = 0.7 * new_feat + 0.3 * self.reid_feature
            self.reid_feature /= (np.linalg.norm(self.reid_feature) + 1e-8)
        self.last_reid_update_idx = frame_idx