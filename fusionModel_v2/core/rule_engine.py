import numpy as np
from .utils import calculate_iob

class RuleEngine:
    def __init__(self):
        # =========================================================
        # 1. 物理阈值配置 (完全对齐 v6_IDfix.py)
        # =========================================================
        self.ACTIVE_SPEED_THRESHOLD = 3.0    # 原版: 3.0
        # self.JUMP_SPEED_THRESHOLD = 10.0     # 原版: 12.0
        self.HEAD_IOB_THRESHOLD = 0.15        # 原版: 0.2
        # self.JUMP_RATIO = 1.2                # 原版: 1.2 (dy > dx * 1.2)
        
    def analyze(self, speed, dx, dy, head_box, bowl_boxes, bowl_types):
        """
        输入: 物理指标
        输出: (Rule_State, Interaction_Type)
        Rule_State 包含: "RESTING", "ACTIVE", "JUMP_CANDIDATE", "INTERACT_CANDIDATE"
        """
        
        # --- 1. 优先检查与碗的交互 (IOB) - [逻辑修正版] ---
        in_bowl = False
        interaction_type = None
        
        # 只有当头部框有效(面积>0)才计算
        if head_box is not None and (head_box[2] - head_box[0]) > 0:
            max_iob = 0.0
            best_type = None
            
            # 遍历所有的碗，寻找 IOB 最大的那个
            for i, b_box in enumerate(bowl_boxes):
                current_iob = calculate_iob(head_box, b_box)
                
                # 记录当前最大的 IOB 和对应的类型
                if current_iob > max_iob:
                    max_iob = current_iob
                    best_type = bowl_types.get(i, "Unknown")
            
            # 只有当最大的 IOB 超过阈值时，才判定为交互
            if max_iob > self.HEAD_IOB_THRESHOLD:
                in_bowl = True
                interaction_type = best_type
        
        # ⚡ 如果判定为在该碗中，返回候选状态
        if in_bowl:
            return "INTERACT_CANDIDATE", interaction_type

        # --- 2. 运动状态判定 (基于速度和向量) ---
        
        # A. 休息判定
        if speed < self.ACTIVE_SPEED_THRESHOLD:
            return "RESTING", None
        
        # B. 跳跃判定
        # if speed > self.JUMP_SPEED_THRESHOLD and (abs(dy) > 10 or abs(dy) > abs(dx) * self.JUMP_RATIO):
        #     print(f"跳跃候选: speed={speed:.2f}, dx={dx:.2f}, dy={dy:.2f}")
        #     return "JUMP_CANDIDATE", None
            
        # C. 默认活跃
        return "ACTIVE", None

