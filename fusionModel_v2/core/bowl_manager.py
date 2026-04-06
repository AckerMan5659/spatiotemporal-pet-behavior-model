import cv2
import numpy as np

class BowlManager:
    def __init__(self, update_interval=30):
        self.bowl_boxes = []
        self.bowl_types = {} # {index: "Food" or "Water"}
        self.update_interval = update_interval
        self.last_update_frame = -update_interval
        
    def update(self, frame, frame_idx, yolo_results):
        """
        定期更新碗的位置和类型
        返回: Boolean (是否刚刚更新了碗)
        """
        # 仅在特定间隔或初始化时更新，节省性能
        if frame_idx >= self.last_update_frame + self.update_interval:
            self.bowl_boxes = []
            self.bowl_types = {}
            
            if yolo_results.boxes is not None:
                for b in yolo_results.boxes:
                    # 假设 class 0 是 bowl (根据你原代码的逻辑)
                    if int(b.cls[0]) == 0:
                        box = b.xyxy[0].cpu().numpy()
                        self.bowl_boxes.append(box)
                        # 判定碗里的内容
                        b_type = self._classify_bowl_content(frame, box)
                        self.bowl_types[len(self.bowl_boxes)-1] = b_type
            
            self.last_update_frame = frame_idx
            return True 
        return False
    
    def _classify_bowl_content(self, frame, box):
        """基于颜色判断是水还是粮 (原版逻辑)"""
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return "Water"
        
        # 简单的颜色阈值判断
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 棕色/狗粮色范围
        lower_brown = np.array([5, 60, 50])
        upper_brown = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv_roi, lower_brown, upper_brown)
        brown_pixel_count = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels == 0: return "Water"
        brown_ratio = brown_pixel_count / total_pixels
        
        # 阈值 0.08 (原版)
        return "Food" if brown_ratio > 0.08 else "Water"
    
    def get_info(self):
        return self.bowl_boxes, self.bowl_types