import numpy as np

# 计算函数

def calculate_iob(inner_box, outer_box):
    """
    计算 Intersection over Box (用于判断头是否在碗里)
    inner_box: [x1, y1, x2, y2] (头部)
    outer_box: [x1, y1, x2, y2] (碗)
    """
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # IOB = 交集 / inner_box 的面积
    innerArea = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    
    return interArea / float(innerArea + 1e-6)

def calculate_iou(boxA, boxB):
    """标准的 Intersection over Union"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])