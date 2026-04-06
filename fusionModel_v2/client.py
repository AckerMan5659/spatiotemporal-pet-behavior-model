import requests
import time
import json
from typing import Optional

# ==================== 配置 ====================
SERVER_URL = "http://ppcapi.ddns.net:9000"
# 替换为你实际想要测试的视频 URL
VIDEO_URL = "https://kayupload.s3.ap-southeast-1.amazonaws.com/petActivities/dog_sneeze2.mp4" 

def start_inference(video_url: str) -> Optional[str]:
    """提交推理任务并返回 task_id"""
    endpoint = f"{SERVER_URL}/infer"
    payload = {"video_url": video_url}
    
    try:
        print(f"🚀 正在提交任务到: {endpoint}...")
        # 设置超时时间，防止下载大视频时挂起
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        task_id = data.get("task_id")
        print(f"✅ 任务已创建，ID: {task_id}")
        return task_id
    except Exception as e:
        print(f"❌ 提交任务失败: {e}")
        return None

def poll_result(task_id: str):
    """轮询任务结果直到完成或失败"""
    endpoint = f"{SERVER_URL}/result/{task_id}"
    print(f"⏳ 开始轮询结果...")
    
    while True:
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            
            if status == "completed":
                # 任务完成后，以 JSON 格式输出
                display_full_info(result)
                break
            elif status == "failed":
                print(f"\n❌ 任务失败: {result.get('error')}")
                break
            else:
                # 打印当前进度状态
                print(f"  > 当前状态: {status}...", end="\r")
                time.sleep(2)  
                
        except Exception as e:
            print(f"❌ 轮询出错: {e}")
            break

def display_full_info(data: dict):
    """将 API 返回的结果转换为指定的 JSON 格式并输出"""
    task_id = data.get("task_id", "")
    status = data.get("status", "")
    results = data.get("results", [])
    
    actions_list = []
    
    for frame in results:
        ts_val = frame.get("timestamp", 0)
        # 将相对秒数转换为 HH:MM:SS 格式字符串
        mins, secs = divmod(int(ts_val), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
        
        agents = frame.get("agents", [])
        for agent in agents:
            agent_id = agent.get("agent_id")
            state = agent.get("state", "").lower()
            probs = agent.get("probs") or {}
            
            # 提取当前状态对应的概率作为置信度
            confidence = probs.get(state, 0.0)
            
            # 容错处理：如果状态在 probs 中找不到（如 ACTIVE），取最大概率值
            if confidence == 0.0 and probs:
                confidence = max(probs.values())

            # 在这里增加 "id" 字段来区分不同的目标
            actions_list.append({
                "id": agent_id,
                "timestamp": time_str,
                "action": state,
                "confidence": round(float(confidence), 3)
            })

    # 构建最终 JSON 结构
    output_json = {
        "request_id": task_id[:8] if task_id else "unknown",
        "message": "Processing successful" if status == "completed" else "Processing failed",
        "result": {
            "actions": actions_list
        }
    }
    
    # 清除轮询的最后一行进度显示
    print("\n" + "="*20 + " JSON OUTPUT " + "="*20)
    print(json.dumps(output_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 1. 提交任务
    tid = start_inference(VIDEO_URL)
    
    # 2. 如果成功，开始获取结果
    if tid:
        poll_result(tid)