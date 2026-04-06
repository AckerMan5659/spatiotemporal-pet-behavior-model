import time
import requests
import cv2
import numpy as np
from locust import HttpUser, task, between, events, constant

# ==================== 配置区 ====================
IMAGE_URLS = [
    f"https://storage.googleapis.com/eye-realtime-video/face-image/frame_000{i}.jpg" for i in range(16)
]
FPS = 1  # 模拟每个客户端的发送频率
CLIENT_ID_PREFIX = "locust_client"

cached_images = []
print("正在预加载 16 张图片到内存...")
for url in IMAGE_URLS:
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            cached_images.append(resp.content)
    except Exception as e:
        pass

if len(cached_images) == 0:
    print("⚠️ 警告：未能下载网络图片，正在生成随机测试图片用于压测...")
    for _ in range(16):
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, encoded_img = cv2.imencode('.jpg', dummy_img)
        cached_images.append(encoded_img.tobytes())
else:
    print(f"✅ 成功加载 {len(cached_images)} 张图片")


# ==================== 压测逻辑 ====================
class PetBehaviorUser(HttpUser):
    wait_time = constant(1 / FPS)

    def on_start(self):
        self.client_id = f"{CLIENT_ID_PREFIX}_{time.time_ns()}"
        self.frame_index = 0

    @task
    def push_frame(self):
        img_data = cached_images[self.frame_index % len(cached_images)]
        files = {'file': ('frame.jpg', img_data, 'image/jpeg')}
        data = {'client_id': self.client_id}

        # catch_response=True 允许我们拦截并自定义这条请求的结果
        # 初始 name 设置一个通用名称，稍后会被覆盖
        with self.client.post("/push_frame", data=data, files=files, name="Processing...", catch_response=True) as response:
            
            if response.status_code == 200:
                try:
                    res_json = response.json()
                    status = res_json.get("status")
                    
                    # 【核心魔法】：根据服务端的真实返回状态，动态修改它在面板上的名字
                    if status == "success":
                        # 这是第 16 帧，包含了完整的推理耗时
                        response.request_meta["name"] = "🎯 /push_frame (Inference E2E)"
                        response.success()
                    
                    elif status == "buffered":
                        # 这是 1-15 帧，只有极短的网络和内存写入耗时
                        response.request_meta["name"] = "⏳ /push_frame (Buffered 1-15)"
                        response.success()
                        
                    else:
                        response.request_meta["name"] = "❌ /push_frame (Unknown)"
                        response.failure(f"Unknown status: {status}")
                        
                except Exception as e:
                    response.request_meta["name"] = "❌ /push_frame (JSON Error)"
                    response.failure(f"JSON Parse Error: {e}")
            else:
                # 记录 HTTP 错误（如 500 服务器崩溃，或 502 网关错误）
                response.request_meta["name"] = f"❌ /push_frame (HTTP {response.status_code})"
                response.failure(f"HTTP Error {response.status_code}")

        self.frame_index += 1