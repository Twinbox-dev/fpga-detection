import requests
import json
import base64
import cv2
import numpy as np
import sys
import time

FPGA_URL = "http://192.168.137.100:8080/predict"

# 读图片
image = cv2.imread(sys.argv[1])
success, encoded = cv2.imencode('.jpg', image)
image_bytes = encoded.tobytes()

# 发请求
print(f"发送图片到 {FPGA_URL} ...")
start = time.time()
resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
cost = time.time() - start

print(f"响应时间: {cost:.3f}s")
print(f"状态码: {resp.status_code}")

# 解析结果
result = json.loads(resp.text)

print("\n=== FPGA 返回结果 ===")
print(f"原始图片路径: {result['image_path']}")
print(f"检测到目标数: {result['objects']}")
print(f"推理耗时: {result['inference_time_ms']} ms")
print(f"检测图片(Base64前10字符): {result['detect_image'][:10]}...")
print(f"输出路径: {result['output_path']}")
print(f"检测图片大小: {result['detect_image_size']} bytes")

# 解码并保存检测图片
img_bytes = base64.b64decode(result['detect_image'])
img_array = np.frombuffer(img_bytes, np.uint8)
detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
out_name = f"result_{int(time.time())}.jpg"
cv2.imwrite(out_name, detect_img)
print(f"\n检测图片已保存: {out_name}")