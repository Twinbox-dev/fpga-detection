import requests
import json
import cv2
import numpy as np
import sys
import time
import os

SERVER_URL = "http://192.168.137.100:9000"

# 读取图片
input_path = sys.argv[1]
image = cv2.imread(input_path)
success, encoded = cv2.imencode('.jpg', image)
image_bytes = encoded.tobytes()

# 构造带文件名的上传字段
filename = os.path.basename(input_path)  # 提取文件名
files = {
    'image_file': (filename, image_bytes, 'image/jpeg')
}

upload_url = f"{SERVER_URL}/upload"
print(f"发送图片到 {upload_url} ...")
start = time.time()
resp = requests.post(upload_url, files=files, timeout=30)
cost = time.time() - start

print(f"响应时间: {cost:.3f}s")
print(f"状态码: {resp.status_code}")

if resp.status_code != 200:
    print(f"[错误] 服务器返回异常: {resp.text}")
    sys.exit(1)

result = resp.json()

print("\n=== 服务器返回结果 ===")
print(f"原始图片路径: {result['image_path']}")
print(f"检测到目标数: {len(result['objects'])}")
print(f"推理耗时: {result['inference_time_ms']} ms")
print(f"输出路径: {result['output_path']}")

for idx, obj in enumerate(result['objects']):
    print(f"  目标{idx+1}: {obj['class_name']} (ID:{obj['class_id']}), "
          f"置信度:{obj['confidence']:.4f}, "
          f"bbox:[{obj['x1']},{obj['y1']},{obj['x2']},{obj['y2']}]")

# 获取结果图片
from os.path import basename
filename_only = basename(result['output_path'])  # 例如 "ca_shang29_detect.jpg"
img_url = f"{SERVER_URL}/result_image?path={filename_only}"
print(f"\n获取结果图片: {img_url}")
img_resp = requests.get(img_url, timeout=30)
if img_resp.status_code == 200:
    out_name = f"result_{int(time.time())}.jpg"
    with open(out_name, 'wb') as f:
        f.write(img_resp.content)
    print(f"检测图片已保存: {out_name} (大小: {len(img_resp.content)} bytes)")
else:
    print(f"[错误] 获取结果图片失败: {img_resp.status_code}")
