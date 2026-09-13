#!/usr/bin/env python3
"""
增强版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣 + 美观Web前端
新增功能：
  - 实时显示机械臂状态（空闲/工作中）
  - 手动复位按钮
  - 显示最后一次动作结果（OK/NG）
  - 美化UI（深色主题，卡片布局）
"""

import sys
import time
import json
import base64
import threading
import os
import numpy as np
import cv2
import requests
from ctypes import *
import serial
from flask import Flask, Response, render_template_string, request, jsonify
from io import BytesIO

sys.path.append("./sdk")
from MvCameraControl_class import *
from Arm_Lib import Arm_Device

# ===================== 用户可配置参数 =====================
# ---- 视觉检测参数 ----
LEFT_PIXELS       = 100
RIGHT_PIXELS      = 100
CANNY_LOW         = 50
CANNY_HIGH        = 150
MIN_EDGE_POINTS   = 50
FPGA_URL          = "http://172.16.68.110:8080/predict"
SEND_COOLDOWN     = 5

# ---- 串口（吸盘）参数 ----
SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 1.0

# ---- 机械臂参数 ----
ARM_SPEED = 800
BUZZER_DELAY = 10
FORCE_TEST = None   # 调试用
# ==========================================================

os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)

# ---------- 全局变量 ----------
last_send_time = 0.0
arm_busy = False
last_action_result = "无"   # 记录最后一次动作结果
latest_frame_raw = None
latest_frame_edge = None
latest_sent_raw = None
latest_detected_img = None
frame_lock = threading.Lock()

# ---------- Flask 应用 ----------
app = Flask(__name__)

# ========== 前端页面 ==========
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>机械臂分拣监控系统</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: #1a1a2e;
        color: #eee;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 20px;
    }
    .container { max-width: 1400px; margin: 0 auto; }
    h1 {
        text-align: center;
        margin-bottom: 20px;
        color: #e94560;
        font-weight: 300;
        letter-spacing: 2px;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }
    .card {
        background: #16213e;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .card-title {
        font-size: 14px;
        color: #a0a0b0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card img {
        width: 100%;
        height: auto;
        border-radius: 8px;
        display: block;
        background: #0f3460;
    }
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f3460;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 20px;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-label {
        color: #888;
        font-size: 14px;
    }
    .status-value {
        font-size: 18px;
        font-weight: bold;
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green { background: #4caf50; }
    .dot-red { background: #e53935; animation: blink 1s infinite; }
    @keyframes blink {
        0%,100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .btn-reset {
        background: #e94560;
        border: none;
        color: white;
        padding: 10px 28px;
        border-radius: 40px;
        font-size: 16px;
        cursor: pointer;
        transition: background 0.2s;
    }
    .btn-reset:hover { background: #d63851; }
    .btn-reset:disabled { background: #555; cursor: not-allowed; }
    .footer {
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-top: 20px;
    }
    @media (max-width: 768px) {
        .grid { grid-template-columns: 1fr; }
        .status-bar { flex-wrap: wrap; gap: 10px; }
    }
</style>
</head>
<body>
<div class="container">
    <h1>🤖 机械臂分拣监控</h1>

    <!-- 状态栏 -->
    <div class="status-bar">
        <div class="status-item">
            <span class="status-label">机械臂状态</span>
            <span id="armStatus" class="status-value">
                <span class="status-dot dot-green"></span> 空闲
            </span>
        </div>
        <div class="status-item">
            <span class="status-label">最后动作</span>
            <span id="lastAction" class="status-value" style="color:#ff9800;">无</span>
        </div>
        <button id="resetBtn" class="btn-reset" onclick="resetArm()">🔄 复位机械臂</button>
    </div>

    <!-- 四个画面 -->
    <div class="grid">
        <div class="card">
            <div class="card-title">📷 原始视频流</div>
            <img src="/video_feed_raw" alt="原始流">
        </div>
        <div class="card">
            <div class="card-title">🔍 Canny 边缘检测</div>
            <img src="/video_feed_edge" alt="边缘流">
        </div>
        <div class="card">
            <div class="card-title">📤 最近发送给 FPGA 的原图</div>
            <img id="sentImg" src="/sent_image" alt="发送原图">
        </div>
        <div class="card">
            <div class="card-title">🎯 FPGA 检测结果</div>
            <img id="detectedImg" src="/detected_image" alt="检测结果">
        </div>
    </div>
    <div class="footer">系统运行中 · 按 Ctrl+C 停止</div>
</div>

<script>
// 定时刷新静态图片和状态
function refreshStatic() {
    document.getElementById('sentImg').src = '/sent_image?' + new Date().getTime();
    document.getElementById('detectedImg').src = '/detected_image?' + new Date().getTime();
}
setInterval(refreshStatic, 2000);

// 定时获取状态
function fetchStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            const statusEl = document.getElementById('armStatus');
            const actionEl = document.getElementById('lastAction');
            const resetBtn = document.getElementById('resetBtn');

            if (data.busy) {
                statusEl.innerHTML = '<span class="status-dot dot-red"></span> 工作中';
                resetBtn.disabled = true;
            } else {
                statusEl.innerHTML = '<span class="status-dot dot-green"></span> 空闲';
                resetBtn.disabled = false;
            }
            actionEl.textContent = data.last_action || '无';
        })
        .catch(() => {});
}
setInterval(fetchStatus, 1000);
fetchStatus();

// 复位机械臂
function resetArm() {
    const btn = document.getElementById('resetBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 复位中...';
    fetch('/reset', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            alert(data.message);
            btn.textContent = '🔄 复位机械臂';
            btn.disabled = false;
        })
        .catch(err => {
            alert('复位失败');
            btn.textContent = '🔄 复位机械臂';
            btn.disabled = false;
        });
}
</script>
</body>
</html>
'''

# ========== Flask 路由 ==========
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/status')
def status():
    """返回机械臂状态和最后动作"""
    global arm_busy, last_action_result
    return jsonify({
        'busy': arm_busy,
        'last_action': last_action_result
    })

@app.route('/reset', methods=['POST'])
def reset():
    """手动复位机械臂（异步执行，不阻塞主循环）"""
    global arm_busy
    if arm_busy:
        return jsonify({'message': '机械臂正在工作中，无法复位'}), 409
    # 在新线程中执行复位
    t = threading.Thread(target=do_reset, daemon=True)
    t.start()
    return jsonify({'message': '复位指令已发送，请等待完成'})

def do_reset():
    """实际复位函数"""
    global arm_busy, last_action_result
    arm_busy = True
    try:
        Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
        time.sleep(ARM_SPEED / 1000 + 0.3)
        last_action_result = "手动复位"
    except Exception as e:
        print(f"复位异常: {e}")
    finally:
        arm_busy = False

# ---------- 视频流生成器 ----------
def gen_frames(get_frame_func):
    while True:
        frame = get_frame_func()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)

def get_raw_frame():
    with frame_lock:
        return latest_frame_raw.copy() if latest_frame_raw is not None else None

def get_edge_frame():
    with frame_lock:
        return latest_frame_edge.copy() if latest_frame_edge is not None else None

@app.route('/video_feed_raw')
def video_feed_raw():
    return Response(gen_frames(get_raw_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_edge')
def video_feed_edge():
    return Response(gen_frames(get_edge_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sent_image')
def sent_image():
    with frame_lock:
        if latest_sent_raw is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_sent_raw)
            if ret:
                return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return '', 204

@app.route('/detected_image')
def detected_image():
    with frame_lock:
        if latest_detected_img is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_detected_img)
            if ret:
                return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return '', 204

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ---------- 串口初始化 ----------
try:
    com = serial.Serial(SUCTION_SERIAL_PORT, SUCTION_SERIAL_BAUD, timeout=1)
    time.sleep(0.1)
except Exception as e:
    print(f"串口打开失败: {e}")
    exit(1)

zhua_qu = bytes.fromhex('A0 01 01 A2')
shi_fang = bytes.fromhex('A0 01 00 A1')

def zhua_qu_xipan():
    try:
        com.write(zhua_qu)
        time.sleep(0.1)
        print("吸盘已启动")
    except Exception as e:
        print(f"吸盘启动失败: {e}")

def shi_fang_xipan():
    try:
        com.write(shi_fang)
        time.sleep(0.1)
        print("吸盘已释放")
    except Exception as e:
        print(f"吸盘释放失败: {e}")

# ---------- 机械臂初始化 ----------
Arm = Arm_Device()
time.sleep(0.1)

# ---------- 定义点位 ----------
p_initial = [90, 90, 90, 0, 90, 90]
p_Z = [90, 38, 26, 22, 90, 90]
p_Z_1 = [90, 90, 0, 90, 90, 90]
p_OK_1 = [90, 90, 90, 0, 90, 90]
p_OK_2 = [-10, 90, 90, 0, 90, 90]
p_OK   = [-10, 65, 0, 30, 90, 90]
p_NG_1 = [90, 90, 90, 0, 90, 90]
p_NG   = [180, 55, 11, 24, 90, 90]

# ---------- 核心动作函数 ----------
def execute_arm_action(action_type):
    global arm_busy, last_action_result
    arm_busy = True
    print(f"\n=== 执行 {action_type} 动作 ===")
    try:
        # 1. 下降到吸取位置
        Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
        time.sleep(ARM_SPEED / 1000 + 0.3)
        # 2. 启动吸盘
        zhua_qu_xipan()
        time.sleep(SUCTION_WAIT_TIME)
        # 3. 抬高
        Arm.Arm_serial_servo_write6(*p_Z_1, ARM_SPEED)
        time.sleep(ARM_SPEED / 1000 + 0.3)
        # 4. 执行运动
        if action_type == "OK":
            Arm.Arm_serial_servo_write6(*p_OK_1, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            Arm.Arm_serial_servo_write6(*p_OK_2, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            Arm.Arm_serial_servo_write6(*p_OK, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
        elif action_type == "NG":
            Arm.Arm_serial_servo_write6(*p_NG_1, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            Arm.Arm_serial_servo_write6(*p_NG, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            Arm.Arm_Buzzer_On(delay=BUZZER_DELAY)
            time.sleep(BUZZER_DELAY * 0.1 + 0.2)
        # 5. 释放吸盘
        shi_fang_xipan()
        # 6. 等待2秒后复位
        time.sleep(2)
        Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
        time.sleep(ARM_SPEED / 1000 + 0.3)
        last_action_result = action_type
        print(f"{action_type} 动作完成，已复位。\n")
    except Exception as e:
        print(f"动作执行异常: {e}")
        last_action_result = f"错误: {e}"
    finally:
        arm_busy = False

def send_to_fpga_and_decide(frame_bgr, timestamp):
    global latest_sent_raw, latest_detected_img
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        return
    image_bytes = encoded.tobytes()
    print(f"\n[触发] 发送图片到 {FPGA_URL} ...")
    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
        if resp.status_code != 200:
            return
        result = json.loads(resp.text)
        objects_count = len(result.get('objects', []))
        inference_ms = result.get('inference_time_ms', 'N/A')
        print(f"[触发] 目标数: {objects_count}, 推理耗时: {inference_ms} ms")

        # 保存 JSON
        json_name = f"json/result_{timestamp}.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存检测图并更新全局变量
        if 'detect_image' in result and result['detect_image']:
            try:
                img_bytes = base64.b64decode(result['detect_image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_name = f"img_detect/detect_{timestamp}.jpg"
                cv2.imwrite(img_name, detect_img)
                with frame_lock:
                    latest_detected_img = detect_img.copy()
            except Exception as e:
                print(f"检测图片保存失败: {e}")

        with frame_lock:
            latest_sent_raw = frame_bgr.copy()

        # 决定动作
        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0:
            action = "OK"
        else:
            action = "NG"

        # 执行机械臂动作（阻塞）
        execute_arm_action(action)

    except Exception as e:
        print(f"[触发] 请求异常: {e}")

# ---------- 主循环 ----------
def main():
    global last_send_time, arm_busy, latest_frame_raw, latest_frame_edge

    # 启动 Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask Web 服务已启动，请访问 http://<本机IP>:5000")

    # 相机初始化
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("未找到相机设备")
        return
    print(f"找到 {deviceList.nDeviceNum} 个相机")
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue
    cam.MV_CC_StartGrabbing()
    data_buf = (c_ubyte * nPayloadSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    print("相机已启动，按 Ctrl+C 停止")

    last_send_time = 0.0
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                continue

            raw = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            scale = 4
            h_small = int(stFrameInfo.nHeight / scale)
            w_small = int(stFrameInfo.nWidth / scale)
            small = cv2.resize(raw, (w_small, h_small))

            edges = cv2.Canny(small, CANNY_LOW, CANNY_HIGH)
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            raw_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

            with frame_lock:
                latest_frame_raw = raw_bgr.copy()
                latest_frame_edge = edge_bgr.copy()

            # 计算平均x
            avg_x = None
            pts = np.column_stack(np.where(edges > 0))
            if len(pts) >= MIN_EDGE_POINTS:
                sorted_pts = pts[np.argsort(pts[:, 1])]
                left_pts = sorted_pts[:min(LEFT_PIXELS, len(sorted_pts))]
                right_pts = sorted_pts[-min(RIGHT_PIXELS, len(sorted_pts)):]
                all_selected = np.vstack([left_pts, right_pts])
                avg_x = int(np.mean(all_selected[:, 1]))

            # 触发检测
            if avg_x is not None:
                half_width = w_small // 2
                now = time.time()
                if avg_x > half_width and (now - last_send_time) >= SEND_COOLDOWN and not arm_busy:
                    last_send_time = now
                    original_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    ts = int(time.time())
                    send_to_fpga_and_decide(original_bgr, ts)

            # 终端 FPS 显示
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                print(f"FPS: {frame_count/elapsed:.1f}")
                frame_count = 0
                fps_start = time.time()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        com.close()
        print("系统已关闭")

if __name__ == "__main__":
    main()