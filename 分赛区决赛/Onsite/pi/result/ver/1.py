#!/usr/bin/env python3
"""
整合版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣 + Flask Web前端
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
from flask import Flask, Response, render_template_string
from io import BytesIO

sys.path.append("./sdk")
from MvCameraControl_class import *
from Arm_Lib import Arm_Device

# ===================== 用户可配置参数 =====================
LEFT_PIXELS       = 100
RIGHT_PIXELS      = 100
CANNY_LOW         = 50
CANNY_HIGH        = 150
MIN_EDGE_POINTS   = 50
FPGA_URL          = "http://172.16.68.110:8080/predict"
SEND_COOLDOWN     = 5

SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 1.0
ARM_SPEED = 800
BUZZER_DELAY = 10
FORCE_TEST = None   # 调试用
# ==========================================================

os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)

# ---------- 全局变量（供Flask访问） ----------
latest_frame_raw = None      # 原始帧 (BGR)
latest_frame_edge = None     # 边缘帧 (BGR)
latest_sent_raw = None       # 最近发送给FPGA的原图 (BGR)
latest_detected_img = None   # FPGA返回的检测图 (BGR)
frame_lock = threading.Lock()

# ---------- Flask 应用 ----------
app = Flask(__name__)

@app.route('/')
def index():
    # 内嵌HTML页面
    html = '''
    <!DOCTYPE html>
    <html>
    <head><title>机械臂分拣监控</title></head>
    <body style="background:#222; color:white; font-family:Arial;">
        <h2>实时监控</h2>
        <table border="0" cellpadding="10">
            <tr>
                <td><b>原始视频流</b></td>
                <td><b>Canny边缘检测</b></td>
            </tr>
            <tr>
                <td><img src="/video_feed_raw" width="640" height="480"></td>
                <td><img src="/video_feed_edge" width="640" height="480"></td>
            </tr>
            <tr>
                <td><b>最近发送给FPGA的原图</b></td>
                <td><b>FPGA检测结果图</b></td>
            </tr>
            <tr>
                <td><img src="/sent_image" width="640" height="480" id="sent_img"></td>
                <td><img src="/detected_image" width="640" height="480" id="detected_img"></td>
            </tr>
        </table>
        <script>
            // 每2秒刷新静态图片
            setInterval(function(){
                document.getElementById('sent_img').src = '/sent_image?'+new Date().getTime();
                document.getElementById('detected_img').src = '/detected_image?'+new Date().getTime();
            }, 2000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

def gen_frames(get_frame_func):
    """生成MJPEG流的生成器"""
    while True:
        frame = get_frame_func()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)  # ~33fps

def get_raw_frame():
    with frame_lock:
        if latest_frame_raw is not None:
            return latest_frame_raw.copy()
        return None

def get_edge_frame():
    with frame_lock:
        if latest_frame_edge is not None:
            return latest_frame_edge.copy()
        return None

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

Arm = Arm_Device()
time.sleep(0.1)

p_initial = [90, 90, 90, 0, 90, 90]
p_Z = [90, 38, 26, 22, 90, 90]
p_Z_1 = [90, 90, 0, 90, 90, 90]
p_OK_1 = [90, 90, 90, 0, 90, 90]
p_OK_2 = [-10, 90, 90, 0, 90, 90]
p_OK   = [-10, 65, 0, 30, 90, 90]
p_NG_1 = [90, 90, 90, 0, 90, 90]
p_NG   = [180, 55, 11, 24, 90, 90]

arm_busy = False

def execute_arm_action(action_type):
    global arm_busy
    arm_busy = True
    print(f"\n=== 执行 {action_type} 动作 ===")
    Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    zhua_qu_xipan()
    time.sleep(SUCTION_WAIT_TIME)
    Arm.Arm_serial_servo_write6(*p_Z_1, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
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
    shi_fang_xipan()
    time.sleep(2)
    Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    print(f"{action_type} 动作完成，已复位。\n")
    arm_busy = False

def send_to_fpga_and_decide(frame_bgr, timestamp):
    global latest_sent_raw, latest_detected_img
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        print("图像编码失败")
        return
    image_bytes = encoded.tobytes()
    print(f"\n[触发] 发送图片到 {FPGA_URL} ...")
    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
        if resp.status_code != 200:
            print(f"[触发] 请求失败: {resp.status_code}")
            return
        result = json.loads(resp.text)
        objects_count = len(result.get('objects', []))
        inference_ms = result.get('inference_time_ms', 'N/A')
        print(f"[触发] 目标数: {objects_count}, 推理耗时: {inference_ms} ms")

        # 保存JSON
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

        # 更新发送的原图
        with frame_lock:
            latest_sent_raw = frame_bgr.copy()

        # 决定动作
        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0:
            action = "OK"
        else:
            action = "NG"
        execute_arm_action(action)

    except Exception as e:
        print(f"[触发] 请求异常: {e}")

def main():
    global last_send_time, arm_busy, latest_frame_raw, latest_frame_edge

    # 启动Flask线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask Web服务已启动，请访问 http://<本机IP>:5000")

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

            # Canny边缘检测
            edges = cv2.Canny(small, CANNY_LOW, CANNY_HIGH)
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # 原始帧转为BGR用于显示
            raw_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

            # 更新全局帧
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

            # 可选：在终端显示FPS
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