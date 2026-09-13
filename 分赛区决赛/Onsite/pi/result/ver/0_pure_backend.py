#!/usr/bin/env python3
"""
整合版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣
当 avg_x > 图像半宽 时触发 FPGA 检测，
若返回目标数 == 0 则执行 OK 动作，否则执行 NG 动作。
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

sys.path.append("./sdk")
from MvCameraControl_class import *
from Arm_Lib import Arm_Device

# ===================== 用户可配置参数 =====================
# ---- 视觉检测参数 ----
LEFT_PIXELS       = 100   # 取最左侧多少个边缘点
RIGHT_PIXELS      = 100   # 取最右侧多少个边缘点
CANNY_LOW         = 50    # Canny 低阈值
CANNY_HIGH        = 150   # Canny 高阈值
MIN_EDGE_POINTS   = 50    # 最少边缘点数，少于该值则不计算平均x
FPGA_URL          = "http://172.16.68.110:8080/predict"  # FPGA 服务地址
SEND_COOLDOWN     = 5     # 两次自动发送的最小间隔（秒）

# ---- 串口（吸盘）参数 ----
SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 1.0   # 吸盘启动后等待吸力达到最大的时间（秒）

# ---- 机械臂参数 ----
ARM_SPEED = 800   # 运动速度（毫秒）
BUZZER_DELAY = 10 # 蜂鸣器持续时间（单位：100ms，10=1000ms）

# ---- 测试状态（仅用于调试，实际由 FPGA 结果决定）----
FORCE_TEST = None   # 设为 "OK" 或 "NG" 可强制指定，None 则使用 FPGA 结果
# ==========================================================

# 创建保存目录
os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)

# ---------- 全局状态 ----------
last_send_time = 0.0
arm_busy = False   # 机械臂是否正在运动中

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
p_initial = [90, 90, 90, 0, 90, 90]   # 复位位置
p_Z = [90, 38, 26, 22, 90, 90]        # 吸取位置
p_Z_1 = [90, 90, 0, 90, 90, 90]       # 吸取后抬高

p_OK_1 = [90, 90, 90, 0, 90, 90]
p_OK_2 = [-10, 90, 90, 0, 90, 90]
p_OK   = [-10, 65, 0, 30, 90, 90]

p_NG_1 = [90, 90, 90, 0, 90, 90]
p_NG   = [180, 55, 11, 24, 90, 90]   # 原196改为180，避免write6越界


def execute_arm_action(action_type):
    """
    执行完整的机械臂+吸盘动作序列
    action_type: 'OK' 或 'NG'
    """
    global arm_busy
    arm_busy = True
    print(f"\n=== 执行 {action_type} 动作 ===")

    # 1. 下降到吸取位置
    print("步骤1: 下降到吸取位置")
    Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)

    # 2. 启动吸盘
    zhua_qu_xipan()
    print(f"等待 {SUCTION_WAIT_TIME} 秒让吸力达到最大...")
    time.sleep(SUCTION_WAIT_TIME)

    # 3. 抬高
    print("步骤2: 抬高")
    Arm.Arm_serial_servo_write6(*p_Z_1, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)

    # 4. 执行 OK 或 NG 运动
    print(f"步骤3: 执行 {action_type} 运动")
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
        # NG 时蜂鸣器响
        print("NG → 蜂鸣器响")
        Arm.Arm_Buzzer_On(delay=BUZZER_DELAY)
        time.sleep(BUZZER_DELAY * 0.1 + 0.2)

    # 5. 释放吸盘
    print("步骤4: 释放吸盘")
    shi_fang_xipan()

    # 6. 等待2秒后复位
    print("等待 2 秒后复位...")
    time.sleep(2)
    Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)

    print(f"{action_type} 动作完成，已复位。\n")
    arm_busy = False


def send_to_fpga_and_decide(frame_bgr, timestamp):
    """
    发送图像到 FPGA，解析结果，并根据目标数决定动作类型
    """
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        print("图像编码失败")
        return

    image_bytes = encoded.tobytes()
    print(f"\n[触发] 发送图片到 {FPGA_URL} ...")
    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
        if resp.status_code != 200:
            print(f"[触发] 请求失败: {resp.status_code} {resp.text}")
            return
        result = json.loads(resp.text)

        objects_count = len(result.get('objects', []))
        inference_ms = result.get('inference_time_ms', 'N/A')
        print(f"[触发] 目标数: {objects_count}, 推理耗时: {inference_ms} ms")

        # 保存 JSON 和检测图片（可选）
        json_name = f"json/result_{timestamp}.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if 'detect_image' in result and result['detect_image']:
            try:
                img_bytes = base64.b64decode(result['detect_image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_name = f"img_detect/detect_{timestamp}.jpg"
                cv2.imwrite(img_name, detect_img)
            except Exception as e:
                print(f"检测图片保存失败: {e}")

        # 根据目标数决定动作
        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0:
            action = "OK"
        else:
            action = "NG"

        # 执行机械臂动作（阻塞，直到完成）
        execute_arm_action(action)

    except Exception as e:
        print(f"[触发] 请求异常: {e}")


def main():
    global last_send_time, arm_busy

    # ---------- 相机初始化 ----------
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

    print("相机已启动，按 q 键退出")
    print(f"触发条件：平均 x 坐标 > 图像宽度的一半，冷却间隔 {SEND_COOLDOWN}s")

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                continue

            # 原始帧（单通道灰度）
            raw = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))

            # 缩放到显示尺寸
            scale = 4
            h_small = int(stFrameInfo.nHeight / scale)
            w_small = int(stFrameInfo.nWidth / scale)
            small = cv2.resize(raw, (w_small, h_small))

            # Canny 边缘检测
            edges = cv2.Canny(small, CANNY_LOW, CANNY_HIGH)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # 计算左右边缘点平均 x
            avg_x = None
            pts = np.column_stack(np.where(edges > 0))
            if len(pts) >= MIN_EDGE_POINTS:
                sorted_pts = pts[np.argsort(pts[:, 1])]
                left_pts = sorted_pts[:min(LEFT_PIXELS, len(sorted_pts))]
                right_pts = sorted_pts[-min(RIGHT_PIXELS, len(sorted_pts)):]
                all_selected = np.vstack([left_pts, right_pts])
                avg_x = int(np.mean(all_selected[:, 1]))

            # 显示 FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "Press q to quit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 画平均 x 竖线
            if avg_x is not None:
                cv2.line(display, (avg_x, 0), (avg_x, h_small-1), (0, 0, 255), 2)
                cv2.putText(display, f"AvgX={avg_x}", (avg_x+5, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 触发条件：avg_x > 半宽 且 冷却时间已过 且 机械臂空闲
                half_width = w_small // 2
                now = time.time()
                if (avg_x > half_width and 
                    (now - last_send_time) >= SEND_COOLDOWN and 
                    not arm_busy):
                    last_send_time = now
                    # 准备原始帧（转为BGR三通道，FPGA接口期望彩色）
                    original_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    ts = int(time.time())
                    # 在主线程中同步执行（阻塞），也可改为线程，但注意 arm_busy 保护
                    send_to_fpga_and_decide(original_bgr, ts)
                    cv2.putText(display, "Action Done", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Integrated System (Edge + Auto Sort)", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        com.close()
        print("系统已关闭")


if __name__ == "__main__":
    main()