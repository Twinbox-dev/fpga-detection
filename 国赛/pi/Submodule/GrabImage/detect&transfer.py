#!/usr/bin/env python3
"""
海康相机视频流 + Canny边缘检测 + 左右边缘点平均x竖线 + 自动FPGA推理（avg_x > 半宽时触发）
结果保存至 img_detect/ 和 json/ 文件夹
按 q 键退出
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

sys.path.append("./sdk")
from MvCameraControl_class import *

# ========== 可调全局参数 ==========
LEFT_PIXELS       = 100   # 取最左侧多少个边缘点
RIGHT_PIXELS      = 100   # 取最右侧多少个边缘点
CANNY_LOW         = 80    # Canny 低阈值
CANNY_HIGH        = 140   # Canny 高阈值
MIN_EDGE_POINTS   = 50    # 最少边缘点数，少于该值则不计算平均x
FPGA_URL          = "http://172.16.68.110:8080/predict"  # FPGA 服务地址
SEND_COOLDOWN     = 5   # 两次自动发送的最小间隔（秒），避免频繁请求
# =================================

# 创建保存目录
os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)

last_send_time = 0.0  # 上次自动发送的时间戳


def send_to_fpga_and_save(frame_bgr, timestamp):
    """
    在新线程中发送图像到 FPGA，保存检测图片和 JSON
    """
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        print("图像编码失败")
        return
    image_bytes = encoded.tobytes()

    print(f"\n[自动发送] 发送图片到 {FPGA_URL} ...")
    start = time.time()
    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
        cost = time.time() - start
        print(f"[自动发送] 响应时间: {cost:.3f}s, 状态码: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[自动发送] 请求失败: {resp.text}")
            return
        result = json.loads(resp.text)

        # 打印简要信息
        objects_count = len(result.get('objects', []))
        inference_ms = result.get('inference_time_ms', 'N/A')
        print(f"[自动发送] 目标数: {objects_count}, 推理耗时: {inference_ms} ms")

        # 保存 JSON
        json_name = f"json/result_{timestamp}.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[自动发送] JSON 已保存: {json_name}")

        # 保存检测图片
        if 'detect_image' in result and result['detect_image']:
            try:
                img_bytes = base64.b64decode(result['detect_image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_name = f"img_detect/detect_{timestamp}.jpg"
                cv2.imwrite(img_name, detect_img)
                print(f"[自动发送] 检测图片已保存: {img_name}")
            except Exception as e:
                print(f"[自动发送] 图片解码保存失败: {e}")
        else:
            print("[自动发送] 返回结果中没有检测图片字段")
        print("==================================\n")

    except Exception as e:
        print(f"[自动发送] 请求异常: {e}")


def main():
    global last_send_time

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
    print(f"自动发送条件：平均 x 坐标 > 图像宽度的一半，冷却间隔 {SEND_COOLDOWN}s")

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

            # ---------- Canny 边缘检测 ----------
            edges = cv2.Canny(small, CANNY_LOW, CANNY_HIGH)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # -----------------------------------

            # ---------- 计算左右边缘点平均 x ----------
            avg_x = None
            pts = np.column_stack(np.where(edges > 0))  # (y, x) 格式
            if len(pts) >= MIN_EDGE_POINTS:
                sorted_pts = pts[np.argsort(pts[:, 1])]  # 按 x 排序
                left_pts = sorted_pts[:min(LEFT_PIXELS, len(sorted_pts))]
                right_pts = sorted_pts[-min(RIGHT_PIXELS, len(sorted_pts)):]
                all_selected = np.vstack([left_pts, right_pts])
                avg_x = int(np.mean(all_selected[:, 1]))
            # -----------------------------------------

            # 计算并显示 FPS
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

                # ---------- 自动发送条件判断 ----------
                half_width = w_small // 2
                now = time.time()
                if avg_x > half_width and (now - last_send_time) >= SEND_COOLDOWN:
                    last_send_time = now
                    # 准备原始帧（转换为BGR三通道，因为FPGA接口期望彩色）
                    original_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    ts = int(time.time())
                    t = threading.Thread(target=send_to_fpga_and_save,
                                         args=(original_bgr, ts))
                    t.start()
                    cv2.putText(display, "Sending...", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                # -------------------------------------

            cv2.imshow("Camera Stream (Canny Edge + Auto FPGA)", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("相机已关闭")


if __name__ == "__main__":
    main()
