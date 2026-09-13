#!/usr/bin/env python3
"""
海康相机视频流 + FPGA 推理（非阻塞版）
按下空格键发送当前帧到 FPGA，保存检测结果和 JSON，视频流不受影响
按 q 键退出
"""

import sys
import time
import json
import base64
import threading
import numpy as np
import cv2
import requests
from ctypes import *

sys.path.append("./sdk")
from MvCameraControl_class import *

FPGA_URL = "http://172.16.68.110:8080/predict"


def send_to_fpga_thread(frame_bgr, timestamp):
    """
    在新线程中发送图像到 FPGA，打印结果并保存
    """
    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        print("图像编码失败")
        return
    image_bytes = encoded.tobytes()

    print(f"\n[线程] 发送图片到 {FPGA_URL} ...")
    start = time.time()
    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=30)
        cost = time.time() - start
        print(f"[线程] 响应时间: {cost:.3f}s, 状态码: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[线程] 请求失败: {resp.text}")
            return
        result = json.loads(resp.text)

        # 打印解析结果
        print("\n=== FPGA 返回结果 ===")
        print(f"原始图片路径: {result.get('image_path', 'N/A')}")
        print(f"检测到目标数: {len(result.get('objects', []))}")
        print(f"推理耗时: {result.get('inference_time_ms', 'N/A')} ms")
        print(f"检测图片(Base64前10字符): {result.get('detect_image', '')[:10]}...")
        print(f"输出路径: {result.get('output_path', 'N/A')}")
        print(f"检测图片大小: {result.get('detect_image_size', 0)} bytes")

        # 保存 JSON
        json_name = f"result_{timestamp}.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[线程] JSON 已保存: {json_name}")

        # 保存检测图片
        if 'detect_image' in result and result['detect_image']:
            try:
                img_bytes = base64.b64decode(result['detect_image'])
                img_array = np.frombuffer(img_bytes, np.uint8)
                detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_name = f"detect_{timestamp}.jpg"
                cv2.imwrite(img_name, detect_img)
                print(f"[线程] 检测图片已保存: {img_name}")
            except Exception as e:
                print(f"[线程] 图片解码保存失败: {e}")
        else:
            print("[线程] 返回结果中没有检测图片字段")
        print("==================================\n")

    except Exception as e:
        print(f"[线程] 请求异常: {e}")


def main():
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

    print("相机已启动，按 空格键 发送当前帧到 FPGA（非阻塞），按 q 键退出")

    frame_count = 0
    fps_start = time.time()
    latest_frame = None  # 保存当前帧（BGR格式，已缩放）

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                continue

            raw = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            small = cv2.resize(raw, (int(stFrameInfo.nWidth/4), int(stFrameInfo.nHeight/4)))
            display = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            latest_frame = display.copy()  # 保存副本供线程使用

            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: send to FPGA | q: quit", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Camera Stream", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if latest_frame is None:
                    print("尚无可用帧")
                    continue
                ts = int(time.time())
                # 在新线程中发送请求，不阻塞主循环
                t = threading.Thread(target=send_to_fpga_thread, args=(latest_frame.copy(), ts))
                t.start()

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