#!/usr/bin/env python3
"""
海康相机视频流展示程序（Canny + 形态学闭运算 + 轮廓筛选 + 最小外接圆）
适用于：纯黑背景，噪点只在圆形内部
运行方式: python3 show_stream.py
按 q 键退出
"""

import sys
import time
import numpy as np
import cv2
from ctypes import *

sys.path.append("./sdk")
from MvCameraControl_class import *


def main():
    # ---------- 全局参数（请根据实际调整）----------
    RADIUS_MIN = 70       # 圆形半径下限（像素）
    RADIUS_MAX = 110      # 圆形半径上限（像素）
    CANNY_LOW = 40
    CANNY_HIGH = 120
    MORPH_KERNEL_SIZE = 5  # 闭运算核大小，用于连接断裂边缘
    # -----------------------------------------------

    # 枚举设备
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
    frame_count = 0
    fps_start = time.time()

    # 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                continue

            # 原始图像
            image = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            # 缩放显示
            image = cv2.resize(image, (int(stFrameInfo.nWidth/4), int(stFrameInfo.nHeight/4)))

            # Canny 边缘检测
            edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH)

            # 形态学闭运算：连接断裂的边缘
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 转换为彩色用于显示
            display = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

            # ---------- 轮廓提取与筛选 ----------
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                # 按面积降序排序
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

                # 遍历轮廓，找到第一个符合半径范围的
                found = False
                for cnt in contours_sorted:
                    area = cv2.contourArea(cnt)
                    if area < 100:  # 忽略极小面积（噪声）
                        continue
                    # 由面积估算等效半径
                    estimated_radius = np.sqrt(area / np.pi)
                    if RADIUS_MIN <= estimated_radius <= RADIUS_MAX:
                        # 使用最小外接圆（不受内部噪点影响）
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        # 再次检查外接圆半径是否在范围内（可选）
                        if RADIUS_MIN <= radius <= RADIUS_MAX:
                            # 绘制圆（绿色边框）
                            cv2.circle(display, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                            # 红色圆心点
                            cv2.circle(display, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                            found = True
                            break  # 只处理最大的符合条件的轮廓

                if not found:
                    # 如果没有找到符合半径的，则用最大轮廓（防止漏检）
                    largest_cnt = contours_sorted[0]
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_cnt)
                    cv2.circle(display, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                    cv2.circle(display, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            # ------------------------------------

            # 显示FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start)
                print(f"FPS: {fps:.1f}")

            cv2.putText(display, f"FPS: {frame_count/(time.time()-fps_start):.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Camera Stream (Robust Circle Detection)", display)

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