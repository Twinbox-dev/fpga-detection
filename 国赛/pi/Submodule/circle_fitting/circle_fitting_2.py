#!/usr/bin/env python3
"""
海康相机视频流展示程序（带Canny边缘检测 + 左右边缘点平均x竖线，含最小点数过滤）
运行方式: python3 show_stream.py
按 q 键退出
"""

import sys
import time
import numpy as np
import cv2
from ctypes import *

# 添加SDK路径（根据实际位置修改）
sys.path.append("./sdk")
from MvCameraControl_class import *

# ========== 可调全局参数 ==========
LEFT_PIXELS       = 100   # 取最左侧多少个边缘点
RIGHT_PIXELS      = 100   # 取最右侧多少个边缘点
CANNY_LOW         = 80    # Canny 低阈值
CANNY_HIGH        = 140   # Canny 高阈值
MIN_EDGE_POINTS   = 50    # 最少边缘点数，少于该值则不计算平均x
# =================================


def main():
    # 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("未找到相机设备")
        return

    print(f"找到 {deviceList.nDeviceNum} 个相机")

    # 选择第一个设备
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # 创建相机
    cam = MvCamera()
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # 获取负载大小
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    # 开始取流
    cam.MV_CC_StartGrabbing()

    data_buf = (c_ubyte * nPayloadSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    print("相机已启动，按 q 键退出")

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                continue

            # 转为灰度图像（原始数据为单通道）
            image = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            # 缩放到合适显示尺寸
            scale = 4
            h_small = int(stFrameInfo.nHeight / scale)
            w_small = int(stFrameInfo.nWidth / scale)
            image = cv2.resize(image, (w_small, h_small))

            # ---------- Canny 边缘检测 ----------
            edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # -----------------------------------

            # ---------- 计算左右边缘点平均 x ----------
            avg_x = None
            pts = np.column_stack(np.where(edges > 0))  # (y, x) 格式
            # 只有当边缘点总数 >= MIN_EDGE_POINTS 时才计算，避免跳动
            if len(pts) >= MIN_EDGE_POINTS:
                sorted_pts = pts[np.argsort(pts[:, 1])]  # 按 x 排序
                left_pts = sorted_pts[:min(LEFT_PIXELS, len(sorted_pts))]
                right_pts = sorted_pts[-min(RIGHT_PIXELS, len(sorted_pts)):]
                all_selected = np.vstack([left_pts, right_pts])
                avg_x = int(np.mean(all_selected[:, 1]))
            # -----------------------------------------

            # 计算并显示 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start)
                print(f"FPS: {fps:.1f}")

            cv2.putText(display, f"FPS: {frame_count/(time.time()-fps_start):.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 画平均 x 竖线（仅当 avg_x 有效时）
            if avg_x is not None:
                cv2.line(display, (avg_x, 0), (avg_x, h_small-1), (0, 0, 255), 2)
                cv2.putText(display, f"AvgX={avg_x}", (avg_x+5, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Camera Stream (Canny Edge)", display)

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
