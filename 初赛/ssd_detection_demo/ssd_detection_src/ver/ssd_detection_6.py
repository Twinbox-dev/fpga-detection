# -*- coding: utf-8 -*-
"""
SSD缺陷检测服务（运行在 Cyclone V SoC 上）
集成了 ctypes 封装 + TCP 服务端，单文件完成所有功能。

用法:
    python ssd_detection.py
    python3 ssd_detection.py --model ssd_mobilenet_v1_opt.nb --port 9000
"""

import argparse
import ctypes
import json
import os
import socket
import struct
import sys
import threading

# ==================== 协议常量 ====================
MAGIC = b'REQ\0'

# ==================== 线程锁 ====================
# C库内部 predictor 是全局变量，非线程安全
detect_lock = threading.Lock()


# ==================== C库封装 ====================

class SSDDetector:
    """SSD缺陷检测器，封装C共享库调用"""

    def __init__(self, model_path, lib_path=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError("模型文件不存在: " + str(model_path))

        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libssd_detection.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError("共享库不存在: " + str(lib_path))

        self._lib = ctypes.CDLL(lib_path)

        self._lib.ssd_init.argtypes = [ctypes.c_char_p]
        self._lib.ssd_init.restype = ctypes.c_int

        self._lib.ssd_detect.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.ssd_detect.restype = ctypes.c_int

        self._lib.ssd_log_start.argtypes = [ctypes.c_char_p]
        self._lib.ssd_log_start.restype = ctypes.c_int

        self._lib.ssd_log_stop.argtypes = []
        self._lib.ssd_log_stop.restype = None

        self._lib.ssd_release.argtypes = []
        self._lib.ssd_release.restype = None

        ret = self._lib.ssd_init(model_path.encode("utf-8"))
        if ret != 0:
            raise RuntimeError("模型初始化失败: " + str(model_path))

        self._released = False

    def log_start(self, log_path):
        self._lib.ssd_log_start(log_path.encode("utf-8"))

    def log_stop(self):
        self._lib.ssd_log_stop()

    def detect(self, img_path, output_path, json_buf_size=8192):
        if self._released:
            raise RuntimeError("检测器已释放")

        if not os.path.exists(img_path):
            raise FileNotFoundError("图片不存在: " + str(img_path))

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_buf = ctypes.create_string_buffer(json_buf_size)
        ret = self._lib.ssd_detect(
            img_path.encode("utf-8"),
            output_path.encode("utf-8"),
            result_buf,
            json_buf_size,
        )
        if ret != 0:
            raise RuntimeError("检测失败: " + str(img_path))

        result_str = result_buf.value.decode("utf-8")
        try:
            return json.loads(result_str)
        except ValueError as e:
            raise RuntimeError("JSON解析失败: " + str(e))

    def release(self):
        if not self._released:
            self._lib.ssd_release()
            self._released = True

    def __del__(self):
        self.release()


# ==================== 网络工具 ====================

def recv_all(sock, size):
    """确保接收指定字节数"""
    buf = b''
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("连接已断开")
        buf += chunk
    return buf


# ==================== 请求处理 ====================

def handle_client(conn, addr, detector, tmp_dir, log_path):
    """处理单个客户端连接"""
    print("[信息] 客户端连接: %s:%d" % addr)

    try:
        # 读取请求: magic + filename_len + filename + image_size + image_data
        magic = recv_all(conn, 4)
        if magic != MAGIC:
            print("[警告] 无效请求头")
            return

        filename_len = struct.unpack('!I', recv_all(conn, 4))[0]
        if filename_len > 256:
            print("[警告] 文件名过长")
            return
        filename = recv_all(conn, filename_len).decode('utf-8')

        image_size = struct.unpack('!I', recv_all(conn, 4))[0]
        if image_size > 50 * 1024 * 1024:
            print("[警告] 图片过大: %d bytes" % image_size)
            return
        image_data = recv_all(conn, image_size)

        print("[信息] 收到: %s (%d bytes)" % (filename, image_size))

        # 保存临时输入图片
        input_path = os.path.join(tmp_dir, filename)
        with open(input_path, 'wb') as f:
            f.write(image_data)

        # 生成输出路径
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(tmp_dir, base + "_detect" + ext)

        # 加锁检测（C库非线程安全）
        with detect_lock:
            detector.log_start(log_path)
            result = detector.detect(input_path, output_path)
            detector.log_stop()

        # 读取日志
        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_content = f.read()

        result["log"] = log_content
        print("[信息] 检测完成: %d 个目标" % len(result.get("objects", [])))

        # 读取结果图片
        with open(output_path, 'rb') as f:
            result_image = f.read()

        # 发送响应: json + image + log
        json_bytes = json.dumps(result, ensure_ascii=False).encode('utf-8')
        log_bytes = log_content.encode('utf-8')

        conn.sendall(struct.pack('!I', len(json_bytes)))
        conn.sendall(json_bytes)
        conn.sendall(struct.pack('!I', len(result_image)))
        conn.sendall(result_image)
        conn.sendall(struct.pack('!I', len(log_bytes)))
        conn.sendall(log_bytes)

        print("[信息] 响应已发送")

    except Exception as e:
        print("[错误] 处理失败: %s" % str(e))


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="SSD缺陷检测服务")
    parser.add_argument("--model", default="model.nb", help="模型文件路径 (默认：model.nb)")
    parser.add_argument("--port", type=int, default=9000, help="监听端口 (默认: 9000)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_detect.log")

    print("[信息] 正在加载模型: %s" % args.model)
    detector = SSDDetector(args.model)
    print("[信息] 模型加载成功")

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print("[信息] 服务已启动，监听 %s:%d" % (args.host, args.port))
    print("[信息] 等待客户端连接...")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, detector, tmp_dir, log_path)
            conn.close()
    except KeyboardInterrupt:
        print("\n[信息] 服务已停止")
    finally:
        detector.release()
        server.close()


if __name__ == "__main__":
    main()
