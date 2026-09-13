# -*- coding: utf-8 -*-
"""
SSD缺陷检测服务（运行在 Cyclone V SoC 上，Python 2.7 兼容版）
集成了 ctypes 封装 + TCP 服务端，单文件完成所有功能。
所有字符串操作均以 bytes 为主，避免 Python 2 的 str/unicode 隐式转换触发 ASCII 解码错误。

用法:
    python ssd_detection.py --model ssd_mobilenet_v1_opt.nb --port 9000
"""

from __future__ import print_function

import argparse
import base64
import ctypes
import json
import os
import socket
import struct
import sys
import threading
import traceback
import uuid
from flask import Flask, request, Response

# ==================== 调试工具 ====================

DEBUG = True  # 设为 False 关闭详细调试

def dbg(tag, msg):
    """调试日志，带标签"""
    if DEBUG:
        print("[DBG:%s] %s" % (tag, msg))

def safe_repr(obj):
    """安全地显示对象信息，不会触发编码错误"""
    try:
        t = type(obj).__name__
        if isinstance(obj, bytes):
            return "<bytes len=%d>" % len(obj)
        elif isinstance(obj, unicode):
            return "<unicode len=%d>" % len(obj)
        else:
            return "<%s %r>" % (t, obj)
    except Exception:
        return "<???>"


# ==================== 协议常量 ====================
MAGIC = b'REQ\0'

# ==================== 线程锁 ====================
# C库内部 predictor 是全局变量，非线程安全
detect_lock = threading.Lock()

# ==================== 模块级状态（main() 中初始化） ====================
g_detector = None
g_tmp_dir = None
g_log_path = None

# ==================== Flask HTTP 服务（供 Jetson 调用） ====================

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """接收 Jetson 发来的图片，执行检测并返回 JSON 结果"""
    if 'image_file' not in request.files:
        dbg("http", "请求中没有 image_file 字段")
        return Response('{"error":"no image_file field"}',
                        mimetype='application/json')

    img_file = request.files['image_file']
    dbg("http", "收到图片: filename=%s" % safe_repr(img_file.filename))

    # 生成唯一文件名，避免并发冲突
    uid = str(uuid.uuid4())
    input_filename = uid + '.jpg'
    input_path = os.path.join(g_tmp_dir, input_filename)
    output_filename = uid + '_detect.jpg'
    output_path = os.path.join(g_tmp_dir, output_filename)

    dbg("http", "input_path=%s" % safe_repr(input_path))

    # 保存上传的图片
    img_file.save(input_path)
    dbg("http", "图片保存完成")

    # 加锁调用检测器
    try:
        with detect_lock:
            g_detector.log_start(g_log_path)
            result = g_detector.detect(input_path, output_path)
            g_detector.log_stop()

        obj_count = len(result.get("objects", []))
        dbg("http", "检测完成: %d 个目标" % obj_count)

	# read_img
	output_path_for_read = output_path
	if isinstance(output_path_for_read, unicode):
	    output_path_for_read = output_path_for_read.encode('utf-8')
        with open(output_path_for_read, 'rb') as f:
            result_image_bytes = f.read()
        result['detect_image'] = base64.b64encode(result_image_bytes).decode('ascii')
        result['detect_image_size'] = len(result_image_bytes)
	dbg("http","image is coded!!!")

        # 清理临时文件
        try:
            os.remove(input_path)
        except Exception:
            pass
        try:
            os.remove(output_path)
        except Exception:
            pass

        # Python 2: json.dumps 返回 bytes (str), ensure_ascii=True 确保纯 ASCII
        json_str = json.dumps(result, ensure_ascii=True)
        if isinstance(json_str, unicode):
            json_str = json_str.encode('utf-8')
        return Response(json_str, mimetype='application/json')

    except Exception as e:
        dbg("http", "检测异常: %s" % str(e))
        traceback.print_exc()
        # 清理临时文件
        try:
            os.remove(input_path)
        except Exception:
            pass
        try:
            os.remove(output_path)
        except Exception:
            pass
        err_json = json.dumps({"error": str(e)}, ensure_ascii=True)
        return Response(err_json, mimetype='application/json')


# ==================== C库封装 ====================

class SSDDetector(object):
    """SSD缺陷检测器，封装C共享库调用"""

    def __init__(self, model_path, lib_path=None):
        dbg("init", "model_path type=%s value=%s" % (type(model_path).__name__, safe_repr(model_path)))

        if not os.path.exists(model_path):
            raise IOError("模型文件不存在: " + repr(model_path))

        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libssd_detection.so")
        if not os.path.exists(lib_path):
            raise IOError("共享库不存在: " + repr(lib_path))

        dbg("init", "lib_path=%s" % lib_path)

        self._lib = ctypes.CDLL(lib_path)
        dbg("init", "CDLL 加载成功")

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

        # Python 2: model_path 如果是 unicode，encode 成 bytes 再传给 ctypes
        model_bytes = self._to_bytes(model_path)
        dbg("init", "model_bytes type=%s len=%d" % (type(model_bytes).__name__, len(model_bytes)))

        ret = self._lib.ssd_init(model_bytes)
        dbg("init", "ssd_init 返回: %d" % ret)
        if ret != 0:
            raise RuntimeError("模型初始化失败: " + repr(model_path))

        self._released = False
        dbg("init", "初始化完成")

    @staticmethod
    def _to_bytes(s):
        """安全地将 str/unicode 转为 bytes，避免 Python 2 的隐式 ASCII decode"""
        if isinstance(s, bytes):
            return s
        elif isinstance(s, unicode):
            return s.encode('utf-8')
        else:
            return bytes(s)

    def log_start(self, log_path):
        lp = self._to_bytes(log_path)
        dbg("log", "log_start: %s" % safe_repr(lp))
        self._lib.ssd_log_start(lp)

    def log_stop(self):
        self._lib.ssd_log_stop()

    def detect(self, img_path, output_path, json_buf_size=8192):
        if self._released:
            raise RuntimeError("检测器已释放")

        dbg("detect", "img_path type=%s val=%s" % (type(img_path).__name__, safe_repr(img_path)))
        dbg("detect", "output_path type=%s val=%s" % (type(output_path).__name__, safe_repr(output_path)))

        if not os.path.exists(img_path):
            raise IOError("图片不存在: " + repr(img_path))

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_bytes = self._to_bytes(img_path)
        out_bytes = self._to_bytes(output_path)
        dbg("detect", "img_bytes=%s out_bytes=%s" % (safe_repr(img_bytes), safe_repr(out_bytes)))

        result_buf = ctypes.create_string_buffer(json_buf_size)
        dbg("detect", "调用 ssd_detect ...")

        ret = self._lib.ssd_detect(img_bytes, out_bytes, result_buf, json_buf_size)
        dbg("detect", "ssd_detect 返回: %d" % ret)

        if ret != 0:
            raise RuntimeError("检测失败: " + repr(img_path))

        # result_buf.value 是 bytes
        raw = result_buf.value
        dbg("detect", "raw result type=%s len=%d" % (type(raw).__name__, len(raw)))
        dbg("detect", "raw result 前200字节: %s" % repr(raw[:200]))

        # 尝试 decode
        try:
            result_str = raw.decode('utf-8')
            dbg("detect", "decode utf-8 成功, len=%d" % len(result_str))
        except UnicodeDecodeError as e:
            dbg("detect", "decode utf-8 失败: %s, 尝试 latin-1" % str(e))
            result_str = raw.decode('latin-1')
            dbg("detect", "decode latin-1 成功, len=%d" % len(result_str))

        # 解析 JSON
        try:
            result = json.loads(result_str)
            dbg("detect", "JSON 解析成功, keys=%s" % list(result.keys()) if isinstance(result, dict) else type(result).__name__)
        except ValueError as e:
            dbg("detect", "JSON 解析失败: %s" % str(e))
            dbg("detect", "result_str 前500字符: %s" % repr(result_str[:500]))
            raise RuntimeError("JSON解析失败: " + str(e))

        return result

    def release(self):
        if not self._released:
            dbg("release", "释放检测器")
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
            raise IOError("连接已断开, 已收 %d/%d bytes" % (len(buf), size))
        buf += chunk
    return buf


# ==================== 请求处理 ====================

def handle_client(conn, addr, detector, tmp_dir, log_path):
    """处理单个客户端连接"""
    dbg("client", "=== 新连接: %s:%d ===" % addr)

    try:
        # ---- 读取 magic ----
        magic = recv_all(conn, 4)
        dbg("client", "magic 收到: %s (期望 %s)" % (repr(magic), repr(MAGIC)))
        if magic != MAGIC:
            print("[警告] 无效请求头: %s" % repr(magic))
            return

        # ---- 读取 filename ----
        raw_fn_len = recv_all(conn, 4)
        filename_len = struct.unpack('!I', raw_fn_len)[0]
        dbg("client", "filename_len = %d" % filename_len)
        if filename_len > 256:
            print("[警告] 文件名过长: %d" % filename_len)
            return

        raw_fn = recv_all(conn, filename_len)
        dbg("client", "filename raw bytes: %s" % repr(raw_fn))

        # Python 2: decode 成 unicode 用于 os.path 操作
        # 如果文件名包含非 ASCII（如中文），这里可能出错
        try:
            filename = raw_fn.decode('utf-8')
            dbg("client", "filename decode utf-8 成功: %s" % repr(filename))
        except UnicodeDecodeError as e:
            dbg("client", "filename decode utf-8 失败: %s, 用 latin-1" % str(e))
            filename = raw_fn.decode('latin-1')
            dbg("client", "filename decode latin-1: %s" % repr(filename))

        # ---- 读取 image ----
        raw_img_size = recv_all(conn, 4)
        image_size = struct.unpack('!I', raw_img_size)[0]
        dbg("client", "image_size = %d bytes" % image_size)
        if image_size > 50 * 1024 * 1024:
            print("[警告] 图片过大: %d bytes" % image_size)
            return

        dbg("client", "开始接收图片数据...")
        image_data = recv_all(conn, image_size)
        dbg("client", "图片接收完成: %d bytes" % len(image_data))

        print("[信息] 收到: %s (%d bytes)" % (filename, image_size))

        # ---- 保存临时输入图片 ----
        # Python 2: os.path.join 对 unicode 和 bytes 混合可能出问题
        # 这里用 bytes 版本的 tmp_dir 和 filename
        try:
            # 尝试用 unicode 路径
            input_path = os.path.join(tmp_dir, filename)
            dbg("client", "input_path (unicode): %s" % repr(input_path))
        except UnicodeDecodeError as e:
            dbg("client", "os.path.join unicode 失败: %s" % str(e))
            # 回退: 用 bytes
            input_path = os.path.join(tmp_dir.encode('utf-8'), raw_fn)
            dbg("client", "input_path (bytes): %s" % repr(input_path))

        # Python 2: open() 对 unicode 路径可能隐式 encode 为 ascii
        # 安全写法: 直接用 bytes 路径
        try:
            with open(input_path, 'wb') as f:
                f.write(image_data)
            dbg("client", "输入图片保存成功: %s" % repr(input_path))
        except UnicodeEncodeError as e:
            dbg("client", "open(input_path) unicode 编码失败: %s" % str(e))
            # 回退: 用 bytes 路径
            input_path_bytes = input_path.encode('utf-8') if isinstance(input_path, unicode) else input_path
            dbg("client", "回退到 bytes 路径: %s" % repr(input_path_bytes))
            with open(input_path_bytes, 'wb') as f:
                f.write(image_data)
            input_path = input_path_bytes
            dbg("client", "输入图片保存成功 (bytes路径)")

        # ---- 生成输出路径 ----
        base, ext = os.path.splitext(filename)
        dbg("client", "base=%s ext=%s" % (repr(base), repr(ext)))

        output_filename = base + u"_detect" + ext
        dbg("client", "output_filename=%s" % repr(output_filename))

        try:
            output_path = os.path.join(tmp_dir, output_filename)
            dbg("client", "output_path (unicode): %s" % repr(output_path))
        except UnicodeDecodeError as e:
            dbg("client", "output_path os.path.join 失败: %s" % str(e))
            output_path = os.path.join(tmp_dir.encode('utf-8'),
                                        (base + "_detect" + ext).encode('utf-8'))
            dbg("client", "output_path (bytes): %s" % repr(output_path))

        # ---- 加锁检测 ----
        dbg("detect", "获取检测锁...")
        with detect_lock:
            dbg("detect", "log_start")
            detector.log_start(log_path)
            dbg("detect", "开始 detect...")
            result = detector.detect(input_path, output_path)
            dbg("detect", "detect 完成")
            detector.log_stop()
        dbg("detect", "释放检测锁")

        # ---- 读取日志 ----
        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                raw_log = f.read()
            try:
                log_content = raw_log.decode('utf-8')
            except UnicodeDecodeError:
                log_content = raw_log.decode('latin-1')
            dbg("client", "日志读取: %d chars" % len(log_content))

        result["log"] = log_content

        obj_count = len(result.get("objects", []))
        print("[信息] 检测完成: %d 个目标" % obj_count)
        dbg("client", "result keys=%s" % list(result.keys()))

        # ---- 读取结果图片 ----
        output_path_for_read = output_path
        if isinstance(output_path_for_read, unicode):
            output_path_for_read_bytes = output_path_for_read.encode('utf-8')
        else:
            output_path_for_read_bytes = output_path_for_read

        dbg("client", "读取结果图: %s" % repr(output_path_for_read_bytes))
        with open(output_path_for_read_bytes, 'rb') as f:
            result_image = f.read()
        dbg("client", "结果图大小: %d bytes" % len(result_image))

        # ---- 构造响应 JSON ----
        # Python 2 重点: json.dumps 返回的是 bytes (str)，不是 unicode
        # ensure_ascii=True 确保输出纯 ASCII bytes，避免后续 encode 问题
        try:
            json_bytes = json.dumps(result, ensure_ascii=True)
            dbg("client", "json.dumps(ensure_ascii=True) 成功, type=%s len=%d" % (type(json_bytes).__name__, len(json_bytes)))
        except Exception as e:
            dbg("client", "json.dumps 失败: %s" % str(e))
            # 如果有非序列化对象，逐字段清理
            clean_result = {}
            for k, v in result.items():
                try:
                    json.dumps(v)
                    clean_result[k] = v
                except Exception:
                    clean_result[k] = repr(v)
            json_bytes = json.dumps(clean_result, ensure_ascii=True)
            dbg("client", "清理后 json.dumps 成功")

        # Python 2: json.dumps 返回 bytes，确保是 bytes
        if isinstance(json_bytes, unicode):
            dbg("client", "json_bytes 是 unicode, encode utf-8")
            json_bytes = json_bytes.encode('utf-8')
        dbg("client", "json_bytes 最终: type=%s len=%d" % (type(json_bytes).__name__, len(json_bytes)))

        # log 也确保是 bytes
        if isinstance(log_content, unicode):
            log_bytes = log_content.encode('utf-8', errors='replace')
        else:
            log_bytes = log_content
        dbg("client", "log_bytes: type=%s len=%d" % (type(log_bytes).__name__, len(log_bytes)))

        # ---- 发送响应 ----
        dbg("client", "发送响应...")
        conn.sendall(struct.pack('!I', len(json_bytes)))
        conn.sendall(json_bytes)
        conn.sendall(struct.pack('!I', len(result_image)))
        conn.sendall(result_image)
        conn.sendall(struct.pack('!I', len(log_bytes)))
        conn.sendall(log_bytes)

        print("[信息] 响应已发送 (json=%d, img=%d, log=%d)" % (
            len(json_bytes), len(result_image), len(log_bytes)))

    except Exception as e:
        print("[错误] 处理失败: %s" % str(e))
        traceback.print_exc()


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="SSD缺陷检测服务 (Python 2.7)")
    parser.add_argument("--model", default="model.nb", help="模型文件路径 (默认: model.nb)")
    parser.add_argument("--port", type=int, default=9000, help="TCP 监听端口 (默认: 9000)")
    parser.add_argument("--host", default="0.0.0.0", help="TCP 监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP 监听端口 (默认: 8080)")
    parser.add_argument("--no-debug", action="store_true", help="关闭调试输出")
    args = parser.parse_args()

    global DEBUG
    if args.no_debug:
        DEBUG = False

    dbg("main", "Python 版本: %s" % sys.version)
    dbg("main", "默认编码: %s" % sys.getdefaultencoding())
    dbg("main", "文件系统编码: %s" % sys.getfilesystemencoding())

    # 尝试设置默认编码为 utf-8（Python 2 默认是 ascii）
    try:
        reload(sys)
        sys.setdefaultencoding('utf-8')
        dbg("main", "已设置默认编码为 utf-8")
    except Exception as e:
        dbg("main", "设置默认编码失败: %s (这通常没问题)" % str(e))

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_detect.log")
    dbg("main", "log_path=%s" % log_path)

    print("[信息] 正在加载模型: %s" % args.model)
    detector = SSDDetector(args.model)
    print("[信息] 模型加载成功")

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    dbg("main", "tmp_dir=%s" % tmp_dir)

    # ---- 设置模块级全局变量，供 Flask 路由使用 ----
    global g_detector, g_tmp_dir, g_log_path
    g_detector = detector
    g_tmp_dir = tmp_dir
    g_log_path = log_path

    # ---- 在独立线程中启动 Flask HTTP 服务 ----
    def run_http():
        app.run(host='0.0.0.0', port=args.http_port,
                debug=False, use_reloader=False)

    http_thread = threading.Thread(target=run_http)
    http_thread.daemon = True
    http_thread.start()
    print("[信息] HTTP 服务已启动，监听 0.0.0.0:%d" % args.http_port)
    print("[信息] Jetson 应连接: http://<FPGA_IP>:%d/predict" % args.http_port)

    # ---- TCP 二进制协议服务（保留原有功能）----
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
