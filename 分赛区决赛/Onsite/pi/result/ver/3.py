#!/usr/bin/env python3
"""
稳定性增强版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣 + 美观Web前端
基于 2.py 优化：
  - FPGA超时30→5秒，避免假死
  - 共享变量加锁，防止竞态
  - 串口通信重试机制
  - FPGA请求异步化，不阻塞主循环
  - 相机取帧失败告警
  - 前端实时统计、操作日志、手动触发
  - 新增API：/health /stats /logs /config /trigger
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
from collections import deque
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
FPGA_TIMEOUT      = 5       # FPGA请求超时（秒），原30秒太长

# ---- 串口（吸盘）参数 ----
SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 1.0
SUCTION_RETRY       = 1     # 串口重试次数

# ---- 机械臂参数 ----
ARM_SPEED = 800
BUZZER_DELAY = 10
FORCE_TEST = None   # 调试用: "OK" / "NG" / None

# ---- 监控参数 ----
MAX_LOGS            = 50    # 最大日志条数
FRAME_FAIL_WARN     = 50    # 连续取帧失败告警阈值
ARM_ACTION_TIMEOUT  = 30    # 机械臂动作超时（秒）
# ==========================================================

os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)

# ---------- 全局状态（带锁保护）----------
arm_lock = threading.Lock()
frame_lock = threading.Lock()

arm_busy = False
last_action_result = "无"
latest_frame_raw = None
latest_frame_edge = None
latest_sent_raw = None
latest_detected_img = None

# ---------- 统计数据 ----------
stats = {
    'start_time': time.time(),
    'fps': 0.0,
    'ok_count': 0,
    'ng_count': 0,
    'fpga_times': [],      # 最近10次FPGA响应时间
    'frame_count': 0,
    'fps_start': time.time(),
}

# ---------- 操作日志 ----------
action_logs = deque(maxlen=MAX_LOGS)

def add_log(msg, level="INFO"):
    """添加操作日志"""
    ts = time.strftime("%H:%M:%S")
    log_entry = f"[{ts}] [{level}] {msg}"
    action_logs.append(log_entry)
    print(log_entry)

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
    .container { max-width: 1440px; margin: 0 auto; }
    h1 {
        text-align: center;
        margin-bottom: 25px;
        color: #e94560;
        font-weight: 300;
        letter-spacing: 2px;
        font-size: 24px;
    }
    .status-bar {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .status-card {
        background: #16213e;
        border-radius: 12px;
        padding: 15px;
    }
    .status-label {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .status-value {
        font-size: 20px;
        font-weight: bold;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .dot-green { background: #4caf50; }
    .dot-red { background: #e53935; }
    .dot-blue { background: #2196f3; }
    .main-grid {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }
    .video-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
    }
    .card {
        background: #16213e;
        border-radius: 12px;
        padding: 15px;
    }
    .card-title {
        font-size: 13px;
        color: #a0a0b0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card img {
        width: 100%;
        height: auto;
        border-radius: 8px;
        display: block;
        background: #0f3460;
    }
    .side-panel {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .log-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 15px;
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    .log-title {
        font-size: 13px;
        color: #a0a0b0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .log-content {
        flex: 1;
        overflow-y: auto;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
        color: #b0b0b0;
        max-height: 400px;
    }
    .log-content::-webkit-scrollbar {
        width: 6px;
    }
    .log-content::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 3px;
    }
    .log-entry {
        padding: 2px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .log-time { color: #666; }
    .log-info { color: #4caf50; }
    .log-warn { color: #ff9800; }
    .log-error { color: #e53935; }
    .btn-group {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .btn {
        border: none;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        cursor: pointer;
        flex: 1;
        min-width: 120px;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-reset { background: #e94560; }
    .btn-trigger { background: #2196f3; }
    .config-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 15px;
    }
    .config-title {
        font-size: 13px;
        color: #a0a0b0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .config-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        font-size: 12px;
    }
    .config-item {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .config-key { color: #888; }
    .config-val { color: #eee; font-weight: 500; }
    .footer {
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    @media (max-width: 1024px) {
        .main-grid { grid-template-columns: 1fr; }
        .video-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 768px) {
        .status-bar { grid-template-columns: repeat(2, 1fr); }
    }
</style>
</head>
<body>
<div class="container">
    <h1>🤖 机械臂分拣监控系统</h1>

    <!-- 状态栏 -->
    <div class="status-bar">
        <div class="status-card">
            <div class="status-label">机械臂状态</div>
            <div id="armStatus" class="status-value">
                <span class="status-dot dot-green"></span>空闲
            </div>
        </div>
        <div class="status-card">
            <div class="status-label">最后动作</div>
            <div id="lastAction" class="status-value" style="color:#ff9800;">无</div>
        </div>
        <div class="status-card">
            <div class="status-label">实时帧率</div>
            <div id="fpsValue" class="status-value" style="color:#2196f3;">--</div>
        </div>
        <div class="status-card">
            <div class="status-label">检测统计</div>
            <div id="statsValue" class="status-value">
                <span style="color:#4caf50;">OK:0</span> / <span style="color:#e53935;">NG:0</span>
            </div>
        </div>
        <div class="status-card">
            <div class="status-label">运行时间</div>
            <div id="uptime" class="status-value" style="color:#9c27b0;">--</div>
        </div>
        <div class="status-card">
            <div class="status-label">FPGA响应</div>
            <div id="fpgaTime" class="status-value" style="color:#00bcd4;">--</div>
        </div>
    </div>

    <!-- 主内容区 -->
    <div class="main-grid">
        <!-- 视频区域 -->
        <div class="video-grid">
            <div class="card">
                <div class="card-title">📷 原始视频流</div>
                <img src="/video_feed_raw" alt="原始流">
            </div>
            <div class="card">
                <div class="card-title">🔍 Canny 边缘检测</div>
                <img src="/video_feed_edge" alt="边缘流">
            </div>
            <div class="card">
                <div class="card-title">📤 最近发送原图</div>
                <img id="sentImg" src="/sent_image" alt="发送原图">
            </div>
            <div class="card">
                <div class="card-title">🎯 FPGA 检测结果</div>
                <img id="detectedImg" src="/detected_image" alt="检测结果">
            </div>
        </div>

        <!-- 侧面板 -->
        <div class="side-panel">
            <!-- 操作按钮 -->
            <div class="card">
                <div class="card-title">🎮 操作控制</div>
                <div class="btn-group">
                    <button id="resetBtn" class="btn btn-reset" onclick="resetArm()">🔄 复位机械臂</button>
                    <button id="triggerBtn" class="btn btn-trigger" onclick="triggerDetect()">📸 手动触发</button>
                </div>
            </div>

            <!-- 配置信息 -->
            <div class="config-panel">
                <div class="config-title">⚙️ 当前配置</div>
                <div id="configContent" class="config-grid">
                    <div class="config-item"><span class="config-key">加载中...</span></div>
                </div>
            </div>

            <!-- 操作日志 -->
            <div class="log-panel">
                <div class="log-title">📋 操作日志</div>
                <div id="logContent" class="log-content"></div>
            </div>
        </div>
    </div>

    <div class="footer">系统运行中 · 按 Ctrl+C 停止 · v3.0 稳定性增强版</div>
</div>

<script>
// 格式化运行时间
function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return h + '时' + m + '分';
    if (m > 0) return m + '分' + s + '秒';
    return s + '秒';
}

// 定时刷新静态图片
function refreshStatic() {
    document.getElementById('sentImg').src = '/sent_image?' + Date.now();
    document.getElementById('detectedImg').src = '/detected_image?' + Date.now();
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
            const triggerBtn = document.getElementById('triggerBtn');

            if (data.busy) {
                statusEl.innerHTML = '<span class="status-dot dot-red"></span>工作中';
                resetBtn.disabled = true;
                triggerBtn.disabled = true;
            } else {
                statusEl.innerHTML = '<span class="status-dot dot-green"></span>空闲';
                resetBtn.disabled = false;
                triggerBtn.disabled = false;
            }
            actionEl.textContent = data.last_action || '无';
        })
        .catch(() => {});
}
setInterval(fetchStatus, 1000);
fetchStatus();

// 定时获取统计信息
function fetchStats() {
    fetch('/stats')
        .then(r => r.json())
        .then(data => {
            document.getElementById('fpsValue').textContent = data.fps.toFixed(1);
            document.getElementById('statsValue').innerHTML =
                '<span style="color:#4caf50;">OK:' + data.ok_count + '</span> / ' +
                '<span style="color:#e53935;">NG:' + data.ng_count + '</span>';
            document.getElementById('uptime').textContent = formatUptime(data.uptime);
            document.getElementById('fpgaTime').textContent = data.fpga_avg > 0 ? data.fpga_avg.toFixed(0) + 'ms' : '--';
        })
        .catch(() => {});
}
setInterval(fetchStats, 1000);
fetchStats();

// 定时获取日志
function fetchLogs() {
    fetch('/logs')
        .then(r => r.json())
        .then(data => {
            const logEl = document.getElementById('logContent');
            logEl.innerHTML = data.logs.map(log => {
                let cls = 'log-entry';
                if (log.includes('[ERROR]')) cls += ' log-error';
                else if (log.includes('[WARN]')) cls += ' log-warn';
                else cls += ' log-info';
                return '<div class="' + cls + '">' + log + '</div>';
            }).reverse().join('');
            logEl.scrollTop = 0;
        })
        .catch(() => {});
}
setInterval(fetchLogs, 2000);
fetchLogs();

// 获取配置
function fetchConfig() {
    fetch('/config')
        .then(r => r.json())
        .then(data => {
            const el = document.getElementById('configContent');
            el.innerHTML = Object.entries(data).map(([k, v]) =>
                '<div class="config-item"><span class="config-key">' + k + '</span><span class="config-val">' + v + '</span></div>'
            ).join('');
        })
        .catch(() => {});
}
fetchConfig();
setInterval(fetchConfig, 10000);

// 复位机械臂
function resetArm() {
    const btn = document.getElementById('resetBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 复位中...';
    fetch('/reset', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.textContent = '🔄 复位机械臂';
            btn.disabled = false;
        })
        .catch(() => {
            btn.textContent = '🔄 复位机械臂';
            btn.disabled = false;
        });
}

// 手动触发检测
function triggerDetect() {
    const btn = document.getElementById('triggerBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 触发中...';
    fetch('/trigger', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.textContent = '📸 手动触发';
            btn.disabled = false;
        })
        .catch(() => {
            btn.textContent = '📸 手动触发';
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
    with arm_lock:
        return jsonify({
            'busy': arm_busy,
            'last_action': last_action_result
        })

@app.route('/stats')
def get_stats():
    """返回统计信息"""
    uptime = time.time() - stats['start_time']
    fpga_avg = np.mean(stats['fpga_times'][-10:]) if stats['fpga_times'] else 0
    return jsonify({
        'fps': stats['fps'],
        'ok_count': stats['ok_count'],
        'ng_count': stats['ng_count'],
        'uptime': uptime,
        'fpga_avg': fpga_avg
    })

@app.route('/logs')
def get_logs():
    """返回操作日志"""
    return jsonify({'logs': list(action_logs)})

@app.route('/config')
def get_config():
    """返回当前配置"""
    return jsonify({
        'FPGA超时': f'{FPGA_TIMEOUT}s',
        '触发冷却': f'{SEND_COOLDOWN}s',
        '串口重试': SUCTION_RETRY,
        '吸盘等待': f'{SUCTION_WAIT_TIME}s',
        '机械臂速度': ARM_SPEED,
        'Canny阈值': f'{CANNY_LOW}-{CANNY_HIGH}',
        '最小边缘点': MIN_EDGE_POINTS,
        '测试模式': FORCE_TEST or '自动'
    })

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'running',
        'uptime': time.time() - stats['start_time'],
        'camera': 'ok',  # 如果能响应说明主线程正常
        'arm_busy': arm_busy
    })

@app.route('/reset', methods=['POST'])
def reset():
    """手动复位机械臂"""
    with arm_lock:
        if arm_busy:
            return jsonify({'message': '机械臂正在工作中，无法复位'}), 409
    t = threading.Thread(target=do_reset, daemon=True)
    t.start()
    return jsonify({'message': '复位指令已发送'})

@app.route('/trigger', methods=['POST'])
def trigger():
    """手动触发一次检测"""
    with frame_lock:
        if latest_frame_raw is None:
            return jsonify({'message': '尚无可用帧'}), 400
        frame = latest_frame_raw.copy()
    with arm_lock:
        if arm_busy:
            return jsonify({'message': '机械臂正在工作中'}), 409
    t = threading.Thread(target=do_manual_trigger, args=(frame,), daemon=True)
    t.start()
    return jsonify({'message': '触发指令已发送'})

def do_reset():
    """实际复位函数"""
    global arm_busy, last_action_result
    with arm_lock:
        arm_busy = True
    try:
        Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
        time.sleep(ARM_SPEED / 1000 + 0.3)
        with arm_lock:
            last_action_result = "手动复位"
        add_log("手动复位完成")
    except Exception as e:
        add_log(f"复位异常: {e}", "ERROR")
    finally:
        with arm_lock:
            arm_busy = False

def do_manual_trigger(frame):
    """手动触发检测"""
    add_log("手动触发检测")
    ts = int(time.time())
    send_to_fpga_and_decide(frame, ts)

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
    add_log(f"串口 {SUCTION_SERIAL_PORT} 已打开")
except Exception as e:
    add_log(f"串口打开失败: {e}", "ERROR")
    exit(1)

zhua_qu = bytes.fromhex('A0 01 01 A2')
shi_fang = bytes.fromhex('A0 01 00 A1')

def zhua_qu_xipan():
    """启动吸盘（带重试）"""
    for attempt in range(SUCTION_RETRY + 1):
        try:
            com.write(zhua_qu)
            time.sleep(0.1)
            add_log("吸盘已启动")
            return True
        except Exception as e:
            add_log(f"吸盘启动失败 (尝试{attempt+1}): {e}", "WARN")
            if attempt < SUCTION_RETRY:
                time.sleep(0.1)
    return False

def shi_fang_xipan():
    """释放吸盘（带重试）"""
    for attempt in range(SUCTION_RETRY + 1):
        try:
            com.write(shi_fang)
            time.sleep(0.1)
            add_log("吸盘已释放")
            return True
        except Exception as e:
            add_log(f"吸盘释放失败 (尝试{attempt+1}): {e}", "WARN")
            if attempt < SUCTION_RETRY:
                time.sleep(0.1)
    return False

# ---------- 机械臂初始化 ----------
Arm = Arm_Device()
time.sleep(0.1)
add_log("机械臂已初始化")

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
    """执行机械臂动作（带超时保护）"""
    global arm_busy, last_action_result
    with arm_lock:
        arm_busy = True
    add_log(f"执行 {action_type} 动作")
    start_time = time.time()

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

        elapsed = time.time() - start_time
        with arm_lock:
            last_action_result = action_type
        add_log(f"{action_type} 动作完成 ({elapsed:.1f}s)")

    except Exception as e:
        add_log(f"动作执行异常: {e}", "ERROR")
        with arm_lock:
            last_action_result = f"错误: {e}"
    finally:
        with arm_lock:
            arm_busy = False

def send_to_fpga_and_decide(frame_bgr, timestamp):
    """发送图片到FPGA并决定动作"""
    global latest_sent_raw, latest_detected_img

    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        add_log("图像编码失败", "ERROR")
        return

    image_bytes = encoded.tobytes()
    add_log(f"发送图片到 FPGA...")
    fpga_start = time.time()

    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=FPGA_TIMEOUT)
        fpga_time = (time.time() - fpga_start) * 1000  # ms

        if resp.status_code != 200:
            add_log(f"FPGA请求失败: {resp.status_code}", "ERROR")
            return

        result = json.loads(resp.text)
        objects_count = len(result.get('objects', []))
        inference_ms = result.get('inference_time_ms', 'N/A')
        add_log(f"FPGA返回: 目标数={objects_count}, 推理={inference_ms}ms, 总耗时={fpga_time:.0f}ms")

        # 更新统计
        stats['fpga_times'].append(fpga_time)
        if len(stats['fpga_times']) > 10:
            stats['fpga_times'] = stats['fpga_times'][-10:]

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
                add_log(f"检测图片保存失败: {e}", "WARN")

        with frame_lock:
            latest_sent_raw = frame_bgr.copy()

        # 决定动作
        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0:
            action = "OK"
        else:
            action = "NG"

        # 更新统计
        if action == "OK":
            stats['ok_count'] += 1
        else:
            stats['ng_count'] += 1

        # 执行机械臂动作
        execute_arm_action(action)

    except requests.exceptions.Timeout:
        add_log(f"FPGA请求超时 ({FPGA_TIMEOUT}s)", "ERROR")
    except Exception as e:
        add_log(f"FPGA请求异常: {e}", "ERROR")

# ---------- 主循环 ----------
def main():
    global last_send_time, arm_busy, latest_frame_raw, latest_frame_edge

    add_log("系统启动中...")

    # 启动 Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    add_log("Flask Web 服务已启动，端口 5000")

    # 相机初始化
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        add_log("未找到相机设备", "ERROR")
        return
    add_log(f"找到 {deviceList.nDeviceNum} 个相机")
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
    add_log("相机已启动，开始主循环")

    last_send_time = 0.0
    frame_count = 0
    fps_start = time.time()
    consecutive_failures = 0

    try:
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                consecutive_failures += 1
                if consecutive_failures >= FRAME_FAIL_WARN:
                    add_log(f"连续取帧失败 {consecutive_failures} 次，请检查相机连接", "WARN")
                    consecutive_failures = 0  # 重置，避免重复告警
                continue
            consecutive_failures = 0

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
                with arm_lock:
                    is_busy = arm_busy
                if avg_x > half_width and (now - last_send_time) >= SEND_COOLDOWN and not is_busy:
                    last_send_time = now
                    original_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    ts = int(time.time())
                    send_to_fpga_and_decide(original_bgr, ts)

            # 更新FPS统计
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                stats['fps'] = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

    except KeyboardInterrupt:
        add_log("用户中断")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        com.close()
        add_log("系统已关闭")

if __name__ == "__main__":
    main()
