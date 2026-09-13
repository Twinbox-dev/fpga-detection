#!/usr/bin/env python3
"""
稳定性增强版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣 + 美观Web前端
基于 3.py 优化：
  - FPGA/机械臂动作异步化，不阻塞视频流
  - 修复emoji显示问题
  - 新增手动控制机械臂接口（5舵机+吸盘）
  - 边缘检测图上显示avg_x位置
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
FPGA_TIMEOUT      = 5
FRAME_RATE        = 60    # 相机帧率 fps（1-120）

# ---- 串口（吸盘）参数 ----
SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 1.0
SUCTION_RETRY       = 1

# ---- 机械臂参数 ----
ARM_SPEED = 800
BUZZER_DELAY = 10
FORCE_TEST = None

# ---- 监控参数 ----
MAX_LOGS            = 50
FRAME_FAIL_WARN     = 50
ARM_ACTION_TIMEOUT  = 30
# ==========================================================

os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs("origin", exist_ok=True)

# ---------- 全局状态（带锁保护）----------
arm_lock = threading.Lock()
frame_lock = threading.Lock()

arm_busy = False
arm_enabled = True   # True=检测+抓取, False=仅检测（扭矩关闭）
current_frame_rate = FRAME_RATE  # 当前帧率，可通过前端滑条调整
cam = None  # 相机句柄（全局，供路由调用）
last_action_result = "无"
last_defect_types = "无"
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
    'fpga_times': [],
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
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .status-card {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .status-label {
        color: #888;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .status-value {
        font-size: 18px;
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
    .main-grid {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }
    .video-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .card {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .card-title {
        font-size: 12px;
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
    .side-panel {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .log-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    .log-title {
        font-size: 12px;
        color: #a0a0b0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .log-content {
        flex: 1;
        overflow-y: auto;
        font-family: 'Consolas', monospace;
        font-size: 11px;
        line-height: 1.5;
        color: #b0b0b0;
        max-height: 300px;
    }
    .log-entry { padding: 2px 0; }
    .log-info { color: #4caf50; }
    .log-warn { color: #ff9800; }
    .log-error { color: #e53935; }
    .btn-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .btn {
        border: none;
        color: white;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 13px;
        cursor: pointer;
        flex: 1;
        min-width: 100px;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-reset { background: #e94560; }
    .btn-trigger { background: #2196f3; }
    .btn-arm { background: #4caf50; }
    .config-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .config-title {
        font-size: 12px;
        color: #a0a0b0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .config-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        font-size: 11px;
    }
    .config-item {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
    }
    .config-key { color: #888; }
    .config-val { color: #eee; }
    .footer {
        text-align: center;
        color: #666;
        font-size: 11px;
        margin-top: 15px;
    }
    /* 手动控制面板 */
    .control-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .control-title {
        font-size: 12px;
        color: #a0a0b0;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .servo-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .servo-label {
        width: 60px;
        font-size: 12px;
        color: #888;
    }
    .servo-slider {
        flex: 1;
        height: 20px;
    }
    .servo-val {
        width: 35px;
        text-align: right;
        font-size: 12px;
    }
    .suction-btns {
        display: flex;
        gap: 8px;
        margin-top: 10px;
    }
    .btn-suction {
        flex: 1;
        padding: 8px;
        border: none;
        border-radius: 6px;
        color: white;
        font-size: 12px;
        cursor: pointer;
    }
    .btn-on { background: #ff9800; }
    .btn-off { background: #607d8b; }
    /* iOS 风格滑钮 */
    .ios-toggle {
        position: relative;
        display: inline-block;
        width: 52px;
        height: 28px;
    }
    .ios-toggle input { display: none; }
    .ios-toggle-slider {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: #555;
        border-radius: 28px;
        cursor: pointer;
        transition: background 0.3s;
    }
    .ios-toggle-slider::before {
        content: '';
        position: absolute;
        left: 3px; top: 3px;
        width: 22px; height: 22px;
        background: white;
        border-radius: 50%;
        transition: transform 0.3s;
    }
    .ios-toggle input:checked + .ios-toggle-slider {
        background: #4caf50;
    }
    .ios-toggle input:checked + .ios-toggle-slider::before {
        transform: translateX(24px);
    }
    @media (max-width: 1024px) {
        .main-grid { grid-template-columns: 1fr; }
    }
</style>
</head>
<body>
<div class="container">
    <h1>机械臂分拣监控系统</h1>

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
                <span style="color:#4caf50;">OK:0</span>/<span style="color:#e53935;">NG:0</span>
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
        <div class="status-card">
            <div class="status-label">缺陷类型</div>
            <div id="defectTypes" class="status-value" style="color:#e53935;">无</div>
        </div>
    </div>

    <!-- 主内容区 -->
    <div class="main-grid">
        <!-- 视频区域 -->
        <div class="video-grid">
            <div class="card">
                <div class="card-title">原始视频流</div>
                <img src="/video_feed_raw" alt="原始流">
            </div>
            <div class="card">
                <div class="card-title">Canny 边缘检测</div>
                <img src="/video_feed_edge" alt="边缘流">
            </div>
            <div class="card">
                <div class="card-title">最近发送原图</div>
                <img id="sentImg" src="/sent_image" alt="发送原图">
            </div>
            <div class="card">
                <div class="card-title">FPGA 检测结果</div>
                <img id="detectedImg" src="/detected_image" alt="检测结果">
            </div>
        </div>

        <!-- 侧面板 -->
        <div class="side-panel">
            <!-- 操作按钮 -->
            <div class="card">
                <div class="card-title">操作控制</div>
                <div class="btn-group">
                    <button id="resetBtn" class="btn btn-reset" onclick="resetArm()">复位机械臂</button>
                    <button id="triggerBtn" class="btn btn-trigger" onclick="triggerDetect()">手动触发</button>
                </div>
                <div style="display:flex; align-items:center; justify-content:space-between; margin-top:10px; padding:6px 4px;">
                    <span style="font-size:13px; color:#a0a0b0;">机械臂使能</span>
                    <label class="ios-toggle">
                        <input type="checkbox" id="armToggle" checked onchange="onArmToggle()">
                        <span class="ios-toggle-slider"></span>
                    </label>
                </div>
            </div>

            <!-- 手动控制面板 -->
            <div class="control-panel">
                <div class="control-title">手动控制</div>
                <div id="servoControls">
                    <div class="servo-row">
                        <span class="servo-label">1号舵机</span>
                        <input type="range" class="servo-slider" id="servo1" min="0" max="180" value="90" oninput="updateServo(1)">
                        <span class="servo-val" id="servo1Val">90</span>
                    </div>
                    <div class="servo-row">
                        <span class="servo-label">2号舵机</span>
                        <input type="range" class="servo-slider" id="servo2" min="0" max="180" value="90" oninput="updateServo(2)">
                        <span class="servo-val" id="servo2Val">90</span>
                    </div>
                    <div class="servo-row">
                        <span class="servo-label">3号舵机</span>
                        <input type="range" class="servo-slider" id="servo3" min="0" max="180" value="90" oninput="updateServo(3)">
                        <span class="servo-val" id="servo3Val">90</span>
                    </div>
                    <div class="servo-row">
                        <span class="servo-label">4号舵机</span>
                        <input type="range" class="servo-slider" id="servo4" min="0" max="180" value="0" oninput="updateServo(4)">
                        <span class="servo-val" id="servo4Val">0</span>
                    </div>
                    <div class="servo-row">
                        <span class="servo-label">5号舵机</span>
                        <input type="range" class="servo-slider" id="servo5" min="0" max="180" value="90" oninput="updateServo(5)">
                        <span class="servo-val" id="servo5Val">90</span>
                    </div>
                </div>
                <div class="suction-btns">
                    <button class="btn-suction btn-on" onclick="suctionOn()">吸盘启动</button>
                    <button class="btn-suction btn-off" onclick="suctionOff()">吸盘释放</button>
                </div>
            </div>

            <!-- 相机帧率控制 -->
            <div class="control-panel">
                <div class="control-title">相机帧率</div>
                <div class="servo-row">
                    <span class="servo-label">帧率 fps</span>
                    <input type="range" class="servo-slider" id="frameRateSlider" min="1" max="120" value="60" oninput="updateFrameRate()">
                    <span class="servo-val" id="frameRateVal">60</span>
                </div>
            </div>

            <!-- 配置信息 -->
            <div class="config-panel">
                <div class="config-title">当前配置</div>
                <div id="configContent" class="config-grid">
                    <div class="config-item"><span class="config-key">加载中...</span></div>
                </div>
            </div>

            <!-- 操作日志 -->
            <div class="log-panel">
                <div class="log-title" style="display:flex; justify-content:space-between; align-items:center;">
                    <span>操作日志</span>
                    <button id="exportBtn" class="btn btn-suction btn-off" onclick="exportLogs()" style="flex:none; min-width:auto; padding:4px 10px; font-size:11px;">导出日志</button>
                </div>
                <div id="logContent" class="log-content"></div>
            </div>
        </div>
    </div>

    <div class="footer">系统运行中 · 按 Ctrl+C 停止 · v4.0</div>
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
            document.getElementById('resetBtn').disabled = data.busy;
            document.getElementById('triggerBtn').disabled = data.busy;

            if (data.busy) {
                statusEl.innerHTML = '<span class="status-dot dot-red"></span>工作中';
            } else {
                statusEl.innerHTML = '<span class="status-dot dot-green"></span>空闲';
            }
            actionEl.textContent = data.last_action || '无';
            document.getElementById('defectTypes').textContent = data.defect_types || '无';
            document.getElementById('armToggle').checked = data.arm_enabled;
            // 使能关闭时禁用所有机械臂手动控件
            const disabled = !data.arm_enabled;
            document.getElementById('resetBtn').disabled = data.busy || disabled;
            document.getElementById('triggerBtn').disabled = data.busy;
            for (let i = 1; i <= 5; i++) {
                document.getElementById('servo' + i).disabled = disabled;
            }
            const suctionBtns = document.querySelectorAll('[onclick="suctionOn()"], [onclick="suctionOff()"]');
            suctionBtns.forEach(b => b.disabled = disabled);
            if (disabled) {
                document.getElementById('resetBtn').style.opacity = '0.4';
                document.querySelectorAll('.servo-slider').forEach(s => s.style.opacity = '0.4');
                suctionBtns.forEach(b => b.style.opacity = '0.4');
            } else {
                document.getElementById('resetBtn').style.opacity = '';
                document.querySelectorAll('.servo-slider').forEach(s => s.style.opacity = '');
                suctionBtns.forEach(b => b.style.opacity = '');
            }
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
                '<span style="color:#4caf50;">OK:' + data.ok_count + '</span>/<span style="color:#e53935;">NG:' + data.ng_count + '</span>';
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

// 复位机械臂
function resetArm() {
    const btn = document.getElementById('resetBtn');
    btn.disabled = true;
    btn.textContent = '复位中...';
    fetch('/reset', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.textContent = '复位机械臂';
            btn.disabled = false;
        })
        .catch(() => {
            btn.textContent = '复位机械臂';
            btn.disabled = false;
        });
}

// 手动触发检测
function triggerDetect() {
    const btn = document.getElementById('triggerBtn');
    btn.disabled = true;
    btn.textContent = '触发中...';
    fetch('/trigger', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.textContent = '手动触发';
            btn.disabled = false;
        })
        .catch(() => {
            btn.textContent = '手动触发';
            btn.disabled = false;
        });
}

// 更新舵机显示值
function updateServo(id) {
    const val = document.getElementById('servo' + id).value;
    document.getElementById('servo' + id + 'Val').textContent = val;
}

// 发送舵机控制
function setServo(id) {
    const angle = document.getElementById('servo' + id).value;
    fetch('/servo/' + id, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({angle: parseInt(angle)})
    });
}

// 监听slider松开事件发送控制
for (let i = 1; i <= 5; i++) {
    document.getElementById('servo' + i).addEventListener('change', function() {
        setServo(i);
    });
}

// 吸盘控制
function suctionOn() {
    fetch('/suction/on', { method: 'POST' });
}
function suctionOff() {
    fetch('/suction/off', { method: 'POST' });
}

// 相机帧率控制
function updateFrameRate() {
    const val = document.getElementById('frameRateSlider').value;
    document.getElementById('frameRateVal').textContent = val;
}
document.getElementById('frameRateSlider').addEventListener('change', function() {
    const fps = parseInt(this.value);
    fetch('/frame_rate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fps: fps})
    });
});

// 导出日志
function exportLogs() {
    const btn = document.getElementById('exportBtn');
    btn.disabled = true;
    btn.textContent = '导出中...';
    fetch('/export_logs', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.textContent = '导出日志';
            btn.disabled = false;
        })
        .catch(() => {
            btn.textContent = '导出日志';
            btn.disabled = false;
        });
}

// 机械臂抓取开关
function onArmToggle() {
    const enabled = document.getElementById('armToggle').checked;
    const mode = enabled ? 'detect_grab' : 'detect_only';
    fetch('/arm_mode/' + mode, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            document.getElementById('armToggle').checked = data.arm_enabled;
        })
        .catch(() => {
            document.getElementById('armToggle').checked = !enabled;
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
    with arm_lock:
        return jsonify({
            'busy': arm_busy,
            'last_action': last_action_result,
            'defect_types': last_defect_types,
            'arm_enabled': arm_enabled
        })

@app.route('/stats')
def get_stats():
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
    return jsonify({'logs': list(action_logs)})

@app.route('/export_logs', methods=['POST'])
def export_logs():
    """导出日志到 log/ 文件夹"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"log/log_{ts}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for line in action_logs:
                f.write(line + '\n')
        add_log(f"日志已导出: {filename}")
        return jsonify({'message': f'日志已导出到 {filename}', 'file': filename})
    except Exception as e:
        return jsonify({'message': f'导出失败: {e}'}), 500

@app.route('/config')
def get_config():
    return jsonify({
        'FPGA超时': f'{FPGA_TIMEOUT}s',
        '触发冷却': f'{SEND_COOLDOWN}s',
        '串口重试': SUCTION_RETRY,
        '吸盘等待': f'{SUCTION_WAIT_TIME}s',
        '机械臂速度': ARM_SPEED,
        'Canny阈值': f'{CANNY_LOW}-{CANNY_HIGH}',
        '最小边缘点': MIN_EDGE_POINTS,
        '测试模式': FORCE_TEST or '自动',
        '相机帧率': f'{current_frame_rate} fps'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'uptime': time.time() - stats['start_time'],
        'camera': 'ok',
        'arm_busy': arm_busy
    })

@app.route('/reset', methods=['POST'])
def reset():
    if not arm_enabled:
        return jsonify({'message': '机械臂使能已关闭'}), 409
    with arm_lock:
        if arm_busy:
            return jsonify({'message': '机械臂正在工作中，无法复位'}), 409
    t = threading.Thread(target=do_reset, daemon=True)
    t.start()
    return jsonify({'message': '复位指令已发送'})

@app.route('/arm_mode/<mode>', methods=['POST'])
def set_arm_mode(mode):
    """切换机械臂模式: detect_only=仅检测(扭矩关闭), detect_grab=检测+抓取(扭矩开启)"""
    global arm_enabled
    if mode == 'detect_only':
        Arm.Arm_serial_set_torque(0)
        arm_enabled = False
        add_log("切换为 [仅检测] 模式，机械臂扭矩已关闭")
        return jsonify({'message': '已切换为仅检测模式', 'arm_enabled': False})
    elif mode == 'detect_grab':
        Arm.Arm_serial_set_torque(1)
        arm_enabled = True
        add_log("切换为 [检测+抓取] 模式，机械臂扭矩已开启")
        return jsonify({'message': '已切换为检测+抓取模式', 'arm_enabled': True})
    return jsonify({'message': '无效模式'}), 400

@app.route('/frame_rate', methods=['POST'])
def set_frame_rate():
    """调整相机帧率"""
    global current_frame_rate, cam
    data = request.get_json()
    fps = data.get('fps', FRAME_RATE)
    fps = max(1, min(120, int(fps)))
    try:
        if cam is not None:
            cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
            current_frame_rate = fps
            add_log(f"相机帧率已调整为 {fps} fps")
            return jsonify({'message': f'帧率已设置为 {fps} fps', 'frame_rate': fps})
        else:
            return jsonify({'message': '相机未初始化'}), 503
    except Exception as e:
        add_log(f"帧率调整失败: {e}", "ERROR")
        return jsonify({'message': f'调整失败: {e}'}), 500

@app.route('/trigger', methods=['POST'])
def trigger():
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

@app.route('/servo/<int:servo_id>', methods=['POST'])
def control_servo(servo_id):
    """控制单个舵机"""
    if servo_id < 1 or servo_id > 5:
        return jsonify({'message': '舵机ID无效，应为1-5'}), 400
    data = request.get_json()
    angle = data.get('angle', 90)
    angle = max(0, min(180, angle))
    if not arm_enabled:
        return jsonify({'message': '机械臂使能已关闭'}), 409
    with arm_lock:
        if arm_busy:
            return jsonify({'message': '机械臂正在工作中'}), 409
    try:
        Arm.Arm_serial_servo_write(servo_id, angle, 500)
        add_log(f"舵机{servo_id} -> {angle}度")
        return jsonify({'message': f'舵机{servo_id}已设置为{angle}度'})
    except Exception as e:
        add_log(f"舵机{servo_id}控制失败: {e}", "ERROR")
        return jsonify({'message': f'控制失败: {e}'}), 500

@app.route('/suction/<action>', methods=['POST'])
def control_suction(action):
    """控制吸盘"""
    if not arm_enabled:
        return jsonify({'message': '机械臂使能已关闭'}), 409
    if action == 'on':
        success = zhua_qu_xipan()
        return jsonify({'message': '吸盘已启动' if success else '吸盘启动失败'})
    elif action == 'off':
        success = shi_fang_xipan()
        return jsonify({'message': '吸盘已释放' if success else '吸盘释放失败'})
    return jsonify({'message': '无效操作'}), 400

def do_reset():
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
    global arm_busy, last_action_result
    with arm_lock:
        arm_busy = True
    add_log(f"===== 开始执行 {action_type} 动作 =====")
    start_time = time.time()

    try:
        # 步骤1: 下降到吸取位置
        add_log(f"[1/6] 机械臂下降到吸取位置 p_Z={p_Z}, 速度={ARM_SPEED}ms")
        Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
        add_log(f"      等待 {ARM_SPEED/1000 + 0.3:.1f}s ...")
        time.sleep(ARM_SPEED / 1000 + 0.3)

        # 步骤2: 启动吸盘
        add_log(f"[2/6] 启动吸盘，等待吸力建立 {SUCTION_WAIT_TIME}s ...")
        zhua_qu_xipan()
        time.sleep(SUCTION_WAIT_TIME)

        # 步骤3: 抬起
        add_log(f"[3/6] 机械臂抬起 p_Z_1={p_Z_1}, 速度={ARM_SPEED}ms")
        Arm.Arm_serial_servo_write6(*p_Z_1, ARM_SPEED)
        add_log(f"      等待 {ARM_SPEED/1000 + 0.3:.1f}s ...")
        time.sleep(ARM_SPEED / 1000 + 0.3)

        # 步骤4: 执行 OK / NG 运动
        if action_type == "OK":
            add_log(f"[4a/6] OK-1: p_OK_1={p_OK_1}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_OK_1, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            add_log(f"[4b/6] OK-2: p_OK_2={p_OK_2}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_OK_2, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            add_log(f"[4c/6] OK-3: p_OK={p_OK}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_OK, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
        elif action_type == "NG":
            add_log(f"[4a/6] NG-1: p_NG_1={p_NG_1}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_NG_1, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            add_log(f"[4b/6] NG-2: p_NG={p_NG}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_NG, ARM_SPEED)
            time.sleep(ARM_SPEED / 1000 + 0.3)
            add_log(f"[4c/6] 蜂鸣器 ON, 持续 {BUZZER_DELAY*100}ms")
            Arm.Arm_Buzzer_On(delay=BUZZER_DELAY)
            time.sleep(BUZZER_DELAY * 0.1 + 0.2)

        # 步骤5: 释放吸盘
        add_log(f"[5/6] 释放吸盘")
        shi_fang_xipan()
        add_log(f"      等待 2s ...")
        time.sleep(2)

        # 步骤6: 复位
        add_log(f"[6/6] 机械臂复位 p_initial={p_initial}, 速度={ARM_SPEED}ms")
        Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
        add_log(f"      等待 {ARM_SPEED/1000 + 0.3:.1f}s ...")
        time.sleep(ARM_SPEED / 1000 + 0.3)

        elapsed = time.time() - start_time
        with arm_lock:
            last_action_result = action_type
        add_log(f"===== {action_type} 动作完成, 总耗时 {elapsed:.1f}s =====")

    except Exception as e:
        add_log(f"动作执行异常: {e}", "ERROR")
        with arm_lock:
            last_action_result = f"错误: {e}"
    finally:
        with arm_lock:
            arm_busy = False

def send_to_fpga_and_decide(frame_bgr, timestamp):
    """发送图片到FPGA并决定动作（在线程中调用）"""
    global latest_sent_raw, latest_detected_img, last_defect_types

    # 保存发送给FPGA的原图
    origin_name = f"origin/origin_{timestamp}.jpg"
    cv2.imwrite(origin_name, frame_bgr)

    success, encoded = cv2.imencode('.jpg', frame_bgr)
    if not success:
        add_log("图像编码失败", "ERROR")
        return

    image_bytes = encoded.tobytes()
    add_log(f"发送图片到 FPGA...")
    fpga_start = time.time()

    try:
        resp = requests.post(FPGA_URL, files={'image_file': image_bytes}, timeout=FPGA_TIMEOUT)
        fpga_time = (time.time() - fpga_start) * 1000

        if resp.status_code != 200:
            add_log(f"FPGA请求失败: {resp.status_code}", "ERROR")
            return

        result = json.loads(resp.text)
        objects = result.get('objects', [])
        objects_count = len(objects)
        inference_ms = result.get('inference_time_ms', 'N/A')

        # 提取缺陷类型（去重）
        defect_types = list(set(obj['class_name'] for obj in objects)) if objects else []
        defect_str = ', '.join(defect_types) if defect_types else '无'
        last_defect_types = defect_str

        add_log(f"FPGA返回: 目标数={objects_count}, 缺陷类型=[{defect_str}], 推理={inference_ms}ms, 总耗时={fpga_time:.0f}ms")

        stats['fpga_times'].append(fpga_time)
        if len(stats['fpga_times']) > 10:
            stats['fpga_times'] = stats['fpga_times'][-10:]

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
                with frame_lock:
                    latest_detected_img = detect_img.copy()
            except Exception as e:
                add_log(f"检测图片保存失败: {e}", "WARN")

        with frame_lock:
            latest_sent_raw = frame_bgr.copy()

        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0:
            action = "OK"
        else:
            action = "NG"

        if action == "OK":
            stats['ok_count'] += 1
        else:
            stats['ng_count'] += 1

        # 执行机械臂动作（仅检测模式时跳过）
        if arm_enabled:
            execute_arm_action(action)
        else:
            add_log(f"仅检测模式，跳过 {action} 抓取动作")

    except requests.exceptions.Timeout:
        add_log(f"FPGA请求超时 ({FPGA_TIMEOUT}s)", "ERROR")
    except Exception as e:
        add_log(f"FPGA请求异常: {e}", "ERROR")

# ---------- 主循环 ----------
def main():
    global last_send_time, arm_busy, latest_frame_raw, latest_frame_edge

    add_log("系统启动中...")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    add_log("Flask Web 服务已启动，端口 5000")

    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        add_log("未找到相机设备", "ERROR")
        return
    add_log(f"找到 {deviceList.nDeviceNum} 个相机")
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    global cam
    cam = MvCamera()
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", FRAME_RATE)
    add_log(f"相机帧率已设置为 {FRAME_RATE} fps")
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
                    consecutive_failures = 0
                continue
            consecutive_failures = 0

            raw = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            # 第一步：压缩到 640x480，减少向 FPGA 传输的数据量
            raw = cv2.resize(raw, (640, 480))
            h, w = raw.shape

            edges = cv2.Canny(raw, CANNY_LOW, CANNY_HIGH)
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

            # 计算平均x
            avg_x = None
            pts = np.column_stack(np.where(edges > 0))
            if len(pts) >= MIN_EDGE_POINTS:
                sorted_pts = pts[np.argsort(pts[:, 1])]
                left_pts = sorted_pts[:min(LEFT_PIXELS, len(sorted_pts))]
                right_pts = sorted_pts[-min(RIGHT_PIXELS, len(sorted_pts)):]
                all_selected = np.vstack([left_pts, right_pts])
                avg_x = int(np.mean(all_selected[:, 1]))

            # 在边缘图上画avg_x线和半宽线
            if avg_x is not None:
                half_width = w // 2
                # 画avg_x竖线（红色）
                cv2.line(edge_bgr, (avg_x, 0), (avg_x, h-1), (0, 0, 255), 2)
                cv2.putText(edge_bgr, f"AvgX={avg_x}", (avg_x+5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # 画半宽线（绿色，作为触发阈值参考）
                cv2.line(edge_bgr, (half_width, 0), (half_width, h-1), (0, 255, 0), 1)
                cv2.putText(edge_bgr, f"Half={half_width}", (half_width+5, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            with frame_lock:
                latest_frame_raw = raw_bgr.copy()
                latest_frame_edge = edge_bgr.copy()

            # 触发检测（异步线程，不阻塞主循环）
            if avg_x is not None:
                half_width = w // 2
                now = time.time()
                with arm_lock:
                    is_busy = arm_busy
                if avg_x > half_width and (now - last_send_time) >= SEND_COOLDOWN and not is_busy:
                    last_send_time = now
                    original_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    ts = int(time.time())
                    # 在新线程中执行FPGA请求和机械臂动作
                    t = threading.Thread(target=send_to_fpga_and_decide,
                                         args=(original_bgr, ts), daemon=True)
                    t.start()

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
