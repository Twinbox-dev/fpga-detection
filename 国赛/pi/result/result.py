#!/usr/bin/env python3
"""
5.1 队列版：海康相机实时流 + Canny边缘检测 + 自动FPGA推理 + 机械臂分拣 + 美观Web前端
基于 5.0.py 改进：
  - 中心基准线 X/Y 位置均可通过全局参数独立调整
  - 检测区域同时约束横轴和纵轴范围（矩形ROI）
  - 前端视频流卡片固定 640x480 避免空白区域
"""

import sys
import time
import json
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
import base64
import subprocess

sys.path.append("./sdk")
from MvCameraControl_class import *
from Arm_Lib import Arm_Device

# ===================== 用户可配置参数 =====================
# ---- 视觉检测参数 ----
CENTER_X            = 380    # ★ 纵向基准线 X 坐标（默认图像水平中心）
CENTER_Y            = 240    # ★ 横向基准线 Y 坐标（默认图像垂直中心）
CENTER_X_RANGE      = 50     # ★ 纵向基准线左右各多少像素（总宽度 = 2 × CENTER_X_RANGE）
CENTER_Y_RANGE      = 200     # ★ 横向基准线上下各多少像素（总高度 = 2 × CENTER_Y_RANGE）
EDGE_COUNT_THRESHOLD= 200    # ★ 中心区域白色像素个数阈值，超过此值触发FPGA
CANNY_LOW           = 50
CANNY_HIGH          = 150
SERVER_URL          = "http://172.16.68.110:9000"
FPGA_URL            = "http://172.16.68.110:9000/upload"
SEND_COOLDOWN       = 3
FPGA_TIMEOUT        = 5
FRAME_RATE          = 60    # 相机帧率 fps（1-120）

# ---- 串口（吸盘）参数 ----
SUCTION_SERIAL_PORT = '/dev/ttyUSB0'
SUCTION_SERIAL_BAUD = 9600
SUCTION_WAIT_TIME   = 0.6
SUCTION_RETRY       = 1

# ---- 串口（报警器）参数 ----
ALARM_SERIAL_PORT   = '/dev/BAOJING'
ALARM_SERIAL_BAUD   = 9600

# ---- 机械臂参数 ----
ARM_SPEED = 350
BUZZER_DELAY = 10
FORCE_TEST = None

# ---- 监控参数 ----
MAX_LOGS            = 100
FRAME_FAIL_WARN     = 50
ARM_ACTION_TIMEOUT  = 30

# ---- AI 助手参数 ----
OLLAMA_URL          = "http://127.0.0.1:11434"   # 本地 Ollama 服务地址
OLLAMA_MODEL        = "qwen3:0.6b"               # 本地模型（4GB 板用最小档，回答更快）
OLLAMA_KEEPALIVE    = "5m"                        # 模型用后 5 分钟卸载（0.6b 很小，保留久一点减少重复加载）
OLLAMA_TIMEOUT      = 120                         # 单次回答超时(秒)
# ==========================================================

os.makedirs("img_detect", exist_ok=True)
os.makedirs("json", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs("origin", exist_ok=True)

# ---------- 全局状态（带锁保护）----------
arm_lock = threading.Lock()
frame_lock = threading.Lock()
arm_queue_lock = threading.Lock()

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

# ---------- 机械臂任务队列 ----------
# 每个元素: {'action': 'OK'|'NG', 'timestamp': int, 'defect_types': str}
arm_queue = deque()
arm_worker_running = True

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

# ---------- 检测历史与统计（仅供 Web 展示，不影响任何控制逻辑）----------
DEFECT_NAME_MAP = {
    "ca_shang": "擦伤",
    "zang_wu": "脏污",
    "zhe_zhou": "褶皱",
    "zhen_kong": "针孔",
    "zheng_chang": "正常",
}
DEFECT_ORDER = ["ca_shang", "zang_wu", "zhe_zhou", "zhen_kong"]

HISTORY_FILE = "defect_history.json"
history_lock = threading.Lock()
detected_images = deque(maxlen=4)   # 最近四次检测结果图 {'ts': int, 'jpeg': bytes}
defect_records = []                 # 缺陷历史 [{'ts': int, 'defects': [str, ...]}, ...]


def load_defect_records():
    """启动时从本地文件加载缺陷历史（用于历史/当月统计图表）"""
    global defect_records
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                defect_records = [r for r in data if isinstance(r, dict) and 'ts' in r]
    except Exception as e:
        print(f"[WARN] 加载缺陷历史失败: {e}")


def save_defect_records():
    """保存缺陷历史到本地文件（调用方需已持有 history_lock）"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(defect_records, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 保存缺陷历史失败: {e}")


def aggregate_defects(records):
    """按缺陷类别聚合计数，返回 {pinyin_name: count}（只统计真正的缺陷，不含正常）"""
    counts = {k: 0 for k in DEFECT_ORDER}
    for r in records:
        for d in r.get('defects', []):
            if d in counts:
                counts[d] += 1
    return counts


def current_month_records(records):
    """筛选当前自然月内的记录（按本地时间）"""
    now = time.localtime()
    keep = []
    for r in records:
        try:
            lt = time.localtime(r['ts'])
        except Exception:
            continue
        if lt.tm_year == now.tm_year and lt.tm_mon == now.tm_mon:
            keep.append(r)
    return keep


def today_records(records):
    """筛选今天（自然日）内的记录"""
    now = time.localtime()
    keep = []
    for r in records:
        try:
            lt = time.localtime(r['ts'])
        except Exception:
            continue
        if lt.tm_year == now.tm_year and lt.tm_mon == now.tm_mon and lt.tm_mday == now.tm_mday:
            keep.append(r)
    return keep


def recent_days_records(records, days):
    """筛选最近 N 天内的记录"""
    cutoff = time.time() - days * 86400
    return [r for r in records if r['ts'] >= cutoff]


def _ok_ng(recs):
    """缺陷为空视为 OK，否则为 NG"""
    ok = sum(1 for r in recs if len(r.get('defects', [])) == 0)
    return ok, len(recs) - ok


def build_stats_digest():
    """生成给 AI 与报告使用的检测数据摘要（只读，不影响控制）"""
    with history_lock:
        records = list(defect_records)
    today = today_records(records)
    week = recent_days_records(records, 7)
    month = current_month_records(records)
    t_ok, t_ng = _ok_ng(today)
    w_ok, w_ng = _ok_ng(week)
    m_ok, m_ng = _ok_ng(month)
    all_ok, all_ng = _ok_ng(records)
    return {
        'today': {'total': len(today), 'ok': t_ok, 'ng': t_ng, 'defects': aggregate_defects(today)},
        'week7': {'total': len(week), 'ok': w_ok, 'ng': w_ng, 'defects': aggregate_defects(week)},
        'month': {'total': len(month), 'ok': m_ok, 'ng': m_ng, 'defects': aggregate_defects(month)},
        'all':   {'total': len(records), 'ok': all_ok, 'ng': all_ng, 'defects': aggregate_defects(records)},
        'session_ok': stats['ok_count'],
        'session_ng': stats['ng_count'],
    }


def _fmt_defects(dc):
    parts = [f"{DEFECT_NAME_MAP[k]}={dc.get(k, 0)}" for k in DEFECT_ORDER if dc.get(k, 0) > 0]
    return "，".join(parts) if parts else "无"


def build_system_prompt(d):
    """根据统计数据生成 AI 系统提示词，让模型基于真实数据回答"""
    return (
        "你是「机械臂分拣系统」的检测数据分析助手，负责基于以下真实检测统计数据回答用户问题。\n"
        "\n"
        f"【今日检测】总数={d['today']['total']}，正常OK={d['today']['ok']}，缺陷NG={d['today']['ng']}；"
        f"缺陷分布：{_fmt_defects(d['today']['defects'])}\n"
        f"【近7天】总数={d['week7']['total']}，OK={d['week7']['ok']}，NG={d['week7']['ng']}；"
        f"缺陷分布：{_fmt_defects(d['week7']['defects'])}\n"
        f"【当月】总数={d['month']['total']}，OK={d['month']['ok']}，NG={d['month']['ng']}；"
        f"缺陷分布：{_fmt_defects(d['month']['defects'])}\n"
        f"【累计历史】总数={d['all']['total']}；缺陷分布：{_fmt_defects(d['all']['defects'])}\n"
        f"【当前运行会话】OK={d['session_ok']}，NG={d['session_ng']}\n"
        "\n"
        "缺陷类别：擦伤、脏污、褶皱、针孔（另有「正常」表示无缺陷）。\n"
        "要求：严格基于以上数据回答，没有的信息不要编造；用简洁中文，需要时列数字；"
        "回答控制在200字以内。"
    )


def call_ollama(system_prompt, question):
    """调用本地 Ollama 模型，返回 (answer, ok)。失败时返回友好提示。"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                'model': OLLAMA_MODEL,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': '/no_think\n' + question},
                ],
                'stream': False,
                'think': False,
                'keep_alive': OLLAMA_KEEPALIVE,
                'options': {'temperature': 0.3, 'num_predict': 320},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code != 200:
            return (f"AI 服务返回错误(状态码 {resp.status_code})。请确认 Ollama 已启动并已执行 ollama pull {OLLAMA_MODEL}。", False)
        answer = resp.json().get('message', {}).get('content', '').strip()
        if not answer:
            return "AI 没有返回内容，请稍后再试。", False
        return answer, True
    except requests.exceptions.ConnectionError:
        return ("无法连接本地 AI 服务。请先在树莓派上安装并启动 Ollama（ollama serve），再执行 ollama pull " + OLLAMA_MODEL + "。", False)
    except requests.exceptions.Timeout:
        return f"AI 响应超时（>{OLLAMA_TIMEOUT}s），4GB 板上小模型较慢，请稍后重试。", False
    except Exception as e:
        return f"调用 AI 失败: {e}", False


def stream_ollama(system_prompt, question):
    """流式调用 Ollama，逐段产出回答文本（生成器，供 /chat 流式返回）"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                'model': OLLAMA_MODEL,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': '/no_think\n' + question},
                ],
                'stream': True,
                'think': False,
                'keep_alive': OLLAMA_KEEPALIVE,
                'options': {'temperature': 0.3, 'num_predict': 320},
            },
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code != 200:
            yield f"[错误] AI 服务返回错误(状态码 {resp.status_code})，请确认 Ollama 已启动并已执行 ollama pull {OLLAMA_MODEL}。"
            return
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
            if chunk.get('done'):
                break
    except requests.exceptions.ConnectionError:
        yield "无法连接本地 AI 服务。请先在树莓派上安装并启动 Ollama（ollama serve），再执行 ollama pull " + OLLAMA_MODEL + "。"
    except requests.exceptions.Timeout:
        yield f"AI 响应超时（>{OLLAMA_TIMEOUT}s），请稍后重试。"
    except Exception as e:
        yield f"调用 AI 失败: {e}"


def ensure_ollama_running():
    """确保本地 Ollama 服务在运行；没跑就后台启动它（仅启动服务，不涉及任何控制逻辑）"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    for cmd in ['ollama', '/usr/local/bin/ollama']:
        try:
            subprocess.Popen([cmd, 'serve'],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             start_new_session=True)
            add_log("已自动启动本地 Ollama 服务（后台）")
            return True
        except FileNotFoundError:
            continue
        except Exception as e:
            add_log(f"启动 Ollama 服务失败: {e}", "WARN")
            return False
    add_log("未找到 ollama 可执行文件，请确认已安装 Ollama", "WARN")
    return False


def build_report_html(d):
    """生成 HTML 检测报告"""
    def stat_block(title, s):
        total = s['total']
        ng_rate = f"{(s['ng'] / total * 100):.1f}%" if total else "0%"
        rows = "".join(
            f"<tr><td>{DEFECT_NAME_MAP[k]}</td><td>{s['defects'].get(k, 0)}</td></tr>"
            for k in DEFECT_ORDER
        )
        return (
            f"<div class='blk'><h3>{title}</h3>"
            f"<p>检测总数：<b>{total}</b> &nbsp; 正常(OK)：<b>{s['ok']}</b> &nbsp; 缺陷(NG)：<b>{s['ng']}</b> &nbsp; 缺陷率：<b>{ng_rate}</b></p>"
            f"<table><tr><th>缺陷类别</th><th>数量</th></tr>{rows}</table></div>"
        )
    gen_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
        "<title>检测报告</title><style>"
        "body{font-family:'Microsoft YaHei',sans-serif;margin:30px;color:#222;}"
        "h1{color:#16213e;}h2{color:#666;font-size:14px;font-weight:normal;}"
        ".blk{border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0;}"
        "table{border-collapse:collapse;margin-top:8px;}"
        "th,td{border:1px solid #ddd;padding:6px 16px;text-align:center;}"
        "th{background:#f2f2f2;}"
        "</style></head><body>"
        "<h1>机械臂分拣系统 · 检测报告</h1>"
        f"<h2>生成时间：{gen_time}</h2>"
        + stat_block("今日检测", d['today'])
        + stat_block("近 7 天", d['week7'])
        + stat_block("当月", d['month'])
        + stat_block("累计历史", d['all'])
        + "</body></html>"
    )


def build_report_md(d):
    """生成 Markdown 检测报告"""
    def stat_block(title, s):
        total = s['total']
        ng_rate = f"{(s['ng'] / total * 100):.1f}%" if total else "0%"
        lines = [
            f"## {title}",
            f"- 检测总数：{total}",
            f"- 正常(OK)：{s['ok']}",
            f"- 缺陷(NG)：{s['ng']}",
            f"- 缺陷率：{ng_rate}",
            "| 缺陷类别 | 数量 |",
            "| --- | --- |",
        ]
        for k in DEFECT_ORDER:
            lines.append(f"| {DEFECT_NAME_MAP[k]} | {s['defects'].get(k, 0)} |")
        return "\n".join(lines)
    gen_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "# 机械臂分拣系统 · 检测报告\n\n"
        f"生成时间：{gen_time}\n\n"
        + stat_block("今日检测", d['today']) + "\n\n"
        + stat_block("近7天", d['week7']) + "\n\n"
        + stat_block("当月", d['month']) + "\n\n"
        + stat_block("累计历史", d['all']) + "\n"
    )


load_defect_records()

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
    /* 任务队列样式 */
    .queue-panel {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .queue-title {
        font-size: 12px;
        color: #a0a0b0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .queue-blocks {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        min-height: 32px;
        align-items: center;
    }
    .queue-block {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: bold;
        color: white;
        white-space: nowrap;
    }
    .queue-ok { background: #4caf50; }
    .queue-ng { background: #e53935; }
    .queue-empty {
        color: #666;
        font-style: italic;
        font-size: 12px;
    }
    .queue-count {
        color: #2196f3;
        font-size: 14px;
        font-weight: bold;
        margin-left: 6px;
    }
    /* 最近四次检测结果 */
    .recent-section {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 20px;
    }
    .recent-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .recent-item {
        background: #0f3460;
        border-radius: 8px;
        overflow: hidden;
    }
    .recent-item img {
        width: 100%;
        height: 160px;
        object-fit: contain;
        display: block;
        background: #0a0a1a;
    }
    .recent-item .recent-label {
        font-size: 11px;
        color: #a0a0b0;
        padding: 6px 8px;
        text-align: center;
        background: #12203c;
    }
    .recent-item .recent-empty {
        height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #555;
        font-size: 12px;
        background: #0a0a1a;
    }
    /* 检测统计模块 */
    .stats-module {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 12px;
        margin-bottom: 20px;
    }
    .chart-card {
        background: #16213e;
        border-radius: 12px;
        padding: 12px;
    }
    .chart-box {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
    }
    .chart-box svg { max-width: 100%; height: auto; }
    .donut-legend {
        display: flex;
        justify-content: center;
        gap: 18px;
        margin-top: 6px;
        font-size: 12px;
    }
    .donut-legend .lg-item { display: flex; align-items: center; gap: 6px; color: #c0c0d0; }
    .donut-legend .lg-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    @media (max-width: 1024px) {
        .stats-module { grid-template-columns: 1fr; }
        .recent-grid { grid-template-columns: repeat(2, 1fr); }
    }
    /* AI 助手对话面板 */
    .chat-messages {
        height: 220px;
        overflow-y: auto;
        background: #0f3460;
        border-radius: 8px;
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 10px;
    }
    .chat-msg {
        max-width: 85%;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .chat-msg.ai { align-self: flex-start; background: #16213e; color: #e0e0ea; }
    .chat-msg.user { align-self: flex-end; background: #2196f3; color: #fff; }
    .chat-input-row { display: flex; gap: 8px; }
    .chat-input {
        flex: 1;
        padding: 8px 12px;
        border: none;
        border-radius: 6px;
        background: #0f3460;
        color: #eee;
        font-size: 13px;
        outline: none;
    }
    .chat-input:focus { box-shadow: 0 0 0 1px #2196f3; }
    .chat-export-row { display: flex; gap: 8px; margin-top: 10px; }
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
                <div class="card-title">自研检测算法</div>
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

            <!-- 任务队列 -->
            <div class="queue-panel">
                <div class="queue-title">机械臂任务队列 <span id="queueCount" class="queue-count">0</span></div>
                <div id="queueBlocks" class="queue-blocks">
                    <span class="queue-empty">队列为空</span>
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

    <!-- 最近四次检测结果 -->
    <div class="recent-section">
        <div class="card-title">最近四次检测结果</div>
        <div class="recent-grid" id="recentGrid">
            <div class="recent-item"><div class="recent-empty">暂无</div><div class="recent-label">检测结果 1</div></div>
            <div class="recent-item"><div class="recent-empty">暂无</div><div class="recent-label">检测结果 2</div></div>
            <div class="recent-item"><div class="recent-empty">暂无</div><div class="recent-label">检测结果 3</div></div>
            <div class="recent-item"><div class="recent-empty">暂无</div><div class="recent-label">检测结果 4</div></div>
        </div>
    </div>

    <!-- 检测统计模块 -->
    <div class="stats-module">
        <div class="chart-card">
            <div class="card-title">正常与缺陷占比（历史）</div>
            <div class="chart-box" id="donutChart"></div>
            <div class="donut-legend">
                <span class="lg-item"><span class="lg-dot" style="background:#4caf50;"></span>正常 <span id="okCount">0</span></span>
                <span class="lg-item"><span class="lg-dot" style="background:#e53935;"></span>缺陷 <span id="ngCount">0</span></span>
            </div>
        </div>
        <div class="chart-card">
            <div class="card-title">历史缺陷分布统计</div>
            <div class="chart-box" id="historyBarChart"></div>
        </div>
        <div class="chart-card">
            <div class="card-title">当月内缺陷分布统计</div>
            <div class="chart-box" id="monthBarChart"></div>
        </div>
    </div>

    <!-- AI 助手 -->
    <div class="recent-section">
        <div class="card-title">AI 助手</div>
        <div id="chatMessages" class="chat-messages">
            <div class="chat-msg ai">你好，我是检测数据分析助手，可以问我「今天的检测结果怎么样」「最近针孔多吗」等问题。</div>
        </div>
        <div class="chat-input-row">
            <input type="text" id="chatInput" class="chat-input" placeholder="输入问题…" onkeydown="if(event.key==='Enter')sendChat()">
            <button id="chatSendBtn" class="btn btn-trigger" onclick="sendChat()">发送</button>
        </div>
        <div class="chat-export-row">
            <button class="btn btn-suction btn-off" onclick="exportReport('html')">导出报告(HTML)</button>
            <button class="btn btn-suction btn-off" onclick="exportReport('md')">导出报告(MD)</button>
        </div>
    </div>

    <div class="footer">系统运行中 · 按 Ctrl+C 停止 · v5.1</div>
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
            // 手动触发不再受 arm_busy 限制
            document.getElementById('resetBtn').disabled = data.busy;
            document.getElementById('triggerBtn').disabled = false;

            if (data.busy) {
                statusEl.innerHTML = '<span class="status-dot dot-red"></span>工作中';
            } else if (data.queue_length > 0) {
                statusEl.innerHTML = '<span class="status-dot dot-green"></span>待处理(' + data.queue_length + ')';
            } else {
                statusEl.innerHTML = '<span class="status-dot dot-green"></span>空闲';
            }
            actionEl.textContent = data.last_action || '无';
            document.getElementById('defectTypes').textContent = data.defect_types || '无';
            document.getElementById('armToggle').checked = data.arm_enabled;
            // 使能关闭时禁用所有机械臂手动控件
            const disabled = !data.arm_enabled;
            document.getElementById('resetBtn').disabled = data.busy || disabled;
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

// 定时获取任务队列
function fetchQueue() {
    fetch('/queue')
        .then(r => r.json())
        .then(data => {
            const blocksEl = document.getElementById('queueBlocks');
            const countEl = document.getElementById('queueCount');
            countEl.textContent = data.length;
            if (data.queue.length === 0) {
                blocksEl.innerHTML = '<span class="queue-empty">队列为空</span>';
            } else {
                blocksEl.innerHTML = data.queue.map(item => {
                    const cls = item.action === 'OK' ? 'queue-ok' : 'queue-ng';
                    const label = item.action === 'OK' ? 'OK' : ('NG [' + (item.defect_types || '?') + ']');
                    return '<span class="queue-block ' + cls + '">' + label + '</span>';
                }).join('');
            }
        })
        .catch(() => {});
}
setInterval(fetchQueue, 1000);
fetchQueue();

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

// ==================== 最近四次检测结果 ====================
function fetchRecentImages() {
    fetch('/recent_images')
        .then(r => r.json())
        .then(data => {
            const grid = document.getElementById('recentGrid');
            const imgs = data.images || [];
            let html = '';
            for (let i = 0; i < 4; i++) {
                const label = '检测结果 ' + (i + 1);
                if (i < imgs.length) {
                    html += '<div class="recent-item"><img src="' + imgs[i].src + '" alt="' + label + '">'
                         + '<div class="recent-label">' + label + '</div></div>';
                } else {
                    html += '<div class="recent-item"><div class="recent-empty">暂无</div>'
                         + '<div class="recent-label">' + label + '</div></div>';
                }
            }
            grid.innerHTML = html;
        })
        .catch(() => {});
}

// ==================== 检测统计图表 ====================
const DEFECT_KEYS = ['ca_shang', 'zang_wu', 'zhe_zhou', 'zhen_kong'];
const DEFECT_LABELS = ['擦伤', '脏污', '褶皱', '针孔'];
const DEFECT_COLORS = ['#42a5f5', '#ffb74d', '#ab47bc', '#26c6da'];

function renderDonutChart(ok, ng) {
    const total = ok + ng;
    const el = document.getElementById('donutChart');
    const r = 60, c = 2 * Math.PI * r, cx = 70, cy = 70;
    let svg = '<svg viewBox="0 0 140 140" width="200" height="200" role="img" aria-label="正常与缺陷占比">';
    svg += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="#26324a" stroke-width="20"></circle>';
    if (total > 0) {
        const okLen = (ok / total) * c;
        const ngLen = (ng / total) * c;
        if (ng > 0) {
            svg += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="#e53935" stroke-width="20" stroke-dasharray="' + ngLen + ' ' + (c - ngLen) + '" stroke-dashoffset="' + (-okLen) + '" transform="rotate(-90 ' + cx + ' ' + cy + ')"></circle>';
        }
        if (ok > 0) {
            svg += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="#4caf50" stroke-width="20" stroke-dasharray="' + okLen + ' ' + (c - okLen) + '" stroke-dashoffset="0" transform="rotate(-90 ' + cx + ' ' + cy + ')"></circle>';
        }
        svg += '<text x="' + cx + '" y="' + (cy - 2) + '" text-anchor="middle" fill="#eee" font-size="22" font-weight="bold">' + total + '</text>';
        svg += '<text x="' + cx + '" y="' + (cy + 16) + '" text-anchor="middle" fill="#888" font-size="11">总检测</text>';
    } else {
        svg += '<text x="' + cx + '" y="' + cy + '" text-anchor="middle" fill="#888" font-size="13">暂无数据</text>';
    }
    svg += '</svg>';
    el.innerHTML = svg;
}

function renderBarChart(elId, counts) {
    const el = document.getElementById(elId);
    const vals = DEFECT_KEYS.map(k => counts[k] || 0);
    const max = Math.max.apply(null, vals.concat([1]));
    const W = 300, H = 200, baseY = 158, barMaxH = 120, barW = 44, gap = 24;
    const totalBarWidth = DEFECT_KEYS.length * barW + (DEFECT_KEYS.length - 1) * gap;
    const startX = (W - totalBarWidth) / 2;
    let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="280" role="img" aria-label="缺陷分布柱状图">';
    svg += '<line x1="20" y1="' + baseY + '" x2="' + (W - 20) + '" y2="' + baseY + '" stroke="#33415c" stroke-width="1"></line>';
    for (let i = 0; i < DEFECT_KEYS.length; i++) {
        const x = startX + i * (barW + gap);
        const v = vals[i];
        const h = v > 0 ? Math.max(4, (v / max) * barMaxH) : 2;
        const y = baseY - h;
        svg += '<rect x="' + x + '" y="' + y + '" width="' + barW + '" height="' + h + '" rx="4" fill="' + DEFECT_COLORS[i] + '">'
            + '<title>' + DEFECT_LABELS[i] + '：' + v + '</title></rect>';
        svg += '<text x="' + (x + barW / 2) + '" y="' + (y - 6) + '" text-anchor="middle" fill="#eee" font-size="12">' + v + '</text>';
        svg += '<text x="' + (x + barW / 2) + '" y="' + (baseY + 16) + '" text-anchor="middle" fill="#a0a0b0" font-size="12">' + DEFECT_LABELS[i] + '</text>';
    }
    svg += '</svg>';
    el.innerHTML = svg;
}

function fetchChartData() {
    fetch('/chart_data')
        .then(r => r.json())
        .then(data => {
            renderDonutChart(data.ok || 0, data.ng || 0);
            document.getElementById('okCount').textContent = data.ok || 0;
            document.getElementById('ngCount').textContent = data.ng || 0;
            renderBarChart('historyBarChart', data.history || {});
            renderBarChart('monthBarChart', data.month || {});
        })
        .catch(() => {});
}

// 初始化并定时刷新
fetchRecentImages();
fetchChartData();
setInterval(fetchRecentImages, 2000);
setInterval(fetchChartData, 3000);

// ==================== AI 助手对话 ====================
function addChatMsg(text, who) {
    const box = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg ' + who;
    div.textContent = text;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

function sendChat() {
    const input = document.getElementById('chatInput');
    const q = input.value.trim();
    if (!q) return;
    const btn = document.getElementById('chatSendBtn');
    input.value = '';
    addChatMsg(q, 'user');

    // 先插入一个空的 AI 气泡，之后往里逐字追加
    const box = document.getElementById('chatMessages');
    const aiDiv = document.createElement('div');
    aiDiv.className = 'chat-msg ai';
    box.appendChild(aiDiv);
    box.scrollTop = box.scrollHeight;

    btn.disabled = true;
    btn.textContent = '思考中…';

    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q})
    })
    .then(resp => {
        if (!resp.body) throw new Error('no stream body');
        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let full = '';
        function read() {
            return reader.read().then(({done, value}) => {
                if (done) {
                    full += decoder.decode();
                    aiDiv.textContent = full;
                    box.scrollTop = box.scrollHeight;
                    btn.disabled = false;
                    btn.textContent = '发送';
                    return;
                }
                full += decoder.decode(value, {stream: true});
                aiDiv.textContent = full;
                box.scrollTop = box.scrollHeight;
                return read();
            });
        }
        return read();
    })
    .catch(() => {
        if (!aiDiv.textContent) aiDiv.textContent = '请求失败，请检查网络或服务状态。';
        btn.disabled = false;
        btn.textContent = '发送';
    });
}

function exportReport(fmt) {
    window.location.href = '/report?format=' + fmt;
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
        busy = arm_busy
    with arm_queue_lock:
        qlen = len(arm_queue)
    return jsonify({
        'busy': busy,
        'queue_length': qlen,
        'last_action': last_action_result,
        'defect_types': last_defect_types,
        'arm_enabled': arm_enabled
    })

@app.route('/queue')
def get_queue():
    """返回当前机械臂任务队列"""
    with arm_queue_lock:
        items = list(arm_queue)
    return jsonify({'queue': items, 'length': len(items)})

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
        '基准线X': f'{CENTER_X}px',
        '基准线Y': f'{CENTER_Y}px',
        '横轴检测范围': f'±{CENTER_X_RANGE}px',
        '纵轴检测范围': f'±{CENTER_Y_RANGE}px',
        '触发像素阈值': EDGE_COUNT_THRESHOLD,
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
    # 5.0: 手动触发不再检查 arm_busy，结果会进入队列
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

# ---------- 检测结果图 / 统计图表数据（只读，供前端展示）----------
@app.route('/recent_images')
def recent_images():
    """返回最近四次检测结果图（base64 data URI，新→旧）"""
    with history_lock:
        items = list(detected_images)
    imgs = []
    for it in reversed(items):
        try:
            b64 = base64.b64encode(it['jpeg']).decode('ascii')
            imgs.append({'ts': it['ts'], 'src': 'data:image/jpeg;base64,' + b64})
        except Exception:
            continue
    return jsonify({'images': imgs})

@app.route('/chart_data')
def chart_data():
    """返回检测统计图表数据：正常/缺陷占比（历史累计）+ 历史与当月缺陷分布"""
    with history_lock:
        records = list(defect_records)
    all_ok, all_ng = _ok_ng(records)
    history_counts = aggregate_defects(records)
    month_counts = aggregate_defects(current_month_records(records))
    return jsonify({
        'ok': all_ok,
        'ng': all_ng,
        'history': history_counts,
        'month': month_counts,
    })

@app.route('/chat', methods=['POST'])
def chat():
    """AI 对话（流式）：基于真实检测数据逐字返回回答（只读，不影响控制）"""
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'answer': '问题不能为空。', 'error': True}), 400
    digest = build_stats_digest()
    resp = Response(stream_ollama(build_system_prompt(digest), question),
                    mimetype='text/plain; charset=utf-8')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp

@app.route('/report')
def report():
    """导出检测报告（HTML 或 Markdown，浏览器下载）"""
    fmt = request.args.get('format', 'html').lower()
    digest = build_stats_digest()
    ts = time.strftime("%Y%m%d_%H%M%S")
    if fmt == 'md':
        content = build_report_md(digest)
        resp = Response(content, mimetype='text/markdown; charset=utf-8')
        resp.headers['Content-Disposition'] = f'attachment; filename="report_{ts}.md"'
        return resp
    content = build_report_html(digest)
    resp = Response(content, mimetype='text/html; charset=utf-8')
    resp.headers['Content-Disposition'] = f'attachment; filename="report_{ts}.html"'
    return resp

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ---------- 串口初始化 ----------
try:
    com = serial.Serial(SUCTION_SERIAL_PORT, SUCTION_SERIAL_BAUD, timeout=1)
    time.sleep(0.1)
    add_log(f"吸盘串口 {SUCTION_SERIAL_PORT} 已打开")
except Exception as e:
    add_log(f"吸盘串口打开失败: {e}", "ERROR")
    exit(1)

# ---------- 报警器初始化 ----------
try:
    alarm_com = serial.Serial(ALARM_SERIAL_PORT, ALARM_SERIAL_BAUD, timeout=0.5)
    time.sleep(0.1)
    add_log(f"报警器串口 {ALARM_SERIAL_PORT} 已打开")
except Exception as e:
    alarm_com = None
    add_log(f"报警器串口打开失败（将跳过报警）: {e}", "WARN")

ALARM_CMD = bytes.fromhex('7E FF 06 03 00 00 01 EF')

def alarm_on():
    """触发一次报警"""
    if alarm_com is None:
        return
    try:
        if not alarm_com.is_open:
            alarm_com.open()
        alarm_com.write(ALARM_CMD)
    except Exception:
        pass

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
p_initial = [90, 60, 45, 15, 90, 90]     #fuwei
p_Z = [90, 33, 26, 15, 90, 90]      # <--- shenxiang lvpian
p_NG_1 = [-10, 60, 45, 15, 90, 90]
p_NG   = [-10, 65, 0, 30, 90, 90]   # <--- OK finil position
p_OK_1 = [180, 60, 45, 15, 90, 90]
p_OK   = [180, 55, 11, 24, 90, 90]  # <--- NG finil position

# ---------- 核心动作函数 ----------
def execute_arm_action(action_type):
    global arm_busy, last_action_result
    with arm_lock:
        arm_busy = True
    add_log(f"===== 开始执行 {action_type} 动作 =====")
    start_time = time.time()

    try:
        time.sleep(3)      # <--- 记得调整
        # 步骤1: 下降到吸取位置
        add_log(f"[1/6] 机械臂下降到吸取位置 p_Z={p_Z}, 速度={ARM_SPEED}ms")
        Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
        add_log(f"      等待 {ARM_SPEED/1000 + 0.3:.1f}s ...")
        time.sleep(ARM_SPEED / 1000 + 0.3)

        # 步骤2: 启动吸盘
        add_log(f"[2/6] 启动吸盘，等待吸力建立 {SUCTION_WAIT_TIME}s ...")
        zhua_qu_xipan()
        time.sleep(SUCTION_WAIT_TIME)


        # 步骤4: 执行 OK / NG 运动
        if action_type == "OK":
            add_log(f"[4a/6] OK-1: p_OK_1={p_OK_1}, 速度={ARM_SPEED}ms")
            Arm.Arm_serial_servo_write6(*p_OK_1, ARM_SPEED)
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
            add_log(f"[4c/6] 蜂鸣器 ON + 报警器 ON, 持续 {BUZZER_DELAY*100}ms")
            Arm.Arm_Buzzer_On(delay=BUZZER_DELAY)
            alarm_on()
            time.sleep(BUZZER_DELAY * 0.1 + 0.2)

        # 步骤5: 释放吸盘
        add_log(f"[5/6] 释放吸盘")
        shi_fang_xipan()
        add_log(f"      等待 1s ...")
        time.sleep(SUCTION_WAIT_TIME)       # <--- 记得调整

        # 步骤6: 复位
        add_log(f"[6/6] 机械臂复位 p_initial={p_initial}, 速度={ARM_SPEED}ms")
        Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)

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

# ---------- 机械臂任务队列 Worker ----------
def arm_worker():
    """独立线程：持续轮询队列，非空时弹出并执行机械臂动作"""
    global arm_busy, arm_worker_running
    add_log("机械臂队列 Worker 已启动")
    while arm_worker_running:
        item = None
        with arm_queue_lock:
            if len(arm_queue) > 0:
                item = arm_queue.popleft()
        if item is not None and arm_enabled:
            add_log(f"从队列取出任务: {item['action']}, 剩余队列长度: {len(arm_queue)}")
            with arm_lock:
                arm_busy = True
            try:
                execute_arm_action(item['action'])
            finally:
                with arm_lock:
                    arm_busy = False
        else:
            time.sleep(0.1)
    add_log("机械臂队列 Worker 已停止")

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
        resp = requests.post(FPGA_URL, files={'image_file': (f'frame_{timestamp}.jpg', image_bytes, 'image/jpeg')}, timeout=FPGA_TIMEOUT)
        fpga_time = (time.time() - fpga_start) * 1000

        if resp.status_code != 200:
            add_log(f"FPGA请求失败: {resp.status_code}", "ERROR")
            return

        result = json.loads(resp.text)
        objects = result.get('objects', [])
        objects_count = len(objects)
        inference_ms = result.get('inference_time_ms', 'N/A')

        # 提取缺陷类型（去重），过滤掉"正常"类别
        defect_types = list(set(obj['class_name'] for obj in objects)) if objects else []
        # 排除 zheng_chang（正常），仅保留真正的缺陷类型用于显示
        real_defects = [d for d in defect_types if d != 'zheng_chang']
        defect_str = ', '.join(real_defects) if real_defects else '无'
        last_defect_types = defect_str

        add_log(f"FPGA返回: 目标数={objects_count}, 缺陷类型=[{defect_str}], 推理={inference_ms}ms, 总耗时={fpga_time:.0f}ms")

        stats['fpga_times'].append(fpga_time)
        if len(stats['fpga_times']) > 10:
            stats['fpga_times'] = stats['fpga_times'][-10:]

        json_name = f"json/result_{timestamp}.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if result.get('output_path'):
            try:
                name_only = os.path.basename(result['output_path'])
                img_resp = requests.get(f"{SERVER_URL}/result_image?path={name_only}", timeout=FPGA_TIMEOUT)
                if img_resp.status_code == 200:
                    img_array = np.frombuffer(img_resp.content, np.uint8)
                    detect_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if detect_img is not None:
                        img_name = f"img_detect/detect_{timestamp}.jpg"
                        cv2.imwrite(img_name, detect_img)
                        with frame_lock:
                            latest_detected_img = detect_img.copy()
                        # 记录最近四次检测结果图（缩略图，供前端展示）
                        try:
                            thumb = detect_img
                            if thumb.shape[1] > 320:
                                tw = 320
                                th_h = int(thumb.shape[0] * tw / thumb.shape[1])
                                thumb = cv2.resize(thumb, (tw, th_h))
                            ok_jpg, jpg_buf = cv2.imencode('.jpg', thumb)
                            if ok_jpg:
                                with history_lock:
                                    detected_images.append({'ts': timestamp, 'jpeg': jpg_buf.tobytes()})
                        except Exception as e:
                            add_log(f"记录检测结果图失败: {e}", "WARN")
                else:
                    add_log(f"获取检测图片失败: {img_resp.status_code}", "WARN")
            except Exception as e:
                add_log(f"检测图片获取失败: {e}", "WARN")

        with frame_lock:
            latest_sent_raw = frame_bgr.copy()

        if FORCE_TEST is not None:
            action = FORCE_TEST
        elif objects_count == 0 or len(real_defects) == 0:
            action = "OK"
        else:
            action = "NG"

        if action == "OK":
            stats['ok_count'] += 1
        else:
            stats['ng_count'] += 1

        # 记录缺陷历史（供统计图表），不影响机械臂控制
        try:
            with history_lock:
                defect_records.append({'ts': timestamp, 'defects': list(real_defects)})
                save_defect_records()
        except Exception as e:
            add_log(f"记录缺陷历史失败: {e}", "WARN")

        # 将任务加入机械臂执行队列（由 arm_worker 线程异步处理）
        if arm_enabled:
            with arm_queue_lock:
                arm_queue.append({
                    'action': action,
                    'timestamp': timestamp,
                    'defect_types': defect_str
                })
            add_log(f"任务已入队: {action} (缺陷: {defect_str}), 队列长度: {len(arm_queue)}")
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

    # 自动启动本地 AI 服务（Ollama），避免每次手动开启；仅启动服务，不涉及控制逻辑
    ensure_ollama_running()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    add_log("Flask Web 服务已启动，端口 5000")

    # 启动机械臂队列 Worker 线程
    arm_thread = threading.Thread(target=arm_worker, daemon=True)
    arm_thread.start()

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

            # 计算中心矩形区域的白色像素个数
            x_start = max(0, CENTER_X - CENTER_X_RANGE)
            x_end   = min(w, CENTER_X + CENTER_X_RANGE)
            y_start = max(0, CENTER_Y - CENTER_Y_RANGE)
            y_end   = min(h, CENTER_Y + CENTER_Y_RANGE)
            center_region = edges[y_start:y_end, x_start:x_end]
            center_edge_count = int(np.count_nonzero(center_region))

            # 在边缘图上画基准线和检测范围矩形
            # 纵向基准线 X = CENTER_X（绿色）
            cv2.line(edge_bgr, (CENTER_X, 0), (CENTER_X, h-1), (0, 255, 0), 1)
            # 横向基准线 Y = CENTER_Y（绿色）
            cv2.line(edge_bgr, (0, CENTER_Y), (w-1, CENTER_Y), (0, 255, 0), 1)
            # 检测范围矩形（蓝色边框）
            cv2.rectangle(edge_bgr, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
            # 显示中心区域边缘点数和阈值
            cv2.putText(edge_bgr, f"ROI={center_edge_count}/{EDGE_COUNT_THRESHOLD}",
                        (CENTER_X+5, CENTER_Y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            with frame_lock:
                latest_frame_raw = raw_bgr.copy()
                latest_frame_edge = edge_bgr.copy()

            # 触发检测（异步线程，不阻塞主循环 — 5.0: 不受 arm_busy 限制）
            now = time.time()
            if center_edge_count >= EDGE_COUNT_THRESHOLD and (now - last_send_time) >= SEND_COOLDOWN:
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
        arm_worker_running = False
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        com.close()
        if alarm_com:
            try:
                alarm_com.close()
            except:
                pass
        add_log("系统已关闭")

if __name__ == "__main__":
    main()
