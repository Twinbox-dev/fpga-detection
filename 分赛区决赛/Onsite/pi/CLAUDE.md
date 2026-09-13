# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotic arm sorting system for the 集创赛 (Integrated Circuit Innovation Competition). The system captures real-time video from a Hikvision industrial camera, uses Canny edge detection to detect object presence/position, sends frames to an FPGA inference server for defect classification, and controls a 6-DOF robotic arm with a suction gripper to sort objects into OK/NG bins.

## Hardware Prerequisites

All scripts assume these are connected to a Linux ARM board (e.g., Raspberry Pi):
- **Hikvision industrial camera** (GigE or USB) — requires the MVS SDK `.so` file present in `./sdk/`
- **6-DOF robotic arm** at I2C address `0x15` (via `smbus`)
- **Suction cup gripper** on serial port `/dev/XIPAN` at 9600 baud
- **FPGA inference server** at `http://172.16.68.110:8080/predict` (HTTP POST, accepts `image_file` multipart, returns JSON with `objects`, `detect_image` base64, `inference_time_ms`)

## Running Scripts

All scripts are standalone Python 3 entry points. Run from the directory containing the script (each script uses `sys.path.append("./sdk")` to find the camera SDK).

```bash
# Camera-only (no detection)
python3 camera/simple_frame.py

# Camera + Canny edge with left/right edge average X line
python3 circle_fitting/circle_fitting.py

# Camera + Canny + morphological close + circle fitting
python3 circle_fitting/circle_fitting_pro.py

# Camera + Canny + auto FPGA trigger (triggers when avg_x > half image width)
python3 GrabImage/detect.py

# Camera + manual FPGA trigger (press space to send frame)
python3 camera/stream_fpga.py

# Camera + non-blocking FPGA trigger (threaded)
python3 camera/multithreading.py

# Arm-only pick-and-place test (OK or NG)
python3 arm/Allarm.py          # includes suction
python3 arm/arm.py             # arm motion only, no suction

# Single servo test
python3 arm/simple_arm.py

# Suction-only test
python3 arm/xipan.py

# Full integrated system (camera + Canny + auto FPGA + arm sorting)
python3 result/0_pure_backend.py

# Full system with Flask web monitoring UI on port 5000
python3 result/1.py

# Test FPGA server with a local image file
python3 transfer/test.py <image_path>
```

## Dependencies

No `requirements.txt` exists. Core dependencies (install manually):
- `numpy`, `opencv-python` (cv2)
- `requests`
- `pyserial`
- `smbus` (or `smbus2`) — I2C for arm control
- `flask` — only for `result/1.py`

## Architecture

### Layered Evolution (simplest → most integrated)

The codebase evolved incrementally. Each module builds on the previous:

1. **Camera layer** (`camera/`, `circle_fitting/`): Raw frame capture via Hikvision MVS SDK → OpenCV processing (Canny, morphological ops, contour analysis, circle fitting)
2. **FPGA inference layer** (`GrabImage/`, `transfer/`): HTTP POST of JPEG frames to FPGA server → parse JSON result (object count, base64-encoded detection image)
3. **Arm control layer** (`arm/`): I2C servo commands (`Arm_Lib.py` driver) + serial suction control
4. **Integration layer** (`result/`): Combines all three layers with auto-trigger logic (avg_x > half_width → capture → FPGA → OK/NG → arm sort)
5. **Web UI layer** (`result/1.py`): Flask MJPEG streaming + static snapshot endpoints for remote monitoring

### Camera SDK (duplicated across modules)

`MvCameraControl_class.py` is the Python ctypes wrapper for Hikvision's C++ MVS SDK. It's duplicated identically in `camera/sdk/`, `circle_fitting/sdk/`, and `GrabImage/sdk/`. Each directory also has supporting files:
- `CameraParams_const.py`, `CameraParams_header.py` — camera parameter constants
- `MvErrorDefine_const.py` — error code definitions
- `PixelType_const.py`, `PixelType_header.py` — pixel format definitions
- `libMvCameraControl.so` — the native shared library

The standard camera initialization pattern (used by every camera script):
1. `MV_CC_EnumDevices()` → enumerate cameras
2. `MV_CC_CreateHandle()` + `MV_CC_OpenDevice()` → connect
3. `MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)` → continuous mode
4. `MV_CC_GetIntValue("PayloadSize")` → get frame buffer size
5. `MV_CC_StartGrabbing()` → begin acquisition
6. Loop: `MV_CC_GetOneFrameTimeout()` → raw byte buffer → `np.asarray().reshape(H, W)`
7. Cleanup: `MV_CC_StopGrabbing()` → `MV_CC_CloseDevice()` → `MV_CC_DestroyHandle()`

The camera outputs single-channel (grayscale) frames; scripts convert to BGR for OpenCV display or for sending to FPGA.

### Detection Trigger Logic

The `GrabImage/detect.py` and `result/` scripts use a Canny edge-based trigger:
1. Compute Canny edges on downscaled frame
2. Find all edge pixel coordinates
3. Sort by X, take leftmost N and rightmost N points
4. If `avg_x > image_width / 2` → object is present on right side → trigger FPGA capture
5. Cooldown period (`SEND_COOLDOWN`, default 5s) prevents repeated triggers

### Arm Control (`arm/Arm_Lib.py`)

`Arm_Device` class communicates with a 6-DOF bus servo controller via I2C (address `0x15`). Key details:
- Servos 1-6; servo 5 has 270° range, others 180°
- Servos 2, 3, 4 are mechanically reversed (angle = 180 - input)
- Position values map: 0-180° → 900-3100 pulse width (servo 5: 0-270° → 380-3700)
- `Arm_serial_servo_write6()` — set all 6 servos simultaneously with a duration
- `Arm_serial_servo_write6_array()` — set from a list
- Arm positions are defined as 6-element lists `[s1, s2, s3, s4, s5, s6]`

### Suction Control

Binary control via serial (`/dev/XIPAN`, 9600 baud):
- Grab: `A0 01 01 A2`
- Release: `A0 01 00 A1`

### FPGA Server API

```
POST http://172.16.68.110:8080/predict
Content-Type: multipart/form-data
Field: image_file (JPEG bytes)

Response JSON:
{
  "image_path": "...",
  "objects": [...],           // detected defect objects
  "inference_time_ms": 123,
  "detect_image": "<base64>", // annotated image
  "detect_image_size": 12345,
  "output_path": "..."
}
```

Decision rule: `len(objects) == 0` → OK (no defects); otherwise → NG.

### OK/NG Sort Positions

Predefined arm joint angles for the two sorting outcomes:
- **OK path**: `[90,90,90,0,90,90]` → `[-10,90,90,0,90,90]` → `[-10,65,0,30,90,90]`
- **NG path**: `[90,90,90,0,90,90]` → `[180,55,11,24,90,90]` + buzzer alert
- **Pick position** (shared): `[90,38,26,22,90,90]`
- **Home position**: `[90,90,90,0,90,90]`

### Defect Classes

The FPGA model detects 4 defect types (transfer/ directory contains test images for each):
- **ca_shang** (擦伤) — Scratches
- **zang_wu** (脏污) — Contamination / dirt
- **zhe_zhou** (褶皱) — Wrinkles
- **zhen_kong** (真空) — Holes / vacuum defects
- **zheng_chang** (正常) — Normal (no defect)

### Flask API Routes (result/2.py through result/4.2.py)

Later integration versions expose a richer REST API on port 5000:

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Monitoring dashboard (inline HTML, dark theme) |
| `/video_feed_raw` | GET | MJPEG stream of raw camera feed |
| `/video_feed_edge` | GET | MJPEG stream of Canny edge detection |
| `/sent_image` | GET | Most recent image sent to FPGA (static JPEG) |
| `/detected_image` | GET | Most recent FPGA detection result (static JPEG) |
| `/status` | GET | JSON: `{busy, last_action, defect_types, arm_enabled}` |
| `/stats` | GET | JSON: FPS, OK/NG counts, uptime, avg FPGA response time |
| `/health` | GET | Health check |
| `/config` | GET | Current configuration parameters |
| `/logs` | GET | Recent operation logs (in-memory deque, max 50 entries) |
| `/export_logs` | POST | Write logs to `log/` directory on disk |
| `/reset` | POST | Manual robotic arm reset to home position |
| `/trigger` | POST | Manual trigger for one detection cycle |
| `/servo/<id>` | POST | Control individual servo (1-5) via `?angle=` |
| `/suction/<action>` | POST | Suction cup on/off |
| `/arm_mode/<mode>` | POST | Toggle `detect_only` / `detect_grab` |
| `/frame_rate` | POST | Adjust camera frame rate via slider |

### result/ Version Evolution

The `result/` directory tracks the evolution from prototype to polished system:

| File | Key Addition |
|------|-------------|
| `0_pure_backend.py` | Baseline: camera + Canny + auto-FPGA + arm (blocking, no UI) |
| `1.py` | Flask web UI with MJPEG streaming (basic HTML table layout) |
| `2.py` | Dark-themed CSS, card layout, status bar, reset button |
| `3.py` | Async FPGA requests, 5s timeout, health/stats/logs/config APIs, retry logic, log export |
| `4.py` | Async arm actions (non-blocking video), manual servo/suction web controls, avg_x overlay on edge feed |
| `4.1.py` | Configurable camera resolution, arm enable/disable switch, contrast slider |
| `4.2.py` | Arm mode toggle (detect_only/detect_grab), defect type display, iOS-style toggle UI, frame rate slider |

## File Purpose Summary

| File | Purpose |
|------|---------|
| `arm/Arm_Lib.py` | I2C driver for 6-DOF robotic arm |
| `arm/Allarm.py` | End-to-end arm test with suction (OK/NG) |
| `arm/arm.py` | Arm-only pick-and-place motion test |
| `arm/simple_arm.py` | Single servo test |
| `arm/xipan.py` | Suction cup serial test |
| `camera/simple_frame.py` | Minimal camera stream |
| `camera/stream_fpga.py` | Camera + FPGA (manual spacebar trigger) |
| `camera/multithreading.py` | Camera + FPGA (non-blocking threaded) |
| `circle_fitting/canny.py` | Camera + Canny edge display |
| `circle_fitting/circle_fitting.py` | Camera + Canny + edge avg X line |
| `circle_fitting/circle_fitting_2.py` | Same with min-points filtering |
| `circle_fitting/circle_fitting_pro.py` | Camera + Canny + morphology + circle fitting |
| `GrabImage/detect.py` | Camera + Canny + auto FPGA trigger |
| `transfer/test.py` | Send local image to FPGA server |
| `result/0_pure_backend.py` | Full integrated system (no web UI) |
| `result/1.py` | Full system + Flask web monitoring |
| `result/2.py` through `result/4.2.py` | Incrementally enhanced versions (see version evolution above) |
