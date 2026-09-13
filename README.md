# 集创赛项目归档 — 基于 Cyclone V SoC FPGA 的铝片缺陷智能检测与分拣系统

本项目为 **全国大学生集成电路创新创业大赛（集创赛）** 参赛作品的全过程归档，包含初赛、分赛区决赛、国赛（总决赛）三个阶段的代码、模型、镜像与文档。

项目整体方案：使用 **PaddleDetection** 训练 SSD 目标检测模型，识别铝片表面三类缺陷（**脏污 / 折痕 / 针孔**，基于 ISDD 数据集并自建标注数据），经 **Paddle Lite** 量化转换后部署到 **Intel Cyclone V SoC FPGA**（ARM + 板载神经网络加速器 NNA）上进行端侧推理；上位机通过 HTTP 服务向 FPGA 请求推理结果，并结合海康工业相机、Canny 边缘检测 / 圆拟合定位，驱动六自由度机械臂（I2C 舵机 + 吸盘）自动分拣 OK / NG 品。

## 目录结构

```
集创赛/
├── 初赛/                    # 初赛阶段
│   ├── 架构图.html          # 系统架构图（SoC 地址映射 / AXI 桥接 / 数据通路）
│   ├── img/                 # 标注数据集（labelme 标注，含脏污/折痕/针孔三类，VOC/JSON 格式）
│   ├── model&output/        # PaddleDetection 训练权重与量化后的 .nb 部署模型
│   ├── SDK/                 # 数据集处理脚本、Paddle Lite 编译命令、辅助工具（json/xml 转换等）
│   └── ssd_detection_demo/  # ARM 端 SSD 检测 C++ Demo（Paddle Lite + OpenCV，CMake 构建）
│
├── 分赛区决赛/              # 分赛区决赛阶段（现场赛板卡环境）
│   ├── C5TB/                # C5TB 板（Cyclone V SoC）FAT32 启动分区：设备树、.rbf、u-boot、zImage 等
│   ├── DE10/                # DE10-Nano 相关：SD 卡 FAT32 分区文件、官方 SystemCD 资料、Linux 内核镜像（已忽略）
│   └── Onsite/              # 现场赛代码：cyclone（FPGA 端）与 pi（上位机端）各一份快照
│
└── 国赛/                    # 国赛（总决赛）阶段
    ├── 总决赛技术文档.pdf    # 总决赛技术文档
    ├── cyclone/             # Cyclone V SoC FPGA 端部署
    │   ├── Arm32本地编译环境/ # Paddle Lite / OpenCV 交叉编译库与头文件
    │   ├── paddle_frame_local/  # 本地推理模式：加载 model.nb + libvnna.so（NPU），cmadrv.ko 驱动
    │   └── paddle_frame_net_2.0/# HTTP 推理服务模式：ssd_http_server + systemd 自启脚本
    └── pi/                  # 树莓派（ARM 上位机）端
        ├── result/          # 完整集成系统：相机采集 → 缺陷检测 → 机械臂分拣（含 Flask 监控界面）
        └── Submodule/       # 分模块：相机、Canny/圆拟合、图像传输、机械臂控制等
```

## 系统架构

```
海康工业相机 ──► 上位机（树莓派/ARM）
                  │  OpenCV（Canny 边缘检测、形态学、圆拟合）判断来料位置
                  │  触发条件（目标越过中线）后 JPEG 编码
                  ▼
        HTTP POST ──► FPGA 推理服务（Cyclone V SoC, ssd_http_server:9000）
                        │  Paddle Lite 加载量化 SSD 模型（model.nb）
                        │  NPU 加速推理（libvnna.so + cmadrv.ko 驱动）
                        ▼
              JSON 返回（缺陷类别/数量、标注图、推理耗时）
                  │
                  ▼
        机械臂分拣：I2C 六自由度舵机 + 串口吸盘，按 OK/NG 分箱
        （可选 Flask Web 界面远程监控 MJPEG 画面与结果）
```

## 技术栈

| 类别 | 内容 |
|------|------|
| 硬件平台 | Intel Cyclone V SoC FPGA（C5TB / DE10-Nano）、树莓派（ARM Linux）、海康工业相机、六自由度机械臂 + 吸盘 |
| 模型训练 | PaddleDetection（SSD）、数据增强与量化（paddleslim），输出 Paddle Lite `.nb` 模型 |
| 数据集 | ISDD 铝片缺陷数据集 + 自建标注（脏污 / 折痕 / 针孔三类，labelme → VOC） |
| 端侧推理 | Paddle Lite（C++ API）、OpenCV 3.1.0（ARM 交叉编译）、NPU 加速（vnna / cma 驱动） |
| 上位机 | Python 3（OpenCV、numpy、requests、pyserial、smbus、Flask） |
| 板级软件 | SoC FPGA Linux（zImage + 设备树 + u-boot + .rbf）、内核模块动态加载 |

## 快速上手

### 1. FPGA 端（Cyclone V SoC）

```bash
# 在板端（例如 /opt/paddle_frame_net）
insmod cmadrv.ko
./ssd_http_server model.nb 9000   # 启动 HTTP 推理服务，监听 9000 端口
```

### 2. 上位机（树莓派 / ARM Linux）

```bash
# 依赖：numpy、opencv-python、requests、pyserial、smbus、flask
# 相机 SDK（海康 MVS）解压至 ./sdk/

python3 result/result.py        # 完整系统：相机 + 自动触发 + FPGA 推理 + 机械臂分拣
python3 result/1.py             # 带 Flask Web 监控界面（端口 5000）
```

### 3. 模型训练（详见 `初赛/SDK/` 内的命令记录）

基于 PaddleDetection release/2.2 训练 SSD，经 paddleslim 量化后导出 `.nb` 部署模型。

## 说明

- `分赛区决赛/DE10/` 为官方资料与系统镜像（体积较大），已加入 `.gitignore`，不上传仓库。
- 部分二进制产物（模型权重、静态库、系统镜像）体积较大，克隆前请留意网络与磁盘占用。
- 本仓库仅作参赛过程归档与学习交流之用。
