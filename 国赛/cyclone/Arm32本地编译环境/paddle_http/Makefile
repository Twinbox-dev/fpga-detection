# ============================================================
# Makefile for ssd_http_server
# 合并 plite-test.cpp + ssd_detection.cpp 的功能
# 替代 ssd_detection_6.cpp + ssd_detection_6.py
# 适配 armv7hf 原生环境（Ubuntu 16.04, gcc 5.4.0）
# ============================================================

# --- 编译器 ---
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++11 -mfpu=neon -march=armv7-a -pthread -fopenmp

# --- Paddle Lite 路径（相对于本 Makefile 所在目录） ---
PADDLE_LITE_DIR ?= ../Paddlelite
PADDLE_INC     ?= $(PADDLE_LITE_DIR)/include
PADDLE_LIB     ?= $(PADDLE_LITE_DIR)/lib

# --- OpenCV 路径 ---
OPENCV_DIR ?= ../ocv3.1.0
OPENCV_INC ?= $(OPENCV_DIR)/include
OPENCV_LIB ?= $(OPENCV_DIR)/lib

# --- httplib.h 所在目录（当前目录） ---
HTTPLIB_INC ?= .

# --- 目标 ---
TARGET  := ssd_http_server
SOURCES := ssd_http_server.cpp
OBJECTS := $(SOURCES:.cpp=.o)

# --- 头文件搜索路径 ---
INCLUDES := -I$(PADDLE_INC) \
            -I$(OPENCV_INC) \
            -I$(HTTPLIB_INC)

# --- 库路径 ---
LDFLAGS := -L$(PADDLE_LIB) \
           -L$(OPENCV_LIB)

# --- 链接库 ---
# 注意：Paddle Lite 库名 paddle_full_api_shared，OpenCV 库名根据实际调整
LIBS := -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lpaddle_full_api_shared \
        -lvnna \
        -ldl -lm -lpthread

# ====================== 编译规则 ======================

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo ""
	@echo "✓ 编译成功: $(TARGET)"
	@echo "  运行: ./$(TARGET) [模型路径] [端口]"
	@echo "  示例: ./$(TARGET) ../ssd_detection_src/ssd_mobilenet_v1_opt.nb 9000"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	@echo "✓ 清理完成"

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	@echo "✓ 已安装到 /usr/local/bin/"

# ====================== 使用说明 ======================
help:
	@echo "============================================"
	@echo " SSD缺陷检测 HTTP 服务器"
	@echo "============================================"
	@echo ""
	@echo "用法:"
	@echo "  make            # 编译"
	@echo "  make clean      # 清理"
	@echo "  make install    # 安装到 /usr/local/bin"
	@echo "  make help       # 显示帮助"
	@echo ""
	@echo "运行:"
	@echo "  ./ssd_http_server [模型路径] [端口]"
	@echo ""
	@echo "API接口:"
	@echo "  GET  /hello             健康检查"
	@echo "  POST /upload            上传图片并检测"
	@echo "  GET  /detect?path=xxx   检测本地图片"
	@echo "  GET  /result_image?path=xxx  获取结果图"
	@echo "  GET  /stats             服务器统计"
	@echo "  GET  /log               查看日志"

.PHONY: all clean install help