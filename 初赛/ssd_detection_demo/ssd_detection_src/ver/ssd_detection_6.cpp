// =================================================================
// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =================================================================

// ==================== 头文件包含 ====================
// C++标准库
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <dirent.h>

// C标准库/系统头文件
#include <stdio.h>
#include <sys/times.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>

// Linux特定头文件
#include <linux/fb.h>
#include <linux/videodev2.h>

// ARM NEON指令集
#include <arm_neon.h>

// OpenCV计算机视觉库
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

// PaddlePaddle Lite推理引擎
#include "paddle_api.h"
#include "intelfpga.h"


// ==================== 命名空间声明 ====================
using namespace paddle::lite_api;
using namespace std;
using namespace cv;

// ==================== ANSI颜色代码定义 ====================
#ifdef _WIN32
    // Windows平台的颜色支持
    #include <windows.h>
    #define COLOR_RESET ""
    #define COLOR_RED ""
    #define COLOR_GREEN ""
    #define COLOR_YELLOW ""
    #define COLOR_BLUE ""
    #define COLOR_MAGENTA ""
    #define COLOR_CYAN ""
    #define COLOR_WHITE ""
    #define COLOR_BOLD ""
#else
    // Linux/Mac平台的颜色支持
    #define COLOR_RESET   "\033[0m"
    #define COLOR_RED     "\033[31m"
    #define COLOR_GREEN   "\033[32m"
    #define COLOR_YELLOW  "\033[33m"
    #define COLOR_BLUE    "\033[34m"
    #define COLOR_MAGENTA "\033[35m"
    #define COLOR_CYAN    "\033[36m"
    #define COLOR_WHITE   "\033[37m"
    #define COLOR_BOLD    "\033[1m"
    #define COLOR_GRAY    "\033[90m"
    #define COLOR_BLUE_BG "\033[44m"
    #define COLOR_GREEN_BG "\033[42m"
    #define COLOR_RED_BG "\033[41m"
#endif

// ==================== 全局常量定义 ====================
const int CPU_THREAD_NUM = 4;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_FULL;
std::stringstream LOG_MEM;

// 全局变量，记录程序启动时间
static auto program_start_time = std::chrono::steady_clock::now();
// ==================== 硬编码配置信息 ====================
// 硬编码的标签类型
const std::vector<std::string> CLASS_NAMES = {
    "ca_shang",   // 0: 擦伤
    "zang_wu",    // 1: 脏污
    "zhe_zhou",   // 2: 褶皱
    "zhen_kong",  // 3: 针孔
    "zheng_chang" // 4: 正常
};

// 从config.txt提取的配置
const std::string DEFAULT_MODEL_PATH = "ssd_mobilenet_v1_opt.nb";  // 默认模型路径
const int DEFAULT_NUM_THREADS = 2;
const std::string DEFAULT_PRECISION = "int8/float3";
const bool DEFAULT_ENABLE_BENCHMARK = false;
const std::string DEFAULT_ARCH = "mobilenetv1";
const std::vector<int> DEFAULT_IMAGE_SHAPE = {3, 300, 300};
const std::vector<int> DEFAULT_RESIZE = {300, 300};
const bool DEFAULT_KEEP_RATIO = false;
const std::vector<float> DEFAULT_MEAN = {127.5f, 127.5f, 127.5f};
const std::vector<float> DEFAULT_STD = {127.502231f, 127.502231f, 127.502231f};
const int DEFAULT_PAD_STRIDE = 0;

// ==================== 全局变量声明 ====================
std::shared_ptr<PaddlePredictor> predictor;
bool g_verbose_mode = false;
bool g_benchmark_mode = false;
bool g_fpga_debug = false;  // FPGA调试信息控制
std::string g_output_dir = ".";  // 默认输出目录
FILE* g_stderr_backup = nullptr;  // 用于保存原始stderr
int g_stdout_backup_fd = -1;     // 用于保存原始stdout的文件描述符
int g_log_fd = -1;               // 日志文件描述符

// ==================== 数据结构定义 ====================
struct Object {
    cv::Rect rec;
    int class_id;
    float prob;
};

struct ImageBlob {
    std::vector<float> im_shape_;
    const float* im_data_;
    std::vector<float> scale_factor_;
    std::vector<float> mean_;
    std::vector<float> scale_;
};

struct DetectionResult {
    std::string image_path;
    std::vector<Object> objects;
    std::string output_path;
    double inference_time;
    
    int total_objects() const { return objects.size(); }
    float max_confidence() const {
        if (objects.empty()) return 0.0f;
        float max_conf = 0.0f;
        for (const auto& obj : objects) {
            if (obj.prob > max_conf) max_conf = obj.prob;
        }
        return max_conf;
    }
};

// ==================== FPGA调试信息控制 ====================

/**
 * @brief 重定向stderr以隐藏FPGA调试信息
 */
void suppress_fpga_debug() {
    if (!g_fpga_debug) {
        // 保存原始stderr
        g_stderr_backup = fdopen(dup(fileno(stderr)), "w");
        // 重定向stderr到/dev/null
        freopen("/dev/null", "w", stderr);
    }
}

// ==================== stdout重定向函数 ====================

/**
 * @brief 重定向stdout到日志文件
 * @param log_file 日志文件路径
 * @return 0=成功, -1=失败
 */
int redirect_stdout_to_file(const char* log_file) {
    if (log_file == nullptr || strlen(log_file) == 0) {
        return -1;
    }

    // 保存原始stdout和stderr文件描述符
    g_stdout_backup_fd = dup(STDOUT_FILENO);
    g_stderr_backup = fdopen(dup(STDERR_FILENO), "w");

    // 打开日志文件
    g_log_fd = open(log_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_log_fd < 0) {
        std::cerr << COLOR_RED << "[警告] 无法打开日志文件: " << log_file << COLOR_RESET << std::endl;
        if (g_stdout_backup_fd >= 0) { close(g_stdout_backup_fd); g_stdout_backup_fd = -1; }
        return -1;
    }

    // 将stdout和stderr都重定向到日志文件
    fflush(stdout);
    fflush(stderr);
    dup2(g_log_fd, STDOUT_FILENO);
    dup2(g_log_fd, STDERR_FILENO);
    close(g_log_fd);
    g_log_fd = -1;

    // 此时stdout/stderr已重定向，后续输出都会写入日志文件
    return 0;
}

/**
 * @brief 恢复stdout和stderr到原始终端
 */
void restore_io() {
    fflush(stdout);
    fflush(stderr);
    if (g_stdout_backup_fd >= 0) {
        dup2(g_stdout_backup_fd, STDOUT_FILENO);
        close(g_stdout_backup_fd);
        g_stdout_backup_fd = -1;
    }
    if (g_stderr_backup) {
        dup2(fileno(g_stderr_backup), STDERR_FILENO);
        fclose(g_stderr_backup);
        g_stderr_backup = nullptr;
    }
}

// ==================== 工具函数实现 ====================
bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

bool is_directory(const std::string& path) {
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0) {
        return false;
    }
    return S_ISDIR(path_stat.st_mode);
}

bool is_image_file(const std::string& filename) {
    std::string ext = "";
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos && dot_pos < filename.length() - 1) {
        ext = filename.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::vector<std::string> supported_exts = {".jpg", ".jpeg", ".png", ".bmp"};
        for (const auto& supported_ext : supported_exts) {
            if (ext == supported_ext) {
                return true;
            }
        }
    }
    return false;
}

std::string get_filename(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(last_slash + 1);
    }
    return full_path;
}

std::string get_directory(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(0, last_slash);
    }
    return ".";
}

/**
 * @brief 获取不带扩展名的文件名
 */
std::string get_stem(const std::string& filename) {
    std::string fname = get_filename(filename);
    size_t last_dot = fname.find_last_of(".");
    if (last_dot != std::string::npos) {
        return fname.substr(0, last_dot);
    }
    return fname;
}

/**
 * @brief 获取当前时间戳（用于日志）
 */
std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

/**
 * @brief 获取当前日期（用于日志）
 */
std::string get_current_date() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d");
    return ss.str();
}

// ==================== 优化的结果输出函数 ====================

/**
 * @brief 优化的单张图片检测结果输出
 * 显示更详细的信息，包括类别名称和中文解释
 */
void print_concise_result(const DetectionResult& result) {
    std::cout << "\n" << COLOR_CYAN << COLOR_BOLD << "╔══════════════════════════════════════════════╗" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << COLOR_BOLD << "║             图片检测结果详情                 ║" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << COLOR_BOLD << "╚══════════════════════════════════════════════╝" << COLOR_RESET << std::endl;
    
    std::cout << COLOR_YELLOW << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_BOLD << COLOR_WHITE << " 输入信息:" << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   图片名称: " << COLOR_BOLD << COLOR_CYAN << get_filename(result.image_path) << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   运行时间: " << get_current_time() << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   检测结果: ";
    if (result.total_objects() > 0) {
        std::cout << COLOR_BOLD << COLOR_GREEN << result.total_objects() << COLOR_RESET << " 个目标";
    } else {
        std::cout << COLOR_BOLD << COLOR_RED << "0" << COLOR_RESET << " 个目标";
    }
    std::cout << std::endl;
    std::cout << COLOR_YELLOW << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    
    if (result.total_objects() > 0) {
        std::cout << "\n" << COLOR_GREEN << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
        std::cout << COLOR_GREEN << "│" << COLOR_BOLD << COLOR_WHITE << " 检测到的目标详情:" << COLOR_RESET << COLOR_GREEN << "                            │" << COLOR_RESET << std::endl;
        std::cout << COLOR_GREEN << "├──────────────────────────────────────────────┤" << COLOR_RESET << std::endl;
        std::cout << COLOR_GREEN << "│" << COLOR_RESET 
                  << std::setw(4) << "序号" 
                  << std::setw(12) << "类别ID"
                  << std::setw(12) << "类别"
                  << std::setw(10) << "置信度"
                  << std::setw(16) << "位置(x,y,w,h)"
                  << COLOR_GREEN << " │" << COLOR_RESET << std::endl;
        std::cout << COLOR_GREEN << "├──────────────────────────────────────────────┤" << COLOR_RESET << std::endl;
        
        for (size_t i = 0; i < result.objects.size(); ++i) {
            const auto& obj = result.objects[i];
            std::string class_name = obj.class_id >= 0 && obj.class_id < static_cast<int>(CLASS_NAMES.size()) 
                                   ? CLASS_NAMES[obj.class_id] 
                                   : "未知";
            
            // 不同类别使用不同颜色
            std::string class_color = COLOR_WHITE;
            std::string chinese_desc = "";
            switch(obj.class_id) {
                case 0: 
                    class_color = COLOR_BLUE;
                    chinese_desc = "(擦伤)"; 
                    break;
                case 1: 
                    class_color = COLOR_RED;
                    chinese_desc = "(脏污)"; 
                    break;
                case 2: 
                    class_color = COLOR_GREEN;
                    chinese_desc = "(褶皱)"; 
                    break;
                case 3: 
                    class_color = COLOR_CYAN;
                    chinese_desc = "(针孔)"; 
                    break;
                case 4: 
                    class_color = COLOR_MAGENTA;
                    chinese_desc = "(正常)"; 
                    break;
                default: 
                    chinese_desc = "(未知)";
            }
            
            // 置信度颜色
            std::string conf_color = COLOR_WHITE;
            if (obj.prob > 0.8) conf_color = COLOR_GREEN;
            else if (obj.prob > 0.6) conf_color = COLOR_YELLOW;
            else conf_color = COLOR_RED;
            
            std::cout << COLOR_GREEN << "│" << COLOR_RESET 
                      << std::setw(4) << i + 1
                      << std::setw(4) << obj.class_id
                      << class_color <<  class_name << COLOR_RESET
                      << conf_color << std::setw(12) << std::fixed << std::setprecision(4) << obj.prob << COLOR_RESET
                      << std::setw(3) << "(" 
                      << std::setw(3) << obj.rec.x << ","
                      << std::setw(3) << obj.rec.y << ","
                      << std::setw(3) << obj.rec.width << ","
                      << std::setw(3) << obj.rec.height << ")"
                      << class_color << std::setw(8) << chinese_desc << COLOR_RESET
                      << COLOR_GREEN << " │" << COLOR_RESET << std::endl;
        }
        std::cout << COLOR_GREEN << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    } else {
        std::cout << "\n" << COLOR_RED << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
        std::cout << COLOR_RED << "│" << COLOR_BOLD << COLOR_WHITE << " 未检测到目标" << COLOR_RESET << COLOR_RED << "                                 │" << COLOR_RESET << std::endl;
        std::cout << COLOR_RED << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    }
    
    if (g_benchmark_mode) {
        std::cout << "\n" << COLOR_CYAN << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_BOLD << COLOR_WHITE << " 性能信息:" << COLOR_RESET << COLOR_CYAN << "                                     │" << COLOR_RESET << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   推理耗时: ";
        
        std::string time_color = COLOR_WHITE;
        if (result.inference_time < 50) time_color = COLOR_GREEN;
        else if (result.inference_time < 100) time_color = COLOR_YELLOW;
        else time_color = COLOR_RED;
        
        std::cout << time_color << std::fixed << std::setprecision(2) 
                  << result.inference_time << COLOR_RESET << " ms" << std::endl;
        if (result.total_objects() > 0) {
            std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   平均目标检测: " << time_color << std::fixed << std::setprecision(2)
                      << (result.inference_time / result.total_objects())
                      << COLOR_RESET << " ms/目标" << std::endl;
        }
        std::cout << COLOR_CYAN << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    }
    
    std::cout << "\n" << COLOR_BLUE << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
    std::cout << COLOR_BLUE << "│" << COLOR_BOLD << COLOR_WHITE << " 输出信息:" << COLOR_RESET << COLOR_BLUE << "                                    │" << COLOR_RESET << std::endl;
    std::cout << COLOR_BLUE << "│" << COLOR_RESET << "   输出图片: " << COLOR_BOLD << COLOR_GREEN << get_filename(result.output_path) << COLOR_RESET << std::endl;
    std::cout << COLOR_BLUE << "│" << COLOR_RESET << "   保存路径: " << result.output_path << std::endl;
    std::cout << COLOR_BLUE << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    
    std::cout << "\n" << COLOR_CYAN << COLOR_BOLD << "══════════════════════════════════════════════" << COLOR_RESET << std::endl;
}

/**
 * @brief 优化的批量处理摘要输出
 */
std::string get_program_uptime_str();
void print_batch_summary(const std::vector<DetectionResult>& results) {
    int total_images = results.size();
    int total_objects = 0;
    int images_with_objects = 0;
    double total_inference_time = 0.0;
    
    // 统计各类别的检测数量
    std::map<int, int> class_count;
    std::map<int, double> class_confidences;
    
    for (const auto& result : results) {
        total_objects += result.total_objects();
        total_inference_time += result.inference_time;
        if (result.total_objects() > 0) {
            images_with_objects++;
        }
        
        for (const auto& obj : result.objects) {
            class_count[obj.class_id]++;
            class_confidences[obj.class_id] += obj.prob;
        }
    }
    
    std::cout << "\n" << COLOR_CYAN << COLOR_BOLD << "╔══════════════════════════════════════════════╗" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << COLOR_BOLD << "║             批量处理统计报告               ║" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << COLOR_BOLD << "╚══════════════════════════════════════════════╝" << COLOR_RESET << std::endl;
    
    std::cout << "\n" << COLOR_YELLOW << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_BOLD << COLOR_WHITE << " 处理概览:" << COLOR_RESET << COLOR_YELLOW << "                                    │" << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   处理时间: " << get_program_uptime_str() << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   处理图片总数: " << COLOR_BOLD << COLOR_CYAN << total_images << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   检测到目标的图片数: " << COLOR_BOLD;
    if (images_with_objects > 0) {
        std::cout << COLOR_GREEN << images_with_objects << COLOR_RESET;
    } else {
        std::cout << COLOR_RED << images_with_objects << COLOR_RESET;
    }
    std::cout << " (" << std::fixed << std::setprecision(1) 
              << (total_images > 0 ? (images_with_objects * 100.0 / total_images) : 0) << "%)" << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   检测到的目标总数: " << COLOR_BOLD << COLOR_CYAN << total_objects << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << "   平均每张图片检测数: " << COLOR_BOLD;
    if (total_objects > 0) {
        std::cout << COLOR_GREEN << std::fixed << std::setprecision(2)
                  << (total_images > 0 ? static_cast<double>(total_objects) / total_images : 0) << COLOR_RESET;
    } else {
        std::cout << COLOR_RED << "0.00" << COLOR_RESET;
    }
    std::cout << std::endl;
    std::cout << COLOR_YELLOW << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    
    if (total_objects > 0) {
        std::cout << "\n" << COLOR_MAGENTA << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
        std::cout << COLOR_MAGENTA << "│" << COLOR_BOLD << COLOR_WHITE << " 类别分布统计:" << COLOR_RESET << COLOR_MAGENTA << "                                │" << COLOR_RESET << std::endl;
        std::cout << COLOR_MAGENTA << "├──────────────────────────────────────────────┤" << COLOR_RESET << std::endl;
        std::cout << COLOR_MAGENTA << "│" << COLOR_RESET 
                  << std::setw(12) << "类别名称"
                  << std::setw(8) << "数量"
                  << std::setw(10) << "占比"
                  << std::setw(12) << "平均置信度"
                  << COLOR_MAGENTA << "            │" << COLOR_RESET << std::endl;
        std::cout << COLOR_MAGENTA << "├──────────────────────────────────────────────┤" << COLOR_RESET << std::endl;
        
        for (int i = 0; i < static_cast<int>(CLASS_NAMES.size()); ++i) {
            int count = class_count[i];
            double avg_conf = (count > 0) ? class_confidences[i] / count : 0.0;
            double percentage = (total_objects > 0) ? (count * 100.0 / total_objects) : 0.0;
            
            // 类别颜色
            std::string class_color = COLOR_WHITE;
            switch(i) {
                case 0: class_color = COLOR_BLUE; break;
                case 1: class_color = COLOR_RED; break;
                case 2: class_color = COLOR_GREEN; break;
                case 3: class_color = COLOR_CYAN; break;
                case 4: class_color = COLOR_MAGENTA; break;
            }
            
            // 数量颜色
            std::string count_color = (count > 0) ? COLOR_GREEN : COLOR_RED;
            
            std::cout << COLOR_MAGENTA << "│" << COLOR_RESET 
                      << class_color << std::setw(12) << CLASS_NAMES[i] << COLOR_RESET
                      << count_color << std::setw(8) << count << COLOR_RESET
                      << std::setw(9) << std::fixed << std::setprecision(1) << percentage << "%"
                      << std::setw(12) << std::fixed << std::setprecision(4) << avg_conf
                      << COLOR_MAGENTA << " │" << COLOR_RESET << std::endl;
        }
        std::cout << COLOR_MAGENTA << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    }
    
    if (g_benchmark_mode && total_images > 0) {
        std::cout << "\n" << COLOR_CYAN << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_BOLD << COLOR_WHITE << " 性能统计:" << COLOR_RESET << COLOR_CYAN << "                                     │" << COLOR_RESET << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   总推理时间: " << std::fixed << std::setprecision(2) 
                  << total_inference_time << " ms" << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   平均推理时间: " << std::fixed << std::setprecision(2) 
                  << (total_inference_time / total_images) << " ms/图片" << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   平均处理速度: " << std::fixed << std::setprecision(2)
                  << (1000.0 / (total_inference_time / total_images)) << " FPS" << std::endl;
        std::cout << COLOR_CYAN << "│" << COLOR_RESET << "   总目标检测速度: " << std::fixed << std::setprecision(2)
                  << (total_objects * 1000.0 / total_inference_time) << " 目标/秒" << std::endl;
        std::cout << COLOR_CYAN << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    }
    
    std::cout << "\n" << COLOR_BLUE << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
    std::cout << COLOR_BLUE << "│" << COLOR_BOLD << COLOR_WHITE << " 输出信息:" << COLOR_RESET << std::endl;
    std::cout << COLOR_BLUE << "│" << COLOR_RESET << "   输出目录: " << COLOR_BOLD << COLOR_CYAN << g_output_dir << COLOR_RESET << std::endl;
    if (total_images > 0) {
        std::cout << COLOR_BLUE << "│" << COLOR_RESET << "   示例输出文件:" << std::endl;
        for (int i = 0; i < std::min(3, static_cast<int>(results.size())); ++i) {
            std::cout << COLOR_BLUE << "│" << COLOR_RESET << "     " 
                      << COLOR_GRAY << get_filename(results[i].image_path) << COLOR_RESET
                      << " → " << COLOR_GREEN << get_filename(results[i].output_path) << COLOR_RESET << std::endl;
        }
        if (results.size() > 3) {
            std::cout << COLOR_BLUE << "│" << COLOR_RESET << "     " << COLOR_GRAY << "... 等 " << results.size() << " 个文件" << COLOR_RESET << std::endl;
        }
    }
    std::cout << COLOR_BLUE << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    
    std::cout << "\n" << COLOR_CYAN << COLOR_BOLD << "══════════════════════════════════════════════" << COLOR_RESET << std::endl;
    std::cout << COLOR_GREEN << COLOR_BOLD << "处理完成！" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << COLOR_BOLD << "══════════════════════════════════════════════" << COLOR_RESET << std::endl;
}

/**
 * @brief 简洁的进度显示，带颜色
 */
void print_progress(int current, int total, const std::string& filename) {
    float percentage = (total > 0) ? (current * 100.0f / total) : 0.0f;
    
    // 根据进度显示不同颜色
    std::string color = COLOR_WHITE;
    if (percentage >= 75) color = COLOR_GREEN;
    else if (percentage >= 50) color = COLOR_YELLOW;
    else if (percentage >= 25) color = COLOR_MAGENTA;
    
    std::cout << "\r" << color;
    std::cout << "[" << std::setw(3) << static_cast<int>(percentage) << "%] ";
    std::cout << COLOR_RESET << "处理: " << color << COLOR_BOLD;
    std::cout << std::setw(30) << std::left << filename.substr(0, 30) << COLOR_RESET;
    std::cout << " " << color << "(" << current << "/" << total << ")" << COLOR_RESET<<std::endl;
    std::cout.flush();
}

// ==================== 核心功能函数 ====================



// 获取程序运行总时间（秒）
double get_program_uptime() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time);
    return duration.count() / 1000.0;
}

// 格式化时间输出
std::string get_program_uptime_str() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time);
    
    auto ms = duration.count();
    auto seconds = ms / 1000;
    auto minutes = seconds / 60;
    auto hours = minutes / 60;
    
    std::ostringstream oss;
    oss << std::setfill('0');
    
    if (hours > 0) {
        oss << hours << "h" << std::setw(2) << minutes % 60 << "m" 
            << std::setw(2) << seconds % 60 << "s";
    } else if (minutes > 0) {
        oss << minutes << "m" << std::setw(2) << seconds % 60 << "s";
    } else if (seconds > 0) {
        oss << seconds << "." << std::setw(3) << ms % 1000 << "s";
    } else {
        oss << ms << "ms";
    }
    
    return oss.str();
}

std::vector<std::string> get_image_files(const std::string& folder_path) {
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(folder_path.c_str());
    if (dir == nullptr) {
        std::cerr << COLOR_RED << "[错误] 无法打开文件夹: " << folder_path << COLOR_RESET << std::endl;
        return image_files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        if (filename == "." || filename == "..") {
            continue;
        }
        
        std::string full_path = folder_path;
        if (folder_path.back() != '/') {
            full_path += "/";
        }
        full_path += filename;
        
        struct stat file_stat;
        if (stat(full_path.c_str(), &file_stat) == 0 && 
            S_ISREG(file_stat.st_mode) && 
            is_image_file(filename)) {
            image_files.push_back(full_path);
        }
    }
    
    closedir(dir);
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

// ==================== 图像处理函数 ====================
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
        std::cerr << COLOR_RED << "[ERROR] mean or scale size must equal to 3" << COLOR_RESET << std::endl;
        exit(1);
    }
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
    float* dout_c0 = dout;
    float* dout_c1 = dout + size;
    float* dout_c2 = dout + size * 2;
    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c0++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c0++) = (*(din++) - mean[2]) * scale[2];
    }
}

// 绘图函数-调用cv2
std::vector<Object> visualize_result(
                        const float* data,
                        int count,
                        float thresh,
                        cv::Mat& image,
                        const std::vector<std::string> &class_names) {
    if (data == nullptr) {
        std::cerr << COLOR_RED << "[ERROR] data can not be nullptr" << COLOR_RESET << std::endl;
        exit(1);
    }
    std::vector<Object> rect_out;
    for (int iw = 0; iw < count; iw++) {
        if (data[1] > thresh) {
            Object obj;
            int x = static_cast<int>(data[2]);
            int y = static_cast<int>(data[3]);
            int w = static_cast<int>(data[4] - data[2] + 1);
            int h = static_cast<int>(data[5] - data[3] + 1);
            cv::Rect rec_clip =
                cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
            obj.class_id = static_cast<int>(data[0]);
            obj.prob = data[1];
            obj.rec = rec_clip;
            if (w > 0 && h > 0 && obj.prob <= 1) {
                rect_out.push_back(obj);
                
                // 检查是否为正常类别（zheng_chang），如果是则跳过可视化
                if (obj.class_id == 4) {  // 正常类别
                    continue;  // 跳过可视化处理
                }
                
                // 不同类别使用不同颜色
                cv::Scalar color;
                switch(obj.class_id) {
                    case 0:  // 擦伤 - 蓝色
                        color = cv::Scalar(255, 0, 0);
                        break;
                    case 1:  // 脏污 - 红色
                        color = cv::Scalar(0, 0, 255);
                        break;
                    case 2:  // 褶皱 - 绿色
                        color = cv::Scalar(128, 0, 128);
                        break;
                    case 3:  // 针孔 - 青色
                        color = cv::Scalar(255, 255, 0);
                        break;
                    default:  // 其他未知类别 - 白色
                        color = cv::Scalar(255, 0, 255);
                        break;
                }
                
                // 绘制边界框
                cv::rectangle(image, rec_clip, color, 2, cv::LINE_AA);
                
                // 计算坐标点
                int x1 = x;
                int y1 = y;
                int x2 = x + w;
                int y2 = y + h;
                
                // 类别标签、置信度和坐标
                std::string label = (obj.class_id >= 0 && obj.class_id < static_cast<int>(class_names.size())) 
                                  ? class_names[obj.class_id] 
                                  : "未知";
                std::string conf_str = std::to_string(obj.prob).substr(0, 4);
                std::string coord_str = "(" + std::to_string(x1) + "," + std::to_string(y1) + ")-(" + 
                                      std::to_string(x2) + "," + std::to_string(y2) + ")";
                std::string text = label + ": " + conf_str + " " + coord_str;
                
                int baseLine;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                
                // 确保标签框不超出图片边界
                int label_y = std::max(y - 5, label_size.height + 5);
                
                // 绘制标签背景
                cv::rectangle(image, 
                            cv::Point(x, label_y - label_size.height - 5), 
                            cv::Point(x + label_size.width, label_y), 
                            color, 
                            cv::FILLED);
                
                // 绘制标签文本
                cv::putText(image, text, 
                           cv::Point(x, label_y - 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(255, 255, 255), 1);
            }
        }
        data += 6;
    }
    return rect_out;
}

// ==================== 模型相关函数 ====================
std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file,
                                           int num_theads) {
    MobileConfig config;
    config.set_model_from_file(model_file);
    config.set_threads(num_theads);
    config.set_power_mode(CPU_POWER_MODE);
    std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);
    return predictor;
}

ImageBlob prepare_imgdata(const cv::Mat& img) {
    ImageBlob img_data;
    
    img_data.im_shape_ = {
        static_cast<float>(DEFAULT_RESIZE[0]),
        static_cast<float>(DEFAULT_RESIZE[1])
    };
    
    img_data.scale_factor_ = {
        static_cast<float>(DEFAULT_RESIZE[0]) / static_cast<float>(img.rows),
        static_cast<float>(DEFAULT_RESIZE[1]) / static_cast<float>(img.cols)
    };
    
    img_data.mean_ = DEFAULT_MEAN;
    img_data.scale_ = DEFAULT_STD;
    
    return img_data;
}

void preprocess(const cv::Mat& img, const ImageBlob img_data, float* data) {
    cv::Mat rgb_img;
    cv::resize(
        img, rgb_img, cv::Size(img_data.im_shape_[0], img_data.im_shape_[1]),
        0.f, 0.f, cv::INTER_CUBIC);

    if(rgb_img.channels() == 4) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGRA2RGB);
    } else if (rgb_img.channels() == 3) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
    }
    
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1);
    
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    neon_mean_scale(
        dimg, data, int(img_data.im_shape_[0] * img_data.im_shape_[1]),
        img_data.mean_, img_data.scale_);
}

// ==================== 信号处理函数（共享库中由调用方管理） ====================

// ==================== 外部C接口（供Python调用） ====================

extern "C" {

/**
 * @brief 初始化SSD检测模型
 * @param model_path 模型文件路径 (*.nb)
 * @return 0=成功, -1=失败
 */
int ssd_init(const char* model_path) {
    if (model_path == nullptr) {
        std::cerr << COLOR_RED << "[错误] 模型路径不能为空" << COLOR_RESET << std::endl;
        return -1;
    }

    std::string model_file(model_path);
    if (!file_exists(model_file)) {
        std::cerr << COLOR_RED << "[错误] 模型文件不存在: " << model_file << COLOR_RESET << std::endl;
        return -1;
    }

    // 加载模型
    try {
        predictor = LoadModel(model_file, DEFAULT_NUM_THREADS);
        if (!predictor) {
            std::cerr << COLOR_RED << "[错误] 模型加载失败" << COLOR_RESET << std::endl;
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << COLOR_RED << "[错误] 模型加载异常: " << e.what() << COLOR_RESET << std::endl;
        return -1;
    }

    std::cout << COLOR_GREEN << "[信息] 模型加载成功: " << model_file << COLOR_RESET << std::endl;
    return 0;
}

/**
 * @brief 开始捕获stdout+stderr到日志文件（每次检测前调用）
 * @param log_file 日志文件路径
 * @return 0=成功, -1=失败
 */
int ssd_log_start(const char* log_file) {
    return redirect_stdout_to_file(log_file);
}

/**
 * @brief 停止捕获，恢复stdout+stderr到终端（每次检测后调用）
 */
void ssd_log_stop() {
    restore_io();
}

/**
 * @brief 检测单张图片
 * @param img_path 输入图片路径
 * @param output_path 输出图片保存路径（含文件名）
 * @param result_json 输出参数，接收结果JSON字符串的缓冲区
 * @param json_buf_size 缓冲区大小
 * @return 0=成功, -1=失败
 */
int ssd_detect(const char* img_path, const char* output_path,
               char* result_json, int json_buf_size) {
    if (img_path == nullptr || output_path == nullptr || result_json == nullptr) {
        std::cerr << COLOR_RED << "[错误] 参数不能为空" << COLOR_RESET << std::endl;
        return -1;
    }

    if (!predictor) {
        std::cerr << COLOR_RED << "[错误] 模型未初始化，请先调用 ssd_init()" << COLOR_RESET << std::endl;
        return -1;
    }

    std::string input_file(img_path);
    std::string output_file(output_path);

    if (!file_exists(input_file)) {
        std::cerr << COLOR_RED << "[错误] 图片文件不存在: " << input_file << COLOR_RESET << std::endl;
        return -1;
    }

    cv::Mat img = imread(input_file, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << COLOR_RED << "[错误] 无法读取图片: " << input_file << COLOR_RESET << std::endl;
        return -1;
    }

    // 准备图像数据
    auto img_data = prepare_imgdata(img);

    auto preprocess_start = std::chrono::steady_clock::now();

    // 设置模型输入
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({1, 2});
    auto* data0 = input_tensor0->mutable_data<float>();
    data0[0] = img_data.im_shape_[0];
    data0[1] = img_data.im_shape_[1];

    std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
    int64_t height = static_cast<int64_t>(img_data.im_shape_[0]);
    int64_t width = static_cast<int64_t>(img_data.im_shape_[1]);
    input_tensor1->Resize({1, 3, height, width});
    auto* data1 = input_tensor1->mutable_data<float>();
    preprocess(img, img_data, data1);

    std::unique_ptr<Tensor> input_tensor2(std::move(predictor->GetInput(2)));
    input_tensor2->Resize({1, 2});
    auto* data2 = input_tensor2->mutable_data<float>();
    data2[0] = img_data.scale_factor_[0];
    data2[1] = img_data.scale_factor_[1];

    auto preprocess_end = std::chrono::steady_clock::now();

    // 推理
    auto inference_start = std::chrono::steady_clock::now();
    predictor->Run();
    auto inference_end = std::chrono::steady_clock::now();

    // 后处理
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor->GetOutput(0)));
    const float* outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    int64_t cnt = 1;
    for (auto& i : shape_out) {
        cnt *= i;
    }

    // 获取检测结果（同时在图片上绘制）
    std::vector<Object> objects = visualize_result(
        outptr, static_cast<int>(cnt / 6), 0.3f, img, CLASS_NAMES);

    // 确保输出目录存在
    std::string output_dir = get_directory(output_file);
    struct stat st;
    if (stat(output_dir.c_str(), &st) == -1) {
        mkdir(output_dir.c_str(), 0755);
    }

    // 保存结果图片
    cv::imwrite(output_file, img);

    // 计算推理耗时
    std::chrono::duration<float> infer_diff = inference_end - inference_start;
    double inference_time_ms = double(infer_diff.count() * 1000);

    // 组装JSON结果
    std::ostringstream json_ss;
    json_ss << "{";
    json_ss << "\"image_path\":\"" << input_file << "\",";
    json_ss << "\"output_path\":\"" << output_file << "\",";
    json_ss << "\"inference_time_ms\":" << std::fixed << std::setprecision(2) << inference_time_ms << ",";
    json_ss << "\"objects\":[";
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];
        std::string class_name = (obj.class_id >= 0 && obj.class_id < static_cast<int>(CLASS_NAMES.size()))
                                 ? CLASS_NAMES[obj.class_id] : "unknown";
        int x1 = obj.rec.x;
        int y1 = obj.rec.y;
        int x2 = obj.rec.x + obj.rec.width;
        int y2 = obj.rec.y + obj.rec.height;

        json_ss << "{";
        json_ss << "\"class_id\":" << obj.class_id << ",";
        json_ss << "\"class_name\":\"" << class_name << "\",";
        json_ss << "\"confidence\":" << std::fixed << std::setprecision(4) << obj.prob << ",";
        json_ss << "\"x1\":" << x1 << ",";
        json_ss << "\"y1\":" << y1 << ",";
        json_ss << "\"x2\":" << x2 << ",";
        json_ss << "\"y2\":" << y2;
        json_ss << "}";
        if (i < objects.size() - 1) json_ss << ",";
    }
    json_ss << "]}";

    std::string json_str = json_ss.str();
    if (static_cast<int>(json_str.size()) >= json_buf_size) {
        std::cerr << COLOR_RED << "[警告] JSON缓冲区不足，结果被截断" << COLOR_RESET << std::endl;
    }
    strncpy(result_json, json_str.c_str(), json_buf_size - 1);
    result_json[json_buf_size - 1] = '\0';

    return 0;
}

/**
 * @brief 释放检测器资源
 */
void ssd_release() {
    predictor = nullptr;
    fpga_release();
    restore_io();
    std::cout << COLOR_GREEN << "[信息] 检测器资源已释放" << COLOR_RESET << std::endl;
}

} // extern "C"

