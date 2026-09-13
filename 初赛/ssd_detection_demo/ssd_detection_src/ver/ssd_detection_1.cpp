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
#include <iomanip>  // 用于格式化输出
#include <sstream>
#include <algorithm>
#include <dirent.h>  // 用于目录操作

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
#include "paddle_api.h"  // NOLINT
#include "intelfpga.h"

// ==================== 命名空间声明 ====================
using namespace paddle::lite_api;  // NOLINT
using namespace std;
using namespace cv;

// ==================== 全局常量定义 ====================
const int CPU_THREAD_NUM = 4;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_FULL;
std::stringstream LOG_MEM;

// ==================== 全局变量声明 ====================
std::shared_ptr<PaddlePredictor> predictor;
bool g_verbose_mode = false;  // 控制详细输出模式
bool g_benchmark_mode = false; // 控制基准测试模式

// ==================== 数据结构定义 ====================

/**
 * @brief 检测结果对象结构体
 * 包含目标框、类别ID和置信度
 */
struct Object {
    cv::Rect rec;       // 目标矩形框
    int class_id;       // 类别ID
    float prob;         // 置信度概率
};

/**
 * @brief 图像预处理数据结构体
 * 存储预处理后的图像数据和相关参数
 */
struct ImageBlob {
    std::vector<float> im_shape_;      // 图像宽高
    const float* im_data_;             // 预处理后的图像数据指针
    std::vector<float> scale_factor_;  // 缩放因子
    std::vector<float> mean_;          // 均值向量
    std::vector<float> scale_;         // 标准差向量
};

/**
 * @brief 检测结果汇总结构体
 * 包含单张图片的检测结果统计
 */
struct DetectionResult {
    std::string image_path;           // 图片路径
    std::vector<Object> objects;      // 检测到的目标
    std::string output_path;          // 输出图片路径
    double inference_time;            // 推理耗时(ms)
    
    // 统计信息
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

// ==================== 计时器类定义 ====================

/**
 * @brief 高精度计时器类（基于chrono）
 * 使用C++11的高分辨率时钟，精度为微秒
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point inTime, outTime;
public:
    void startTimer() { inTime = std::chrono::high_resolution_clock::now(); }
    
    // 获取耗时（单位：毫秒）
    float getCostTimer() {
        outTime = std::chrono::high_resolution_clock::now();
        return static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime)
            .count() / 1e+3);
    }
};

// ==================== 交互界面函数 ====================

/**
 * @brief 显示程序标题和帮助信息
 */
void show_header() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "    PaddleDetection SSD模型推理程序" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "支持功能：" << std::endl;
    std::cout << "  1. 单张图片检测" << std::endl;
    std::cout << "  2. 文件夹批量检测" << std::endl;
    std::cout << "  3. 切换输出模式（简洁/详细）" << std::endl;
    std::cout << "  4. 启用/禁用基准测试" << std::endl;
    std::cout << "  5. 退出程序" << std::endl;
    std::cout << "==========================================" << std::endl;
}

/**
 * @brief 显示模式选择菜单
 * @return 用户选择模式
 */
int show_mode_menu() {
    int choice = 0;
    std::cout << "\n请选择处理模式：" << std::endl;
    std::cout << "1. 单张图片检测" << std::endl;
    std::cout << "2. 文件夹批量检测" << std::endl;
    std::cout << "3. 切换输出模式（简洁/详细）" << std::endl;
    std::cout << "4. 启用/禁用基准测试" << std::endl;
    std::cout << "5. 退出程序" << std::endl;
    std::cout << "\n请输入选择 (1-5): ";
    
    while (true) {
        std::cin >> choice;
        if (std::cin.fail() || choice < 1 || choice > 5) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "输入无效，请重新输入 (1-5): ";
        } else {
            std::cin.ignore(10000, '\n');
            break;
        }
    }
    return choice;
}

/**
 * @brief 切换输出模式
 */
void toggle_output_mode() {
    g_verbose_mode = !g_verbose_mode;
    std::cout << "\n输出模式已切换为: " 
              << (g_verbose_mode ? "详细模式" : "简洁模式") << std::endl;
}

/**
 * @brief 切换基准测试模式
 */
void toggle_benchmark_mode() {
    g_benchmark_mode = !g_benchmark_mode;
    std::cout << "\n基准测试模式: " 
              << (g_benchmark_mode ? "已启用" : "已禁用") << std::endl;
}

/**
 * @brief 获取文件或文件夹路径
 * @param is_folder 是否为文件夹模式
 * @return 路径字符串
 */
std::string get_input_path(bool is_folder) {
    std::string path;
    
    if (is_folder) {
        std::cout << "\n请输入包含图片的文件夹路径: ";
    } else {
        std::cout << "\n请输入图片文件路径: ";
    }
    
    std::getline(std::cin, path);
    
    // 检查路径是否存在
    struct stat path_stat;
    while (stat(path.c_str(), &path_stat) != 0) {
        std::cout << "路径不存在，请重新输入: ";
        std::getline(std::cin, path);
    }
    
    if (is_folder && !S_ISDIR(path_stat.st_mode)) {
        std::cout << "输入路径不是文件夹，请重新输入: ";
        return get_input_path(is_folder);
    } else if (!is_folder && !S_ISREG(path_stat.st_mode)) {
        std::cout << "输入路径不是文件，请重新输入: ";
        return get_input_path(is_folder);
    }
    
    return path;
}

// ==================== 工具函数实现 ====================

/**
 * @brief 判断文件是否存在
 * @param filename 文件名
 * @return 是否存在
 */
bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

/**
 * @brief 判断是否为目录
 * @param path 路径
 * @return 是否为目录
 */
bool is_directory(const std::string& path) {
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0) {
        return false;
    }
    return S_ISDIR(path_stat.st_mode);
}

/**
 * @brief 判断是否为图片文件
 * @param filename 文件名
 * @return 是否为支持的图片格式
 */
bool is_image_file(const std::string& filename) {
    std::string ext = "";
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos && dot_pos < filename.length() - 1) {
        ext = filename.substr(dot_pos);
        
        // 转换为小写
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        // 支持的图片格式
        std::vector<std::string> supported_exts = {".jpg", ".jpeg", ".png", ".bmp"};
        for (const auto& supported_ext : supported_exts) {
            if (ext == supported_ext) {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief 从完整路径中提取文件名
 * @param full_path 完整路径
 * @return 文件名
 */
std::string get_filename(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(last_slash + 1);
    }
    return full_path;
}

/**
 * @brief 从完整路径中提取目录路径
 * @param full_path 完整路径
 * @return 目录路径
 */
std::string get_directory(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(0, last_slash);
    }
    return ".";
}

/**
 * @brief 从文件名中提取不带扩展名的部分
 * @param filename 文件名
 * @return 不带扩展名的文件名
 */
std::string get_stem(const std::string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    size_t last_dot = filename.find_last_of(".");
    
    std::string name_only = filename;
    if (last_slash != std::string::npos) {
        name_only = filename.substr(last_slash + 1);
    }
    
    if (last_dot != std::string::npos && last_dot > last_slash) {
        return name_only.substr(0, last_dot - (last_slash == std::string::npos ? 0 : last_slash + 1));
    }
    return name_only;
}

// ==================== 结果输出函数 ====================

/**
 * @brief 简洁输出单张图片检测结果
 * @param result 检测结果
 */
void print_concise_result(const DetectionResult& result) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "输入图片: " << get_filename(result.image_path) << std::endl;
    std::cout << "检测结果: " << result.total_objects() << " 个目标" << std::endl;
    
    if (result.total_objects() > 0) {
        std::cout << "\n检测到的目标:" << std::endl;
        std::cout << std::setw(6) << "序号" 
                  << std::setw(10) << "类别" 
                  << std::setw(8) << "置信度" 
                  << std::setw(15) << "位置(x,y,w,h)" << std::endl;
        std::cout << std::string(45, '-') << std::endl;
        
        for (size_t i = 0; i < result.objects.size(); ++i) {
            const auto& obj = result.objects[i];
            std::cout << std::setw(6) << i + 1
                      << std::setw(10) << obj.class_id
                      << std::setw(8) << std::fixed << std::setprecision(3) << obj.prob
                      << std::setw(4) << "(" 
                      << obj.rec.x << "," << obj.rec.y << "," 
                      << obj.rec.width << "," << obj.rec.height << ")" 
                      << std::endl;
        }
    } else {
        std::cout << "未检测到目标" << std::endl;
    }
    
    if (g_benchmark_mode) {
        std::cout << "\n推理耗时: " << std::fixed << std::setprecision(2) 
                  << result.inference_time << " ms" << std::endl;
    }
    
    std::cout << "输出图片: " << get_filename(result.output_path) << std::endl;
    std::cout << "==========================================" << std::endl;
}

/**
 * @brief 输出批量处理摘要
 * @param results 所有检测结果
 */
void print_batch_summary(const std::vector<DetectionResult>& results) {
    int total_images = results.size();
    int total_objects = 0;
    int images_with_objects = 0;
    double total_inference_time = 0.0;
    
    for (const auto& result : results) {
        total_objects += result.total_objects();
        total_inference_time += result.inference_time;
        if (result.total_objects() > 0) {
            images_with_objects++;
        }
    }
    
    std::cout << "\n══════════════════════════════════════════════" << std::endl;
    std::cout << "              批量处理完成" << std::endl;
    std::cout << "══════════════════════════════════════════════" << std::endl;
    std::cout << "处理图片总数: " << total_images << std::endl;
    std::cout << "检测到目标的图片数: " << images_with_objects << std::endl;
    std::cout << "检测到的目标总数: " << total_objects << std::endl;
    
    if (g_benchmark_mode && total_images > 0) {
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) 
                  << (total_inference_time / total_images) << " ms/图片" << std::endl;
    }
    
    std::cout << "\n输出目录结构:" << std::endl;
    for (const auto& result : results) {
        std::cout << "  " << get_filename(result.image_path) 
                  << " -> " << get_filename(result.output_path) << std::endl;
    }
    std::cout << "══════════════════════════════════════════════" << std::endl;
}

// ==================== 核心功能函数 ====================

/**
 * @brief 获取文件夹中所有图片文件
 * @param folder_path 文件夹路径
 * @return 图片文件路径列表
 */
std::vector<std::string> get_image_files(const std::string& folder_path) {
    std::vector<std::string> image_files;
    
    DIR* dir = opendir(folder_path.c_str());
    if (dir == nullptr) {
        std::cerr << "[错误] 无法打开文件夹: " << folder_path << std::endl;
        return image_files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // 跳过.和..
        if (filename == "." || filename == "..") {
            continue;
        }
        
        // 构建完整路径
        std::string full_path = folder_path;
        if (folder_path.back() != '/') {
            full_path += "/";
        }
        full_path += filename;
        
        // 检查是否为常规文件且是图片文件
        struct stat file_stat;
        if (stat(full_path.c_str(), &file_stat) == 0 && 
            S_ISREG(file_stat.st_mode) && 
            is_image_file(filename)) {
            image_files.push_back(full_path);
        }
    }
    
    closedir(dir);
    
    // 按文件名排序
    std::sort(image_files.begin(), image_files.end());
    return image_files;
}

/**
 * @brief 加载标签文件
 * @param path 标签文件路径
 * @return 标签字符串向量
 */
std::vector<std::string> LoadLabels(const std::string &path) {
    std::ifstream file;
    std::vector<std::string> labels;
    file.open(path);
    while (file) {
        std::string line;
        std::getline(file, line);
        std::string::size_type pos = line.find(" ");
        if (pos != std::string::npos) {
            line = line.substr(pos);
        }
        labels.push_back(line);
    }
    file.clear();
    file.close();
    return labels;
}

/**
 * @brief 加载配置文件
 * @param config_path 配置文件路径
 * @return 配置键值对映射
 */
std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
    std::ifstream in(config_path);
    std::string line;
    std::map<std::string, std::string> dict;
    
    if (in) {
        while (getline(in, line)) {
            std::stringstream ss(line);
            std::string key, value;
            if (ss >> key >> value) {
                dict[key] = value;
            }
        }
    } else {
        std::cerr << "[错误] 无法打开配置文件: " << config_path << std::endl;
        exit(1);
    }
    return dict;
}

// ==================== 图像处理函数 ====================

/**
 * @brief NEON加速的均值方差归一化处理
 * 将图像从NHWC布局转为NCHW，并应用均值方差归一化
 */
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
        std::cerr << "[ERROR] mean or scale size must equal to 3\n";
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

/**
 * @brief 可视化检测结果
 */
std::vector<Object> visualize_result(
                        const float* data,
                        int count,
                        float thresh,
                        cv::Mat& image,
                        const std::vector<std::string> &class_names) {
    if (data == nullptr) {
        std::cerr << "[ERROR] data can not be nullptr\n";
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
                
                // 绘制边界框
                cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                
                // 添加标签和置信度
                std::string label = class_names[obj.class_id];
                std::string conf_str = std::to_string(obj.prob).substr(0, 4);
                std::string text = label + ": " + conf_str;
                
                int baseLine;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                
                // 绘制标签背景
                cv::rectangle(image, 
                            cv::Point(x, y - label_size.height - 5), 
                            cv::Point(x + label_size.width, y), 
                            cv::Scalar(0, 0, 255), 
                            cv::FILLED);
                
                // 绘制标签文本
                cv::putText(image, text, 
                           cv::Point(x, y - 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(255, 255, 255), 1);
            }
        }
        data += 6;
    }
    return rect_out;
}

// ==================== 模型相关函数 ====================

/**
 * @brief 加载Paddle Lite模型
 */
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

/**
 * @brief 准备图像数据
 */
ImageBlob prepare_imgdata(const cv::Mat& img,
                          std::map<std::string,
                          std::string> config) {
    ImageBlob img_data;
    
    // 解析目标尺寸
    std::vector<int> target_size_;
    std::string resize_str = config.at("Resize");
    size_t comma_pos = resize_str.find(',');
    if (comma_pos != std::string::npos) {
        target_size_.push_back(std::stoi(resize_str.substr(0, comma_pos)));
        target_size_.push_back(std::stoi(resize_str.substr(comma_pos + 1)));
    } else {
        target_size_ = {300, 300};  // 默认尺寸
    }
    
    img_data.im_shape_ = {
        static_cast<float>(target_size_[0]),
        static_cast<float>(target_size_[1])
    };
    
    img_data.scale_factor_ = {
        static_cast<float>(target_size_[0]) / static_cast<float>(img.rows),
        static_cast<float>(target_size_[1]) / static_cast<float>(img.cols)
    };
    
    // 解析均值和标准差
    std::vector<float> mean_, scale_;
    std::string mean_str = config.at("mean");
    std::string std_str = config.at("std");
    
    // 解析均值
    comma_pos = 0;
    size_t start_pos = 0;
    while ((comma_pos = mean_str.find(',', start_pos)) != std::string::npos) {
        mean_.push_back(std::stof(mean_str.substr(start_pos, comma_pos - start_pos)));
        start_pos = comma_pos + 1;
    }
    mean_.push_back(std::stof(mean_str.substr(start_pos)));
    
    // 解析标准差
    start_pos = 0;
    while ((comma_pos = std_str.find(',', start_pos)) != std::string::npos) {
        scale_.push_back(std::stof(std_str.substr(start_pos, comma_pos - start_pos)));
        start_pos = comma_pos + 1;
    }
    scale_.push_back(std::stof(std_str.substr(start_pos)));
    
    img_data.mean_ = mean_;
    img_data.scale_ = scale_;
    return img_data;
}

/**
 * @brief 图像预处理函数
 */
void preprocess(const cv::Mat& img, const ImageBlob img_data, float* data) {
    cv::Mat rgb_img;
    cv::resize(
        img, rgb_img, cv::Size(img_data.im_shape_[0], img_data.im_shape_[1]),
        0.f, 0.f, cv::INTER_CUBIC);

    // 处理RGBA图像
    if(rgb_img.channels() == 4) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGRA2RGB);
    } else if (rgb_img.channels() == 3) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
    }
    
    // 转换为浮点数
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1);
    
    // NEON加速的归一化处理
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    neon_mean_scale(
        dimg, data, int(img_data.im_shape_[0] * img_data.im_shape_[1]),
        img_data.mean_, img_data.scale_);
}

// ==================== 主推理函数 ====================

/**
 * @brief 处理单张图片
 * @param config 配置信息
 * @param img_path 图片路径
 * @param class_names 类别名称
 * @param result 返回检测结果
 * @return 是否处理成功
 */
bool process_single_image(std::map<std::string, std::string>& config,
                         const std::string& img_path,
                         const std::vector<std::string>& class_names,
                         DetectionResult& result) {
    // 检查图片文件是否存在
    if (!file_exists(img_path)) {
        std::cerr << "[错误] 图片文件不存在: " << img_path << std::endl;
        return false;
    }
    
    // 读取图片
    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[错误] 无法读取图片: " << img_path << std::endl;
        return false;
    }
    
    result.image_path = img_path;
    
    auto img_data = prepare_imgdata(img, config);
    
    auto preprocess_start = std::chrono::steady_clock::now();
    
    // 设置模型输入
    #if 1
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({1, 2});
    auto* data0 = input_tensor0->mutable_data<float>();
    data0[0] = img_data.im_shape_[0];
    data0[1] = img_data.im_shape_[1];
    
    std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
    input_tensor1->Resize({1, 3, img_data.im_shape_[0], img_data.im_shape_[1]});
    auto* data1 = input_tensor1->mutable_data<float>();
    preprocess(img, img_data, data1);

    std::unique_ptr<Tensor> input_tensor2(std::move(predictor->GetInput(2)));
    input_tensor2->Resize({1, 2});
    auto* data2 = input_tensor2->mutable_data<float>();
    data2[0] = img_data.scale_factor_[0];
    data2[1] = img_data.scale_factor_[1];
    #else
    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
    input_tensor->Resize({1, 3, img_data.im_shape_[0], img_data.im_shape_[1]});
    auto* data = input_tensor->mutable_data<float>();
    preprocess(img, img_data, data);
    #endif
    
    auto preprocess_end = std::chrono::steady_clock::now();
    
    // 推理
    auto inference_start = std::chrono::steady_clock::now();
    predictor->Run();
    auto inference_end = std::chrono::steady_clock::now();
    
    // 后处理
    auto postprocess_start = std::chrono::steady_clock::now();
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor->GetOutput(0)));
    const float* outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    int64_t cnt = 1;
    for (auto& i : shape_out) {
        cnt *= i;
    }
    
    // 获取检测结果
    result.objects = visualize_result(
        outptr, static_cast<int>(cnt / 6), 0.5f, img, class_names);
    
    // 生成输出路径
    std::string dir_path = get_directory(img_path);
    std::string filename = get_stem(img_path) + "_result.jpg";
    result.output_path = dir_path + "/" + filename;
    
    // 保存结果图片
    cv::imwrite(result.output_path, img);
    auto postprocess_end = std::chrono::steady_clock::now();
    
    // 计算耗时
    std::chrono::duration<float> infer_diff = inference_end - inference_start;
    result.inference_time = double(infer_diff.count() * 1000);
    
    return true;
}

/**
 * @brief 批量处理图片
 * @param config 配置信息
 * @param folder_path 文件夹路径
 * @return 所有图片的检测结果
 */
std::vector<DetectionResult> process_folder(std::map<std::string, std::string>& config,
                                            const std::string& folder_path) {
    std::vector<DetectionResult> all_results;
    
    // 加载标签
    std::vector<std::string> class_names = LoadLabels(config["label_path"]);
    
    // 获取所有图片文件
    std::vector<std::string> image_files = get_image_files(folder_path);
    
    if (image_files.empty()) {
        std::cout << "文件夹中没有找到支持的图片文件 (.jpg, .jpeg, .png, .bmp)" << std::endl;
        return all_results;
    }
    
    std::cout << "找到 " << image_files.size() << " 张图片，开始处理..." << std::endl;
    
    // 加载模型
    predictor = LoadModel(config["model_file"], stoi(config["num_threads"]));
    
    int processed = 0;
    for (const auto& img_path : image_files) {
        DetectionResult result;
        
        if (g_verbose_mode) {
            std::cout << "\n处理中: " << get_filename(img_path) 
                      << " (" << ++processed << "/" << image_files.size() << ")" << std::endl;
        } else {
            std::cout << ".";
            std::cout.flush();
        }
        
        if (process_single_image(config, img_path, class_names, result)) {
            all_results.push_back(result);
            
            if (g_verbose_mode) {
                print_concise_result(result);
            }
        } else {
            std::cerr << "\n[错误] 处理失败: " << img_path << std::endl;
        }
    }
    
    if (!g_verbose_mode) {
        std::cout << "\n";  // 换行
    }
    
    return all_results;
}

// ==================== 信号处理函数 ====================

/**
 * @brief 中断信号处理函数
 */
void int_handler(int sig) {
    fflush(stdout);
    fpga_release();  // 释放FPGA资源
    std::cout << "\n\n程序被用户中断" << std::endl;
    exit(0);
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    // 设置信号处理
    signal(SIGINT, int_handler);
    
    // 显示标题
    show_header();
    
    // 检查命令行参数
    if (argc < 2) {
        std::cerr << "\n使用方法: " << argv[0] << " <配置文件路径> [图片/文件夹路径]" << std::endl;
        std::cerr << "示例1: " << argv[0] << " config.txt" << std::endl;
        std::cerr << "示例2: " << argv[0] << " config.txt test.jpg" << std::endl;
        std::cerr << "示例3: " << argv[0] << " config.txt ./images/" << std::endl;
        return 1;
    }
    
    std::string config_path = argv[1];
    
    // 加载配置
    if (!file_exists(config_path)) {
        std::cerr << "[错误] 配置文件不存在: " << config_path << std::endl;
        return 1;
    }
    
    auto config = LoadConfigTxt(config_path);
    
    if (g_verbose_mode) {
        std::cout << "\n配置信息:" << std::endl;
        std::cout << "模型文件: " << config["model_file"] << std::endl;
        std::cout << "标签文件: " << config["label_path"] << std::endl;
        std::cout << "输入尺寸: " << config["Resize"] << std::endl;
        std::cout << "线程数: " << config["num_threads"] << std::endl;
    }
    
    // 判断运行模式
    int mode = 1;  // 默认单张图片模式
    std::string input_path = "";
    
    if (argc >= 3) {
        // 如果有第二个参数，使用命令行模式
        input_path = argv[2];
        
        if (file_exists(input_path)) {
            if (is_directory(input_path)) {
                mode = 2;  // 文件夹模式
                std::cout << "检测到文件夹路径: " << input_path << std::endl;
            } else {
                mode = 1;  // 单张图片模式
                std::cout << "检测到图片文件: " << input_path << std::endl;
            }
        } else {
            std::cerr << "[错误] 输入路径不存在: " << input_path << std::endl;
            return 1;
        }
    } else {
        // 交互式模式
        while (true) {
            mode = show_mode_menu();
            
            if (mode == 5) {
                std::cout << "\n退出程序" << std::endl;
                return 0;
            }
            
            // 其他选项
            if (mode == 3) {
                toggle_output_mode();
                continue;
            } else if (mode == 4) {
                toggle_benchmark_mode();
                continue;
            } else if (mode == 1 || mode == 2) {
                input_path = get_input_path(mode == 2);
                break;
            }
        }
    }
    
    // 根据模式处理
    std::vector<DetectionResult> results;
    
    if (mode == 1) {
        // 单张图片模式
        std::vector<std::string> class_names = LoadLabels(config["label_path"]);
        predictor = LoadModel(config["model_file"], stoi(config["num_threads"]));
        
        DetectionResult result;
        if (process_single_image(config, input_path, class_names, result)) {
            print_concise_result(result);
        } else {
            std::cerr << "[错误] 处理图片失败" << std::endl;
            return 1;
        }
        
        results.push_back(result);
        
    } else if (mode == 2) {
        // 文件夹批处理模式
        results = process_folder(config, input_path);
        
        if (!results.empty()) {
            print_batch_summary(results);
        }
    }
    
    // 清理资源
    fpga_release();
    
    std::cout << "\n处理完成！" << std::endl;
    return 0;
}