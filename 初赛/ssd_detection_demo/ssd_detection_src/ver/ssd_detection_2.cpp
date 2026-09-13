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
#include <getopt.h>

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

// ==================== 全局常量定义 ====================
const int CPU_THREAD_NUM = 4;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_FULL;
std::stringstream LOG_MEM;

// ==================== 硬编码配置信息 ====================
// 从config.txt提取的配置（除model_file外）
const std::string DEFAULT_LABEL_PATH = "./labels/label_list";
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
std::string g_output_dir = ".";  // 默认输出目录

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

// ==================== 计时器类定义 ====================
class Timer {
private:
    std::chrono::high_resolution_clock::time_point inTime, outTime;
public:
    void startTimer() { inTime = std::chrono::high_resolution_clock::now(); }
    
    float getCostTimer() {
        outTime = std::chrono::high_resolution_clock::now();
        return static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime)
            .count() / 1e+3);
    }
};

// ==================== 命令行参数解析 ====================

/**
 * @brief 显示用法信息
 */
void show_usage(const std::string& program_name) {
    std::cout << "\n使用方法: " << program_name << " [选项]\n\n";
    std::cout << "选项:\n";
    std::cout << "  -I, --img <path>        输入图片文件路径\n";
    std::cout << "  -M, --model <path>      模型文件路径 (*.nb文件)\n";
    std::cout << "  -D, --dir <path>        输入图片文件夹路径\n";
    std::cout << "  -O, --output <path>     输出文件夹路径 (默认: 当前文件夹)\n";
    std::cout << "  -h, --help              显示此帮助信息\n";
    std::cout << "\n示例:\n";
    std::cout << "  " << program_name << " -I test.jpg -M ssd_mobilenet_v1_opt.nb\n";
    std::cout << "  " << program_name << " -D ./images -M model.nb -O ./results\n";
    std::cout << "  " << program_name << " --img input.jpg --model model.nb --output ./output\n";
}

/**
 * @brief 解析命令行参数
 */
bool parse_arguments(int argc, char** argv, 
                     std::string& img_path,
                     std::string& model_path,
                     std::string& dir_path,
                     std::string& output_dir) {
    
    // 命令行选项定义
    static struct option long_options[] = {
        {"img", required_argument, 0, 'I'},
        {"model", required_argument, 0, 'M'},
        {"dir", required_argument, 0, 'D'},
        {"output", required_argument, 0, 'O'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    bool has_input = false;
    
    while ((opt = getopt_long(argc, argv, "I:M:D:O:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'I':
                img_path = optarg;
                has_input = true;
                break;
            case 'M':
                model_path = optarg;
                break;
            case 'D':
                dir_path = optarg;
                has_input = true;
                break;
            case 'O':
                output_dir = optarg;
                break;
            case 'h':
                show_usage(argv[0]);
                exit(0);
            case '?':
                std::cerr << "未知选项或缺少参数\n";
                return false;
            default:
                return false;
        }
    }
    
    // 检查必要参数
    if (!has_input) {
        std::cerr << "错误: 必须指定输入图片(-I)或输入文件夹(-D)\n";
        return false;
    }
    
    if (model_path.empty()) {
        std::cerr << "错误: 必须指定模型文件路径(-M)\n";
        return false;
    }
    
    if (!img_path.empty() && !dir_path.empty()) {
        std::cerr << "错误: 不能同时指定图片文件(-I)和文件夹(-D)\n";
        return false;
    }
    
    return true;
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
 * 修复: 正确提取不含扩展名的文件名
 */
std::string get_stem(const std::string& filename) {
    std::string fname = get_filename(filename);
    size_t last_dot = fname.find_last_of(".");
    if (last_dot != std::string::npos) {
        return fname.substr(0, last_dot);
    }
    return fname;
}

// ==================== 结果输出函数 ====================
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
    
    std::cout << "\n输出目录: " << g_output_dir << std::endl;
    std::cout << "══════════════════════════════════════════════" << std::endl;
}

// ==================== 核心功能函数 ====================
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

// ==================== 图像处理函数 ====================
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
                
                cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                
                std::string label = class_names[obj.class_id];
                std::string conf_str = std::to_string(obj.prob).substr(0, 4);
                std::string text = label + ": " + conf_str;
                
                int baseLine;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                
                cv::rectangle(image, 
                            cv::Point(x, y - label_size.height - 5), 
                            cv::Point(x + label_size.width, y), 
                            cv::Scalar(0, 0, 255), 
                            cv::FILLED);
                
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
    
    // 使用硬编码的配置
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

// ==================== 核心推理函数 ====================
bool process_single_image(const std::string& model_path,
                         const std::string& img_path,
                         const std::vector<std::string>& class_names,
                         DetectionResult& result) {
    if (!file_exists(img_path)) {
        std::cerr << "[错误] 图片文件不存在: " << img_path << std::endl;
        return false;
    }
    
    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[错误] 无法读取图片: " << img_path << std::endl;
        return false;
    }
    
    result.image_path = img_path;
    
    auto img_data = prepare_imgdata(img);
    
    // 加载模型
    if (!predictor) {
        predictor = LoadModel(model_path, DEFAULT_NUM_THREADS);
    }
    
    auto preprocess_start = std::chrono::steady_clock::now();
    
    // 设置模型输入
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
    
    // 获取检测结果
    result.objects = visualize_result(
        outptr, static_cast<int>(cnt / 6), 0.5f, img, class_names);
    
    // 生成输出路径 - 修复文件名问题
    std::string filename_stem = get_stem(img_path);
    std::string output_filename = filename_stem + "_result.jpg";
    
    // 确保输出目录存在
    struct stat st;
    if (stat(g_output_dir.c_str(), &st) == -1) {
        mkdir(g_output_dir.c_str(), 0755);
    }
    
    result.output_path = g_output_dir + "/" + output_filename;
    
    // 保存结果图片
    cv::imwrite(result.output_path, img);
    
    // 计算耗时
    std::chrono::duration<float> infer_diff = inference_end - inference_start;
    result.inference_time = double(infer_diff.count() * 1000);
    
    return true;
}

std::vector<DetectionResult> process_folder(const std::string& model_path,
                                            const std::string& folder_path) {
    std::vector<DetectionResult> all_results;
    
    std::vector<std::string> class_names = LoadLabels(DEFAULT_LABEL_PATH);
    std::vector<std::string> image_files = get_image_files(folder_path);
    
    if (image_files.empty()) {
        std::cout << "文件夹中没有找到支持的图片文件 (.jpg, .jpeg, .png, .bmp)" << std::endl;
        return all_results;
    }
    
    std::cout << "找到 " << image_files.size() << " 张图片，开始处理..." << std::endl;
    
    predictor = LoadModel(model_path, DEFAULT_NUM_THREADS);
    
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
        
        if (process_single_image(model_path, img_path, class_names, result)) {
            all_results.push_back(result);
            
            if (g_verbose_mode) {
                print_concise_result(result);
            }
        } else {
            std::cerr << "\n[错误] 处理失败: " << img_path << std::endl;
        }
    }
    
    if (!g_verbose_mode) {
        std::cout << "\n";
    }
    
    return all_results;
}

// ==================== 信号处理函数 ====================
void int_handler(int sig) {
    fflush(stdout);
    fpga_release();
    std::cout << "\n\n程序被用户中断" << std::endl;
    exit(0);
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    signal(SIGINT, int_handler);
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "    PaddleDetection SSD模型推理程序" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // 解析命令行参数
    std::string img_path, model_path, dir_path;
    
    if (!parse_arguments(argc, argv, img_path, model_path, dir_path, g_output_dir)) {
        show_usage(argv[0]);
        return 1;
    }
    
    // 验证模型文件
    if (!file_exists(model_path)) {
        std::cerr << "[错误] 模型文件不存在: " << model_path << std::endl;
        return 1;
    }
    
    // 显示配置信息
    std::cout << "\n配置信息:" << std::endl;
    std::cout << "模型文件: " << model_path << std::endl;
    std::cout << "标签文件: " << DEFAULT_LABEL_PATH << std::endl;
    std::cout << "输入尺寸: " << DEFAULT_RESIZE[0] << "," << DEFAULT_RESIZE[1] << std::endl;
    std::cout << "均值: " << DEFAULT_MEAN[0] << "," << DEFAULT_MEAN[1] << "," << DEFAULT_MEAN[2] << std::endl;
    std::cout << "标准差: " << DEFAULT_STD[0] << "," << DEFAULT_STD[1] << "," << DEFAULT_STD[2] << std::endl;
    std::cout << "线程数: " << DEFAULT_NUM_THREADS << std::endl;
    std::cout << "输出目录: " << g_output_dir << std::endl;
    std::cout << "==========================================" << std::endl;
    
    std::vector<DetectionResult> results;
    
    if (!img_path.empty()) {
        // 单张图片模式
        std::vector<std::string> class_names = LoadLabels(DEFAULT_LABEL_PATH);
        
        if (!file_exists(img_path)) {
            std::cerr << "[错误] 图片文件不存在: " << img_path << std::endl;
            return 1;
        }
        
        DetectionResult result;
        if (process_single_image(model_path, img_path, class_names, result)) {
            print_concise_result(result);
        } else {
            std::cerr << "[错误] 处理图片失败" << std::endl;
            return 1;
        }
        
        results.push_back(result);
        
    } else if (!dir_path.empty()) {
        // 文件夹批处理模式
        if (!file_exists(dir_path) || !is_directory(dir_path)) {
            std::cerr << "[错误] 文件夹不存在或不是有效目录: " << dir_path << std::endl;
            return 1;
        }
        
        results = process_folder(model_path, dir_path);
        
        if (!results.empty()) {
            print_batch_summary(results);
        }
    }
    
    // 清理资源
    fpga_release();
    
    std::cout << "\n处理完成！" << std::endl;
    return 0;
}