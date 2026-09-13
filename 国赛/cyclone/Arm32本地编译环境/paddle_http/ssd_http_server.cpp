// =================================================================
// SSD缺陷检测 HTTP 服务器
// 
// 功能：将 plite-test.cpp (HTTP服务器) 与 ssd_detection.cpp (SSD检测)
//       合并为一个独立的可执行文件，替代 ssd_detection_6.cpp + 
//       ssd_detection_6.py 的组合方案。
//
// 编译示例 (ARM / Cyclone V SoC):
//   g++ ssd_http_server.cpp -o ssd_http_server \
//       -I<opencv_include> -I<paddle_lite_include> \
//       -L<opencv_lib> -L<paddle_lite_lib> \
//       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//       -lpaddle_light_api_shared -lintelfpga \
//       -lpthread -ldl -lm -lstdc++ \
//       -O2 -mfpu=neon -march=armv7-a
//
// 运行:
//   ./ssd_http_server [模型路径] [端口]
//   ./ssd_http_server ssd_mobilenet_v1_opt.nb 9000
//
// HTTP接口:
//   GET  /hello          -> 健康检查
//   POST /upload         -> 上传图片并检测 (form-data, field: "image_file")
//   GET  /detect?path=xxx -> 对服务器本地图片执行检测
//   GET  /stats          -> 获取服务器统计信息
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
#include <mutex>

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

// HTTP服务器
#include "httplib.h"

// ==================== 命名空间声明 ====================
using namespace paddle::lite_api;
using namespace std;
using namespace cv;

// ==================== ANSI颜色代码定义 ====================
#ifdef _WIN32
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
#endif

// ==================== 全局常量 ====================
const int CPU_THREAD_NUM = 4;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_FULL;

const std::vector<std::string> CLASS_NAMES = {
    "ca_shang",   // 0: 擦伤
    "zang_wu",    // 1: 脏污
    "zhe_zhou",   // 2: 褶皱
    "zhen_kong",  // 3: 针孔
    "zheng_chang" // 4: 正常
};

const std::string DEFAULT_MODEL_PATH = "ssd_mobilenet_v1_opt.nb";
const int DEFAULT_NUM_THREADS = 2;
const std::vector<int> DEFAULT_RESIZE = {300, 300};
const std::vector<float> DEFAULT_MEAN = {127.5f, 127.5f, 127.5f};
const std::vector<float> DEFAULT_STD = {127.502231f, 127.502231f, 127.502231f};

// ==================== 全局变量 ====================
std::shared_ptr<PaddlePredictor> g_predictor;
bool g_fpga_debug = false;
std::string g_output_dir = "output";
std::string g_log_file = "detection.log";
FILE* g_stderr_backup = nullptr;

// 线程安全锁（predictor非线程安全）
std::mutex g_detect_mutex;

// 统计信息
static auto g_server_start_time = std::chrono::steady_clock::now();
int g_total_requests = 0;
int g_total_detections = 0;
double g_total_inference_time = 0.0;
int g_total_objects_detected = 0;

// ==================== 数据结构 ====================
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

// ==================== FPGA调试控制 ====================
void suppress_fpga_debug() {
    if (!g_fpga_debug) {
        g_stderr_backup = fdopen(dup(fileno(stderr)), "w");
        (void)freopen("/dev/null", "w", stderr);
    }
}

void restore_stderr() {
    if (!g_fpga_debug && g_stderr_backup) {
        fflush(stderr);
        dup2(fileno(g_stderr_backup), fileno(stderr));
        fclose(g_stderr_backup);
        g_stderr_backup = nullptr;
    }
}

// ==================== 工具函数 ====================
bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

bool is_image_file(const std::string& filename) {
    std::string ext = "";
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos != std::string::npos && dot_pos < filename.length() - 1) {
        ext = filename.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::vector<std::string> supported_exts = {".jpg", ".jpeg", ".png", ".bmp"};
        for (const auto& supported_ext : supported_exts) {
            if (ext == supported_ext) return true;
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

std::string get_stem(const std::string& filename) {
    std::string fname = get_filename(filename);
    size_t last_dot = fname.find_last_of(".");
    if (last_dot != std::string::npos) {
        return fname.substr(0, last_dot);
    }
    return fname;
}

std::string get_directory(const std::string& full_path) {
    size_t last_slash = full_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return full_path.substr(0, last_slash);
    }
    return ".";
}

std::string get_current_time_str() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

std::string get_uptime_str() {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_server_start_time).count();
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
    } else {
        oss << seconds << "." << std::setw(3) << ms % 1000 << "s";
    }
    return oss.str();
}

// ==================== 日志 ====================
void log_detection_result(const std::string& filename, const DetectionResult& result) {
    static std::ofstream log_stream;
    static bool first_write = true;

    if (first_write) {
        log_stream.open(g_log_file, std::ios::out | std::ios::trunc);
        if (log_stream.is_open()) {
            log_stream << "timestamp\tfile_name\tdefect_name\tdefect_order\tx1\ty1\tx2\ty2\tconfidence\n";
            first_write = false;
            std::cout << COLOR_GREEN << "[信息] 日志文件已创建: " << g_log_file << COLOR_RESET << std::endl;
        }
    }

    if (!log_stream.is_open()) return;

    std::string base_filename = get_filename(filename);
    int defect_order = 0;

    for (const auto& obj : result.objects) {
        if (obj.class_id == 4) continue; // 跳过正常类别

        std::string defect_name = (obj.class_id >= 0 && obj.class_id < static_cast<int>(CLASS_NAMES.size()))
                                 ? CLASS_NAMES[obj.class_id] : "unknown";
        defect_order++;

        log_stream << get_current_time_str() << "\t"
                   << base_filename << "\t"
                   << defect_name << "\t"
                   << defect_order << "\t"
                   << obj.rec.x << "\t"
                   << obj.rec.y << "\t"
                   << (obj.rec.x + obj.rec.width) << "\t"
                   << (obj.rec.y + obj.rec.height) << "\t"
                   << std::fixed << std::setprecision(4) << obj.prob << "\n";
    }
    log_stream.flush();
}

// ==================== NEON预处理 ====================
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
        vst1q_f32(dout_c0, vmulq_f32(vsub0, vscale0));
        vst1q_f32(dout_c1, vmulq_f32(vsub1, vscale1));
        vst1q_f32(dout_c2, vmulq_f32(vsub2, vscale2));
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

// ==================== 可视化 ====================
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
            cv::Rect rec_clip = cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
            obj.class_id = static_cast<int>(data[0]);
            obj.prob = data[1];
            obj.rec = rec_clip;
            if (w > 0 && h > 0 && obj.prob <= 1) {
                rect_out.push_back(obj);

                if (obj.class_id == 4) continue; // 正常类别不画框

                cv::Scalar color;
                switch(obj.class_id) {
                    case 0: color = cv::Scalar(255, 0, 0); break;    // 擦伤-蓝
                    case 1: color = cv::Scalar(0, 0, 255); break;    // 脏污-红
                    case 2: color = cv::Scalar(128, 0, 128); break;  // 褶皱-绿
                    case 3: color = cv::Scalar(255, 255, 0); break;  // 针孔-青
                    default: color = cv::Scalar(255, 0, 255); break;
                }

                cv::rectangle(image, rec_clip, color, 2, cv::LINE_AA);

                std::string label = (obj.class_id >= 0 && obj.class_id < static_cast<int>(class_names.size()))
                                   ? class_names[obj.class_id] : "未知";
                std::string text = label + ": " + std::to_string(obj.prob).substr(0, 4);

                int baseLine;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int label_y = std::max(y - 5, label_size.height + 5);
                cv::rectangle(image,
                            cv::Point(x, label_y - label_size.height - 5),
                            cv::Point(x + label_size.width, label_y),
                            color, cv::FILLED);
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

// ==================== 模型加载 ====================
std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file, int num_threads) {
    MobileConfig config;
    config.set_model_from_file(model_file);
    config.set_threads(num_threads);
    config.set_power_mode(CPU_POWER_MODE);
    return CreatePaddlePredictor<MobileConfig>(config);
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
    cv::resize(img, rgb_img, cv::Size(img_data.im_shape_[0], img_data.im_shape_[1]),
                0.f, 0.f, cv::INTER_CUBIC);
    if (rgb_img.channels() == 4) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGRA2RGB);
    } else if (rgb_img.channels() == 3) {
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
    }
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    neon_mean_scale(dimg, data, int(img_data.im_shape_[0] * img_data.im_shape_[1]),
                    img_data.mean_, img_data.scale_);
}

// ==================== 核心检测函数 ====================

/**
 * @brief 对指定路径的图片执行SSD检测
 * @param img_path 输入图片路径
 * @param output_path 输出图片路径
 * @param result 检测结果输出
 * @return true=成功, false=失败
 */
bool detect_image(const std::string& img_path,
                  const std::string& output_path,
                  DetectionResult& result) {
    if (!file_exists(img_path)) {
        std::cerr << COLOR_RED << "[错误] 图片文件不存在: " << img_path << COLOR_RESET << std::endl;
        return false;
    }

    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << COLOR_RED << "[错误] 无法读取图片: " << img_path << COLOR_RESET << std::endl;
        return false;
    }

    result.image_path = img_path;
    auto img_data = prepare_imgdata(img);

    // 加锁（predictor全局唯一，非线程安全）
    std::lock_guard<std::mutex> lock(g_detect_mutex);

    if (!g_predictor) {
        std::cerr << COLOR_RED << "[错误] 模型未初始化" << COLOR_RESET << std::endl;
        return false;
    }

    // 输入0: im_shape
    std::unique_ptr<Tensor> input_tensor0(std::move(g_predictor->GetInput(0)));
    input_tensor0->Resize({1, 2});
    auto* data0 = input_tensor0->mutable_data<float>();
    data0[0] = img_data.im_shape_[0];
    data0[1] = img_data.im_shape_[1];

    // 输入1: image data
    std::unique_ptr<Tensor> input_tensor1(std::move(g_predictor->GetInput(1)));
    int64_t height = static_cast<int64_t>(img_data.im_shape_[0]);
    int64_t width = static_cast<int64_t>(img_data.im_shape_[1]);
    input_tensor1->Resize({1, 3, height, width});
    auto* data1 = input_tensor1->mutable_data<float>();
    preprocess(img, img_data, data1);

    // 输入2: scale_factor
    std::unique_ptr<Tensor> input_tensor2(std::move(g_predictor->GetInput(2)));
    input_tensor2->Resize({1, 2});
    auto* data2 = input_tensor2->mutable_data<float>();
    data2[0] = img_data.scale_factor_[0];
    data2[1] = img_data.scale_factor_[1];

    // 推理
    auto inference_start = std::chrono::steady_clock::now();
    g_predictor->Run();
    auto inference_end = std::chrono::steady_clock::now();

    // 后处理
    std::unique_ptr<const Tensor> output_tensor(std::move(g_predictor->GetOutput(0)));
    const float* outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    int64_t cnt = 1;
    for (auto& i : shape_out) cnt *= i;

    result.objects = visualize_result(
        outptr, static_cast<int>(cnt / 6), 0.3f, img, CLASS_NAMES);

    // 确保输出目录存在
    std::string out_dir = get_directory(output_path);
    if (!out_dir.empty() && !file_exists(out_dir)) {
        mkdir(out_dir.c_str(), 0755);
    }

    result.output_path = output_path;
    cv::imwrite(output_path, img);

    // 日志
    log_detection_result(img_path, result);

    // 统计
    std::chrono::duration<float> infer_diff = inference_end - inference_start;
    result.inference_time = double(infer_diff.count() * 1000);

    // 更新全局统计
    g_total_detections++;
    g_total_inference_time += result.inference_time;
    g_total_objects_detected += result.total_objects();

    return true;
}

// ==================== JSON组装 ====================

/**
 * @brief 将检测结果转为JSON字符串
 */
std::string detection_result_to_json(const DetectionResult& result) {
    std::ostringstream js;
    js << "{";
    js << "\"image_path\":\"" << result.image_path << "\",";
    js << "\"output_path\":\"" << result.output_path << "\",";
    js << "\"inference_time_ms\":" << std::fixed << std::setprecision(2) << result.inference_time << ",";
    js << "\"objects\":[";
    for (size_t i = 0; i < result.objects.size(); ++i) {
        const auto& obj = result.objects[i];
        std::string class_name = (obj.class_id >= 0 && obj.class_id < static_cast<int>(CLASS_NAMES.size()))
                                 ? CLASS_NAMES[obj.class_id] : "unknown";
        js << "{";
        js << "\"class_id\":" << obj.class_id << ",";
        js << "\"class_name\":\"" << class_name << "\",";
        js << "\"confidence\":" << std::fixed << std::setprecision(4) << obj.prob << ",";
        js << "\"x1\":" << obj.rec.x << ",";
        js << "\"y1\":" << obj.rec.y << ",";
        js << "\"x2\":" << (obj.rec.x + obj.rec.width) << ",";
        js << "\"y2\":" << (obj.rec.y + obj.rec.height);
        js << "}";
        if (i < result.objects.size() - 1) js << ",";
    }
    js << "]}";
    return js.str();
}

/**
 * @brief 读取文件全部内容到string
 */
std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return "";
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

/**
 * @brief 读取文件全部内容到vector<uint8_t>
 */
std::vector<uint8_t> read_file_to_bytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
}

// ==================== HTTP响应辅助 ====================

/**
 * @brief 发送JSON响应
 */
void send_json_response(httplib::Response &res, const std::string& json_str, int status = 200) {
    res.status = status;
    res.set_content(json_str, "application/json; charset=utf-8");
}

/**
 * @brief 发送错误响应
 */
void send_error_response(httplib::Response &res, const std::string& msg, int status = 400) {
    std::ostringstream js;
    js << "{\"error\":\"" << msg << "\"}";
    res.status = status;
    res.set_content(js.str(), "application/json; charset=utf-8");
}

// ==================== 信号处理 ====================
void int_handler(int sig) {
    std::cout << "\n\n" << COLOR_RED << "服务器被中断，正在清理..." << COLOR_RESET << std::endl;
    fpga_release();
    restore_stderr();
    exit(0);
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    signal(SIGINT, int_handler);

    // ========== 解析命令行参数 ==========
    std::string model_path = DEFAULT_MODEL_PATH;
    int port = 9000;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) port = atoi(argv[2]);

    // ========== 验证模型文件 ==========
    if (!file_exists(model_path)) {
        std::cerr << COLOR_RED << "[错误] 模型文件不存在: " << model_path << COLOR_RESET << std::endl;
        return 1;
    }

    // ========== 抑制FPGA调试输出 ==========
    suppress_fpga_debug();

    // ========== 加载模型 ==========
    std::cout << COLOR_CYAN << COLOR_BOLD << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     SSD缺陷检测 HTTP 服务器 (PaddleLite)      ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝" << COLOR_RESET << std::endl;
    std::cout << std::endl;

    std::cout << COLOR_YELLOW << "┌──────────────────────────────────────────────┐" << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 模型文件: " << COLOR_CYAN << model_path << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 监听端口: " << COLOR_CYAN << port << COLOR_RESET << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 类别数量: " << CLASS_NAMES.size() << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 支持类别: ";
    for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        std::string cc;
        switch(i) { case 0: cc=COLOR_BLUE; break; case 1: cc=COLOR_RED; break;
                     case 2: cc=COLOR_GREEN; break; case 3: cc=COLOR_CYAN; break;
                     case 4: cc=COLOR_MAGENTA; break; default: cc=COLOR_WHITE; }
        std::cout << cc << CLASS_NAMES[i] << COLOR_RESET;
        if (i < CLASS_NAMES.size()-1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 输入尺寸: " << DEFAULT_RESIZE[0] << "×" << DEFAULT_RESIZE[1] << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 线程数:   " << DEFAULT_NUM_THREADS << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 输出目录: " << g_output_dir << std::endl;
    std::cout << COLOR_YELLOW << "│" << COLOR_RESET << " 日志文件: " << g_log_file << std::endl;
    std::cout << COLOR_YELLOW << "└──────────────────────────────────────────────┘" << COLOR_RESET << std::endl;
    std::cout << std::endl;

    std::cout << COLOR_GREEN << "[信息] 正在加载模型..." << COLOR_RESET << std::endl;
    try {
        g_predictor = LoadModel(model_path, DEFAULT_NUM_THREADS);
        if (!g_predictor) {
            std::cerr << COLOR_RED << "[错误] 模型加载失败" << COLOR_RESET << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << COLOR_RED << "[错误] 模型加载异常: " << e.what() << COLOR_RESET << std::endl;
        return 1;
    }
    std::cout << COLOR_GREEN << "[信息] 模型加载成功 ✓" << COLOR_RESET << std::endl;

    // 确保输出目录存在
    if (!file_exists(g_output_dir)) {
        mkdir(g_output_dir.c_str(), 0755);
    }

    // ========== 创建HTTP服务器 ==========
    httplib::Server srv;

    // ---------- GET /hello ----------
    srv.Get("/hello", [](const httplib::Request& req, httplib::Response& res) {
        g_total_requests++;
        std::ostringstream js;
        js << "{";
        js << "\"status\":\"ok\",";
        js << "\"message\":\"SSD Detection Server is running\",";
        js << "\"uptime\":\"" << get_uptime_str() << "\"";
        js << "}";
        send_json_response(res, js.str());
        std::cout << COLOR_GREEN << "[GET /hello] " << get_current_time_str() << COLOR_RESET << std::endl;
    });

    // ---------- POST /upload ----------
    // 接收 multipart/form-data 上传的图片，执行检测，返回JSON结果
    srv.Post("/upload", [](const httplib::Request& req, httplib::Response& res) {
        g_total_requests++;
        std::cout << COLOR_CYAN << "[POST /upload] " << get_current_time_str() << COLOR_RESET << std::endl;

        // 检查是否有文件上传
        if (!req.has_file("image_file")) {
            // 也尝试其他常见字段名
            bool found = false;
            for (const auto& pair : req.files) {
                std::cout << "  field: " << pair.first << std::endl;
                found = true;
            }
            if (!found) {
                send_error_response(res, "未找到上传的文件字段 (期望 'image_file')", 400);
                return;
            }
        }

        const auto& file = req.get_file_value("image_file");
        std::string filename = file.filename;
        if (filename.empty()) filename = "uploaded_image.jpg";

        std::cout << "  文件名: " << filename << std::endl;
        std::cout << "  类型: " << file.content_type << std::endl;
        std::cout << "  大小: " << file.content.size() << " bytes" << std::endl;

        // 验证是否为图片
        if (!is_image_file(filename)) {
            send_error_response(res, "不支持的文件类型，请上传 .jpg/.jpeg/.png/.bmp 图片", 400);
            return;
        }

        // 保存上传的图片到临时文件
        std::string input_path = g_output_dir + "/upload_" + filename;
        std::ofstream ofs(input_path, std::ios::binary);
        if (!ofs) {
            send_error_response(res, "无法保存上传的文件", 500);
            return;
        }
        ofs.write(file.content.data(), file.content.size());
        ofs.close();

        // 生成输出路径
        std::string stem = get_stem(filename);
        std::string output_path = g_output_dir + "/" + stem + "_detect.jpg";

        // 执行检测
        DetectionResult result;
        bool success = detect_image(input_path, output_path, result);

        if (!success) {
            send_error_response(res, "图片检测失败", 500);
            return;
        }

        // 读取结果图片，转为base64内嵌到JSON中
        std::vector<uint8_t> result_image = read_file_to_bytes(output_path);

        // 组装JSON响应
        std::string json_result = detection_result_to_json(result);

        // 在JSON中附加图片数据（base64编码）
        std::string json_with_image;
        // 简单做法：先发JSON，图片通过单独接口获取
        // 这里直接在JSON中加output_path字段，客户端再请求图片

        std::cout << COLOR_GREEN << "  ✓ 检测完成: " << result.total_objects() << " 个目标"
                  << " (" << std::fixed << std::setprecision(1) << result.inference_time << "ms)" << COLOR_RESET << std::endl;

        // 设置自定义头，告诉客户端结果图片路径
        res.set_header("X-Detection-Count", std::to_string(result.total_objects()));
        res.set_header("X-Inference-Time", std::to_string((int)result.inference_time) + "ms");
        res.set_header("X-Result-Image", "/result_image?path=" + stem + "_detect.jpg");

        send_json_response(res, json_result);
    });

    // ---------- GET /detect?path=xxx ----------
    // 对服务器本地已存在的图片执行检测
    srv.Get("/detect", [](const httplib::Request& req, httplib::Response& res) {
        g_total_requests++;
        std::cout << COLOR_CYAN << "[GET /detect] " << get_current_time_str() << COLOR_RESET << std::endl;

        std::string img_path;
        if (req.has_param("path")) {
            img_path = req.get_param_value("path");
        } else if (req.has_param("file")) {
            img_path = req.get_param_value("file");
        } else {
            send_error_response(res, "缺少参数 'path' 或 'file'", 400);
            return;
        }

        // 安全检查：防止路径遍历
        if (img_path.find("..") != std::string::npos ||
            img_path.find('/') == 0) {
            send_error_response(res, "非法的文件路径", 403);
            return;
        }

        // 如果只给了文件名，在输出目录中查找
        std::string full_path;
        if (file_exists(img_path)) {
            full_path = img_path;
        } else {
            full_path = g_output_dir + "/" + img_path;
        }

        if (!file_exists(full_path)) {
            send_error_response(res, "图片文件不存在: " + img_path, 404);
            return;
        }

        std::string stem = get_stem(get_filename(full_path));
        std::string output_path = g_output_dir + "/" + stem + "_detect.jpg";

        DetectionResult result;
        bool success = detect_image(full_path, output_path, result);

        if (!success) {
            send_error_response(res, "检测失败", 500);
            return;
        }

        std::cout << COLOR_GREEN << "  ✓ 检测完成: " << result.total_objects() << " 个目标"
                  << " (" << std::fixed << std::setprecision(1) << result.inference_time << "ms)" << COLOR_RESET << std::endl;

        std::string json_result = detection_result_to_json(result);
        send_json_response(res, json_result);
    });

    // ---------- GET /result_image?path=xxx ----------
    // 获取检测结果图片
    srv.Get("/result_image", [](const httplib::Request& req, httplib::Response& res) {
        std::string img_path;
        if (req.has_param("path")) {
            img_path = req.get_param_value("path");
        } else {
            send_error_response(res, "缺少参数 'path'", 400);
            return;
        }

        // 安全检查
        if (img_path.find("..") != std::string::npos ||
            img_path.find('/') == 0) {
            send_error_response(res, "非法的文件路径", 403);
            return;
        }

        std::string full_path = g_output_dir + "/" + img_path;
        if (!file_exists(full_path)) {
            send_error_response(res, "结果图片不存在", 404);
            return;
        }

        std::vector<uint8_t> img_data = read_file_to_bytes(full_path);
        if (img_data.empty()) {
            send_error_response(res, "无法读取图片", 500);
            return;
        }

        // 根据扩展名设置Content-Type
        std::string content_type = "image/jpeg";
        if (img_path.find(".png") != std::string::npos) content_type = "image/png";
        else if (img_path.find(".bmp") != std::string::npos) content_type = "image/bmp";

        res.status = 200;
	res.set_content(reinterpret_cast<const char*>(img_data.data()), img_data.size(), content_type.c_str());
    });

    // ---------- GET /stats ----------
    // 获取服务器运行统计
    srv.Get("/stats", [](const httplib::Request& req, httplib::Response& res) {
        std::ostringstream js;
        js << "{";
        js << "\"uptime\":\"" << get_uptime_str() << "\",";
        js << "\"total_requests\":" << g_total_requests << ",";
        js << "\"total_detections\":" << g_total_detections << ",";
        js << "\"total_objects_detected\":" << g_total_objects_detected << ",";
        js << "\"total_inference_time_ms\":" << std::fixed << std::setprecision(2) << g_total_inference_time << ",";
        if (g_total_detections > 0) {
            js << "\"avg_inference_time_ms\":" << std::fixed << std::setprecision(2)
               << (g_total_inference_time / g_total_detections) << ",";
            js << "\"avg_fps\":" << std::fixed << std::setprecision(2)
               << (1000.0 / (g_total_inference_time / g_total_detections)) << ",";
        } else {
            js << "\"avg_inference_time_ms\":0,";
            js << "\"avg_fps\":0,";
        }
        js << "\"output_dir\":\"" << g_output_dir << "\"";
        js << "}";
        send_json_response(res, js.str());
    });

    // ---------- GET /log ----------
    // 获取检测日志内容
    srv.Get("/log", [](const httplib::Request& req, httplib::Response& res) {
        if (!file_exists(g_log_file)) {
            send_error_response(res, "日志文件不存在", 404);
            return;
        }
        std::string content = read_file_to_string(g_log_file);
        res.status = 200;
        res.set_content(content, "text/plain; charset=utf-8");
    });

    // ---------- GET / (主页) ----------
    srv.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        std::ostringstream html;
        html << "<!DOCTYPE html>\n<html>\n<head>\n";
        html << "<title>SSD缺陷检测服务</title>\n";
        html << "<style>\nbody{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5;}\n";
        html << "h1{color:#333;} .endpoint{background:#fff;padding:15px;margin:10px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}\n";
        html << "code{background:#e8e8e8;padding:2px 6px;border-radius:3px;}\n";
        html << ".method{display:inline-block;padding:3px 8px;border-radius:3px;color:#fff;font-weight:bold;font-size:12px;}\n";
        html << ".get{background:#61affe;}.post{background:#49cc90;}\n";
        html << "</style></head>\n<body>\n";
        html << "<h1>🔍 SSD缺陷检测 HTTP 服务器</h1>\n";
        html << "<p>服务运行中，已运行时间: <b>" << get_uptime_str() << "</b></p>\n";
        html << "<h2>API 接口</h2>\n";
        html << "<div class='endpoint'><span class='method get'>GET</span> <code>/hello</code> - 健康检查</div>\n";
        html << "<div class='endpoint'><span class='method post'>POST</span> <code>/upload</code> - 上传图片并检测 (form-data, field: image_file)</div>\n";
        html << "<div class='endpoint'><span class='method get'>GET</span> <code>/detect?path=xxx.jpg</code> - 检测服务器本地图片</div>\n";
        html << "<div class='endpoint'><span class='method get'>GET</span> <code>/result_image?path=xxx_detect.jpg</code> - 获取检测结果图片</div>\n";
        html << "<div class='endpoint'><span class='method get'>GET</span> <code>/stats</code> - 获取服务器统计信息</div>\n";
        html << "<div class='endpoint'><span class='method get'>GET</span> <code>/log</code> - 查看检测日志</div>\n";
        html << "<h2>支持的缺陷类别</h2>\n<ul>\n";
        for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
            html << "<li><b>" << CLASS_NAMES[i] << "</b> (class_id=" << i << ")</li>\n";
        }
        html << "</ul>\n</body>\n</html>";
        res.status = 200;
        res.set_content(html.str(), "text/html; charset=utf-8");
    });

    // ========== 启动服务器 ==========
    std::cout << std::endl;
    std::cout << COLOR_GREEN << COLOR_BOLD << "✓ 服务器已启动!" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "  监听地址: 0.0.0.0:" << port << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "  访问主页: http://localhost:" << port << "/" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "  健康检查: http://localhost:" << port << "/hello" << COLOR_RESET << std::endl;
    std::cout << COLOR_CYAN << "  上传检测: POST http://localhost:" << port << "/upload" << COLOR_RESET << std::endl;
    std::cout << std::endl;
    std::cout << COLOR_YELLOW << "按 Ctrl+C 停止服务器" << COLOR_RESET << std::endl;
    std::cout << "───────────────────────────────────────────────" << std::endl;

    if (!srv.listen("0.0.0.0", port)) {
        std::cerr << COLOR_RED << "[错误] 无法绑定端口 " << port << COLOR_RESET << std::endl;
        fpga_release();
        restore_stderr();
        return 1;
    }

    // 正常退出清理
    std::cout << "\n" << COLOR_YELLOW << "服务器已停止" << COLOR_RESET << std::endl;
    std::cout << "  总请求数: " << g_total_requests << std::endl;
    std::cout << "  总检测数: " << g_total_detections << std::endl;
    std::cout << "  总检测目标: " << g_total_objects_detected << std::endl;
    if (g_total_detections > 0) {
        std::cout << "  平均推理时间: " << std::fixed << std::setprecision(2)
                  << (g_total_inference_time / g_total_detections) << " ms" << std::endl;
    }

    fpga_release();
    restore_stderr();
    return 0;
}
