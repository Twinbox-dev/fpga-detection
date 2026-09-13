#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "paddle_api.h"  // NOLINT
#include <stdio.h>
#include <sys/times.h>
#include <unistd.h>
#include "json.hpp"
#include "httplib.h"
#include "time.h"
#include <ctime>

using json = nlohmann::json;
using namespace httplib;
using namespace cv;
using namespace paddle::lite_api;  // NOLINT
using namespace std;

std::shared_ptr<paddle::lite_api::PaddlePredictor> g_predictor;
std::vector<string> class_names;
const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

// Object for storing all preprocessed data
struct ImageBlob {
  // image width and height
  std::vector<float> im_shape_;
  // Buffer for image data after preprocessing
  const float* im_data_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_;
  std::vector<float> mean_;
  std::vector<float> scale_;
};

// 加载缺陷列表
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

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

// 加载配置文件
std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleDetection lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "===End of PaddleDetection lite demo config===" << std::endl;
}


// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
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

// 推理结果处理
std::string visualize_result(
                        const float* data,
                        int count,
                        float thresh,
                        cv::Mat& image,
                        const std::vector<std::string> &class_names,
                        double time) {
  if (data == nullptr) {
    std::cerr << "[ERROR] data can not be nullptr\n";
    exit(1);
  }
  std::string out_put = "";
  int num = 0;
  json out;
  json res_all;
  std::vector<Object> rect_out;
  for (int iw = 0; iw < count; iw++) {
    json res;
    int oriw = image.cols;
    int orih = image.rows;
    if (data[1] > thresh) {
      num += 1;
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
        int x2 = x + w;
        int y2 = y + h;
        std::string class_name = class_names[obj.class_id];
        res["score"] = obj.prob;
        res["class_name"] = class_name;
        res["loc"] = {x, y, x2, y2};
        res["prediction_time"] = time;
        res_all.push_back(res);
      }
    }
    data += 6;
  }
  out["len"] = num;
  out["result"] = res_all;
  out_put = out.dump();
  return out_put;
}

// Load Model and create model predictor
std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file,
                                           int num_theads) {
   MobileConfig config;
   config.set_threads(num_theads);
   config.set_model_from_file(model_file);
   config.set_threads(CPU_THREAD_NUM);
   config.set_power_mode(CPU_POWER_MODE);
   std::shared_ptr<PaddlePredictor> predictor =
   CreatePaddlePredictor<MobileConfig>(config);
  return predictor;

}

// 图片信息处理
ImageBlob prepare_imgdata(const cv::Mat& img,
                          std::map<std::string,
                          std::string> config) {
  ImageBlob img_data;
  std::vector<int> target_size_;
  std::vector<std::string> size_str = split(config.at("Resize"), ",");
  transform(size_str.begin(), size_str.end(), back_inserter(target_size_),
            [](std::string const& s){return stoi(s);});
  int width = target_size_[0];
  int height = target_size_[1];
  img_data.im_shape_ = {
      static_cast<float>(target_size_[0]),
      static_cast<float>(target_size_[1])
  };

  img_data.scale_factor_ = {
    static_cast<float>(target_size_[0]) / static_cast<float>(img.rows),
    static_cast<float>(target_size_[1]) / static_cast<float>(img.cols)
  };
  std::vector<float> mean_;
  std::vector<float> scale_;
  std::vector<std::string> mean_str = split(config.at("mean"), ",");
  std::vector<std::string> std_str = split(config.at("std"), ",");
  transform(mean_str.begin(), mean_str.end(), back_inserter(mean_),
            [](std::string const& s){return stof(s);});
  transform(std_str.begin(), std_str.end(), back_inserter(scale_),
            [](std::string const& s){return stof(s);});
  img_data.mean_ = mean_;
  img_data.scale_ = scale_;
  return img_data;
}

// 预处理
void preprocess(const cv::Mat& img, const ImageBlob img_data, float* data) {
  cv::Mat rgb_img;
  cv::resize(
      img, rgb_img, cv::Size(img_data.im_shape_[0],img_data.im_shape_[1]),
      0.f, 0.f, cv::INTER_CUBIC);

  if(rgb_img.channels() == 4) {
    cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGRA2RGB);
  }
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1/*1 / 255.fi*/);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(
    dimg, data, int(img_data.im_shape_[0] * img_data.im_shape_[1]),
    img_data.mean_, img_data.scale_);
}

// 初始化模型
void init(std::map<std::string, std::string> config) {
  std::string model_file = config.at("model_file");
  auto predictor = LoadModel(model_file, stoi(config.at("num_threads")));
  g_predictor = predictor;
}

// 模型推理
std::string RunModel(std::map<std::string, std::string> config,
              cv::Mat &img) {
  clock_t start, start_run, t1, t2, t3;
  t1 = clock();
  std::string label_path = config.at("label_path");
  // Load Labels
  std::vector<std::string> class_names = LoadLabels(label_path);
  std::cout << "Load label Time: " << (double)(clock() - t1)/1000<<" ms" << std::endl;

  start = clock();
  auto img_data = prepare_imgdata(img, config);
  std::cout << "Read image Time: " << (double)(clock() - start)/1000<<" ms" << std::endl;

  // 1. Prepare input data from image
  // input 0
  t2 = clock();
  std::unique_ptr<Tensor> input_tensor0(std::move(g_predictor->GetInput(0)));
  input_tensor0->Resize({1, 2});
  auto* data0 = input_tensor0->mutable_data<float>();
  data0[0] = img_data.im_shape_[0];
  data0[1] = img_data.im_shape_[1];

  // input1
  std::unique_ptr<Tensor> input_tensor1(std::move(g_predictor->GetInput(1)));
  input_tensor1->Resize({1, 3, img_data.im_shape_[0], img_data.im_shape_[1]});
  auto* data1 = input_tensor1->mutable_data<float>();
  preprocess(img, img_data, data1);

  // input2
  std::unique_ptr<Tensor> input_tensor2(std::move(g_predictor->GetInput(2)));
  input_tensor2->Resize({1, 2});
  auto* data2 = input_tensor2->mutable_data<float>();
  data2[0] = img_data.scale_factor_[0];
  data2[1] = img_data.scale_factor_[1];
  std::cout << "Input Time: " << (double)(clock() - t2)/1000<<" ms" << std::endl;

  start_run = clock();
  g_predictor->Run();
  double prediction_time = (double)(clock() - start_run)/1000;
  std::cout << "prediction_time: " << prediction_time<<" ms" << std::endl;

  // 3. Get output and post process
  t3 = clock();
  std::unique_ptr<const Tensor> output_tensor(
      std::move(g_predictor->GetOutput(0)));
  const float* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t cnt = 1;
  for (auto& i : shape_out) {
    cnt *= i;
  }
  std::string out_put = visualize_result(
      outptr, static_cast<int>(cnt / 6), 0.3f, img, class_names, prediction_time);
  std::cout << "Output Time: " << (double)(clock() - t3)/1000<<" ms" << std::endl;
  return out_put;
}

std::string dump_headers(const Headers &headers) {
  std::string s;
  char buf[BUFSIZ];

  for (auto it = headers.begin(); it != headers.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
    s += buf;
  }

  return s;
}

// 日志打印
std::string log(const Request &req, const Response &res) {
  std::string s;
  char buf[BUFSIZ];

  s += "================================\n";

  snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
           req.version.c_str(), req.path.c_str());
  s += buf;

  std::string query;
  for (auto it = req.params.begin(); it != req.params.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%c%s=%s",
             (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
             x.second.c_str());
    query += buf;
  }
  snprintf(buf, sizeof(buf), "%s\n", query.c_str());
  s += buf;

  s += dump_headers(req.headers);

  s += "--------------------------------\n";

  snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
  s += buf;
  s += dump_headers(res.headers);
  s += "\n";

  if (!res.body.empty()) { s += res.body; }

  s += "\n";

  return s;
}

// html简单界面
const char *html = R"(

<form id="formElem">
  <input type="file" name="image_file" onchange="uploadImg(this) " accept="image/*">
  <input type="submit">
</form>
<div style="float:top;border:1px dashed;background:#F0F8FF">
<pre>REST API Request: curl -F image_file=@test.jpg http://{ip}:8080/predict</pre>
<pre>Response:{"len":2,"result":[{"class_name":"red","loc":[677,174,866,569],"score":0.972812831401825},{"class_name":"red","loc":[25,163,215,582],"score":0.9500260949134827}]}
</div>
<img id="predictImg" width="768px" style="float:left"/>
<pre id="result" style="float:left;margin-left:150px;margin-top:100px"></pre>
<script src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js"></script>
<script>
  formElem.onsubmit = async (e) => {
    e.preventDefault();
    let res = await fetch('/predict', {
      method: 'POST',
      body: new FormData(formElem)
    });
    var result = await res.text();
    result =  eval('(' + result + ')');
    $("#result").text(JSON.stringify(result, null, 4));
    console.log(result);
  };
  function uploadImg(obj) {
    var file = obj.files[0];
    var reader = new FileReader();
    reader.onload = function (e) {
      var img = document.getElementById("predictImg");
      img.src = e.target.result;
    }
    reader.readAsDataURL(file)
  }

</script>

)";


// 主程序
int main(int argc, char** argv) {
  std::string config_path = "ssd_detection_src/config_ppyolo_tiny.txt";
  std::string img_path = "images/915.jpg";
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);

  // load config
  auto config = LoadConfigTxt(config_path);
  PrintConfig(config);

  // 初始化模型并推理
  init(config);
  RunModel(config, img);

  // httplib使用
  Server svr;
  svr.new_task_queue = [] { return new ThreadPool(1); };
  svr.Get("/", [](const Request & /*req*/, Response &res) {
    res.set_content(html, "text/html");
  });
  svr.Post("/predict", [](const Request &req, Response &res) {
    time_t now = time(0);
    tm* local_time = localtime(&now);
    std::cout << asctime(local_time) << std::endl;
    clock_t start;
    start = clock();
	// 获取上传的图片
    auto image_file = req.get_file_value("image_file");
    cv::Mat img_decode;
	// 对获取的图片进行处理
    std::vector<uchar> data(image_file.content.begin(), image_file.content.end());
    img_decode = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
    std::cout << "Read image to cv time: " << (double)(clock() - start)/1000<<" ms" << std::endl;
	// 加载配置文件
    std::string config_path = "ssd_detection_src/config_ppyolo_tiny.txt";
    auto config = LoadConfigTxt(config_path);

    // 进行模型推理
    std::string out_put = RunModel(config, img_decode);
    res.set_content(out_put, "text/plain");
    std::cout << "Running All Time: " << (double)(clock() - start)/1000<<" ms" << std::endl;
  });
  svr.set_logger([](const Request &req, const Response &res) {
    printf("%s", log(req, res).c_str());
  });
  svr.listen("0.0.0.0", 8080);

  return 0;
}
