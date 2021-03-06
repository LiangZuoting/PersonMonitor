#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <limits>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "paddle_api.h"
#include "httplib.h"

const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = { 1, 3, 320, 320 };
const std::vector<float> INPUT_MEAN = { 0.5f, 0.5f, 0.5f };
const std::vector<float> INPUT_STD = { 0.5f, 0.5f, 0.5f };
const float SCORE_THRESHOLD = 0.5f;
const int PERSON_CLASS_ID = 1;

inline int64_t get_current_us() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

void preprocess(cv::Mat &input_image, const std::vector<float> &input_mean,
	const std::vector<float> &input_std, int input_width,
	int input_height, float *input_data) {
	cv::Mat resize_image;
	cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0, 0);
	if (resize_image.channels() == 4) {
		cv::cvtColor(resize_image, resize_image, CV_BGRA2RGB);
	}
	cv::Mat norm_image;
	resize_image.convertTo(norm_image, CV_32FC3, 1 / 255.f);
	// NHWC->NCHW
	int image_size = input_height * input_width;
	const float *image_data = reinterpret_cast<const float *>(norm_image.data);
	float32x4_t vmean0 = vdupq_n_f32(input_mean[0]);
	float32x4_t vmean1 = vdupq_n_f32(input_mean[1]);
	float32x4_t vmean2 = vdupq_n_f32(input_mean[2]);
	float32x4_t vscale0 = vdupq_n_f32(1.0f / input_std[0]);
	float32x4_t vscale1 = vdupq_n_f32(1.0f / input_std[1]);
	float32x4_t vscale2 = vdupq_n_f32(1.0f / input_std[2]);
	float *input_data_c0 = input_data;
	float *input_data_c1 = input_data + image_size;
	float *input_data_c2 = input_data + image_size * 2;
	int i = 0;
	for (; i < image_size - 3; i += 4) {
		float32x4x3_t vin3 = vld3q_f32(image_data);
		float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
		float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
		float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
		float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
		float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
		float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
		vst1q_f32(input_data_c0, vs0);
		vst1q_f32(input_data_c1, vs1);
		vst1q_f32(input_data_c2, vs2);
		image_data += 12;
		input_data_c0 += 4;
		input_data_c1 += 4;
		input_data_c2 += 4;
	}
	for (; i < image_size; i++) {
		*(input_data_c0++) = (*(image_data++) - input_mean[0]) / input_std[0];
		*(input_data_c1++) = (*(image_data++) - input_mean[1]) / input_std[1];
		*(input_data_c2++) = (*(image_data++) - input_mean[2]) / input_std[2];
	}
}

bool process(cv::Mat &input_image,
	std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
	// preprocess
	std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
		std::move(predictor->GetInput(0)));
	input_tensor->Resize(INPUT_SHAPE);
	int input_width = INPUT_SHAPE[3];
	int input_height = INPUT_SHAPE[2];
	auto *input_data = input_tensor->mutable_data<float>();
	preprocess(input_image, INPUT_MEAN, INPUT_STD, input_width, input_height,
		input_data);
	// predict
	predictor->Run();
	// postprocess
	std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
		std::move(predictor->GetOutput(0)));
	const float *output_data = output_tensor->mutable_data<float>();
	int64_t output_size = 1;
	for (auto dim : output_tensor->shape()) {
		output_size *= dim;
	}
	bool has_person = false;
	for (int64_t i = 0; i < output_size; i += 6) {
		int class_id = static_cast<int>(output_data[i]);
		if (output_data[i + 1] < SCORE_THRESHOLD || class_id != PERSON_CLASS_ID) {
			continue;
		}
		// person found
		has_person = true;
		break;
	}
	return has_person;
}

//
// 07:00:00 to 18:59:59 is daytime.
static const int START_HOUR = 7;
static const int END_HOUR = 19;

std::pair<bool, tm *> is_day(const std::chrono::system_clock::time_point &tp)
{
	auto time = std::chrono::system_clock::to_time_t(tp);
	auto tm = localtime(&time);
	return { tm->tm_hour < END_HOUR && tm->tm_hour >= START_HOUR, tm };
}

// next start time
std::chrono::system_clock::time_point get_next_start_tp()
{
	auto now = std::chrono::system_clock::now();
	auto time = std::chrono::system_clock::to_time_t(now);
	auto tm = localtime(&time);
	tm->tm_hour = END_HOUR;
	tm->tm_min = 0;
	tm->tm_sec = 0;
	return std::chrono::system_clock::from_time_t(mktime(tm));
}

int main(int argc, char **argv) {
	if (argc < 2) {
		printf(
			"Usage: \n"
			"./PersonMonitor model_file_path");
		return -1;
	}

	std::string model_path = argv[1];

	paddle::lite_api::MobileConfig config;
	config.set_model_from_file(model_path);
	config.set_threads(CPU_THREAD_NUM);
	config.set_power_mode(CPU_POWER_MODE);

	std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor =
		paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);

	bool stop = false;
	std::mutex mat_mutex;
	cv::Mat latest_mat;
	std::condition_variable capture_cv;
	std::mutex capture_cv_mutex;
	std::thread capture_thread([&] {
		while (!stop)
		{
			cv::VideoCapture cap;
			// try to open camera each 5 sec until it is connected.
			while (!is_day(std::chrono::system_clock::now()).first && !stop && !cap.isOpened())
			{
				cap.open(0);
				if (!cap.isOpened())
				{
					std::this_thread::sleep_for(std::chrono::seconds(5));
					continue;
				}
				cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
				cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
				cap.set(cv::CAP_PROP_FPS, 4);
				printf("camera opened.\n");
			}
			while (!is_day(std::chrono::system_clock::now()).first && !stop)
			{
				std::lock_guard<std::mutex> lock(mat_mutex);
				cap >> latest_mat;
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
			if (cap.isOpened())
			{
				cap.release();
			}
			auto next_start_tp = get_next_start_tp();
			{
				std::unique_lock<std::mutex> lock(capture_cv_mutex);
				capture_cv.wait_until(lock, next_start_tp);
			}
		}
		});
	
	std::condition_variable detection_cv;
	std::mutex detection_cv_mutex;
	httplib::Client smartHome("http://192.168.3.27");
	smartHome.set_keep_alive(true);
	std::thread detection_thread([&] {
		while (!stop)
		{
			while (!is_day(std::chrono::system_clock::now()).first && !stop)
			{
				cv::Mat mat;
				{
					std::lock_guard<std::mutex> lock(mat_mutex);
					mat = latest_mat;
				}
				if (mat.empty())
				{
					continue;
				}
				bool has_person = process(mat, predictor);
				std::string json = R"({"from": "monitor", "protocol": "miot", "ip": "192.168.3.29", "siid": 2, "piid": 1, "value": )";
				json += has_person ? R"(true})" : R"(false})";
				smartHome.Post("/", json, "application/json");
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
			auto next_start_tp = get_next_start_tp();
			{
				std::unique_lock<std::mutex> lock(detection_cv_mutex);
				detection_cv.wait_until(lock, next_start_tp);
			}
		}
		});

	capture_thread.join();
	detection_thread.join();
	return 0;
}
