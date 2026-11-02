#ifndef OPL_HUMAN_VISION__HAILO_MODEL_HPP_
#define OPL_HUMAN_VISION__HAILO_MODEL_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "opl_human_vision/hailo_tensor.hpp"

namespace hailort {
class VDevice;
class InferModel;
class ConfiguredInferModel;
}  // namespace hailort

namespace opl_human_vision {

struct HailoModelOptions {
  // Open the VDevice through hailort_service so several processes (e.g. this
  // stack plus another Hailo app) can share the single Hailo-8L. Costs an
  // extra buffer copy; leave false when everything runs in one container.
  bool multi_process_service{false};
  // Scheduler priority 0..31 (HailoRT default 16). Higher wins when several
  // models are ready at the same time. <0 keeps the default.
  int scheduler_priority{-1};
  // Dequantise every non-NMS output on the host. Leave false to keep native
  // uint8/uint16 outputs and dequantise lazily via TensorView::at().
  bool float_outputs{false};
  // On-chip NMS overrides (ignored for models without NMS). <0 keeps HEF value.
  float nms_score_threshold{-1.0f};
  float nms_iou_threshold{-1.0f};
};

// One HEF configured on the process-wide shared Hailo VDevice. All models in
// the process share that VDevice, and the HailoRT round-robin scheduler
// time-slices the accelerator between them.
//
// Thread safety: run() is serialised internally, but input()/output() buffers
// are owned by the instance, so one HailoModel should be driven by one caller
// at a time (fill input -> run -> read outputs).
class HailoModel {
public:
  static std::unique_ptr<HailoModel> create(
    const std::string & hef_path, const HailoModelOptions & options, std::string & error);

  // Process-wide shared instance per HEF path. Lets e.g. the tracker and the
  // recognizer use ONE configured re-ID network instead of two, which saves a
  // network-group switch on the time-shared Hailo-8L. Callers must hold
  // useLock() across fill-input -> run() -> read-outputs.
  static std::shared_ptr<HailoModel> shared(
    const std::string & hef_path, const HailoModelOptions & options, std::string & error);

  std::unique_lock<std::mutex> useLock() {return std::unique_lock<std::mutex>(use_mutex_);}

  ~HailoModel();
  HailoModel(const HailoModel &) = delete;
  HailoModel & operator=(const HailoModel &) = delete;

  const std::string & path() const {return hef_path_;}
  const std::string & name() const {return name_;}

  int inputWidth() const {return input_w_;}
  int inputHeight() const {return input_h_;}
  int inputChannels() const {return input_c_;}

  // uint8 NHWC RGB input buffer wrapped as a cv::Mat (no copy).
  cv::Mat inputMat() {return input_mat_;}

  size_t numOutputs() const {return outputs_.size();}
  const TensorView & output(size_t i) const {return outputs_[i];}
  const std::vector<TensorView> & outputs() const {return outputs_;}

  // Raw NMS-by-class buffer and its geometry (only for models with on-chip NMS).
  bool hasNms() const {return nms_index_ >= 0;}
  const float * nmsData() const;
  size_t nmsFloats() const {return nms_index_ < 0 ? 0 : outputs_[static_cast<size_t>(nms_index_)].size();}
  int nmsNumClasses() const {return nms_classes_;}
  int nmsMaxBoxesPerClass() const {return nms_max_boxes_;}

  bool run(std::string & error, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

private:
  HailoModel() = default;

  struct AlignedDeleter {void operator()(uint8_t * p) const;};
  using AlignedBuffer = std::unique_ptr<uint8_t, AlignedDeleter>;
  static AlignedBuffer allocate(size_t bytes);

  std::string hef_path_;
  std::string name_;

  // Declaration order matters: the VDevice must outlive the configured model.
  std::shared_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> infer_model_;
  std::unique_ptr<hailort::ConfiguredInferModel> configured_;
  struct BindingsHolder;
  std::unique_ptr<BindingsHolder> bindings_;

  AlignedBuffer input_buffer_;
  size_t input_bytes_{0};
  cv::Mat input_mat_;
  int input_w_{0};
  int input_h_{0};
  int input_c_{0};

  std::vector<AlignedBuffer> output_buffers_;
  std::vector<TensorView> outputs_;
  int nms_index_{-1};
  int nms_classes_{0};
  int nms_max_boxes_{0};

  std::mutex run_mutex_;
  std::mutex use_mutex_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__HAILO_MODEL_HPP_
