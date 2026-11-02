#include "opl_human_vision/hailo_model.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>

#include <hailo/hailort.hpp>

namespace opl_human_vision {

namespace {

constexpr size_t kPageSize = 4096;
constexpr const char * kSharedGroupId = "opl_human_vision";

std::string statusText(hailo_status status)
{
  std::ostringstream os;
  os << hailo_get_status_message(status) << " (" << static_cast<int>(status) << ")";
  return os.str();
}

// One VDevice per process, shared by every model so the HailoRT scheduler can
// time-slice them. Released automatically once the last model is destroyed.
std::shared_ptr<hailort::VDevice> acquireVDevice(bool multi_process, std::string & error)
{
  static std::mutex mutex;
  static std::weak_ptr<hailort::VDevice> shared;
  static bool shared_is_multi_process = false;

  std::lock_guard<std::mutex> lock(mutex);
  if (auto existing = shared.lock()) {
    if (existing && multi_process != shared_is_multi_process) {
      std::cerr << "[HailoModel] VDevice already open with multi_process_service="
                << (shared_is_multi_process ? "true" : "false")
                << "; ignoring conflicting request." << std::endl;
    }
    return existing;
  }

  hailo_vdevice_params_t params;
  hailo_status status = hailo_init_vdevice_params(&params);
  if (status != HAILO_SUCCESS) {
    error = "hailo_init_vdevice_params failed: " + statusText(status);
    return nullptr;
  }
  params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
  if (multi_process) {
    params.multi_process_service = true;
    params.group_id = kSharedGroupId;
  }

  auto vdevice = hailort::VDevice::create_shared(params);
  if (!vdevice) {
    error = "Failed to open Hailo VDevice: " + statusText(vdevice.status()) +
      (multi_process ? " (is hailort.service running?)" :
      " (is another process holding /dev/hailo0? set hailo_multi_process:=true to share)");
    return nullptr;
  }
  auto device = vdevice.release();
  shared = device;
  shared_is_multi_process = multi_process;
  return device;
}

TensorDType toDType(hailo_format_type_t type)
{
  switch (type) {
    case HAILO_FORMAT_TYPE_UINT8: return TensorDType::kUInt8;
    case HAILO_FORMAT_TYPE_UINT16: return TensorDType::kUInt16;
    default: return TensorDType::kFloat32;
  }
}

}  // namespace

struct HailoModel::BindingsHolder {
  explicit BindingsHolder(hailort::ConfiguredInferModel::Bindings && b)
  : bindings(std::move(b)) {}
  hailort::ConfiguredInferModel::Bindings bindings;
};

void HailoModel::AlignedDeleter::operator()(uint8_t * p) const {std::free(p);}

HailoModel::AlignedBuffer HailoModel::allocate(size_t bytes)
{
  const size_t rounded = ((bytes + kPageSize - 1) / kPageSize) * kPageSize;
  void * p = std::aligned_alloc(kPageSize, std::max(rounded, kPageSize));
  if (p) {std::memset(p, 0, rounded);}
  return AlignedBuffer(static_cast<uint8_t *>(p));
}

HailoModel::~HailoModel()
{
  // Tear down in dependency order before the buffers are freed.
  if (configured_) {
    configured_->shutdown();
  }
  bindings_.reset();
  configured_.reset();
  infer_model_.reset();
  vdevice_.reset();
}

std::unique_ptr<HailoModel> HailoModel::create(
  const std::string & hef_path, const HailoModelOptions & options, std::string & error)
{
  std::unique_ptr<HailoModel> model(new HailoModel());
  model->hef_path_ = hef_path;

  model->vdevice_ = acquireVDevice(options.multi_process_service, error);
  if (!model->vdevice_) {return nullptr;}

  auto infer_model = model->vdevice_->create_infer_model(hef_path);
  if (!infer_model) {
    error = "Failed to load HEF '" + hef_path + "': " + statusText(infer_model.status());
    return nullptr;
  }
  model->infer_model_ = infer_model.release();
  auto & im = *model->infer_model_;

  if (im.inputs().size() != 1) {
    error = "Expected exactly one input in '" + hef_path + "', found " +
      std::to_string(im.inputs().size());
    return nullptr;
  }
  auto input = im.input();
  if (!input) {
    error = "Failed to query input: " + statusText(input.status());
    return nullptr;
  }
  input->set_format_type(HAILO_FORMAT_TYPE_UINT8);
  input->set_format_order(HAILO_FORMAT_ORDER_NHWC);
  const auto in_shape = input->shape();
  model->input_h_ = static_cast<int>(in_shape.height);
  model->input_w_ = static_cast<int>(in_shape.width);
  model->input_c_ = static_cast<int>(in_shape.features);
  if (model->input_c_ != 3) {
    error = "Only 3-channel inputs are supported; '" + hef_path + "' has " +
      std::to_string(model->input_c_);
    return nullptr;
  }

  // Configure output formats before configure().
  for (const auto & out_name : im.get_output_names()) {
    auto out = im.output(out_name);
    if (!out) {
      error = "Failed to query output '" + out_name + "': " + statusText(out.status());
      return nullptr;
    }
    if (out->is_nms()) {
      out->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
      out->set_format_order(HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS);
      if (options.nms_score_threshold >= 0.0f) {
        out->set_nms_score_threshold(options.nms_score_threshold);
      }
      if (options.nms_iou_threshold >= 0.0f) {
        out->set_nms_iou_threshold(options.nms_iou_threshold);
      }
      continue;
    }
    const auto order = out->format().order;
    if (order != HAILO_FORMAT_ORDER_NC && order != HAILO_FORMAT_ORDER_NHWC) {
      out->set_format_order(HAILO_FORMAT_ORDER_NHWC);
    }
    // Per-channel quantisation can't be expressed by a single scale/zp pair.
    if (options.float_outputs || out->get_quant_infos().size() != 1) {
      out->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }
  }

  auto configured = im.configure();
  if (!configured) {
    error = "Failed to configure '" + hef_path + "' on the Hailo device: " +
      statusText(configured.status());
    return nullptr;
  }
  model->configured_ = std::make_unique<hailort::ConfiguredInferModel>(configured.release());
  if (options.scheduler_priority >= 0) {
    const auto prio = static_cast<uint8_t>(
      std::min(options.scheduler_priority, static_cast<int>(HAILO_SCHEDULER_PRIORITY_MAX)));
    const auto status = model->configured_->set_scheduler_priority(prio);
    if (status != HAILO_SUCCESS) {
      std::cerr << "[HailoModel] set_scheduler_priority failed: " << statusText(status) << std::endl;
    }
  }

  auto bindings = model->configured_->create_bindings();
  if (!bindings) {
    error = "Failed to create bindings: " + statusText(bindings.status());
    return nullptr;
  }
  model->bindings_ = std::make_unique<BindingsHolder>(bindings.release());
  auto & b = model->bindings_->bindings;

  // Input buffer.
  model->input_bytes_ = input->get_frame_size();
  model->input_buffer_ = allocate(model->input_bytes_);
  if (!model->input_buffer_) {
    error = "Out of memory allocating input buffer";
    return nullptr;
  }
  model->input_mat_ = cv::Mat(
    model->input_h_, model->input_w_, CV_8UC3, model->input_buffer_.get());
  {
    auto in_stream = b.input();
    if (!in_stream) {
      error = "Failed to bind input: " + statusText(in_stream.status());
      return nullptr;
    }
    const auto status = in_stream->set_buffer(
      hailort::MemoryView(model->input_buffer_.get(), model->input_bytes_));
    if (status != HAILO_SUCCESS) {
      error = "Failed to set input buffer: " + statusText(status);
      return nullptr;
    }
  }

  // Output buffers (formats are final after configure()).
  for (const auto & out_name : im.get_output_names()) {
    auto out = im.output(out_name);
    const size_t bytes = out->get_frame_size();
    auto buffer = allocate(bytes);
    if (!buffer) {
      error = "Out of memory allocating output buffer";
      return nullptr;
    }
    auto out_stream = b.output(out_name);
    if (!out_stream) {
      error = "Failed to bind output '" + out_name + "': " + statusText(out_stream.status());
      return nullptr;
    }
    const auto status = out_stream->set_buffer(hailort::MemoryView(buffer.get(), bytes));
    if (status != HAILO_SUCCESS) {
      error = "Failed to set output buffer: " + statusText(status);
      return nullptr;
    }

    TensorView view;
    view.name = out_name;
    view.data = buffer.get();
    if (out->is_nms()) {
      auto nms_shape = out->get_nms_shape();
      if (!nms_shape) {
        error = "Failed to query NMS shape: " + statusText(nms_shape.status());
        return nullptr;
      }
      model->nms_index_ = static_cast<int>(model->outputs_.size());
      model->nms_classes_ = static_cast<int>(nms_shape->number_of_classes);
      model->nms_max_boxes_ = static_cast<int>(nms_shape->max_bboxes_per_class);
      view.dtype = TensorDType::kFloat32;
      view.channels = static_cast<int>(bytes / sizeof(float));
    } else {
      const auto shape = out->shape();
      const auto format = out->format();
      view.height = static_cast<int>(shape.height);
      view.width = static_cast<int>(shape.width);
      view.channels = static_cast<int>(shape.features);
      view.dtype = toDType(format.type);
      if (view.dtype != TensorDType::kFloat32) {
        const auto quant = out->get_quant_infos().front();
        view.scale = quant.qp_scale;
        view.zero_point = quant.qp_zp;
      }
    }
    model->outputs_.push_back(view);
    model->output_buffers_.push_back(std::move(buffer));
  }

  model->name_ = im.hef().get_network_groups_names().empty() ?
    hef_path : im.hef().get_network_groups_names().front();
  return model;
}

std::shared_ptr<HailoModel> HailoModel::shared(
  const std::string & hef_path, const HailoModelOptions & options, std::string & error)
{
  static std::mutex mutex;
  static std::map<std::string, std::weak_ptr<HailoModel>> registry;

  std::lock_guard<std::mutex> lock(mutex);
  if (auto existing = registry[hef_path].lock()) {
    return existing;
  }
  std::shared_ptr<HailoModel> model(create(hef_path, options, error).release());
  if (model) {
    registry[hef_path] = model;
  }
  return model;
}

const float * HailoModel::nmsData() const
{
  if (nms_index_ < 0) {return nullptr;}
  return static_cast<const float *>(outputs_[static_cast<size_t>(nms_index_)].data);
}

bool HailoModel::run(std::string & error, std::chrono::milliseconds timeout)
{
  std::lock_guard<std::mutex> lock(run_mutex_);
  const auto status = configured_->run(bindings_->bindings, timeout);
  if (status != HAILO_SUCCESS) {
    error = "Hailo inference failed for '" + name_ + "': " + statusText(status);
    return false;
  }
  return true;
}

}  // namespace opl_human_vision
