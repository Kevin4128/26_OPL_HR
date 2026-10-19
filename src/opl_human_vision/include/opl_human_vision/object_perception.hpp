#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace opl_human_vision::objects {
inline const std::vector<std::string> labels = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

struct Detection {cv::Rect box; int class_id; float confidence;};
struct Letterbox {float scale; int left; int top;};
inline Letterbox letterbox(const cv::Mat &image, cv::Mat &blob) {
  const float scale = std::min(640.0f / image.cols, 640.0f / image.rows);
  const int w = std::lround(image.cols * scale), h = std::lround(image.rows * scale);
  cv::Mat padded(640, 640, CV_8UC3, cv::Scalar(114, 114, 114)), resized;
  cv::resize(image, resized, cv::Size(w, h));
  const int left = (640-w)/2, top = (640-h)/2;
  resized.copyTo(padded(cv::Rect(left, top, w, h)));
  cv::dnn::blobFromImage(padded, blob, 1.0/255, cv::Size(), cv::Scalar(), true, false);
  return {scale, left, top};
}
inline std::vector<Detection> decode(const float *data, int channels, int anchors,
    cv::Size size, Letterbox pad, float confidence, float nms) {
  if (!data || channels != 84 || anchors != 8400 || pad.scale <= 0)
    throw std::invalid_argument("Expected YOLO11 COCO float32 [1,84,8400]");
  std::vector<Detection> result;
  std::vector<std::vector<cv::Rect>> boxes(80);
  std::vector<std::vector<float>> scores(80);
  for (int i=0; i<anchors; ++i) {
    int best=0;
    float score=data[4*anchors+i];
    for (int c=1; c<80; ++c) {
      const float value=data[(c+4)*anchors+i];
      if (value>score) {score=value; best=c;}
    }
    if (!std::isfinite(score) || score<confidence || score>1) continue;
    const float x=data[i], y=data[anchors+i], w=data[2*anchors+i], h=data[3*anchors+i];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h) ||
        w<=0 || h<=0 || x<0 || y<0 || x>640 || y>640 || w>1280 || h>1280) continue;
    const int x1=std::clamp(static_cast<int>(std::floor((x-w/2-pad.left)/pad.scale)),0,size.width);
    const int y1=std::clamp(static_cast<int>(std::floor((y-h/2-pad.top)/pad.scale)),0,size.height);
    const int x2=std::clamp(static_cast<int>(std::ceil((x+w/2-pad.left)/pad.scale)),0,size.width);
    const int y2=std::clamp(static_cast<int>(std::ceil((y+h/2-pad.top)/pad.scale)),0,size.height);
    if (x2<=x1 || y2<=y1) continue;
    boxes[best].emplace_back(x1,y1,x2-x1,y2-y1); scores[best].push_back(score);
  }
  for (int cls=0; cls<80; ++cls) {
    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes[cls],scores[cls],confidence,nms,kept);
    for (int i:kept) result.push_back({boxes[cls][i],cls,scores[cls][i]});
  }
  return result;
}
inline float median(std::vector<float> values) {
  auto mid=values.begin()+values.size()/2;
  std::nth_element(values.begin(),mid,values.end());
  return *mid;
}
struct Surface {
  bool valid=false;
  std::string status="missing_depth";
  cv::Point3d point{};
  cv::Mat mask;
  unsigned pixels=0;
  float mad=0;
};
inline Surface surface(const cv::Mat &depth, cv::Rect box, double fx, double fy,
    double cx, double cy, double min_depth, double max_depth) {
  Surface out;
  if (!(std::isfinite(fx)&&std::isfinite(fy)&&std::isfinite(cx)&&std::isfinite(cy)) || fx<=0 || fy<=0) {
    out.status="invalid_calibration"; return out;
  }
  if (depth.empty()) return out;
  if (depth.type()!=CV_16UC1 && depth.type()!=CV_32FC1) {
    out.status="unsupported_depth_encoding"; return out;
  }
  box &= cv::Rect(0,0,depth.cols,depth.rows);
  if (box.area()<16) {out.status="insufficient_depth"; return out;}
  const cv::Rect inner(box.x+box.width/10,box.y+box.height/10,
    box.width-2*(box.width/10),box.height-2*(box.height/10));
  const cv::Rect center(box.x+box.width/3,box.y+box.height/3,
    std::max(1,box.width/3),std::max(1,box.height/3));
  struct Sample {float z; int x,y;};
  std::vector<Sample> samples;
  for (int y=inner.y;y<inner.y+inner.height;++y) for (int x=inner.x;x<inner.x+inner.width;++x) {
    const float z=depth.type()==CV_16UC1?depth.at<uint16_t>(y,x)*0.001f:depth.at<float>(y,x);
    if (std::isfinite(z)&&z>=min_depth&&z<=max_depth) samples.push_back({z,x,y});
  }
  if (samples.size()<10) {out.status="insufficient_depth"; return out;}
  std::sort(samples.begin(),samples.end(),[](auto &a,auto &b){return a.z<b.z;});
  struct Cluster {size_t begin,end; unsigned center_count;};
  std::vector<Cluster> clusters;
  size_t start=0;
  for (size_t i=1;i<=samples.size();++i) {
    // Limit total cluster width too: a continuous background ramp is not one surface.
    if (i<samples.size() && samples[i].z-samples[i-1].z<0.04f &&
        samples[i].z-samples[start].z<0.12f+0.03f*samples[start].z) continue;
    unsigned central=0;
    for (size_t j=start;j<i;++j) if (center.contains({samples[j].x,samples[j].y})) ++central;
    if (i-start>=std::max<size_t>(10,inner.area()/20) && central>=3)
      clusters.push_back({start,i,central});
    start=i;
  }
  if (clusters.empty()) {out.status="no_supported_foreground"; return out;}
  const auto chosen=clusters.front();
  for (size_t i=1;i<clusters.size();++i) {
    if (clusters[i].center_count>=0.7*chosen.center_count) {
      out.status="ambiguous_depth"; return out;
    }
  }
  std::vector<float> zs;
  for (size_t i=chosen.begin;i<chosen.end;++i) zs.push_back(samples[i].z);
  const float z=median(zs);
  for (auto &v:zs) v=std::abs(v-z);
  out.mad=median(zs);
  if (out.mad>0.04f+0.01f*z) {out.status="depth_spread"; return out;}
  out.mask=cv::Mat::zeros(depth.size(),CV_8UC1);
  size_t representative=chosen.begin;
  double best=1e30;
  for (size_t i=chosen.begin;i<chosen.end;++i) {
    const auto &s=samples[i];
    out.mask.at<uint8_t>(s.y,s.x)=255;
    const double distance=std::pow((s.x-box.x-box.width/2.0)/box.width,2)+
      std::pow((s.y-box.y-box.height/2.0)/box.height,2)+std::pow((s.z-z)/0.05,2);
    if (distance<best) {best=distance; representative=i;}
  }
  const auto &s=samples[representative];
  out.point={(s.x-cx)*s.z/fx,(s.y-cy)*s.z/fy,s.z};
  if (!std::isfinite(out.point.x) || !std::isfinite(out.point.y) || !std::isfinite(out.point.z)) {
    out.mask.release(); out.status="nonfinite_projection"; return out;
  }
  out.pixels=chosen.end-chosen.begin; out.valid=true; out.status="valid_surface_point";
  return out;
}
struct Color {std::string name="unknown"; float confidence=0;};
inline Color color(const cv::Mat &image, cv::Rect box, const cv::Mat &foreground) {
  cv::Mat hsv; cv::cvtColor(image(box),hsv,cv::COLOR_BGR2HSV);
  const std::vector<std::string> names={"black","white","gray","red","orange","yellow","green","blue","purple"};
  std::vector<unsigned> votes(names.size(),0);
  unsigned total=0;
  for (int y=0;y<hsv.rows;++y) for (int x=0;x<hsv.cols;++x) {
    if (!foreground.empty()) {if (!foreground.at<uint8_t>(y+box.y,x+box.x)) continue;}
    else if (x<hsv.cols/5 || x>=hsv.cols-hsv.cols/5 || y<hsv.rows/5 || y>=hsv.rows-hsv.rows/5) continue;
    const auto p=hsv.at<cv::Vec3b>(y,x); const int h=p[0],s=p[1],v=p[2];
    int bin=v<45?0:s<40?(v>200?1:2):h<10||h>=170?3:h<23?4:h<35?5:h<85?6:h<135?7:8;
    ++votes[bin]; ++total;
  }
  if (total<10) return {};
  auto best=std::max_element(votes.begin(),votes.end());
  const float fraction=static_cast<float>(*best)/total;
  if (fraction<0.6f) return {"unknown",fraction};
  return {names[best-votes.begin()],fraction};
}
}  // namespace opl_human_vision::objects
