#include "opl_human_vision/object_perception.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <iostream>
using namespace opl_human_vision::objects;
int main() {
  assert(labels.size()==80 && labels[39]=="bottle" && labels[41]=="cup" && labels[56]=="chair" && labels[73]=="book");
  cv::Mat image(480,640,CV_8UC3,cv::Scalar(0,0,255)),input(640,640,CV_8UC3);
  const auto pad=letterbox(image,input); assert(pad.scale==1 && pad.top==80 && pad.left==0);
  // Padding is grey 114, and the BGR frame is written as RGB for the HEF.
  assert(input.at<cv::Vec3b>(0,0)==cv::Vec3b(114,114,114));
  assert(input.at<cv::Vec3b>(320,320)==cv::Vec3b(255,0,0));
  // Hailo NMS-by-class: per class {count, count x [ymin,xmin,ymax,xmax,score]}.
  auto nms_buffer=[](std::vector<std::pair<int,std::vector<float>>> boxes) {
    std::vector<float> out;
    for (int c=0;c<80;++c) {
      std::vector<float> mine;
      for (auto &[cls,b]:boxes) if (cls==c) mine.insert(mine.end(),b.begin(),b.end());
      out.push_back(static_cast<float>(mine.size()/5)); out.insert(out.end(),mine.begin(),mine.end());
    }
    out.resize(out.size()+64,0.0f);  // trailing slack like the real max-size buffer
    return out;
  };
  const float n=1.0f/640.0f;
  auto output=nms_buffer({{39,{270*n,270*n,370*n,370*n,0.9f}},{41,{0.1f,0.1f,0.2f,0.2f,0.7f}},
    {41,{0.3f,0.3f,0.4f,0.4f,0.2f}}});
  auto detections=decodeHailoNms(output.data(),output.size(),80,640,640,image.size(),pad,0.35f);
  assert(detections.size()==2 && detections[0].class_id==39 && detections[1].class_id==41);
  assert(detections[0].box==cv::Rect(270,190,100,100));
  assert(std::abs(detections[0].confidence-0.9f)<1e-6f);
  bool rejected=false;
  try {decodeHailoNms(output.data(),3,80,640,640,image.size(),pad,0.35f);} catch(const std::invalid_argument &) {rejected=true;}
  assert(rejected);
  cv::Mat depth(100,100,CV_32FC1,cv::Scalar(3));
  depth(cv::Rect(35,10,30,80)).setTo(1);
  auto point=surface(depth,{0,0,100,100},100,100,50,50,0.2,10);
  assert(point.valid && std::abs(point.point.z-1)<1e-6 && cv::countNonZero(point.mask)>0);
  cv::Mat depth16; depth.convertTo(depth16,CV_16UC1,1000);
  assert(surface(depth16,{0,0,100,100},100,100,50,50,0.2,10).valid);
  auto invalid=surface(depth,{0,0,100,100},0,100,50,50,0.2,10);
  assert(!invalid.valid && invalid.status=="invalid_calibration");
  assert(!surface(depth,{0,0,40,100},1e-308,100,50,50,0.2,10).valid);
  depth.setTo(std::numeric_limits<float>::quiet_NaN());
  assert(!surface(depth,{0,0,100,100},100,100,50,50,0.2,10).valid);
  depth.setTo(3); depth(cv::Rect(0,0,50,100)).setTo(1);
  invalid=surface(depth,{0,0,100,100},100,100,50,50,0.2,10);
  assert(!invalid.valid && invalid.status=="ambiguous_depth");
  assert(color(image,{0,0,100,100},{}).name=="red");
  image(cv::Rect(0,0,50,100)).setTo(cv::Scalar(255,0,0));
  assert(color(image,{0,0,100,100},{}).name=="unknown");
  cv::Mat mask=cv::Mat::zeros(image.size(),CV_8UC1); mask(cv::Rect(0,0,50,100)).setTo(255);
  assert(color(image,{0,0,100,100},mask).name=="blue");
  std::cout << "Hailo NMS decoder, letterbox, HSV and foreground depth checks passed\n";
}
