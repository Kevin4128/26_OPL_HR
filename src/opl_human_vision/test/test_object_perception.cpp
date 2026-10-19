#include "opl_human_vision/object_perception.hpp"
#include <cassert>
#include <iostream>
using namespace opl_human_vision::objects;
int main() {
  assert(labels.size()==80 && labels[39]=="bottle" && labels[41]=="cup" && labels[56]=="chair" && labels[73]=="book");
  cv::Mat image(480,640,CV_8UC3,cv::Scalar(0,0,255)),blob;
  const auto pad=letterbox(image,blob); assert(pad.scale==1 && pad.top==80 && pad.left==0);
  std::vector<float> output(84*8400,0);
  auto anchor=[&](int i,int c,float score) {
    output[i]=320; output[8400+i]=320; output[2*8400+i]=100; output[3*8400+i]=100;
    output[(4+c)*8400+i]=score;
  };
  anchor(0,39,0.9); anchor(1,39,0.8); anchor(2,41,0.7);
  auto detections=decode(output.data(),84,8400,image.size(),pad,0.35,0.5);
  assert(detections.size()==2 && detections[0].class_id==39 && detections[1].class_id==41);
  assert(detections[0].box==cv::Rect(270,190,100,100));
  bool rejected=false; try {decode(output.data(),56,8400,image.size(),pad,0.35,0.5);} catch(const std::invalid_argument &) {rejected=true;}
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
  std::cout << "Object decoder, classwise NMS, letterbox, HSV and foreground depth checks passed\n";
}
