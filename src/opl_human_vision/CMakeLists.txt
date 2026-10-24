cmake_minimum_required(VERSION 3.18)
project(opl_human_vision)

# Single-config colcon builds otherwise default to unoptimized CPU code.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(opl_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

# Search TensorRT & CUDA paths
set(TRT_SEARCH_PATHS
  /usr/lib/x86_64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  /usr/local/cuda/lib64
  $ENV{TENSORRT_ROOT}/lib
)

set(TRT_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/x86_64-linux-gnu
  /usr/include/aarch64-linux-gnu
  /usr/local/cuda/include
  $ENV{TENSORRT_ROOT}/include
)

find_path(NVINFER_INCLUDE_DIR NvInfer.h HINTS ${TRT_INCLUDE_SEARCH_PATHS})
find_library(NVINFER_LIB nvinfer HINTS ${TRT_SEARCH_PATHS})
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TRT_SEARCH_PATHS})

if(NVINFER_LIB AND NVONNXPARSER_LIB AND NVINFER_INCLUDE_DIR)
  message(STATUS ">>> TensorRT Found! Compiling with GPU Acceleration.")
  add_compile_definitions(HAVE_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  set(TRT_LIBRARIES CUDA::cudart ${NVINFER_LIB} ${NVONNXPARSER_LIB})
  include_directories(${NVINFER_INCLUDE_DIR})
else()
  message(WARNING ">>> TensorRT NOT found. Compiling in Simulation/Laptop Mode.")
  set(TRT_LIBRARIES "")
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# Component Libraries
# ------------------------------------------------------------------------------

# Independent object perception component; human algorithms remain separate.
add_library(object_detection_node SHARED src/object_detection_node.cpp)
target_link_libraries(object_detection_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(object_detection_node rclcpp rclcpp_components rclcpp_lifecycle
  sensor_msgs cv_bridge opl_interfaces tf2_ros tf2_geometry_msgs diagnostic_msgs)
rclcpp_components_register_node(object_detection_node
  PLUGIN "opl_human_vision::ObjectDetectionNode" EXECUTABLE object_detection_node_exe
  EXECUTOR MultiThreadedExecutor)

# 1. Human Detector Component Library
add_library(human_detector_node SHARED
  src/human_detector_node.cpp
  src/trt_engine.cpp
)
target_link_libraries(human_detector_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(human_detector_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(human_detector_node PLUGIN "opl_human_vision::HumanDetectorNode" EXECUTABLE human_detector_node_exe)

# 2. Human Tracker Component Library
add_library(human_tracker_node SHARED
  src/human_tracker_node.cpp
  src/tracker_engine.cpp
  src/osnet_extractor.cpp
)
target_link_libraries(human_tracker_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(human_tracker_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(human_tracker_node PLUGIN "opl_human_vision::HumanTrackerNode" EXECUTABLE human_tracker_node_exe)

# 3. Face & Re-ID Recognizer Component Library (Modularized)
add_library(face_recognizer_node SHARED
  src/face_recognizer_node.cpp
  src/face_engine.cpp
  src/scrfd_detector.cpp
  src/adaface_extractor.cpp
  src/osnet_extractor.cpp
)
target_link_libraries(face_recognizer_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(face_recognizer_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(face_recognizer_node PLUGIN "opl_human_vision::FaceRecognizerNode" EXECUTABLE face_recognizer_node_exe)

# ------------------------------------------------------------------------------
# Installation Rules
# ------------------------------------------------------------------------------
install(TARGETS object_detection_node human_detector_node human_tracker_node face_recognizer_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

foreach(directory config models laptop_models)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${directory}")
    install(DIRECTORY ${directory} DESTINATION share/${PROJECT_NAME})
  endif()
endforeach()

install(PROGRAMS
  scripts/webcam_publisher.py
  scripts/object_visualizer.py
  scripts/visualizer.py
  scripts/lifecycle_activator.py
  scripts/following_target.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export include directories and libraries for dependent packages
ament_export_include_directories(include)
ament_export_libraries(human_detector_node human_tracker_node face_recognizer_node)

if(BUILD_TESTING)
  add_executable(test_tracker_rgb_node test/test_tracker_rgb_node.cpp)
  target_link_libraries(test_tracker_rgb_node human_tracker_node ${OpenCV_LIBS})
  ament_target_dependencies(test_tracker_rgb_node rclcpp rclcpp_lifecycle cv_bridge opl_interfaces)
  target_compile_options(test_tracker_rgb_node PRIVATE -UNDEBUG)
  add_test(NAME tracker_rgb_node COMMAND test_tracker_rgb_node)
  add_executable(test_rgb_buffer test/test_rgb_buffer.cpp)
  target_link_libraries(test_rgb_buffer ${OpenCV_LIBS})
  ament_target_dependencies(test_rgb_buffer rclcpp cv_bridge)
  target_compile_options(test_rgb_buffer PRIVATE -UNDEBUG)
  add_test(NAME rgb_buffer COMMAND test_rgb_buffer)
  # Exercise lifecycle/message logic without GPU models, using the same node sources.
  add_executable(test_recognizer_dropout test/test_recognizer_dropout.cpp
    src/face_recognizer_node.cpp src/face_engine.cpp src/scrfd_detector.cpp
    src/adaface_extractor.cpp src/osnet_extractor.cpp)
  target_link_libraries(test_recognizer_dropout ${OpenCV_LIBS})
  ament_target_dependencies(test_recognizer_dropout rclcpp rclcpp_components rclcpp_lifecycle
    sensor_msgs cv_bridge opl_interfaces)
  target_compile_options(test_recognizer_dropout PRIVATE -UHAVE_TENSORRT -UNDEBUG)
  add_test(NAME recognizer_dropout COMMAND test_recognizer_dropout)
  add_executable(test_identity_continuity test/test_identity_continuity.cpp)
  target_link_libraries(test_identity_continuity face_recognizer_node ${OpenCV_LIBS})
  ament_target_dependencies(test_identity_continuity opl_interfaces)
  target_compile_options(test_identity_continuity PRIVATE -UNDEBUG)
  add_test(NAME identity_continuity COMMAND test_identity_continuity)

  enable_testing()
  add_executable(test_object_perception test/test_object_perception.cpp)
  target_link_libraries(test_object_perception ${OpenCV_LIBS})
  target_compile_options(test_object_perception PRIVATE -UNDEBUG)
  add_test(NAME object_perception COMMAND test_object_perception)
  add_executable(test_pose_nms test/test_pose_nms.cpp)
  target_link_libraries(test_pose_nms ${OpenCV_LIBS})
  target_compile_options(test_pose_nms PRIVATE -UNDEBUG)
  add_test(NAME pose_nms COMMAND test_pose_nms)
  add_executable(test_tracker_occlusion test/test_tracker_occlusion.cpp)
  target_link_libraries(test_tracker_occlusion human_tracker_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
  ament_target_dependencies(test_tracker_occlusion opl_interfaces)
  target_compile_options(test_tracker_occlusion PRIVATE -UNDEBUG)
  add_test(NAME tracker_occlusion COMMAND test_tracker_occlusion)
endif()

ament_package()
