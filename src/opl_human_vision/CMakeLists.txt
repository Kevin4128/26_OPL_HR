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

# HailoRT (Raspberry Pi 5 + Hailo-8L). Installed by the hailort .deb; provides
# HailoRT::libhailort and the C++ InferModel API.
# (HailoRTConfig uses exact-version matching, so check the version separately.)
find_package(HailoRT REQUIRED)
if(HailoRT_VERSION VERSION_LESS 4.23)
  message(FATAL_ERROR "HailoRT >= 4.23 required (InferModel API), found ${HailoRT_VERSION}")
endif()
message(STATUS ">>> HailoRT ${HailoRT_VERSION} found: building for the Hailo accelerator.")
set(HAILO_LIBRARIES HailoRT::libhailort)

# Tune for the Pi 5's Cortex-A76 cores when building natively on aarch64.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a76" OPL_HAS_MCPU_A76)
  if(OPL_HAS_MCPU_A76)
    add_compile_options("$<$<CONFIG:Release>:-mcpu=cortex-a76>")
  endif()
endif()
add_compile_options("$<$<CONFIG:Release>:-O3>")

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# ------------------------------------------------------------------------------
# Component Libraries
# ------------------------------------------------------------------------------

# Independent object perception component; human algorithms remain separate.
# Shared HailoRT wrapper: one process-wide VDevice + per-HEF configured models.
add_library(opl_hailo SHARED src/hailo_model.cpp)
target_link_libraries(opl_hailo ${HAILO_LIBRARIES} ${OpenCV_LIBS})

add_library(object_detection_node SHARED src/object_detection_node.cpp)
target_link_libraries(object_detection_node opl_hailo ${OpenCV_LIBS})
ament_target_dependencies(object_detection_node rclcpp rclcpp_components rclcpp_lifecycle
  sensor_msgs cv_bridge opl_interfaces tf2_ros tf2_geometry_msgs diagnostic_msgs)
rclcpp_components_register_node(object_detection_node
  PLUGIN "opl_human_vision::ObjectDetectionNode" EXECUTABLE object_detection_node_exe
  EXECUTOR MultiThreadedExecutor)

# 1. Human Detector Component Library
add_library(human_detector_node SHARED
  src/human_detector_node.cpp
)
target_link_libraries(human_detector_node opl_hailo ${OpenCV_LIBS})
ament_target_dependencies(human_detector_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(human_detector_node PLUGIN "opl_human_vision::HumanDetectorNode" EXECUTABLE human_detector_node_exe)

# 2. Human Tracker Component Library
add_library(human_tracker_node SHARED
  src/human_tracker_node.cpp
  src/tracker_engine.cpp
  src/reid_extractor.cpp
)
target_link_libraries(human_tracker_node opl_hailo ${OpenCV_LIBS})
ament_target_dependencies(human_tracker_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(human_tracker_node PLUGIN "opl_human_vision::HumanTrackerNode" EXECUTABLE human_tracker_node_exe)

# 3. Face & Re-ID Recognizer Component Library (Modularized)
add_library(face_recognizer_node SHARED
  src/face_recognizer_node.cpp
  src/face_engine.cpp
  src/scrfd_detector.cpp
  src/face_embedder.cpp
  src/reid_extractor.cpp
)
target_link_libraries(face_recognizer_node opl_hailo ${OpenCV_LIBS})
ament_target_dependencies(face_recognizer_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(face_recognizer_node PLUGIN "opl_human_vision::FaceRecognizerNode" EXECUTABLE face_recognizer_node_exe)

# ------------------------------------------------------------------------------
# Installation Rules
# ------------------------------------------------------------------------------
install(TARGETS opl_hailo object_detection_node human_detector_node human_tracker_node face_recognizer_node
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

foreach(directory config models)
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
ament_export_libraries(opl_hailo human_detector_node human_tracker_node face_recognizer_node)

if(BUILD_TESTING)
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
  target_link_libraries(test_tracker_occlusion human_tracker_node ${OpenCV_LIBS})
  ament_target_dependencies(test_tracker_occlusion opl_interfaces)
  target_compile_options(test_tracker_occlusion PRIVATE -UNDEBUG)
  add_test(NAME tracker_occlusion COMMAND test_tracker_occlusion)

  # Pure decoder tests with synthetic tensors (no accelerator needed).
  add_executable(test_hailo_decoders test/test_hailo_decoders.cpp)
  target_link_libraries(test_hailo_decoders ${OpenCV_LIBS})
  target_compile_options(test_hailo_decoders PRIVATE -UNDEBUG)
  add_test(NAME hailo_decoders COMMAND test_hailo_decoders)

  # On-device smoke test; exits 77 (skipped) without /dev/hailo0, the HEFs,
  # or when another process holds the accelerator.
  add_executable(test_hailo_models test/test_hailo_models.cpp
    src/scrfd_detector.cpp src/face_embedder.cpp src/reid_extractor.cpp)
  target_link_libraries(test_hailo_models opl_hailo ${OpenCV_LIBS})
  ament_target_dependencies(test_hailo_models rclcpp)
  target_compile_options(test_hailo_models PRIVATE -UNDEBUG)
  add_test(NAME hailo_models COMMAND test_hailo_models
    ${CMAKE_CURRENT_SOURCE_DIR}/models/hailo8l /usr/local/hailo/resources/images)
  set_tests_properties(hailo_models PROPERTIES SKIP_RETURN_CODE 77)
endif()

ament_package()
