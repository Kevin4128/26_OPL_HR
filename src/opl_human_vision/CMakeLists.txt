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
find_package(OpenSSL REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

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
ament_target_dependencies(human_tracker_node std_msgs rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces tf2_ros tf2_geometry_msgs)
rclcpp_components_register_node(human_tracker_node PLUGIN "opl_human_vision::HumanTrackerNode" EXECUTABLE human_tracker_node_exe)

# 3. Face & Re-ID Recognizer Component Library (Modularized)
add_library(face_recognizer_node SHARED
  src/face_recognizer_node.cpp
  src/face_engine.cpp
  src/identity_store.cpp
  src/scrfd_detector.cpp
  src/adaface_extractor.cpp
  src/osnet_extractor.cpp
)
target_link_libraries(face_recognizer_node ${TRT_LIBRARIES} ${OpenCV_LIBS} OpenSSL::Crypto)
get_filename_component(OPL_WORKSPACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_compile_definitions(face_recognizer_node PRIVATE OPL_IDENTITY_STORE_DEFAULT="${OPL_WORKSPACE_DIR}/.private/identities")
ament_target_dependencies(face_recognizer_node rclcpp rclcpp_components rclcpp_lifecycle sensor_msgs cv_bridge opl_interfaces)
rclcpp_components_register_node(face_recognizer_node PLUGIN "opl_human_vision::FaceRecognizerNode" EXECUTABLE face_recognizer_node_exe)

# ------------------------------------------------------------------------------
# Installation Rules
# ------------------------------------------------------------------------------
install(TARGETS human_detector_node human_tracker_node face_recognizer_node
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
  scripts/visualizer.py
  scripts/lifecycle_activator.py
  scripts/following_target.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export include directories and libraries for dependent packages
ament_export_include_directories(include)
ament_export_libraries(human_detector_node human_tracker_node face_recognizer_node)

if(BUILD_TESTING)
  enable_testing()
  add_test(NAME visualizer_label COMMAND /usr/bin/python3 -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/test -p test_visualizer_label.py)
  add_executable(test_identity_regressions test/test_identity_regressions.cpp)
  target_link_libraries(test_identity_regressions face_recognizer_node ${OpenCV_LIBS})
  ament_target_dependencies(test_identity_regressions opl_interfaces)
  target_compile_options(test_identity_regressions PRIVATE -UNDEBUG)
  foreach(scenario namespace geometry names)
    add_test(NAME identity_${scenario}_regression COMMAND test_identity_regressions ${scenario})
  endforeach()
  add_library(test_directory_sync_failure SHARED test/test_directory_sync_failure.cpp)
  target_link_libraries(test_directory_sync_failure ${CMAKE_DL_LIBS})
  add_executable(test_identity_commit_failure test/test_identity_commit_failure.cpp)
  target_link_libraries(test_identity_commit_failure face_recognizer_node ${OpenCV_LIBS})
  target_compile_options(test_identity_commit_failure PRIVATE -UNDEBUG)
  add_test(NAME identity_commit_failure COMMAND ${CMAKE_COMMAND} -E env
    LD_PRELOAD=$<TARGET_FILE:test_directory_sync_failure> OPL_TEST_FAIL_DIRECTORY_FSYNC=1
    $<TARGET_FILE:test_identity_commit_failure>)
  add_executable(test_identity_policy test/test_identity_policy.cpp)
  target_link_libraries(test_identity_policy face_recognizer_node ${OpenCV_LIBS})
  ament_target_dependencies(test_identity_policy opl_interfaces)
  target_compile_options(test_identity_policy PRIVATE -UNDEBUG)
  add_test(NAME identity_policy COMMAND test_identity_policy)
  add_executable(test_tracking_sequences test/test_tracking_sequences.cpp)
  target_link_libraries(test_tracking_sequences human_tracker_node ${OpenCV_LIBS})
  ament_target_dependencies(test_tracking_sequences opl_interfaces)
  target_compile_options(test_tracking_sequences PRIVATE -UNDEBUG)
  add_test(NAME tracking_sequences COMMAND test_tracking_sequences)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME camera_configuration COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_camera_configuration.py
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/test_detector.launch.py)
  add_executable(test_pose_nms test/test_pose_nms.cpp)
  target_link_libraries(test_pose_nms ${OpenCV_LIBS})
  target_compile_options(test_pose_nms PRIVATE -UNDEBUG)
  add_test(NAME pose_nms COMMAND test_pose_nms)
  add_executable(test_tracker_occlusion test/test_tracker_occlusion.cpp)
  target_link_libraries(test_tracker_occlusion human_tracker_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
  ament_target_dependencies(test_tracker_occlusion opl_interfaces)
  target_compile_options(test_tracker_occlusion PRIVATE -UNDEBUG)
  add_test(NAME tracker_occlusion COMMAND test_tracker_occlusion)
  add_executable(test_rgbd_observation test/test_rgbd_observation.cpp)
  target_link_libraries(test_rgbd_observation human_tracker_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
  ament_target_dependencies(test_rgbd_observation opl_interfaces cv_bridge tf2_ros tf2_geometry_msgs)
  target_compile_options(test_rgbd_observation PRIVATE -UNDEBUG)
  add_test(NAME rgbd_observation COMMAND test_rgbd_observation)
  add_executable(test_rgbd_tracker_node test/test_rgbd_tracker_node.cpp)
  target_link_libraries(test_rgbd_tracker_node human_tracker_node ${TRT_LIBRARIES} ${OpenCV_LIBS})
  ament_target_dependencies(test_rgbd_tracker_node rclcpp rclcpp_lifecycle opl_interfaces tf2_ros tf2_geometry_msgs)
  target_compile_options(test_rgbd_tracker_node PRIVATE -UNDEBUG)
  add_test(NAME rgbd_tracker_node COMMAND test_rgbd_tracker_node)
  set_tests_properties(rgbd_tracker_node PROPERTIES TIMEOUT 15)
endif()

ament_package()
