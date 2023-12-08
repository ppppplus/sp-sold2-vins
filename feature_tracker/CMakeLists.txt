cmake_minimum_required(VERSION 3.0.2)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -g")


# set(cv_bridge_DIR "/home/nvidia/cv_bridge_ws/devel/share/cv_bridge/cmake")
find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
    std_msgs
    sensor_msgs
    camera_model
    message_filters
    message_generation
    message_runtime
    cv_bridge
    )

find_package(cv_bridge REQUIRED)
find_package(NumCpp 2.12.0 REQUIRED)
# Set(OpenCV_DIR "/home/nvidia/package/opencv-3.2.0/build")
find_package(OpenCV)
find_package(Eigen3  REQUIRED)

add_message_files(
    FILES
    Featuremap.msg
)
generate_messages(
    DEPENDENCIES 
    std_msgs
)

catkin_package(CATKIN_DEPENDS
message_runtime)

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
#   /home/plus/plvins_ws/src/PL-VINS/feature_tracker/src/line_descriptor/include # find LSD 
)

# file(GLOB_RECURSE OLD_OPENCV "/usr/lib/aarch64-linux-gnu/libopencv*")
#file(GLOB_RECURSE OLD_CV_BRIDGE "/opt/ros/noetic/lib/libcv_bridge.so")
# list(REMOVE_ITEM catkin_LIBRARIES ${OLD_OPENCV})
#list(REMOVE_ITEM catkin_LIBRARIES ${OLD_CV_BRIDGE})

# add_executable(postprocess
#     src/postprocess_node.cpp
#     src/postprocess_parameters.cpp
#     src/postprocess.cpp
#     # src/postprocess.h
#     )
# add_dependencies(postprocess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(postprocess ${catkin_LIBRARIES} ${OpenCV_LIBS} NumCpp::NumCpp)

add_executable(origin_pointfeature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )
# # add_executable(feature_tracker
#     # src/feature_tracker_node.py
#     # src/parameters.cpp
#     # src/feature_tracker.py
#     # )
target_link_libraries(origin_pointfeature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${cv_bridge_LIBS} ${GPERFTOOLS_PROFILER})

# line detect and match
# add_executable(origin_linefeature_tracker    
#     src/origin/linefeature_tracker_node.cpp
#     src/origin/parameters.cpp
#     src/origin/linefeature_tracker.cpp
#     src/origin/tic_toc.h
# 
#     src/line_descriptor/src/binary_descriptor_custom.cpp
#     src/line_descriptor/src/binary_descriptor_matcher.cpp
#     src/line_descriptor/src/bitarray_custom.hpp
#     src/line_descriptor/src/bitops_custom.hpp
#     src/line_descriptor/src/draw_custom.cpp
#     src/line_descriptor/src/LSDDetector_custom.cpp
#     src/line_descriptor/src/precomp_custom.hpp
#     src/line_descriptor/src/types_custom.hpp
# )
# /home/healer/PL-VINS/PLVINS/PL-VINS/feature_tracker
# target_link_libraries(origin_linefeature_tracker 
# ${catkin_LIBRARIES} 
# ${OpenCV_LIBS}
# ${cv_bridge_LIBS}
# /home/nnplvio_ws/src/nn-pl-vio/feature_tracker/src/line_descriptor/lib/liblinedesc.so  # line feature
# )
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "    cv_bridge include path: ${cv_bridge_INCLUDE_DIRS}")
#message(STATUS "    LIBRARIES: ${OpenCV_LIBRARIES}")
#message(STATUS "    cv_bridge_LIBRARIES: ${cv_bridge_LIBRARIES}")
#message(STATUS "    catkin_LIBRARIES: ${catkin_LIBRARIES}")