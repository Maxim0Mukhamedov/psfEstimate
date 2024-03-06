cmake_minimum_required(VERSION 3.27)
project(psfEstimate)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)

add_executable(psfEstimate main.cpp)

target_link_libraries(psfEstimate PRIVATE opencv_ml opencv_dnn opencv_core opencv_flann opencv_imgcodecs opencv_highgui)