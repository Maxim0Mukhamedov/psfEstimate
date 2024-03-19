add_library(psfc psfc.cpp psfc.hpp)

find_package(OpenCV CONFIG REQUIRED)

target_link_libraries(psfc PUBLIC
        opencv_ml
        opencv_dnn
        opencv_core
        opencv_flann
        opencv_imgcodecs
        opencv_highgui
)
target_include_directories(psfc PUBLIC ".")