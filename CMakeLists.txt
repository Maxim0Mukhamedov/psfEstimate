cmake_minimum_required(VERSION 3.5)
project(psfEstimate)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)


add_subdirectory(psfc)
add_executable(synthData generateSynthData.cpp)
add_executable(psfEstimate plots.cpp
        bindings/imgui_impl_glfw.h
        bindings/imgui_impl_glfw.cpp
        bindings/imgui_impl_opengl3_loader.h
        bindings/imgui_impl_opengl3.cpp
        bindings/imgui_impl_opengl3.h
)
add_executable(generator generator.cpp
        bindings/imgui_impl_glfw.h
        bindings/imgui_impl_glfw.cpp
        bindings/imgui_impl_opengl3_loader.h
        bindings/imgui_impl_opengl3.cpp
        bindings/imgui_impl_opengl3.h
)

target_include_directories(psfEstimate PUBLIC bindings)

target_link_libraries(psfEstimate PUBLIC
        opencv_ml
        opencv_dnn
        opencv_core
        opencv_flann
        opencv_imgcodecs
        opencv_highgui
        glfw
        OpenGL::GL
        glad::glad
        implot::implot
        imgui::imgui
        psfc
)

target_link_libraries(synthData PUBLIC
        opencv_ml
        opencv_dnn
        opencv_core
        opencv_flann
        opencv_imgcodecs
        opencv_highgui
        psfc
)

target_include_directories(generator PUBLIC bindings)
target_link_libraries(generator PUBLIC
        opencv_ml
        opencv_dnn
        opencv_core
        opencv_flann
        opencv_imgcodecs
        opencv_highgui
        glfw
        OpenGL::GL
        glad::glad
        implot::implot
        imgui::imgui
        psfc
)


