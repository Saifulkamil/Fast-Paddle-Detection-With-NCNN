project(paddle_detection_jni)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(paddle_detection_jni SHARED
    paddle_detection_jni.cpp
    ppdet_pico.cpp
)

target_link_libraries(paddle_detection_jni
    ncnn
    ${OpenCV_LIBS}
    jnigraphics
    log
    android
)
