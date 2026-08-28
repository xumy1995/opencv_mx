cmake_minimum_required(VERSION 3.18)
project(opencv_mx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

find_path(CVCUDA_INCLUDE_DIR cvcuda/OpResize.h
  HINTS ENV CVCUDA_ROOT
  PATH_SUFFIXES include)
find_library(CVCUDA_LIBRARY NAMES cvcuda
  HINTS ENV CVCUDA_ROOT
  PATH_SUFFIXES lib lib64)
find_library(NVCV_TYPES_LIBRARY NAMES nvcv_types
  HINTS ENV CVCUDA_ROOT
  PATH_SUFFIXES lib lib64)

# Native MACA Runtime used for device allocation, copies and streams.
find_path(MX_RUNTIME_INCLUDE_DIR mcr/mc_runtime_api.h
  HINTS ${MACA_PATH} ENV MACA_PATH
  PATHS /opt/maca /opt/maca-3.8.2 /opt/maca-3.8.0
  PATH_SUFFIXES include)
find_library(MX_RUNTIME_LIBRARY NAMES mcruntime
  HINTS ${MACA_PATH} ENV MACA_PATH
  PATHS /opt/maca /opt/maca-3.8.2 /opt/maca-3.8.0
  PATH_SUFFIXES lib lib64)
find_library(MX_CUDA_BRIDGE_LIBRARY NAMES runtime_cu
  HINTS ${MACA_PATH} ENV MACA_PATH
  PATHS /opt/maca /opt/maca-3.8.2 /opt/maca-3.8.0
  PATH_SUFFIXES lib lib64)
find_path(MX_CUDA_BRIDGE_INCLUDE_DIR cuda_runtime.h
  HINTS ${MACA_PATH} ENV MACA_PATH
  PATHS /opt/maca /opt/maca-3.8.2 /opt/maca-3.8.0
  PATH_SUFFIXES tools/cu-bridge/include include)

foreach(var CVCUDA_INCLUDE_DIR CVCUDA_LIBRARY NVCV_TYPES_LIBRARY
            MX_RUNTIME_INCLUDE_DIR MX_RUNTIME_LIBRARY MX_CUDA_BRIDGE_INCLUDE_DIR
            MX_CUDA_BRIDGE_LIBRARY)
  if(NOT ${var})
    message(FATAL_ERROR "${var} not found; configure inside the MX CV-CUDA container and set CVCUDA_ROOT/MACA_PATH if needed")
  endif()
endforeach()

add_library(opencv_mx
  src/gpumat.cpp
  src/resize.cpp)
target_include_directories(opencv_mx
  PUBLIC
    include
    ${MX_RUNTIME_INCLUDE_DIR}
    ${MX_RUNTIME_INCLUDE_DIR}/mcr
  PRIVATE
    ${CVCUDA_INCLUDE_DIR}
    ${MX_CUDA_BRIDGE_INCLUDE_DIR})
target_link_libraries(opencv_mx
  PUBLIC ${OpenCV_LIBS} ${MX_RUNTIME_LIBRARY} ${MX_CUDA_BRIDGE_LIBRARY}
  PRIVATE ${CVCUDA_LIBRARY} ${NVCV_TYPES_LIBRARY})

add_executable(mx_resize_test tests/resize_test.cpp)
target_link_libraries(mx_resize_test PRIVATE opencv_mx)
add_test(NAME mx_resize_test COMMAND mx_resize_test)

add_executable(mx_resize_example examples/resize.cpp)
target_link_libraries(mx_resize_example PRIVATE opencv_mx)

enable_testing()
add_test(NAME mx_resize_smoke COMMAND mx_resize_example
  ${CMAKE_CURRENT_SOURCE_DIR}/testdata/input.jpg
  ${CMAKE_CURRENT_BINARY_DIR}/output.jpg)
