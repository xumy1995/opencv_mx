cmake_minimum_required(VERSION 3.18)
project(opencv_mx VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

find_path(CVCUDA_INCLUDE_DIR cvcuda/OpResize.h
  HINTS ${CVCUDA_ROOT} ENV CVCUDA_ROOT
  PATH_SUFFIXES include)
find_library(CVCUDA_LIBRARY NAMES cvcuda
  HINTS ${CVCUDA_ROOT} ENV CVCUDA_ROOT
  PATH_SUFFIXES lib lib64)
find_library(NVCV_TYPES_LIBRARY NAMES nvcv_types
  HINTS ${CVCUDA_ROOT} ENV CVCUDA_ROOT
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
  src/resize.cpp
  src/cvtcolor.cpp)
target_include_directories(opencv_mx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${MX_RUNTIME_INCLUDE_DIR}
    ${MX_RUNTIME_INCLUDE_DIR}/mcr
  PRIVATE
    ${CVCUDA_INCLUDE_DIR}
    ${MX_CUDA_BRIDGE_INCLUDE_DIR})
target_link_libraries(opencv_mx
  PUBLIC ${OpenCV_LIBS} ${MX_RUNTIME_LIBRARY} ${MX_CUDA_BRIDGE_LIBRARY}
  PRIVATE ${CVCUDA_LIBRARY} ${NVCV_TYPES_LIBRARY})

add_library(OpenCVMX::opencv_mx ALIAS opencv_mx)

option(BUILD_TESTS "Build MX backend tests" ON)
option(BUILD_EXAMPLES "Build MX backend examples" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

if(BUILD_TESTS)
  enable_testing()
  add_executable(mx_resize_test tests/resize_test.cpp)
  target_link_libraries(mx_resize_test PRIVATE opencv_mx)
  add_test(NAME mx_resize_test COMMAND mx_resize_test)
  set_tests_properties(mx_resize_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(mx_gpumat_test tests/gpumat_test.cpp)
  target_link_libraries(mx_gpumat_test PRIVATE opencv_mx)
  add_test(NAME mx_gpumat_test COMMAND mx_gpumat_test)
  set_tests_properties(mx_gpumat_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(mx_cvtcolor_test tests/cvtcolor_test.cpp)
  target_link_libraries(mx_cvtcolor_test PRIVATE opencv_mx)
  add_test(NAME mx_cvtcolor_test COMMAND mx_cvtcolor_test)
endif()

if(BUILD_EXAMPLES)
  add_executable(mx_resize_example examples/resize.cpp)
  target_link_libraries(mx_resize_example PRIVATE opencv_mx)
endif()

if(BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(opencv_mx_python MODULE python/module.cpp)
  target_link_libraries(opencv_mx_python PRIVATE opencv_mx)
  set_target_properties(opencv_mx_python PROPERTIES OUTPUT_NAME opencv_mx)
endif()

install(TARGETS opencv_mx EXPORT OpenCVMXTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT OpenCVMXTargets
        FILE OpenCVMXTargets.cmake
        NAMESPACE OpenCVMX::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenCVMX)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenCVMXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OpenCVMXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenCVMX)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/OpenCVMXConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/OpenCVMXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/OpenCVMXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenCVMX)
