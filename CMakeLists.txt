cmake_minimum_required(VERSION 3.18)
project(cpu_gpu_image_enhancement LANGUAGES CXX CUDA)

# ---- Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Warnings (nice to have) ----
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- OpenCV ----
# On Windows, pass -DOpenCV_DIR="C:/opencv/build" (folder containing OpenCVConfig.cmake)
find_package(OpenCV 4 REQUIRED core imgproc imgcodecs highgui)

# ---- Output dirs (bin/ under build/) ----
foreach(OUT RUNTIME LIBRARY ARCHIVE)
  set(CMAKE_${OUT}_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ========== CUDA demo ==========
add_executable(rgb2gray ${SRC_DIR}/CUDA_rgbtograyscale.cu)
target_link_libraries(rgb2gray PRIVATE ${OpenCV_LIBS})
target_include_directories(rgb2gray PRIVATE ${OpenCV_INCLUDE_DIRS})

# Optional: pin GPU architectures (e.g., Ampere 86)
# set_property(TARGET rgb2gray PROPERTY CUDA_ARCHITECTURES 86)

# ========== CPU algorithms (build each as its own exe) ==========
set(ALGO_TARGETS
    bhep
    bhepDD
    bhepSO
    bhepUO
    bhepvid
)
foreach(tgt IN LISTS ALGO_TARGETS)
  add_executable(${tgt} ${SRC_DIR}/${tgt}.cpp)
  target_link_libraries(${tgt} PRIVATE ${OpenCV_LIBS})
  target_include_directories(${tgt} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()

# ---- Install (optional) ----
install(TARGETS rgb2gray ${ALGO_TARGETS} RUNTIME DESTINATION bin)

# ---- Helpful status ----
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
