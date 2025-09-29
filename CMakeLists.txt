# cmake_minimum_required(VERSION 2.8)
# project(convolution2d)

# #set(OpenCV_DIR /usr/local/share/OpenCV)
# find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)

# file(GLOB SOURCES "*.cu")

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGCXX)
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
# endif()

# if(CUDA_FOUND)
# 	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -std=c++11")
# 	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61")

# 	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
# 		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G;-g;-lineinfo;--ptxas-options=-v")
# 	else()
# 		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-g;-lineinfo;--ptxas-options=-v")
# 	endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# 	if(OpenCV_FOUND)
# 		CUDA_INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# 		include_directories(${OpenCV_INCLUDE_DIRS})

# 		CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})

# 		target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# 	endif(OpenCV_FOUND)
# endif(CUDA_FOUND)

project(cudasamples)

# compiler options, set c++11 (requires cmake 3.2 or higher)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11) 


# FIND LIBRARIES
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
# set(agc_ "agc.cu")
# cuda_add_executable(agc ${agc_})
# target_link_libraries(agc ${OpenCV_LIBS})


# set(histogram_ "histogram.cu")
# cuda_add_executable(histogram ${histogram_})
# target_link_libraries(histogram ${OpenCV_LIBS})

# set(bhep_ "bhep.cu")
# cuda_add_executable(bhep ${bhep_})
# target_link_libraries(bhep ${OpenCV_LIBS})

set(agc_vid_ "agc_video.cu")
cuda_add_executable(agc_vid ${agc_vid_})
target_link_libraries(agc_vid ${OpenCV_LIBS})

set(bhep_vid_ "bhep_video.cu")
cuda_add_executable(bhep_vid ${bhep_vid_})
target_link_libraries(bhep_vid ${OpenCV_LIBS})

# set(qhep_ "qhep.cu")
# cuda_add_executable(qhep ${qhep_})
# target_link_libraries(qhep ${OpenCV_LIBS})

set(qhep_vid_ "qhep_vid.cu")
cuda_add_executable(qhep_vid ${qhep_vid_})
target_link_libraries(qhep_vid ${OpenCV_LIBS})

# set(aieli_ "aieli.cu")
# cuda_add_executable(aieli ${aieli_})
# target_link_libraries(aieli ${OpenCV_LIBS})

set(aieli_vid_ "aieli_vid.cu")
cuda_add_executable(aieli_vid ${aieli_vid_})
target_link_libraries(aieli_vid ${OpenCV_LIBS})

set(clahedgc_ "clahedgc.cu")
cuda_add_executable(clahedgc ${clahedgc_})
target_link_libraries(clahedgc ${OpenCV_LIBS})

