cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
set( CMAKE_CXX_FLAGS "-O2 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")
list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=sm_52 -expt-relaxed-constexpr")


if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -DCOMPILER_MSVC")
else(WIN32)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
endif(WIN32)
find_package(CUDA REQUIRED)




set(splinegrid_srcs
	"splinegrid_cpu.cc"
	"splinegrid_gradient_cpu.cc"
)
set(splinegrid_gpu_srcs
	"splinegrid_gpu.cu.cc"
	"splinegrid_gradient_gpu.cu.cc"
)

execute_process(COMMAND python -c "import sys,tensorflow; sys.stdout.write(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIR)
execute_process(COMMAND python -c "import sys,tensorflow; sys.stdout.write(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIBRARY)

MESSAGE( STATUS "CMAKE_BINARY_DIR:         " ${TF_INCLUDE_DIR} )

include_directories(${TF_INCLUDE_DIR})
include_directories(${TF_INCLUDE_DIR}/external/nsync/public)

set(TARGET splinegrid)
set(TARGET_GPU "${TARGET}_gpu")

set_source_files_properties(${splinegrid_gpu_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)


add_library(${TARGET} STATIC ${splinegrid_srcs})
cuda_add_library(${TARGET_GPU} STATIC ${splinegrid_gpu_srcs})

add_library(splines SHARED splines.cc)

if(WIN32)
set (PYWRAP ${TF_LIBRARY}/python/pywrap_tensorflow_internal.lib)
else(WIN32)
set (PYWRAP ${TF_LIBRARY}/python/libpywrap_tensorflow_internal.so)
endif(WIN32)

target_link_libraries(splines ${TARGET} ${TARGET_GPU})
target_link_libraries(splines ${CUDA_LIBRARIES})
target_link_libraries(splines ${PYWRAP})

set_target_properties(splines PROPERTIES PREFIX "")
