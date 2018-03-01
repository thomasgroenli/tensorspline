cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
set( CMAKE_CXX_FLAGS "-std=c++11 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")
find_package(CUDA REQUIRED)


list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=sm_52 -expt-relaxed-constexpr")


if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -DCOMPILER_MSVC")
endif(WIN32)





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

MESSAGE( STATUS "TF_LIBRARY:         " ${TF_LIBRARY} )

include_directories(${TF_INCLUDE_DIR})
include_directories(${TF_INCLUDE_DIR}/external/nsync/public)

set(TARGET splinegrid)
set(TARGET_GPU "${TARGET}_gpu")

set_source_files_properties(${splinegrid_gpu_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)


add_library(${TARGET} STATIC ${splinegrid_srcs})
cuda_add_library(${TARGET_GPU} STATIC ${splinegrid_gpu_srcs})

add_library(splines SHARED splines.cc)

if(WIN32)
file(COPY ${TF_LIBRARY}/python/pywrap_tensorflow_internal.lib
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pywrap)
set (PYWRAP ${CMAKE_CURRENT_BINARY_DIR}/pywrap/pywrap_tensorflow_internal.lib)

else(WIN32)
file(COPY ${TF_LIBRARY}/python/_pywrap_tensorflow_internal.so
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pywrap)
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/pywrap/_pywrap_tensorflow_internal.so libpywrap_tensorflow_internal.so)
set (PYWRAP ${CMAKE_CURRENT_BINARY_DIR}/pywrap/libpywrap_tensorflow_internal.so)
endif(WIN32)

target_link_libraries(splines ${TARGET} ${TARGET_GPU})
target_link_libraries(splines ${CUDA_LIBRARIES})
target_link_libraries(splines ${PYWRAP})

set_target_properties(splines PROPERTIES PREFIX "")
