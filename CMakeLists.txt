cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
set( CMAKE_CXX_FLAGS "-std=c++11 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=sm_52 -expt-relaxed-constexpr")


if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -DCOMPILER_MSVC")
endif(WIN32)





set(splinegrid_srcs
	"src/splinegrid_cpu.cc"
)
set(splinegrid_gpu_srcs
	"src/splinegrid_gpu.cu.cc"
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

add_library(splines SHARED src/splines.cc)

find_library(PYWRAP pywrap_tensorflow_internal PATHS ${TF_LIBRARY}/python)

target_link_libraries(splines ${TARGET} ${TARGET_GPU})
target_link_libraries(splines ${CUDA_LIBRARIES})
target_link_libraries(splines ${PYWRAP})

set_target_properties(splines PROPERTIES PREFIX "")

add_custom_command(
        TARGET splines POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/setup.py
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/setup.py)

add_custom_command(
        TARGET splines POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorspline/__init__.py
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/tensorspline/__init__.py)


add_custom_command(
        TARGET splines POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/splines.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/tensorspline/lib/splines.dll)