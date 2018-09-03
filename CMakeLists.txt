cmake_minimum_required(VERSION 3.5)
project("TensorSpline")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")

option(ENABLE_GPU "Enable GPU support" ON)
option(ENABLE_MULTITHREAD "Enable multithreaded CPU support" ON)


if(ENABLE_MULTITHREAD)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MULTITHREAD")
endif(ENABLE_MULTITHREAD)

if(ENABLE_GPU)
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "--gpu-architecture=sm_52 -expt-relaxed-constexpr")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU")
endif(ENABLE_GPU)



if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -DCOMPILER_MSVC")
endif(WIN32)


execute_process(COMMAND python -c "import sys,tensorflow; sys.stdout.write(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE_DIR)
execute_process(COMMAND python -c "import sys,tensorflow; sys.stdout.write(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIBRARY)

MESSAGE( STATUS "TF_LIBRARY:         " ${TF_LIBRARY} )

set(splines_srcs
	"src/splines.cc"
)

set(splinegrid_srcs
	"src/splinegrid_cpu.cc"
)
set(splinegrid_gpu_srcs
	"src/splinegrid_gpu.cu.cc"
)



include_directories(${TF_INCLUDE_DIR})
include_directories(${TF_INCLUDE_DIR}/external/nsync/public)
find_library(PYWRAP pywrap_tensorflow_internal PATHS ${TF_LIBRARY}/python)

set(TARGET splines)
add_library(${TARGET} SHARED ${splines_srcs})

set(TARGET_CPU "${TARGET}_cpu")
add_library(${TARGET_CPU} STATIC ${splinegrid_srcs})

target_link_libraries(${TARGET} ${TARGET_CPU} ${PYWRAP})


if(ENABLE_GPU)
set(TARGET_GPU "${TARGET}_gpu")
set_source_files_properties(${splinegrid_gpu_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_library(${TARGET_GPU} STATIC ${splinegrid_gpu_srcs})

target_link_libraries(${TARGET} ${CUDA_LIBRARIES} ${TARGET_GPU})
endif(ENABLE_GPU)

set_target_properties(${TARGET} PROPERTIES PREFIX "")

add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/setup.py
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/setup.py)

add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorspline/__init__.py
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/tensorspline/__init__.py)

add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/$<TARGET_FILE_NAME:${TARGET}>
                ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/tensorspline/lib/$<TARGET_FILE_NAME:${TARGET}>)
