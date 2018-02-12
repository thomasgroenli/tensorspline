TF_INC=-I/home/thomg/miniconda3/lib/python3.6/site-packages/tensorflow/include
TF_NSYNC=$(TF_INC)/external/nsync/public
TF_LIB=-L/home/thomg/miniconda3/lib/python3.6/site-packages/tensorflow


FLAGS=-O2 -D_GLIBCXX_USE_CXX11_ABI=0


all: build test Makefile

build: splines.so


splinegrid_gpu.cu.a: splinegrid_gpu.cu.cc splines.h
	nvcc -std=c++11 --expt-relaxed-constexpr --gpu-architecture=sm_52 -c -o splinegrid_gpu.cu.a splinegrid_gpu.cu.cc $(TF_INC) $(TF_NSYNC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

splinegrid_gradient_gpu.cu.a: splinegrid_gradient_gpu.cu.cc splines.h
	nvcc -std=c++11 --expt-relaxed-constexpr --gpu-architecture=sm_52 -c -o splinegrid_gradient_gpu.cu.a splinegrid_gradient_gpu.cu.cc $(TF_INC) $(TF_NSYNC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


splinegrid_cpu.a: splinegrid_cpu.cc splines.h
	g++ -std=c++11 splinegrid_cpu.cc -c -o splinegrid_cpu.a -fPIC $(TF_INC) $(TF_NSYNC) $(TF_LIB) -ltensorflow_framework $(FLAGS)

splinegrid_gradient_cpu.a: splinegrid_gradient_cpu.cc splines.h
	g++ -std=c++11 splinegrid_gradient_cpu.cc -c -o splinegrid_gradient_cpu.a -fPIC $(TF_INC) $(TF_NSYNC) $(TF_LIB) -ltensorflow_framework $(FLAGS)

splines.so: splines.cc splines.h splinegrid_cpu.a splinegrid_gradient_cpu.a splinegrid_gpu.cu.a splinegrid_gradient_gpu.cu.a
	g++ -std=c++11 -shared splines.cc splinegrid_cpu.a splinegrid_gradient_cpu.a splinegrid_gpu.cu.a splinegrid_gradient_gpu.cu.a -o splines.so -fPIC $(TF_INC) $(TF_NSYNC) $(TF_LIB) -ltensorflow_framework $(FLAGS)




test: splines.so
	python SplineTest.py

