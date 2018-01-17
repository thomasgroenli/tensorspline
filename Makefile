TF_INC=-I/home/thomg/miniconda3/lib/python3.6/site-packages/tensorflow/include
TF_NSYNC=$(TF_INC)/external/nsync/public
TF_LIB=-L/home/thomg/miniconda3/lib/python3.6/site-packages/tensorflow

FLAGS=-O2 -D_GLIBCXX_USE_CXX11_ABI=0

SplineGrid:
	g++ -std=c++11 -shared splines.cc -o splines.so -fPIC $(TF_INC) $(TF_NSYNC) $(TF_LIB) -ltensorflow_framework $(FLAGS)
	python SplineTest.py
