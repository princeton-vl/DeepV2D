TF_CFLAGS="-I/home/zach/virtualenvs/tf2/local/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0"
TF_LFLAGS="-L/home/zach/virtualenvs/tf2/local/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -L/usr/local/cuda-9.2/lib64"

nvcc -std=c++11 -c -o backproject_op_gpu.cu.o backproject_op_gpu.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o backproject.so backproject_op.cc \
  backproject_op_gpu.cu.o ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC -lcudart ${TF_LFLAGS[@]}
