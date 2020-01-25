TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


CUDALIB=/usr/local/cuda-9.2/lib64/

nvcc -std=c++11 -c -o backproject_op_gpu.cu.o backproject_op_gpu.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o backproject.so backproject_op.cc \
  backproject_op_gpu.cu.o ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC -lcudart  -L${CUDALIB} ${TF_LFLAGS[@]}

