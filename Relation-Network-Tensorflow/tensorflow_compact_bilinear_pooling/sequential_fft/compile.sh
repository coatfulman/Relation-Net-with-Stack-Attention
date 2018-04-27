TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
  
g++ -std=c++11 -shared -o ./build/sequential_batch_fft.so \
  sequential_batch_fft.cc \
  -I $TF_INC -fPIC \
  -lcudart -lcufft -L/usr/local/cuda/lib64


