import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
import mxnet as mx
import numpy as np
from mxnet.test_utils import check_consistency, set_default_context
from numpy.testing import assert_allclose
import time

set_default_context(mx.gpu(0))
del test_support_vector_machine_l1_svm
del test_support_vector_machine_l2_svm


'''
##############################
Tensor sketching:
Given input x \in R^n, h \in R^n, \in {1,...d}; s \in R^n, \in {+1, -1};
Output y \in R^d  = TS(x)
       y[h[i]] += s[i]x[i] i = 1,2,...n
##############################
Tensor sketching of an outer product of two vectors:
TS(<x1,x2>) = TS(x1) * TS(x2) = FFT^{-1}(FFT(TS(x1)) .\times FFT(TS(x2)))
##############################
Input: feature matrices X1, X2 \in R^{N \times in_dim}, N: number of samples, in_dim: feature dimension, 
       each row contains the two vectors of one outer product;
       index hash h \in R^n; 
       sign hash s \in R^n;
       out_dim d
Output: Y \in R^{N \times d} = TS(<X1,X2>)
       
'''

N = 5
in_dim = 20
out_dim = 6
assert(in_dim > out_dim)

in_shape = (N, in_dim)
hash_shape = (1, in_dim)

X1 = np.random.uniform(-10, 10, in_shape)
X2 = np.random.uniform(-10, 10, in_shape)
h = np.random.randint(0, out_dim, hash_shape)
s = np.random.randint(0, 2, hash_shape)*2-np.ones(hash_shape)

shape = [in_shape, hash_shape, hash_shape]

#    TS(X1)
print('TS(X1):')
sym = mx.sym.CountSketch(name='countsketch',out_dim = out_dim) 

arr = [mx.nd.empty(shape[i]) for i in range(3)]
arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
arr[0][:] = X1                          #input x
arr[1][:] = h                           #hash h
arr[2][:] = s                           #hash s

exe = sym.bind(mx.gpu(0), arr, arr_grad)
exe.forward(is_train=False)
output1 = exe.outputs[0].asnumpy()


#    TS(X2)
print('TS(X2):')
sym = mx.sym.CountSketch(name='countsketch',out_dim = out_dim) 

arr = [mx.nd.empty(shape[i]) for i in range(3)]
arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
arr[0][:] = X2                          #input x
arr[1][:] = h                           #hash h
arr[2][:] = s                           #hash s

exe = sym.bind(mx.gpu(0), arr, arr_grad)
exe.forward(is_train=False)
output2 = exe.outputs[0].asnumpy()




assert(output1.shape == output2.shape)
fft_shape = output1.shape
grad_req='write'

#    FFT(TS(X1))   
sym = mx.sym.FFT(name='fft', compute_size = 128) 
ctx_list = {'ctx': mx.gpu(0),'fft_data': fft_shape, 'type_dict': {'fft_data': np.float32}}
exe = sym.simple_bind(grad_req=grad_req, **ctx_list)

for arr, iarr in zip(exe.arg_arrays, [output1]):
    arr[:] = iarr.astype(arr.dtype)

exe.forward(is_train=False)
fft_output1 = exe.outputs[0]

#    FFT(TS(X2))   
sym = mx.sym.FFT(name='fft', compute_size = 128) 
ctx_list = {'ctx': mx.gpu(0),'fft_data': fft_shape, 'type_dict': {'fft_data': np.float32}}
exe = sym.simple_bind(grad_req=grad_req, **ctx_list)

for arr, iarr in zip(exe.arg_arrays, [output2]):
    arr[:] = iarr.astype(arr.dtype)
      
exe.forward(is_train=False)
fft_output2 = exe.outputs[0]

#Elementwise multiplication
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a * b

y = c.bind(default_context(), args={'a': fft_output1, 'b' : fft_output2})
y.forward()
temp_Y = y.outputs[0].asnumpy()


print('temp_Y')
print(temp_Y)
assert(temp_Y.shape == (fft_shape[0],2*fft_shape[1]))
#    IFFT()
ifft_shape = temp_Y.shape

sym = mx.sym.IFFT(name='ifft', compute_size = 128) 
ctx_list = {'ctx': mx.gpu(0),'ifft_data': ifft_shape, 'type_dict': {'ifft_data': np.float32}}
exe = sym.simple_bind(grad_req=grad_req, **ctx_list)

for arr, iarr in zip(exe.arg_arrays, [temp_Y]):
    arr[:] = iarr.astype(arr.dtype)
      
exe.forward(is_train=False)
Y = exe.outputs[0].asnumpy()/fft_shape[1]
print('Y')
print(Y)
