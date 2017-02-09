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

def MCB_unit(data1,data2,N,in_dim,out_dim,IFFT=True):
#input data1,data2: flattened tensor,np.ndarray type, size N by in_dim
#for resnet: N = length*height*batch_size, in_dim = number of filters
#output Y: flattened tensor, size N by out_dim (real numbers)

    assert(in_dim > out_dim)
    in_shape = (N,in_dim)
    hash_shape = (1, in_dim)
    h1 = np.random.randint(0, out_dim, hash_shape)
    s1 = np.random.randint(0, 2, hash_shape)*2-np.ones(hash_shape)
    h2 = np.random.randint(0, out_dim, hash_shape)
    s2 = np.random.randint(0, 2, hash_shape)*2-np.ones(hash_shape)

    shape = [in_shape, hash_shape, hash_shape]

    #Step 1: count sketch
    sym = mx.sym.CountSketch(name='countsketch',out_dim = out_dim) 
    arr = [mx.nd.empty(shape[i]) for i in range(3)]
    arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
    arr[0][:] = X1                          #input x
    arr[1][:] = h1                           #hash h
    arr[2][:] = s1                           #hash s
    exe = sym.bind(mx.gpu(0), arr, arr_grad)
    exe.forward(is_train=False)
    output1 = exe.outputs[0].asnumpy()

    sym = mx.sym.CountSketch(name='countsketch',out_dim = out_dim) 
    arr = [mx.nd.empty(shape[i]) for i in range(3)]
    arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
    arr[0][:] = X2                          #input x
    arr[1][:] = h2                           #hash h
    arr[2][:] = s2                           #hash s
    exe = sym.bind(mx.gpu(0), arr, arr_grad)
    exe.forward(is_train=False)
    output2 = exe.outputs[0].asnumpy()

    #Step 2: FFT
    assert(output1.shape == output2.shape)
    fft_shape = output1.shape
    grad_req='write'

    sym = mx.sym.FFT(name='fft', compute_size = 128) 
    ctx_list = {'ctx': mx.gpu(0),'fft_data': fft_shape, 'type_dict': {'fft_data': np.float32}}
    exe = sym.simple_bind(grad_req=grad_req, **ctx_list)
    for arr, iarr in zip(exe.arg_arrays, [output1]):
        arr[:] = iarr.astype(arr.dtype)
    exe.forward(is_train=False)
    fft_output1 = exe.outputs[0]

    sym = mx.sym.FFT(name='fft', compute_size = 128) 
    ctx_list = {'ctx': mx.gpu(0),'fft_data': fft_shape, 'type_dict': {'fft_data': np.float32}}
    exe = sym.simple_bind(grad_req=grad_req, **ctx_list)
    for arr, iarr in zip(exe.arg_arrays, [output2]):
        arr[:] = iarr.astype(arr.dtype)      
    exe.forward(is_train=False)
    fft_output2 = exe.outputs[0]

    #Step 3: Elementwise multiplication
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = a * b

    y = c.bind(default_context(), args={'a': fft_output1, 'b' : fft_output2})
    y.forward()
    temp_Y = y.outputs[0].asnumpy()

    #Step 4: IFFT
    if IFFT:
        print('ifft:')
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
        return Y
    else:
        return temp_Y[:,0::2]






N = 5
in_dim = 20
out_dim = 6
in_shape = (N, in_dim)
ifft_flag = False

X1 = np.random.uniform(-10, 10, in_shape)
X2 = np.random.uniform(-10, 10, in_shape)
Y = MCB_unit(X1,X2,N,in_dim,out_dim,ifft_flag)
print('Y:')
print(Y)
