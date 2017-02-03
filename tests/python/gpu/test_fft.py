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



print('test:')
#shape = (2,4,3,2)
shape = (3,2)
grad_req='write'
   
print("fft input:")
init = [np.random.normal(size=shape, scale=1.0)]
print(init)
    
sym = mx.sym.FFT(name='fft', compute_size = 128) 
ctx_list = [{'ctx': mx.gpu(0),'fft_data': shape, 'type_dict': {'fft_data': np.float32}}]
exe_list = [sym.simple_bind(grad_req=grad_req, **ctx) for ctx in ctx_list]
   
for exe in exe_list:
    print(exe.arg_arrays)
    for arr, iarr in zip(exe.arg_arrays, init):
        print(arr)
        arr[:] = iarr.astype(arr.dtype)
        
#forward predict
for exe in exe_list:
    exe.forward(is_train=False)

outputs = [exe.outputs[0].asnumpy() for exe in exe_list]
   
print('fft output')
print(outputs)

desire_out = np.fft.fft(init, n=None, axis=-1, norm=None)
print('fft desired output')
print(desire_out)


print("ifft input:")
init = outputs
print(outputs)

sym = mx.sym.IFFT(name='ifft', compute_size = 128) 
#ctx_list = [{'ctx': mx.gpu(0),'ifft_data': (shape[0],shape[1],shape[2],2*shape[3]), 'type_dict': {'ifft_data': np.float32}}]
ctx_list = [{'ctx': mx.gpu(0),'ifft_data': (shape[0],shape[1]*2), 'type_dict': {'ifft_data': np.float32}}]

exe_list = [sym.simple_bind(grad_req=grad_req, **ctx) for ctx in ctx_list]
   
for exe in exe_list:
    for arr, iarr in zip(exe.arg_arrays, init):
        arr[:] = iarr.astype(arr.dtype)
    
#forward predict
for exe in exe_list:
    exe.forward(is_train=False)

outputs = [exe.outputs[0].asnumpy()/shape[1] for exe in exe_list]
    
print('ifft Output')
print(outputs

desire_out = np.fft.ifft(desire_out, n=None, axis=-1, norm=None)
print('ifft desired output')
print(desire_out)




