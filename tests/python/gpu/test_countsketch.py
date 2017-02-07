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




print('start')
out_dim = 3
sym = mx.sym.CountSketch(name='countsketch',out_dim = out_dim) 
ctx_list = [{'ctx':mx.gpu(0)}]

batch_size = 10
n = 803  #number of count sketch
in_dim = 20   #input dimension
assert(in_dim > out_dim)
shape = [(n,in_dim), (1,in_dim),(1,in_dim)]     #shape of input x, hash h and hash s

arr = [mx.nd.empty(shape[i]) for i in range(3)]
arr_grad = [mx.nd.empty(shape[i]) for i in range(3)]
x = np.random.uniform(-10, 10, shape[0])
arr[0][:] = x                                 #input x
h = np.random.randint(0, out_dim, shape[1])
arr[1][:] = h                                 #hash h
s = np.random.randint(0, 2, shape[2])*2-np.ones(shape[2])
arr[2][:] = s                                 #hash s


exe_list = [sym.bind(mx.gpu(0), arr, arr_grad)]

for exe in exe_list:
    exe.forward(is_train=False)

outputs = [exe.outputs[0].asnumpy() for exe in exe_list]

########### ground truth ################
gt = np.zeros((n,out_dim))
temp = np.multiply(x, s)
for num_sample in np.arange(0,n):
    for idx in np.arange(0,in_dim):
        gt[num_sample][h[0][idx]] += temp[num_sample][idx]

#print(gt)
#print(outputs)
print(reldiff(gt, outputs) )
assert(reldiff(gt, outputs) < 1e-6 )
print('success!')

######### Backward test ###############
out_grad = mx.nd.empty((n,out_dim))
out_grad[:] = np.random.normal(-3, 3, (n,out_dim))
for exe in exe_list:
    exe.backward([out_grad])  
outputs_back = arr_grad[0].asnumpy()
    
gt_back = np.zeros((n,in_dim))
for j in np.arange(0,n):
    for i in np.arange(0,in_dim):
        gt_back[j,i] = out_grad.asnumpy()[j, h[0,i]] * s[0,i]
print(reldiff(gt_back, outputs_back) )
assert(reldiff(gt_back, outputs_back) < 1e-6 )
print('success!')