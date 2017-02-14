import mxnet as mx
import numpy as np

def tensorsketching_poly(data1,out_dim,order,name,compute_size = 128, ifftflag = True):
    c = mx.sym.Variable('output')
    for i in range(order):
        cs = mx.sym.CountSketch( data = data1,name= 'cs'+i,out_dim = out_dim) 
        fft = mx.sym.FFT(data = cs, name='fft'+i, compute_size = compute_size) 
        c = c * fft
    if ifftflag:
        ifft = mx.sym.IFFT(data = c, name=name+'_ifft', compute_size = compute_size) 
        return ifft
    else:     
        return c