import mxnet as mx
import numpy as np

cs1 = mx.sym.CountSketch(name='cs1',out_dim = 6) 
cs2 = mx.sym.CountSketch(name='cs2',out_dim = 6) 
fft1 = mx.sym.FFT(data = cs1, name='fft1', compute_size = 128) 
fft2 = mx.sym.FFT(data = cs2, name='fft2', compute_size = 128) 
c = fft1 * fft2
#y = c.bind(default_context(), args={'fft1': fft1, 'fft2' : fft2})
mx.viz.plot_network(c)    
    
