import mxnet as mx
import numpy as np
out_dim = 6
compute_size = 128
D1 = mx.sym.Variable('data1')
S1 = mx.sym.Variable('s1',init = mx.init.One(),shape = (1,20))
H1 = mx.sym.Variable('h1',init = mx.init.One(),shape = (1,20))
D2 = mx.sym.Variable('data2')
S2 = mx.sym.Variable('s2',init = mx.init.One(),shape = (1,20))
H2 = mx.sym.Variable('h2',init = mx.init.One(),shape = (1,20))

cs1 = mx.sym.CountSketch( data = D1,s=S1, h = H1 ,name='cs1',out_dim = out_dim) 
cs2 = mx.sym.CountSketch( data = D2,s=S2, h = H2 ,name='cs2',out_dim = out_dim) 
fft1 = mx.sym.FFT(data = cs1, name='fft1', compute_size = compute_size) 
fft2 = mx.sym.FFT(data = cs2, name='fft2', compute_size = compute_size) 
c = fft1 * fft2
out = mx.sym.IFFT(data = c, name='ifft', compute_size = compute_size) 

mod = mx.mod.Module(symbol=out, 
                    context=mx.gpu(0),
                    data_names=['data1','data2']
                    )

data = mx.io.NDArrayIter({'data1':np.random.uniform(-1, 1, (10,20)),'data2':np.random.uniform(-1, 1, (10,20))})
#mod.bind(data_shapes=[('data1', (10, 20)),('data2', (10, 20)),('s1', (1, 20)),('s2', (1, 20)),('h1', (1, 20)),('h2', (1, 20))])
#mod.init_params()
#mod.fit(data, num_epoch=1)
'''
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
mod.bind(data_shapes=[('data1', (10, 20)),('data2', (10, 20))],for_training=False,)
mod.init_params()
mod.forward(Batch(data))
'''
mod.fit(data, num_epoch=1)