import numpy as np
import mxnet as mx
import random
from numpy.testing import assert_allclose
from mxnet.test_utils import *




a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a * b
baseline = lambda a, b: a * b
sample_num = 200


shape = (3,2)

d =  [np.random.random(shape), np.random.random(shape)]
print('d:')
print(d)
print(d[0].shape)
x = baseline(d[0], d[1])
print('x:')
print(x)
y = c.bind(default_context(), args={'a': mx.nd.array(d[0]), 'b' : mx.nd.array(d[1])})
y.forward()
print('y:')
print(y.outputs[0].asnumpy())
assert_allclose(x, y.outputs[0].asnumpy(), rtol=1e-3, atol=1e-5)
        