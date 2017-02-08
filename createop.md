## Example of creating an operator in C++ platform
Let's take fft for example. Detailed codes can be found [here](https://github.com/shiyangdaisy/mxnet/tree/master/src/operator/tensor)

You need to create fft.h, fft.cc, and fft.cu three files. fft.h is the head file. fft.cc and fft.cu include functions you need for running on CPU and GPU correspondently.

### fft.h 
----------
* Define parameters in the operator:
```c++
struct FFTParam : public dmlc::Parameter<FFTParam> {
	......
};
```
* Define operator class: Include Forward and Backward functions:

  Forward function is straight forward. output = op(input).
  
  Backward function calculates the in_gradient given input, output and out_gradient. in_gradient = out_gradient*(\partial(ouput) / \partial(input))

* Define operator property:

  Some properties you may have are: ListofInputArgument, ListofOutput etc.

### fft.cc 
----------
* Register the operator:
```c++
MXNET_REGISTER_OP_PROPERTY(FFT, FFTProp)
.describe("Apply FFT to input.")
.add_argument("data", "Symbol", "Input data to the FFTOp.")
.add_arguments(FFTParam::__FIELDS__());
```
* Create executable object of the operator:
```c++
Operator *FFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
	......
}
```
### fft.cu 
----------
* Create an object of the operator:
```c++
Operator* CreateOp<gpu>(FFTParam param, int dtype) {
    Operator *op = NULL;
    ......
};
```
