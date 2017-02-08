## Example of creating an operator in C++ platform
Let's take fft for example. Detailed codes can be found [here](https://github.com/shiyangdaisy/mxnet/tree/master/src/operator/tensor)

You need to create fft.h, fft.cc, and fft.cu three files. fft.h is the head file. fft.cc and fft.cu include functions you need for running on CPU and GPU correspondently.

### fft.h 
----------
* Define parameters in the operator:
```c++
struct FFTParam : public dmlc::Parameter<FFTParam> {
    int compute_size; // the maximum size of sub-batch to be forwarded through FFT in one time
    DMLC_DECLARE_PARAMETER(FFTParam){        // declare the parameter you defined
        DMLC_DECLARE_FIELD(compute_size).set_default(128)
        .describe("Maximum size of sub-batch to be forwarded at one time");
    }
};
```
* Define operator class: Include Forward and Backward functions:

  Forward function is straight forward. output = op(input).
  
  Backward function calculates the in_gradient given input, output and out_gradient. in_gradient = out_gradient*($\partial$(ouput) / $\partial$(input))
```c++
template<typename xpu, typename DType>
class FFTOp : public Operator {
public:
  explicit FFTOp(FFTParam p){
    this->param_ = p;
    init_cufft_ = false;
    dim_ = 0;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req, 
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args){
		       ......
  }
     
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args){
			......
  }
private:
  FFTParam param_;
  int dim_, stride_, num_compute, n_ffts;
  bool init_cufft_;
};
```

* Declare Factory Function(used for dispatch specialization)
```c++
template<typename xpu>
Operator* CreateOp(FFTParam param, int dtype);
```

* Define operator property:

  Some properties you may have are: ListofInputArgument, ListofOutput etc.
```c++
class FFTProp : public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }
  ......
}
```
### fft.cc 
----------
* Register the operator:
```c++
DMLC_REGISTER_PARAMETER(FFTParam);
MXNET_REGISTER_OP_PROPERTY(FFT, FFTProp)
.describe("Apply FFT to input.")
.add_argument("data", "Symbol", "Input data to the FFTOp.")
.add_arguments(FFTParam::__FIELDS__());
```
* Create executable object of the operator:
```c++
template<>
Operator *CreateOp<cpu>(FFTParam param, int dtype){
	Operator *op = NULL;
        op = new FFTOp<cpu, DType>(param);
	return op;
}
Operator *FFTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
	std::vector<TShape> out_shape, aux_shape;
	std::vector<int> out_type, aux_type;
	CHECK(InferType(in_type, &out_type, &aux_type));
	CHECK(InferShape(in_shape, &out_shape, &aux_shape));
	DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}
```
### fft.cu 
----------
* Create an object of the operator:
```c++
Operator* CreateOp<gpu>(FFTParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {op = new FFTOp<gpu, DType>(param);})
    return op;
}
```
* Add cuda kernels if you need
