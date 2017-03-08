/*!
 * Copyright (c) 2017 by Contributors
 * \file mxnet_op.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_MXNET_OP_H_
#define MXNET_OPERATOR_MXNET_OP_H_

#include <mxnet/base.h>
#include <algorithm>

namespace mxnet {
namespace op {
namespace mxnet_op {
#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif


template<typename OP, typename xpu>
struct Kernel;

template<typename OP>
struct Kernel<OP, cpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<cpu> *s, int N, Args... args) {
#if (MXNET_USE_CUDA == 0)
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
  }
};

#ifdef __CUDACC__
template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template<typename OP>
struct Kernel<OP, gpu> {
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
  }
};
#endif  // __CUDACC__

/*! \brief operator request type switch */
#define MXNET_ASSIGN_REQ_SWITCH(req, ReqType, ...)  \
  switch (req) {                                    \
  case kNullOp:                                     \
    break;                                          \
  case kWriteInplace:                               \
  case kWriteTo:                                    \
    {                                               \
      const int ReqType = kWriteTo;                 \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kAddTo:                                      \
    {                                               \
      const int ReqType = kAddTo;                   \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    break;                                          \
  }

/*!
 * \brief assign the val to out according
 * to request in Kernel::Launch
 * \param out the data to be assigned
 * \param req the assignment request
 * \param val the value to be assigned to out
 * \tparam OType output type
 * \tparam VType value type
 */
#define KERNEL_ASSIGN(out, req, val)  \
  {                                   \
    switch (req) {                    \
      case kNullOp:                   \
        break;                        \
      case kWriteTo:                  \
      case kWriteInplace:             \
        (out) = (val);                \
        break;                        \
      case kAddTo:                    \
        (out) += (val);               \
        break;                        \
      default:                        \
        break;                        \
    }                                 \
  }

struct clip {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = a_max;
    } else if (data < a_min) {
      out[i] = a_min;
    } else {
      out[i] = data;
    }
  }
};

struct clip_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* grad, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = 0;
    } else if (data < a_min) {
      out[i] = 0;
    } else {
      out[i] = grad[i];
    }
  }
};

#define REVERSE_MAX_DIM 10

struct reverse {
  MSHADOW_XINLINE static int ReverseIndex(index_t idx,
                                          index_t nreversedim,
                                          const index_t * stride_,
                                          const index_t * trailing_) {
    index_t outputIndex = idx;
    for (index_t i = 0; i < nreversedim; ++i) {
      const index_t low = outputIndex % trailing_[i];
      index_t high = outputIndex / trailing_[i];
      const index_t x = high%stride_[i];
      high /= stride_[i];
      outputIndex = (high*stride_[i] + stride_[i] - 1 - x)*trailing_[i] + low;
    }
    return outputIndex;
  }
#ifdef __CUDACC__
  template<typename DType>
  __device__  static void Map(int index, index_t nreversedim, const DType *src, DType *dst,
                              const index_t * stride_,
                              const index_t * trailing_) {
    __shared__ index_t stride_share[REVERSE_MAX_DIM];
    __shared__ index_t trailing_share[REVERSE_MAX_DIM];
    if (threadIdx.x < REVERSE_MAX_DIM) {
      stride_share[threadIdx.x] = stride_[threadIdx.x];
      trailing_share[threadIdx.x] = trailing_[threadIdx.x];
    }
    __syncthreads();
    index_t new_idx = ReverseIndex(index, nreversedim, stride_share, trailing_share);
    dst[new_idx] = src[index];
  }
#else
  template<typename DType>
  MSHADOW_XINLINE  static void Map(int index, index_t nreversedim, const DType *src, DType *dst,
                                   const index_t * stride_,
                                   const index_t * trailing_) {
    index_t new_idx = ReverseIndex(index, nreversedim, stride_, trailing_);
    dst[new_idx] = src[index];
  }
#endif
};

}  // namespace mxnet_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MXNET_OP_H_
