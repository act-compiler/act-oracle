#include <iostream>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace nb = nanobind;
namespace ffi = xla::ffi;

void print_buffer_recursive(const double *data, const std::vector<int64_t> &dims,
                            std::vector<int64_t> &indices, int depth = 0) {
  if (depth == dims.size()) {
    int64_t linear_index = 0;
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
      linear_index += indices[i] * stride;
      stride *= dims[i];
    }
    std::cout << data[linear_index];
    return;
  }

  if (depth == 0) {
    std::cout << "shape: (";
    for (size_t i = 0; i < dims.size(); ++i) {
      std::cout << dims[i] << (i == dims.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
  }

  std::cout << std::string(depth * 2, ' ') << "[";

  if (depth == dims.size() - 1) {
    for (int64_t i = 0; i < dims[depth]; ++i) {
      indices[depth] = i;
      print_buffer_recursive(data, dims, indices, depth + 1);
      if (i < dims[depth] - 1)
        std::cout << ", ";
    }
  } else {
    for (int64_t i = 0; i < dims[depth]; ++i) {
      if (i > 0)
        std::cout << std::endl;
      indices[depth] = i;
      print_buffer_recursive(data, dims, indices, depth + 1);
    }
  }

  std::cout << "]";
}

void print_buffer(const double *data, const std::vector<int64_t> &dims) {
  if (dims.empty()) {
    std::cout << "scalar: " << data[0] << std::endl;
    return;
  }

  std::vector<int64_t> indices(dims.size(), 0);
  print_buffer_recursive(data, dims, indices);
  std::cout << std::endl;
}

// Print a host buffer under its prefix
void print_prefixed(const uint8_t *prefix, const double *data, ffi::Span<const int64_t> dims) {
  std::vector<int64_t> dimensions(dims.begin(), dims.end());
  std::cout << prefix << ": " << std::endl;
  print_buffer(data, dimensions);
}

auto PrintImpl(ffi::Buffer<ffi::DataType::U8> prefix, ffi::Buffer<ffi::DataType::F64> buf,
               ffi::Result<ffi::Buffer<ffi::DataType::S32>> out) {
  print_prefixed(prefix.typed_data(), buf.typed_data(), buf.dimensions());
  return ffi::Error::Success();
}

// Define and export the FFI handler symbol.
XLA_FFI_DEFINE_HANDLER_SYMBOL(PrintHandler, PrintImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F64>>()
                                  .Ret<ffi::Buffer<ffi::DataType::S32>>());

#ifdef WITH_CUDA
// Copy both buffers to host on the call's stream, then print
auto PrintCudaImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::U8> prefix,
                   ffi::Buffer<ffi::DataType::F64> buf,
                   ffi::Result<ffi::Buffer<ffi::DataType::S32>> out) {
  std::vector<uint8_t> host_prefix(prefix.element_count());
  std::vector<double> host_buf(buf.element_count());
  cudaError_t err = cudaMemcpyAsync(host_prefix.data(), prefix.typed_data(), prefix.size_bytes(),
                                    cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(host_buf.data(), buf.typed_data(), buf.size_bytes(),
                          cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess)
    err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess)
    return ffi::Error::Internal(cudaGetErrorString(err));
  print_prefixed(host_prefix.data(), host_buf.data(), buf.dimensions());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PrintCudaHandler, PrintCudaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F64>>()
                                  .Ret<ffi::Buffer<ffi::DataType::S32>>());
#endif

// Nanobind module definition: targets by platform, then by name
NB_MODULE(print_handler_ext, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    nb::dict cpu;
    cpu["print_handler"] = nb::capsule(reinterpret_cast<void *>(PrintHandler));
    registrations["cpu"] = cpu;
#ifdef WITH_CUDA
    nb::dict cuda;
    cuda["print_handler"] = nb::capsule(reinterpret_cast<void *>(PrintCudaHandler));
    registrations["CUDA"] = cuda;
#endif
    return registrations;
  });
}
