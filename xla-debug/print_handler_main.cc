#include <iostream>
#include <numeric>
#include <vector>
#include <iomanip>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

void print_buffer_recursive(const double* data, const std::vector<int64_t>& dims, 
                           std::vector<int64_t>& indices, int depth = 0) {
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
      if (i < dims[depth] - 1) std::cout << ", ";
    }
  } else {
    for (int64_t i = 0; i < dims[depth]; ++i) {
      if (i > 0) std::cout << std::endl;
      indices[depth] = i;
      print_buffer_recursive(data, dims, indices, depth + 1);
    }
  }
  
  std::cout << "]";
}

void print_buffer(const double* data, const std::vector<int64_t>& dims) {
  if (dims.empty()) {
    std::cout << "scalar: " << data[0] << std::endl;
    return;
  }

  std::vector<int64_t> indices(dims.size(), 0);
  print_buffer_recursive(data, dims, indices);
  std::cout << std::endl;
}

auto PrintImpl(ffi::Buffer<ffi::DataType::U8> prefix, ffi::Buffer<ffi::DataType::F64> buf, ffi::Result<ffi::Buffer<ffi::DataType::S32>> out) {
  auto dims = buf.dimensions();
  std::vector<int64_t> dimensions(dims.begin(), dims.end());
  std::cout << prefix.typed_data() << ": " << std::endl;
  print_buffer(buf.typed_data(), dimensions);
  return ffi::Error::Success();
}

// Define and export the FFI handler symbol.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrintHandler, PrintImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U8>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());

// Nanobind module definition
NB_MODULE(print_handler_ext, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["print_handler"] = nb::capsule(reinterpret_cast<void*>(PrintHandler));
        return registrations;
    });
}
