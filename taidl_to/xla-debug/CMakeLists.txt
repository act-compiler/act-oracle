cmake_minimum_required(VERSION 3.15)
project(print_handler_ext CXX)

# Find Python and JAX/nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

# Add the JAX include directory
include_directories(${JAXLIB_INCLUDE_DIR})

# Define the nanobind module
nanobind_add_module(
    print_handler_ext
    print_handler_main.cc
)

# Find CUDA, if any, in the nvidia wheel jax loads it from
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nvidia.cu13; print(nvidia.cu13.__path__[0])"
  OUTPUT_VARIABLE CUDA_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

# Add the CUDA handler only where CUDA is found
if(CUDA_DIR)
  target_compile_definitions(print_handler_ext PRIVATE WITH_CUDA)
  target_include_directories(print_handler_ext PRIVATE ${CUDA_DIR}/include)
  target_link_libraries(print_handler_ext PRIVATE ${CUDA_DIR}/lib/libcudart.so.13)
endif()
