# ACT Generator: Test Oracle (Functional Simulator)

This repository contains the source code and documentation for `act-oracle`, a tool generator in the [ACT Ecosystem](https://github.com/act-compiler/act) that automatically generates test oracles just from ISA specifications written in [TAIDL](https://github.com/act-compiler/taidl).

For more details about the ACT Ecosystem, refer to the top-level repository: [act-compiler/act](https://github.com/act-compiler/act).

## TAIDL: Tensor Accelerator ISA Definition Language

TAIDL is a domain-specific language designed to define instruction set architectures (ISAs) for tensor accelerators.
It is published at [MICRO 2025](https://doi.org/10.1145/3725843.3756075).
TAIDL not only standardizes the way tensor accelerator ISAs are specified but also enables automated generation of tools such as test oracles (functional simulators) and compiler backends, significantly reducing the effort required to develop and maintain these components.

## Test Oracle Generation

`act-oracle` is one of the tool generators in the [ACT Ecosystem](https://github.com/act-compiler/act) that consumes TAIDL specifications and emits out a fast & scalable test oracle instantaneously.

Details on automatically generating fast & scalable test oracles from TAIDL definitions is present in Section 6 of our [MICRO 2025](https://doi.org/10.1145/3725843.3756075) paper.
For detailed evaluations against existing baselines, refer to our [MICRO 2025 Artifact](https://github.com/act-compiler/taidl-artifact-micro25).

## Installation

```bash
pip install git+https://github.com/act-compiler/act-oracle.git@v2.6
```

Generating an oracle compiles its debug handler, which needs a C++ compiler and Python's development headers on the machine, while CMake and nanobind are dependencies pip installs and jax's FFI headers come from jax.

## Usage

```python
from taidl_to import generate_oracle

generate_oracle(acc)  # acc: a TAIDL Accelerator
```

`generate_oracle(acc)` writes the oracle of the accelerator `acc` into `targets/<name>/oracle/` under the accelerator's `base_dir`, by default the directory of the file that creates the accelerator.
The oracle is a Python package: `api.py` holds one function per instruction, named after it, `decorator.py` holds `kernel`, which declares a kernel's HBM size and its inputs, constants and outputs, and `build/` holds the debug handler.
A kernel `k` decorated with `kernel` is launched by mode: `k('fsim-compile')()` traces its instruction calls into one StableHLO module and compiles it with XLA through jax, and `k('fsim')(*inputs)` runs the compiled module on the inputs and returns its outputs and the run's time in milliseconds.
A generation that is refused leaves the oracle an earlier generation wrote in place.
TAIDL-TO pins TAIDL v2.13 beside jax and jaxlib 0.11.2.

## Examples

- `examples/mini_trn/`: `mini_trn.py`, an accelerator with two on-chip buffers, and its `attention`, `identity`, `matmul` and `softmax` kernels under `assembly/`
- `examples/mini_trn_double/`: `mini_trn_double.py`, a variant with a third on-chip buffer, and its four attention kernels, `attention_k1.py` to `attention_k4.py`, under `assembly/`

Running a spec, such as `python examples/mini_trn/mini_trn.py`, generates its oracle into `examples/mini_trn/targets/mini_trn/oracle/`.
Each kernel file defines a function that takes the oracle's `kernel` and `api` and returns the decorated kernel; in the five attention kernels it is called `attention`.
`tests/conftest.py` generates each oracle in a temporary directory and runs its kernels on the matrices under `data/`.
The tests hold each attention kernel to its golden there and to a NumPy reference, and `identity`, `matmul` and `softmax` to the input, the product and the softmax NumPy computes.

## Platforms

An oracle simulates on the CPU unless the generated oracle's `decorator.set_simulation_backend('GPU')` moves it to an NVIDIA GPU; any other platform name raises `ValueError`.
Setting a platform, which the first kernel launch does with the CPU if nothing has, also turns on jax's `jax_enable_x64` and sets `jax_default_matmul_precision` to `highest` for the whole process.

## Testing

```bash
./tests/pytest.sh                 # every test, every oracle on the CPU
./tests/pytest.sh --platform GPU  # the same tests on an NVIDIA GPU
```

`tests/pytest.sh` runs the tests in `devanshdvj/act-env:py3.14-jax0.11.2`, the image `docker/act.env` names, in its tag for the host's architecture, amd64 or arm64, and in its CUDA 13 tag where `nvidia-smi` succeeds and docker lists the nvidia runtime; the CI pulls its amd64 tag.
Ahead of the tests, the CI checks the formatting and lint against `pyproject.toml`, `.editorconfig` and `.clang-format` with `./docker/exec.sh "ruff format --check && ruff check && shfmt -d . && clang-format --dry-run -Werror $(git ls-files '*.cc' '*.cpp' '*.h' '*.hpp' | tr '\n' ' ')"`.
On the CPU an attention kernel equals its golden bit for bit; on a GPU the goldens are skipped, and an attention kernel is held to its NumPy reference within four bf16 ulps of the largest magnitude in V, twice the largest deviation measured on an NVIDIA H100 NVL.
