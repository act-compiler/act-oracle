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
pip install git+https://github.com/act-compiler/act-oracle.git@v2.8
```

Generating an oracle compiles its debug handler, which needs a C++ compiler and Python's development headers on the machine, while CMake and nanobind are dependencies pip installs and jax's FFI headers come from jax.

## Usage

```python
from taidl_to import generate_oracle

generate_oracle(acc)  # acc: a TAIDL Accelerator
```

`generate_oracle(acc)` writes the oracle of the accelerator `acc` into `targets/<name>/oracle/` under the accelerator's `base_dir`, by default the directory of the file that creates the accelerator.
The oracle is a Python package: `api.py` holds one function per instruction, named after it, `decorator.py` holds `kernel`, which declares a kernel's HBM size and its inputs, constants and outputs, and `build/` holds the debug handler.
A kernel `k` decorated with `kernel` is launched by mode: `k('fsim-compile')()` traces its instruction calls into one StableHLO module, compiles it with XLA through jax and returns the kernel's declared inputs and its own compile time in milliseconds, and `k('fsim')(*inputs)` runs the compiled module on the inputs and returns its outputs and the run's time in milliseconds.
A generation that is refused leaves no oracle, not even one an earlier generation wrote.
TAIDL-TO pins TAIDL v2.13 beside jax and jaxlib 0.11.2.

## Examples

- `examples/mini_trn/`: `mini_trn.py`, an AWS Trainium-like accelerator design with two on-chip buffers, and its `attention`, `identity`, `matmul` and `softmax` kernels under `assembly/`
- `examples/mini_trn_double/`: `mini_trn_double.py`, a variant with a third on-chip buffer, and its four attention kernels, `attention_k1.py` to `attention_k4.py`, under `assembly/`

Running a spec, such as `python examples/mini_trn/mini_trn.py`, generates its oracle into `examples/mini_trn/targets/mini_trn/oracle/`.
Each kernel file defines a function that takes the oracle's `kernel` and `api` and returns the decorated kernel; in the five attention kernels it is called `attention`.
`tests/conftest.py` generates each oracle in a temporary directory and runs its kernels on the matrices under `data/`.
The tests hold each attention kernel to its golden there and to a NumPy reference, and `identity`, `matmul` and `softmax` to the input, the product and the softmax NumPy computes.

## What an oracle refuses

TAIDL v2.13 refuses with a `SyntaxError`, as the spec loads, a semantics whose form the language does not accept, and leaves whether its shapes and types agree and whether an operation is supported to each generator.
`generate_oracle` then refuses, before it writes anything, a spec it cannot turn into an oracle, with an error that names the instruction or data model.
It raises `NotImplementedError` for what the oracle does not support:

- an operation word at an element type that `docs/operations.md` does not list for it, or an index of `dynamic_slice`, `dynamic_update_slice` or `gather` that is not of an integer type
- `dot` or `concatenate` at `s2` or `u2`, though the oracle lowers every element type the grammar lexes
- an expression in backticks in an attribute that is not a list, such as `iota_dimension` or `epsilon`, which the oracle lowers from a literal alone
- an instruction that updates the machine state
- a name the generated oracle keeps for itself, as an instruction, attribute, data model or value

It raises `ValueError` for a spec that contradicts itself:

- an instruction with output regions and no semantics, or a `%In<k>` or `%Out<k>` in one with no region of its kind
- a tile typed unlike its data model, or holding more elements than its region
- a line whose shapes, element types or attributes disagree, where the spec writes them as integers
- a `compare` whose `type` does not fit its operands, a `constant` literal past its type's range, or a `batch_norm_inference` epsilon past the range of `f32`

It raises `TypeError` for a `constant` number that is not an integer at an integer type, or a `pred` literal at another type.

## Operations

An oracle lowers every operation word TAIDL v2.13 lexes, each to the StableHLO operation that means the same or to a CHLO one where StableHLO has none, and `docs/operations.md` gives each word's operation and the element types at which it lowers.
On the CPU the tests hold every word exactly but the transcendental ones, held within 2 ulps at `bf16` and `f16` and 8 at `f32` and `f64`, and the one word that reaches past its bound within twice the most it was measured to reach.
On a GPU those bounds double and an exact word stays exact, but `sqrt` at `f32` is held within 2 ulps.

## Platforms

An oracle simulates on the CPU unless the generated oracle's `decorator.set_simulation_backend('GPU')` moves it to an NVIDIA GPU; any other platform name raises `ValueError`.
Setting a platform, which the first kernel launch does with the CPU if nothing has, also turns on jax's `jax_enable_x64` and sets `jax_default_matmul_precision` to `highest` for the whole process.
Setting the CPU also sets jax's `jax_platforms` to `cpu` unless the caller has set it.
If no jax backend has started before, jax then starts only its CPU platform and never initialises CUDA on a GPU host, and setting the GPU afterwards raises `ValueError`.
Every dot an oracle emits carries the precision `HIGHEST`, so a GPU computes an `f32` dot at `f32`, as the CPU does.
Every `triangular_solve` an oracle emits at `f16` solves at `f32` and converts its result back to `f16`, since XLA's GPU backend fails a solve at `f16`.

## Testing

```bash
./tests/pytest.sh                 # every test, every oracle on the CPU
./tests/pytest.sh --platform GPU  # the same tests on an NVIDIA GPU
```

`tests/pytest.sh` runs the tests in `devanshdvj/act-env:py3.14-jax0.11.2`, the image `docker/act.env` names, in its tag for the host's architecture, amd64 or arm64, and in its CUDA 13 tag where `nvidia-smi` succeeds and docker lists the nvidia runtime; the CI pulls its amd64 and arm64 tags.
Ahead of the tests, the CI checks the formatting and lint against `pyproject.toml`, `.editorconfig` and `.clang-format` with `./docker/exec.sh "ruff format --check && ruff check && shfmt -d . && clang-format --dry-run -Werror $(git ls-files '*.cc' '*.cpp' '*.h' '*.hpp' | tr '\n' ' ')"`.
On the CPU an attention kernel equals its golden bit for bit; on a GPU the goldens are skipped, and an attention kernel is held to its NumPy reference within four bf16 ulps of the largest magnitude in V, twice the largest deviation measured on an NVIDIA H100 NVL.
`docker/exec.sh`, `docker/run.sh` and `docker/cleanup.sh` take `--cpu` or `--cuda13` to choose the image, as in `docker/exec.sh --cpu "<cmd>"`; without either they pick the `cuda13` image on a GPU host and the CPU image elsewhere, and a CPU image is given no GPU.

## Documentation

`docs/operations.md` says what the oracle lowers: the StableHLO or CHLO operation each operation word becomes, the element types at which it lowers, what generation refuses by name, and how closely the tests hold each word to its reference.
The language itself, its declarations, instructions, semantics, expressions and operation words, is documented in [TAIDL's `docs/`](https://github.com/act-compiler/taidl/tree/master/docs).
