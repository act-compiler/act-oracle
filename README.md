# ACT Generator: Test Oracle (Functional Simulator)

This repository contains the source code and documentation for `act-oracle`, a tool generator in the [ACT Ecosystem](https://github.com/act-compiler/act) that automatically generates test oracles just from ISA specifications written in [TAIDL](https://github.com/act-compiler/taidl).

For more details about the ACT Ecosystem, refer to the top-level repository: [act-compiler/act](https://github.com/act-compiler/act).

## TAIDL: Tensor Accelerator ISA Definition Language

TAIDL is a domain-specific language designed to define instruction set architectures (ISAs) for tensor accelerators. It is published at [MICRO 2025](https://doi.org/10.1145/3725843.3756075).
TAIDL not only standardizes the way tensor accelerator ISAs are specified but also enables automated generation of tools such as test oracles (functional simulators) and compiler backends, significantly reducing the effort required to develop and maintain these components.

## Test Oracle Generation

`act-oracle` is one of the tool generators in the [ACT Ecosystem](https://github.com/act-compiler/act) that consumes TAIDL specifications and emits out a fast & scalable test oracle instantaneously.

Details on automatically generating fast & scalable test oracles from TAIDL definitions is present in Section 6 of our [MICRO 2025](https://doi.org/10.1145/3725843.3756075) paper.
For detailed evaluations against existing baselines, refer to our [MICRO 2025 Artifact](https://github.com/act-compiler/taidl-artifact-micro25).

## Installation

```bash
pip install git+https://github.com/act-compiler/act-oracle.git@v2.4
```

## Platforms

An oracle simulates on the CPU unless the generated oracle's `decorator.set_simulation_backend('GPU')` moves it to an NVIDIA GPU; any other platform name raises `ValueError`.
Setting a platform, which the first kernel launch does with the CPU if nothing has, also turns on jax's `jax_enable_x64` and sets `jax_default_matmul_precision` to `highest` for the whole process.

## Testing

```bash
./tests/pytest.sh                 # every test, every oracle on the CPU
./tests/pytest.sh --platform GPU  # the same tests on an NVIDIA GPU
```

`tests/pytest.sh` runs the tests in `devanshdvj/act-env:py3.14-jax0.10.2`, the image `docker/act.env` names, in its tag for the host's architecture, amd64 or arm64, and in its CUDA 13 tag where `nvidia-smi` succeeds and docker lists the nvidia runtime; the CI pulls its amd64 tag.
On the CPU an attention kernel equals its golden bit for bit; on a GPU the goldens are skipped, and an attention kernel is held to its NumPy reference within four bf16 ulps of the largest magnitude in V, twice the largest deviation measured on an NVIDIA H100 NVL.
