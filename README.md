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
