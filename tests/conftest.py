"""Shared test utilities for TAIDL-TO oracle tests"""

import os
import sys
import json
import shutil
import tempfile
import importlib
import importlib.util

import numpy as np
import ml_dtypes
import jax
import jax.numpy as jnp


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
# The platform every oracle simulates on, set by --platform
PLATFORM = "CPU"


def pytest_addoption(parser):
    parser.addoption("--platform", choices=["CPU", "GPU"], default="CPU",
                     help="simulate the oracles on the CPU or the GPU")


def pytest_configure(config):
    global PLATFORM
    PLATFORM = config.getoption("--platform")


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def import_oracle(target_dir, name):
    """Import the oracle generated under target_dir as package `name` and return (decorator, api)"""
    oracle_dir = os.path.join(target_dir, "oracle")
    spec = importlib.util.spec_from_file_location(name, os.path.join(oracle_dir, "__init__.py"),
                                                  submodule_search_locations=[oracle_dir])
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    decorator = importlib.import_module(f"{name}.decorator")
    decorator.set_simulation_backend(PLATFORM)
    return decorator, importlib.import_module(f"{name}.api")


def generate_example(example_dir):
    """Generate the example's oracle into a temp dir and return its target dir"""
    work_dir = tempfile.mkdtemp()

    # Copy spec into work_dir
    for name in ["QKV.py", "QKV_new.py"]:
        src = os.path.join(example_dir, name)
        if os.path.exists(src):
            shutil.copy(src, work_dir)

    os.chdir(work_dir)
    for name in ["QKV.py", "QKV_new.py"]:
        path = os.path.join(work_dir, name)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("QKV_spec", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break

    return os.path.join(work_dir, "targets", "QKV")


def generate_and_import_oracle(example_dir):
    """Generate oracle into a temp dir and return (kernel, api)"""
    oracle_dir = generate_example(example_dir)
    sys.path.insert(0, oracle_dir)

    oracle_decorator = importlib.import_module("oracle.decorator")
    oracle_api = importlib.import_module("oracle.api")
    oracle_decorator.set_simulation_backend(PLATFORM)

    return oracle_decorator.kernel, oracle_api


def run_kernel(kernel, api, kernel_module_name, assembly_dir, data_dir):
    """Run a kernel on the example's Q, K, V and return its output"""
    sys.path.insert(0, assembly_dir)
    kernel_module = importlib.import_module(kernel_module_name)
    qkv_kernel = kernel_module.qkv(kernel, api)
    qkv_kernel('fsim-compile')()

    Q = load_bf16_matrix(os.path.join(data_dir, "Q.dat"), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, "K.dat"), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, "V.dat"), (64, 64))

    outputs, _ = qkv_kernel('fsim')(Q, K, V)
    return outputs[0]


def load_golden(data_dir, kernel_name):
    """Return a kernel's golden and the record of where it came from"""
    with open(os.path.join(data_dir, "goldens.json")) as f:
        goldens = json.load(f)
    name, record = next((n, r) for n, r in goldens.items() if kernel_name in r["kernels"])
    return load_bf16_matrix(os.path.join(data_dir, name), (64, 64)), record


def bf16(x):
    """Round float64 values to bf16, kept in float64"""
    return x.astype(ml_dtypes.bfloat16).astype(np.float64)


def attention_reference(data_dir):
    """Return softmax(Q K^T) V in numpy, rounded to bf16 where the spec rounds, and its tolerance"""
    Q, K, V = (np.fromfile(os.path.join(data_dir, f"{n}.dat"), dtype=ml_dtypes.bfloat16)
               .reshape(64, 64).astype(np.float64) for n in "QKV")
    S = bf16(Q @ K.T)
    E = bf16(np.exp(S))
    P = bf16(E / bf16(E.sum(axis=1, keepdims=True)))
    # Two bf16 ulps at max|V| bound every output on the CPU
    ulp = 2.0 ** (np.floor(np.log2(np.abs(V).max())) - 7)
    # On the GPU, twice the two ulps an H100 reaches
    tol = (2 if PLATFORM == "CPU" else 4) * ulp
    return bf16(P @ V), tol
