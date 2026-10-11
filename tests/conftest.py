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


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def generate_and_import_oracle(example_dir):
    """Generate oracle into a temp dir and return (kernel, api)"""
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

    oracle_dir = os.path.join(work_dir, "targets", "QKV")
    sys.path.insert(0, oracle_dir)

    oracle_decorator = importlib.import_module("oracle.decorator")
    oracle_api = importlib.import_module("oracle.api")
    oracle_decorator.set_simulation_backend('CPU')

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
    # Two bf16 ulps at max|V|, which bounds every output
    tol = 2 * 2.0 ** (np.floor(np.log2(np.abs(V).max())) - 7)
    return bf16(P @ V), tol
