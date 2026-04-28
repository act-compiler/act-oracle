"""Test QKV oracle with all kernels"""

import os
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, generate_and_import_oracle, run_kernel

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV")
kernel, api = generate_and_import_oracle(QKV_DIR)


def test_attention():
    max_diff = run_kernel(kernel, api, "attention",
                          os.path.join(QKV_DIR, "assembly"),
                          os.path.join(QKV_DIR, "data"))
    assert max_diff == 0, f"Max diff: {max_diff}"


def test_identity():
    from conftest import load_bf16_matrix
    import sys, importlib
    sys.path.insert(0, os.path.join(QKV_DIR, "assembly"))
    identity_mod = importlib.import_module("identity")
    k = identity_mod.identity(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, "data", "Q.dat"), (64, 64))
    outputs, _ = k('fsim')(A)
    assert outputs[0].shape == (64, 64)


def test_matmul():
    from conftest import load_bf16_matrix
    import sys, importlib
    sys.path.insert(0, os.path.join(QKV_DIR, "assembly"))
    matmul_mod = importlib.import_module("matmul")
    k = matmul_mod.matmul(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, "data", "Q.dat"), (64, 64))
    B = load_bf16_matrix(os.path.join(QKV_DIR, "data", "K.dat"), (64, 64))
    outputs, _ = k('fsim')(A, B)
    assert outputs[0].shape == (64, 64)


def test_softmax():
    from conftest import load_bf16_matrix
    import sys, importlib
    sys.path.insert(0, os.path.join(QKV_DIR, "assembly"))
    softmax_mod = importlib.import_module("softmax")
    k = softmax_mod.softmax(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, "data", "Q.dat"), (64, 64))
    outputs, _ = k('fsim')(A)
    assert outputs[0].shape == (64, 64)
