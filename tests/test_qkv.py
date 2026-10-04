"""Test QKV oracle with all kernels"""

import os
import pytest
import numpy as np
import jax.numpy as jnp
from conftest import (
    EXAMPLES_DIR,
    PLATFORM,
    generate_and_import_oracle,
    run_kernel,
    load_golden,
    attention_reference,
    bf16,
)

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')
kernel, api = generate_and_import_oracle(QKV_DIR)


@pytest.mark.skipif(PLATFORM == 'GPU', reason='the goldens are exact on the CPU alone')
def test_attention():
    output = run_kernel(
        kernel, api, 'attention', os.path.join(QKV_DIR, 'assembly'), os.path.join(QKV_DIR, 'data')
    )
    golden, record = load_golden(os.path.join(QKV_DIR, 'data'), 'attention')
    max_diff = float(jnp.max(jnp.abs(output - golden)))
    assert max_diff == 0, (
        f'Max diff: {max_diff} (golden from jax {record["jax"]} on {record["platform"]} {record["machine"]})'
    )


def test_attention_reference():
    output = run_kernel(
        kernel, api, 'attention', os.path.join(QKV_DIR, 'assembly'), os.path.join(QKV_DIR, 'data')
    )
    reference, tol = attention_reference(os.path.join(QKV_DIR, 'data'))
    max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
    assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'


def assert_near(output, reference):
    """Hold an output to its reference within two bf16 ulps at the reference's largest magnitude"""
    tol = 2 * 2.0 ** (np.floor(np.log2(np.abs(reference).max())) - 7)
    max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
    assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'


def test_identity():
    from conftest import load_bf16_matrix
    import sys, importlib

    sys.path.insert(0, os.path.join(QKV_DIR, 'assembly'))
    identity_mod = importlib.import_module('identity')
    k = identity_mod.identity(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, 'data', 'Q.dat'), (64, 64))
    outputs, _ = k('fsim')(A)
    # A load and a store move the bytes unchanged
    assert np.asarray(outputs[0]).tobytes() == np.asarray(A).tobytes()


def test_matmul():
    from conftest import load_bf16_matrix
    import sys, importlib

    sys.path.insert(0, os.path.join(QKV_DIR, 'assembly'))
    matmul_mod = importlib.import_module('matmul')
    k = matmul_mod.matmul(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, 'data', 'Q.dat'), (64, 64))
    B = load_bf16_matrix(os.path.join(QKV_DIR, 'data', 'K.dat'), (64, 64))
    outputs, _ = k('fsim')(A, B)
    # The dot rounds to bf16 once
    assert_near(outputs[0], bf16(np.asarray(A, np.float64) @ np.asarray(B, np.float64)))


def test_softmax():
    from conftest import load_bf16_matrix
    import sys, importlib

    sys.path.insert(0, os.path.join(QKV_DIR, 'assembly'))
    softmax_mod = importlib.import_module('softmax')
    k = softmax_mod.softmax(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(QKV_DIR, 'data', 'Q.dat'), (64, 64))
    outputs, _ = k('fsim')(A)
    # A times I is A; each later step rounds to bf16
    E = bf16(np.exp(np.asarray(A, np.float64)))
    assert_near(outputs[0], bf16(E / bf16(E.sum(axis=1, keepdims=True))))
