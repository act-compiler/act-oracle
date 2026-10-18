"""Test the mini_trn oracle with all kernels"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    EXAMPLES_DIR,
    PLATFORM,
    attention_reference,
    bf16,
    generate_and_import_oracle,
    load_golden,
    run_kernel,
)

MINI_TRN_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn')
kernel, api = generate_and_import_oracle(MINI_TRN_DIR)


@pytest.mark.skipif(PLATFORM == 'GPU', reason='the goldens are exact on the CPU alone')
def test_attention():
    output = run_kernel(
        kernel,
        api,
        'attention',
        os.path.join(MINI_TRN_DIR, 'assembly'),
        os.path.join(MINI_TRN_DIR, 'data'),
    )
    golden, record = load_golden(os.path.join(MINI_TRN_DIR, 'data'), 'attention')
    max_diff = float(jnp.max(jnp.abs(output - golden)))
    assert max_diff == 0, (
        f'Max diff: {max_diff} (golden from jax {record["jax"]} on {record["platform"]} {record["machine"]})'
    )


def test_attention_reference():
    output = run_kernel(
        kernel,
        api,
        'attention',
        os.path.join(MINI_TRN_DIR, 'assembly'),
        os.path.join(MINI_TRN_DIR, 'data'),
    )
    reference, tol = attention_reference(os.path.join(MINI_TRN_DIR, 'data'))
    max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
    assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'


def assert_near(output, reference):
    """Hold an output to its reference within two bf16 ulps at the reference's largest magnitude"""
    tol = 2 * 2.0 ** (np.floor(np.log2(np.abs(reference).max())) - 7)
    max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
    assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'


def test_identity():
    import importlib
    import sys

    from conftest import load_bf16_matrix

    sys.path.insert(0, os.path.join(MINI_TRN_DIR, 'assembly'))
    identity_mod = importlib.import_module('identity')
    k = identity_mod.identity(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(MINI_TRN_DIR, 'data', 'Q.dat'), (64, 64))
    outputs, _ = k('fsim')(A)
    assert outputs[0].shape == (64, 64)
    # A load and a store move the bytes unchanged
    assert np.asarray(outputs[0]).tobytes() == np.asarray(A).tobytes()


def test_matmul():
    import importlib
    import sys

    from conftest import load_bf16_matrix

    sys.path.insert(0, os.path.join(MINI_TRN_DIR, 'assembly'))
    matmul_mod = importlib.import_module('matmul')
    k = matmul_mod.matmul(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(MINI_TRN_DIR, 'data', 'Q.dat'), (64, 64))
    B = load_bf16_matrix(os.path.join(MINI_TRN_DIR, 'data', 'K.dat'), (64, 64))
    outputs, _ = k('fsim')(A, B)
    assert outputs[0].shape == (64, 64)
    # The dot rounds to bf16 once
    assert_near(outputs[0], bf16(np.asarray(A, np.float64) @ np.asarray(B, np.float64)))


def test_softmax():
    import importlib
    import sys

    from conftest import load_bf16_matrix

    sys.path.insert(0, os.path.join(MINI_TRN_DIR, 'assembly'))
    softmax_mod = importlib.import_module('softmax')
    k = softmax_mod.softmax(kernel, api)
    k('fsim-compile')()

    A = load_bf16_matrix(os.path.join(MINI_TRN_DIR, 'data', 'Q.dat'), (64, 64))
    outputs, _ = k('fsim')(A)
    assert outputs[0].shape == (64, 64)
    # A @ I is A; later steps round to bf16
    E = bf16(np.exp(np.asarray(A, np.float64)))
    assert_near(outputs[0], bf16(E / bf16(E.sum(axis=1, keepdims=True))))
