"""Test QKV_new oracle with all 4 attention kernel variants"""

import os
import pytest
import numpy as np
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, PLATFORM, generate_and_import_oracle, run_kernel, load_golden, attention_reference

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV_new")
kernel, api = generate_and_import_oracle(QKV_DIR)


@pytest.mark.skipif(PLATFORM == "GPU", reason="the goldens are exact on the CPU alone")
@pytest.mark.parametrize("kernel_name", ["attention_k1", "attention_k2", "attention_k3", "attention_k4"])
def test_attention(kernel_name):
    output = run_kernel(kernel, api, kernel_name,
                        os.path.join(QKV_DIR, "assembly"),
                        os.path.join(QKV_DIR, "data"))
    golden, record = load_golden(os.path.join(QKV_DIR, "data"), kernel_name)
    max_diff = float(jnp.max(jnp.abs(output - golden)))
    assert max_diff == 0, f"{kernel_name}: Max diff: {max_diff} (golden from jax {record['jax']} on {record['platform']} {record['machine']})"


@pytest.mark.parametrize("kernel_name", ["attention_k1", "attention_k2", "attention_k3", "attention_k4"])
def test_attention_reference(kernel_name):
    output = run_kernel(kernel, api, kernel_name,
                        os.path.join(QKV_DIR, "assembly"),
                        os.path.join(QKV_DIR, "data"))
    reference, tol = attention_reference(os.path.join(QKV_DIR, "data"))
    max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
    assert max_diff <= tol, f"{kernel_name}: Max diff from reference: {max_diff} > {tol}"
