"""Test QKV_new oracle with all 4 attention kernel variants"""

import os
import pytest
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, generate_and_import_oracle, run_kernel, load_golden

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV_new")
kernel, api = generate_and_import_oracle(QKV_DIR)


@pytest.mark.parametrize("kernel_name", ["attention_k1", "attention_k2", "attention_k3", "attention_k4"])
def test_attention(kernel_name):
    output = run_kernel(kernel, api, kernel_name,
                        os.path.join(QKV_DIR, "assembly"),
                        os.path.join(QKV_DIR, "data"))
    golden, record = load_golden(os.path.join(QKV_DIR, "data"), kernel_name)
    max_diff = float(jnp.max(jnp.abs(output - golden)))
    assert max_diff == 0, f"{kernel_name}: Max diff: {max_diff} (golden from jax {record['jax']} on {record['platform']} {record['machine']})"
