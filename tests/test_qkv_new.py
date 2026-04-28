"""Test QKV_new oracle with all 4 attention kernel variants"""

import os
import pytest
from conftest import EXAMPLES_DIR, generate_and_import_oracle, run_kernel

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV_new")
kernel, api = generate_and_import_oracle(QKV_DIR)


@pytest.mark.parametrize("kernel_name", ["attention_k1", "attention_k2", "attention_k3", "attention_k4"])
def test_attention(kernel_name):
    max_diff = run_kernel(kernel, api, kernel_name,
                          os.path.join(QKV_DIR, "assembly"),
                          os.path.join(QKV_DIR, "data"))
    assert max_diff == 0, f"{kernel_name}: Max diff: {max_diff}"
