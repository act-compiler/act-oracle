"""Test the kernel API of the generated oracle"""

import os

import pytest
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, generate_and_import_oracle

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV")
kernel, api = generate_and_import_oracle(QKV_DIR)

# Q, K and V in, one output after them
INPUT = [{'addr': a, 'shape': (64, 64), 'dtype': jnp.bfloat16} for a in (0, 8192, 16384)]
OUTPUT = [{'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16}]


@pytest.mark.parametrize("mode", ["simulate", "FSIM"])
def test_unknown_mode_refused(mode):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError, match=f"unknown oracle mode '{mode}'"):
        k(mode)
