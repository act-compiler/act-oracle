"""Test the kernel API of the generated oracle"""

import os
import glob
import time

import numpy as np
import pytest
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, generate_and_import_oracle, generate_example, import_oracle, load_bf16_matrix

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV")
kernel, api = generate_and_import_oracle(QKV_DIR)

# Q, K and V in, one output after them
INPUT = [{'addr': a, 'shape': (64, 64), 'dtype': jnp.bfloat16} for a in (0, 8192, 16384)]
OUTPUT = [{'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16}]


def load_qkv():
    return [load_bf16_matrix(os.path.join(QKV_DIR, "data", f"{n}.dat"), (64, 64)) for n in "QKV"]


@pytest.mark.parametrize("mode", ["simulate", "FSIM"])
def test_unknown_mode_refused(mode):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError, match=f"unknown oracle mode '{mode}'"):
        k(mode)


def test_debug_prints_values(capfd):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.debug(prefix="d1", data="d1[0]")
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    Q, K, V = load_qkv()
    k('fsim')(Q, K, V)
    # Row 0 of d1, which holds Q
    lines = capfd.readouterr().out.splitlines()
    shape = lines.index("shape: (1, 64)")
    assert lines[shape - 1] == "d1: "
    values = lines[shape + 1].strip("[] ").split(",")
    np.testing.assert_allclose([float(v) for v in values], np.asarray(Q[0], np.float64), rtol=1e-5)


def test_generation_rebuilds_the_debug_handler():
    start = time.time()
    build_dir = os.path.join(generate_example(QKV_DIR), "oracle", "build")
    handler = glob.glob(os.path.join(build_dir, "print_handler_ext*.so"))
    # Built by this generation, not one before it
    assert handler and os.path.getmtime(handler[0]) >= start


def test_debug_before_any_instruction(capfd):
    # This oracle imported afresh, so debug is the first call it traces
    decorator, fresh = import_oracle(os.path.dirname(os.path.dirname(api.__file__)), "oracle_fresh")

    @decorator.kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        fresh.debug(prefix="d1", data="d1[0]")
        fresh.load_rm(n=64, addr_in=0, addr_out=0)
        fresh.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    k('fsim')(*load_qkv())
    # Nothing is loaded into d1 yet
    lines = capfd.readouterr().out.splitlines()
    values = lines[lines.index("d1: ") + 2].strip("[] ").split(",")
    assert [float(v) for v in values] == [0.0] * 64
