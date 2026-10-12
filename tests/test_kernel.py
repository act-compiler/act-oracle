"""Test the kernel API of the generated oracle"""

import os
import glob
import time

import numpy as np
import pytest
import jax.numpy as jnp
from conftest import (EXAMPLES_DIR, PLATFORM, generate_and_import_oracle, generate_example, import_oracle,
                      load_bf16_matrix, run_kernel, attention_reference)

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


def test_unended_loop_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop("i", 0, 4)
        api.load_rm(n=16, addr_in="2048 * %i", addr_out="16 * %i")

    with pytest.raises(ValueError, match="loop %i has no end_loop"):
        k('fsim-compile')()


def test_unstarted_loop_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.end_loop()

    with pytest.raises(ValueError, match="end_loop has no start_loop"):
        k('fsim-compile')()


def looped():
    """The attention kernel with its Q and V loads and its store in loops"""
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop("i", 0, 4)
        api.load_rm(n=16, addr_in="2048 * %i", addr_out="16 * %i")
        api.end_loop()
        api.load_cm(n=64, addr_in=8192, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.softmax(n=64, addr=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop("i", 0, 4)
        api.load_rm(n=16, addr_in="16384 + 2048 * %i", addr_out="64 + 16 * %i")
        api.end_loop()
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop("j", 0, 8)
        api.store_rm(n=8, addr_in="8 * %j", addr_out="24576 + 1024 * %j")
        api.end_loop()
    return k


def nested():
    """The attention kernel with its Q and V loads in two-deep loops"""
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        # Each half of Q in two inner loops of eight rows
        api.start_loop("io", 0, 2)
        api.start_loop("ii", 0, 2)
        api.load_rm(n=8, addr_in="4096 * %io + 1024 * %ii", addr_out="32 * %io + 8 * %ii")
        api.end_loop()
        api.start_loop("jj", 0, 2)
        api.load_rm(n=8, addr_in="4096 * %io + 2048 + 1024 * %jj", addr_out="32 * %io + 16 + 8 * %jj")
        api.end_loop()
        api.end_loop()
        api.load_cm(n=64, addr_in=8192, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.softmax(n=64, addr=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop("io", 0, 2)
        api.start_loop("ii", 0, 2)
        api.load_rm(n=16, addr_in="16384 + 4096 * %io + 2048 * %ii", addr_out="64 + 32 * %io + 16 * %ii")
        api.end_loop()
        api.end_loop()
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)
    return k


@pytest.mark.parametrize("twin", [looped, nested])
def test_loop_matches_unrolled(twin):
    k = twin()
    k('fsim-compile')()
    outputs, _ = k('fsim')(*load_qkv())
    if PLATFORM == "GPU":
        # No bit equality on the GPU: the twin is held to the reference
        reference, tol = attention_reference(os.path.join(QKV_DIR, "data"))
        assert np.max(np.abs(np.asarray(outputs[0], np.float64) - reference)) <= tol
        return
    unrolled = run_kernel(kernel, api, "attention",
                          os.path.join(QKV_DIR, "assembly"), os.path.join(QKV_DIR, "data"))
    assert np.asarray(outputs[0]).tobytes() == np.asarray(unrolled).tobytes()
