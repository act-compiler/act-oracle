"""Test the oracle one instruction at a time"""

import os
import sys
import tempfile
import importlib

import numpy as np
import jax.numpy as jnp
from taidl import Accelerator
from taidl_to import generate_oracle

# One instruction per check, reading and writing HBM
ops = Accelerator("OPS", base_dir=tempfile.mkdtemp())

# Contracting dims listed out of order
instr = ops.add_instruction("dot_perm", [], ["addr_a", "addr_b", "addr_out"])
instr.set_inputs([["d0", ["@a.addr_a"], ["8192"]], ["d0", ["@a.addr_b"], ["8192"]]])
instr.set_outputs([["d0", ["@a.addr_out"], ["8192"]]])
instr.add_semantics("""
ENTRY dot_perm {
    %In1 = u8[8192] parameter(0);
    %In2 = u8[8192] parameter(1);
    %a = u8[8,8,64,2] reshape(%In1);
    %b = u8[8,8,64,2] reshape(%In2);
    %c = bf16[8,8,64] bitcast_convert(%a);
    %d = bf16[8,8,64] bitcast_convert(%b);
    %e = bf16[64,64] dot(%c, %d), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,0};
    %f = u8[64,64,2] bitcast_convert(%e);
    ROOT %Out0 = u8[8192] reshape(%f);
}
""")

generate_oracle(ops)
sys.path.insert(0, os.path.join(ops.base_dir, "targets", "OPS"))
decorator = importlib.import_module("oracle.decorator")
api = importlib.import_module("oracle.api")
decorator.set_simulation_backend('CPU')

# Inputs at fixed HBM offsets, output after them
IN_ADDRS = [0, 16384]
OUT_ADDR = 32768


def run(instr, args, out_shape, out_dtype):
    """Run one instruction on args and return its output"""
    @decorator.kernel(
        hbm=OUT_ADDR + 16384,
        input=[{'addr': a, 'shape': x.shape, 'dtype': x.dtype} for a, x in zip(IN_ADDRS, args)],
        constant=[],
        output=[{'addr': OUT_ADDR, 'shape': out_shape, 'dtype': out_dtype}]
    )
    def k():
        getattr(api, instr)(*IN_ADDRS[:len(args)], OUT_ADDR)

    k('fsim-compile')()
    outputs, _ = k('fsim')(*args)
    return np.asarray(outputs[0]).astype(np.float64)


# Small integers keep every bf16 sum exact
rng = np.random.default_rng(0)


def small_ints(shape):
    return jnp.asarray(rng.integers(-2, 3, shape), dtype=jnp.bfloat16)


def test_dot_dimension_order():
    a, b = small_ints((8, 8, 64)), small_ints((8, 8, 64))
    out = run("dot_perm", [a, b], (64, 64), jnp.bfloat16)
    expected = np.einsum("pqi,qpj->ij", np.asarray(a, np.float64), np.asarray(b, np.float64))
    np.testing.assert_array_equal(out, expected)
