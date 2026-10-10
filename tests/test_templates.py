"""Test the oracle one instruction at a time"""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    LOWERED,
    PLATFORM,
    ULPS,
    add_typed_instruction,
    each_type,
    import_oracle,
    run_typed,
    ulps,
)
from taidl import Accelerator

from taidl_to import generate_oracle

# One instruction per check, reading and writing HBM
ops = Accelerator('OPS', base_dir=tempfile.mkdtemp())

# Contracting dims listed out of order
instr = ops.add_instruction('dot_perm', [], ['addr_a', 'addr_b', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_a'], ['8192']], ['d0', ['@a.addr_b'], ['8192']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['8192']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[8192] parameter(0);
    %In1 = u8[8192] parameter(1);
    %a = u8[8,8,64,2] reshape(%In0);
    %b = u8[8,8,64,2] reshape(%In1);
    %c = bf16[8,8,64] bitcast_convert(%a);
    %d = bf16[8,8,64] bitcast_convert(%b);
    %e = bf16[64,64] dot(%c, %d), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,0};
    %f = u8[64,64,2] bitcast_convert(%e);
    ROOT %Out0 = u8[8192] reshape(%f);
}
""")

# A row sum per type: four integer types, three floats
TYPES = {
    's8': jnp.int8,
    's32': jnp.int32,
    'u8': jnp.uint8,
    'u32': jnp.uint32,
    'f32': jnp.float32,
    'f16': jnp.float16,
    'bf16': jnp.bfloat16,
}
for t, dtype in TYPES.items():
    w = jnp.dtype(dtype).itemsize
    unit = f',{w}' if w > 1 else ''
    instr = ops.add_instruction(f'sum_{t}', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], [f'{64 * 64 * w}']]])
    instr.set_outputs([['d0', ['@a.addr_out'], [f'{64 * w}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[{64 * 64 * w}] parameter(0);
    %a = u8[64,64{unit}] reshape(%In0);
    %b = {t}[64,64] bitcast_convert(%a);
    %c = {t}[64] reduce_add(%b), dimensions={{1}};
    %d = u8[64{unit}] bitcast_convert(%c);
    ROOT %Out0 = u8[{64 * w}] reshape(%d);
}}
""")

# Broadcasts of a bf16 input, with what each must produce
BROADCASTS = {
    'bcast_rows': (
        """%r = bf16[64] reduce_add(%x), dimensions={1};
    %y = bf16[64,64] broadcast(%r), dimensions={0};""",
        lambda x: np.broadcast_to(x.sum(axis=1)[:, None], x.shape),
    ),
    'bcast_cols': (
        """%r = bf16[64] reduce_add(%x), dimensions={0};
    %y = bf16[64,64] broadcast(%r), dimensions={1};""",
        lambda x: np.broadcast_to(x.sum(axis=0)[None, :], x.shape),
    ),
    'bcast_transpose': ("""%y = bf16[64,64] broadcast(%x), dimensions={1,0};""", lambda x: x.T),
}
for name, (body, _) in BROADCASTS.items():
    instr = ops.add_instruction(name, [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['8192']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['8192']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[8192] parameter(0);
    %a = u8[64,64,2] reshape(%In0);
    %x = bf16[64,64] bitcast_convert(%a);
    {body}
    %b = u8[64,64,2] bitcast_convert(%y);
    ROOT %Out0 = u8[8192] reshape(%b);
}}
""")

# A dot with a batch dimension
instr = ops.add_instruction('dot_batched', [], ['addr_a', 'addr_b', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_a'], ['8192']], ['d0', ['@a.addr_b'], ['8192']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['4096']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[8192] parameter(0);
    %In1 = u8[8192] parameter(1);
    %a = u8[8,16,32,2] reshape(%In0);
    %b = u8[8,32,16,2] reshape(%In1);
    %c = bf16[8,16,32] bitcast_convert(%a);
    %d = bf16[8,32,16] bitcast_convert(%b);
    %e = bf16[8,16,16] dot(%c, %d), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1};
    %f = u8[8,16,16,2] bitcast_convert(%e);
    ROOT %Out0 = u8[4096] reshape(%f);
}
""")

# A copy whose result type is not its operand's
instr = ops.add_instruction('copy_retyped', [], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['8192']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['8192']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[8192] parameter(0);
    %a = u8[64,64,2] reshape(%In0);
    %x = bf16[64,64] bitcast_convert(%a);
    %y = f16[64,64] copy(%x);
    %b = u8[64,64,2] bitcast_convert(%y);
    ROOT %Out0 = u8[8192] reshape(%b);
}
""")

# An integer buffer, stored to HBM as it starts
ops.add_data_model('d1', [64], [64], 's32')
instr = ops.add_instruction('store_d1', [], ['addr_out'])
instr.set_inputs([['d1', ['0'], ['64']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['16384']]])
instr.add_semantics("""
ENTRY {
    %In0 = s32[64,64] parameter(0);
    %a = u8[64,64,4] bitcast_convert(%In0);
    ROOT %Out0 = u8[16384] reshape(%a);
}
""")

# A value named out, as the kernel's outputs are
instr = ops.add_instruction('named_out', [], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['8192']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['8192']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[8192] parameter(0);
    %out = u8[8192] copy(%In0);
    ROOT %Out0 = u8[8192] copy(%out);
}
""")

# A float through a rank-0 value and back
instr = ops.add_instruction('rank0', [], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['4']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['4']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[4] parameter(0);
    %a = u8[1,4] reshape(%In0);
    %b = f32[1] bitcast_convert(%a);
    %c = f32[] reshape(%b);
    %d = f32[1] reshape(%c);
    %e = u8[1,4] bitcast_convert(%d);
    ROOT %Out0 = u8[4] reshape(%e);
}
""")

# Two attributes, one name beginning the other
instr = ops.add_instruction('prefix', ['n', 'n_rows'], ['addr', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr'], ['@c.n_rows * 128 / @c.n']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n_rows * 128 / @c.n']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n_rows * 128 / @c.n`] parameter(0);
    ROOT %Out0 = u8[`@c.n_rows * 128 / @c.n`] copy(%In0);
}
""")

# Two byte inputs xored as preds
instr = ops.add_instruction('pred_xor', [], ['addr_a', 'addr_b', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_a'], ['4']], ['d0', ['@a.addr_b'], ['4']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['4']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[4] parameter(0);
    %In1 = u8[4] parameter(1);
    %a = pred[4] bitcast_convert(%In0);
    %b = pred[4] bitcast_convert(%In1);
    %c = pred[4] xor(%a, %b);
    ROOT %Out0 = u8[4] bitcast_convert(%c);
}
""")

# A pred to and from a wider type
instr = ops.add_instruction('pred_wide', [], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['4']]])
instr.set_outputs([['d0', ['@a.addr_out'], ['4']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u16[2] bitcast_convert(%a);
    %c = pred[2,2] bitcast_convert(%b);
    %d = u16[2] bitcast_convert(%c);
    %e = u8[2,2] bitcast_convert(%d);
    ROOT %Out0 = u8[4] reshape(%e);
}
""")

# Free names: a star-imported module, two locals, a debug value's
instr = ops.add_instruction('np', [], ['line', 'output'])
instr.set_inputs([['d0', ['@a.line'], ['4']]])
instr.set_outputs([['d0', ['@a.output'], ['4']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[4] parameter(0);
    %call = u8[4] copy(%In0);
    ROOT %Out0 = u8[4] add(%call, %In0);
}
""")

# Each lowered word: its line, %y's shape, NumPy's %y
LINES = {
    'add': ('%y = {t}[2,2] add(%x0, %x1);', [2, 2], lambda x, y: x + y),
    'subtract': ('%y = {t}[2,2] subtract(%x0, %x1);', [2, 2], lambda x, y: x - y),
    'multiply': ('%y = {t}[2,2] multiply(%x0, %x1);', [2, 2], lambda x, y: x * y),
    'divide': (
        '%y = {t}[2,2] divide(%x0, %x1);',
        [2, 2],
        lambda x, y: x / y if x.dtype.kind == 'f' else x // y,
    ),
    'maximum': ('%y = {t}[2,2] maximum(%x0, %x1);', [2, 2], np.maximum),
    'minimum': ('%y = {t}[2,2] minimum(%x0, %x1);', [2, 2], np.minimum),
    'xor': ('%y = {t}[2,2] xor(%x0, %x1);', [2, 2], np.bitwise_xor),
    'exponential': ('%y = {t}[2,2] exponential(%x0);', [2, 2], lambda x, y: np.exp(x)),
    'convert': (
        '%w = f32[2,2] convert(%x0);\n    %y = {t}[2,2] convert(%w);',
        [2, 2],
        lambda x, y: x,
    ),
    'bitcast_convert': ('%y = {t}[2,2] bitcast_convert(%x0);', [2, 2], lambda x, y: x),
    'copy': ('%y = {t}[2,2] copy(%x0);', [2, 2], lambda x, y: x),
    'reshape': ('%y = {t}[4] reshape(%x0);', [4], lambda x, y: x.reshape(4)),
    'transpose': ('%y = {t}[2,2] transpose(%x0), dimensions={{1,0}};', [2, 2], lambda x, y: x.T),
    'broadcast': (
        '%y = {t}[2,2,2] broadcast(%x0), dimensions={{1,2}};',
        [2, 2, 2],
        lambda x, y: np.broadcast_to(x, (2, 2, 2)),
    ),
    'dot': (
        '%y = {t}[2,2] dot(%x0, %x1), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}};',
        [2, 2],
        lambda x, y: x @ y,
    ),
    'reduce_add': (
        '%y = {t}[2] reduce_add(%x0), dimensions={{1}};',
        [2],
        lambda x, y: x.sum(axis=1),
    ),
}

# One instruction per lowered word and type
for word, (line, shape, _) in LINES.items():
    for t in LOWERED[word]:
        inputs = [(t, [2, 2]), (t, [2, 2])]
        add_typed_instruction(ops, f'{word}_{t}', inputs, line.format(t=t), (t, shape))

generate_oracle(ops)
decorator, api = import_oracle(os.path.join(ops.base_dir, 'targets', 'OPS'), 'oracle_OPS')

# Inputs at fixed HBM offsets, output after them
IN_ADDRS = [0, 16384]
OUT_ADDR = 32768


def run(instr, args, out_shape, out_dtype):
    """Run one instruction on args and return its output"""

    @decorator.kernel(
        hbm=OUT_ADDR + 16384,
        input=[{'addr': a, 'shape': x.shape, 'dtype': x.dtype} for a, x in zip(IN_ADDRS, args)],
        constant=[],
        output=[{'addr': OUT_ADDR, 'shape': out_shape, 'dtype': out_dtype}],
    )
    def k():
        getattr(api, instr)(*IN_ADDRS[: len(args)], OUT_ADDR)

    k('fsim-compile')()
    outputs, _ = k('fsim')(*args)
    return np.asarray(outputs[0]).astype(np.float64)


# Small integers keep every bf16 sum exact
rng = np.random.default_rng(0)


def small_ints(shape):
    return jnp.asarray(rng.integers(-2, 3, shape), dtype=jnp.bfloat16)


def test_dot_dimension_order():
    a, b = small_ints((8, 8, 64)), small_ints((8, 8, 64))
    out = run('dot_perm', [a, b], (64, 64), jnp.bfloat16)
    expected = np.einsum('pqi,qpj->ij', np.asarray(a, np.float64), np.asarray(b, np.float64))
    np.testing.assert_array_equal(out, expected)


def test_dot_batch_dimensions():
    a, b = small_ints((8, 16, 32)), small_ints((8, 32, 16))
    out = run('dot_batched', [a, b], (8, 16, 16), jnp.bfloat16)
    expected = np.einsum('bik,bkj->bij', np.asarray(a, np.float64), np.asarray(b, np.float64))
    np.testing.assert_array_equal(out, expected)


def test_copy_of_another_type_refused():
    with pytest.raises(ValueError, match='does not have the result type'):
        run('copy_retyped', [small_ints((64, 64))], (64, 64), jnp.float16)


def test_integer_buffer_starts_at_zero():
    # The store overwrites the ones laid where it writes
    spec = {'addr': 0, 'shape': (64, 64), 'dtype': jnp.int32}

    @decorator.kernel(hbm=16384, input=[spec], constant=[], output=[spec])
    def k():
        api.store_d1(0)

    k('fsim-compile')()
    outputs, _ = k('fsim')(jnp.ones((64, 64), jnp.int32))
    np.testing.assert_array_equal(np.asarray(outputs[0]), np.zeros((64, 64)))


def test_value_named_out():
    x = small_ints((64, 64))
    out = run('named_out', [x], (64, 64), jnp.bfloat16)
    np.testing.assert_array_equal(out, np.asarray(x, np.float64))


@pytest.mark.parametrize('t', TYPES)
def test_reduce_add_types(t):
    x = jnp.asarray(rng.integers(0, 2, (64, 64)), dtype=TYPES[t])
    out = run(f'sum_{t}', [x], (64,), TYPES[t])
    np.testing.assert_array_equal(out, np.asarray(x, np.float64).sum(axis=1))


@pytest.mark.parametrize('word', LINES)
def test_operation_at_each_type_it_takes(word):
    _, shape, reference = LINES[word]

    def check(t):
        # Small values keep results exact; a pred's 0 or 1
        low, high = (0, 2) if t == 'pred' else (1, 4)
        x, y = (rng.integers(low, high, (2, 2)).astype(DTYPES[t]) for _ in range(2))
        out = run_typed(decorator, api, f'{word}_{t}', [x, y], (t, shape))
        wide = np.float64 if t in FLOATS else np.int64
        expected = np.asarray(reference(x.astype(wide), y.astype(wide)))
        if word == 'exponential':
            # Twice the CPU's ulps on a GPU
            assert ulps(out, expected, t) <= ULPS[t] * (2 if PLATFORM == 'GPU' else 1)
        else:
            exact = expected.astype(DTYPES[t]).astype(np.float64)
            np.testing.assert_array_equal(out.astype(np.float64), exact)

    each_type(LOWERED[word], check)


def test_names_the_oracle_does_not_reserve():
    x = jnp.asarray(rng.integers(0, 256, 4), dtype=jnp.uint8)
    out = run('np', [x], (4,), jnp.uint8)
    np.testing.assert_array_equal(out, np.asarray(x, np.int64) * 2 % 256)


@pytest.mark.parametrize('name', BROADCASTS)
def test_broadcast_dimensions(name):
    x = small_ints((64, 64))
    out = run(name, [x], (64, 64), jnp.bfloat16)
    np.testing.assert_array_equal(out, BROADCASTS[name][1](np.asarray(x, np.float64)))


def test_rank0_result():
    x = jnp.asarray([3.25], dtype=jnp.float32)
    np.testing.assert_array_equal(run('rank0', [x], (1,), jnp.float32), [3.25])


def test_pred_to_and_from_bytes():
    a, b = (jnp.asarray(rng.integers(0, 2, 4), dtype=jnp.uint8) for _ in range(2))
    out = run('pred_xor', [a, b], (4,), jnp.uint8)
    np.testing.assert_array_equal(out, np.asarray(a) ^ np.asarray(b))


def test_pred_to_and_from_a_wider_type():
    x = jnp.asarray([1, 0, 0, 1], dtype=jnp.uint8)
    np.testing.assert_array_equal(run('pred_wide', [x], (4,), jnp.uint8), np.asarray(x))


def test_bool_kernel_input_and_output():
    a, b = (jnp.asarray(rng.integers(0, 2, 4), dtype=jnp.bool_) for _ in range(2))
    out = run('pred_xor', [a, b], (4,), jnp.bool_)
    np.testing.assert_array_equal(out, np.asarray(a) ^ np.asarray(b))


def test_attribute_named_by_the_start_of_another():
    @decorator.kernel(
        hbm=OUT_ADDR + 16384,
        input=[{'addr': IN_ADDRS[0], 'shape': (8192,), 'dtype': np.uint8}],
        constant=[],
        output=[{'addr': OUT_ADDR, 'shape': (8192,), 'dtype': np.uint8}],
    )
    def k():
        api.prefix(1, 64, IN_ADDRS[0], OUT_ADDR)

    x = jnp.asarray(rng.integers(0, 256, 8192), dtype=jnp.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), np.asarray(x))


def test_state_variable_in_the_api():
    acc = Accelerator('STATE', base_dir=tempfile.mkdtemp())
    acc.add_initial_state('mode', 3)
    generate_oracle(acc)
    with open(os.path.join(acc.base_dir, 'targets', 'STATE', 'oracle', 'api.py')) as file:
        assert "'mode': 3," in file.read()
