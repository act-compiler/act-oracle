"""Test constant and iota at each element type, against NumPy"""

import re
import tempfile

import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    LOWERED,
    add_typed_instruction,
    each_type,
    import_oracle,
    run_bytes,
)
from taidl import Accelerator

from taidl_to import generate_oracle

# A literal of each type and the value it means
LITERALS = {
    'f64': ('-0.5', -0.5),
    'f32': ('1e-3', 1e-3),
    'f16': ('3', 3.0),
    'bf16': ('0.1', 0.1),
    'u8': ('255', 255),
    'u16': ('65535', 65535),
    'u32': ('4294967295', 4294967295),
    'u64': ('18446744073709551615', 18446744073709551615),
    's8': ('-128', -128),
    's16': ('-32768', -32768),
    's32': ('2147483647', 2147483647),
    's64': ('-9223372036854775808', -9223372036854775808),
    'pred': ('true', True),
}

# Float literals: non-finite, a signed zero, two that round
SPECIALS = ['inf', '-inf', 'nan', '-nan', '-0.0', '65504', '1e-30']
# Each float's specials, an 8-bit float's in its range
SPECIAL = {
    **dict.fromkeys(FLOATS, SPECIALS),
    'f8e4m3fn': ['nan', '-nan', '-0.0', '1e-30'],
    'f8e5m2': ['inf', '-inf', 'nan', '-nan', '-0.0', '1e-30'],
}
# Literals past a type's largest value that round to it
LARGEST = {
    'f32': '3.4028235e38',
    'f16': '65519',
    'bf16': '3.39e38',
    'f8e4m3fn': '464',
    'f8e5m2': '61439',
}
# A pred takes a number too
PREDS = ['false', '0', '1']

SHAPES = {'matrix': (2, 3), 'scalar': ()}


def add(acc, name, line, t, dims, attributes=()):
    """An instruction whose semantics is line, a t[dims] named %y, written to HBM"""
    add_typed_instruction(acc, name, [], line, (t, list(dims)), attributes)


ops = Accelerator('CONSTANTS', base_dir=tempfile.mkdtemp())
for t, (text, _) in LITERALS.items():
    for shape, dims in SHAPES.items():
        typed = f'{t}[{",".join(map(str, dims))}]'
        add(ops, f'constant_{t}_{shape}', f'%y = {typed} constant({text});', t, dims)
for t, texts in SPECIAL.items():
    for k, text in enumerate(texts):
        add(ops, f'special_{t}_{k}', f'%y = {t}[2] constant({text});', t, (2,))
for t, text in LARGEST.items():
    add(ops, f'largest_{t}', f'%y = {t}[2] constant({text});', t, (2,))
for k, text in enumerate(PREDS):
    add(ops, f'pred_{k}', f'%y = pred[2] constant({text});', 'pred', (2,))
# Expressions, evaluated at each call
add(ops, 'expression_s32', '%y = s32[2] constant(`@c.k * 2 - 1`);', 's32', (2,), ['k'])
add(ops, 'expression_f32', '%y = f32[2] constant(`@c.k`);', 'f32', (2,), ['k'])
add(ops, 'expression_s8', '%y = s8[2] constant(`@c.k`);', 's8', (2,), ['k'])
for t in LOWERED['iota']:
    for dimension in (0, 1):
        line = f'%y = {t}[3,4] iota(), iota_dimension={dimension};'
        add(ops, f'iota_{t}_{dimension}', line, t, (3, 4))
# A dimension with a leading zero, which the grammar takes
add(ops, 'iota_leading_zero', '%y = s32[3,4] iota(), iota_dimension=01;', 's32', (3, 4))

generate_oracle(ops)
decorator, api = import_oracle(f'{ops.base_dir}/targets/CONSTANTS', 'oracle_CONSTANTS')


def run(name, expected, *calls):
    """Run one instruction once per call's attributes, each writing after the last, and hold the
    bytes they write to expected's"""
    calls = calls or [()]
    nbytes = expected.nbytes

    def body():
        for j, attributes in enumerate(calls):
            getattr(api, name)(*attributes, j * nbytes // len(calls))

    # Bytes, holding a nan and a signed zero too
    out = run_bytes(decorator, body, [], 0, nbytes)
    np.testing.assert_array_equal(out, expected.reshape(-1).view(np.uint8))


def test_constant():
    def check(t):
        for shape, dims in SHAPES.items():
            run(f'constant_{t}_{shape}', np.full(dims, LITERALS[t][1], DTYPES[t]))

    each_type(LOWERED['constant'], check)


def test_constant_special():
    def check(t):
        # A float rounds to its type, to nearest
        for k, text in enumerate(SPECIAL[t]):
            with np.errstate(under='ignore'):
                run(f'special_{t}_{k}', np.full(2, float(text), DTYPES[t]))

    each_type(SPECIAL, check)


def test_constant_largest():
    def check(t):
        run(f'largest_{t}', np.full(2, float(LARGEST[t]), DTYPES[t]))

    each_type(LARGEST, check)


def test_constant_pred():
    for k, text in enumerate(PREDS):
        run(f'pred_{k}', np.full(2, text in ('true', '1')))


def test_constant_expression():
    # Each call evaluates the expression afresh
    run('expression_s32', np.repeat(np.int32([9, 13]), 2), [5], [7])
    run('expression_f32', np.repeat(np.float32([3, -2]), 2), [3], [-2])


def test_constant_expression_out_of_range():
    with pytest.raises(ValueError, match='expression_s8: constant 300 at s8 is out of its range'):
        run('expression_s8', np.zeros(2, np.int8), [300])


def test_iota():
    def check(t):
        for dimension in (0, 1):
            index = np.arange(4)[None, :] if dimension else np.arange(3)[:, None]
            run(f'iota_{t}_{dimension}', np.broadcast_to(index, (3, 4)).astype(DTYPES[t]))

    each_type(LOWERED['iota'], check)


def test_iota_dimension_with_a_leading_zero():
    run('iota_leading_zero', np.broadcast_to(np.arange(4, dtype=np.int32), (3, 4)))


def refused(line, t):
    """An oracle generated from one instruction writing line's %y, a t[2]"""
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    add(acc, 'refused', line, t, (2,), ['k'])
    generate_oracle(acc)


# Literals each type cannot hold, and what the refusal says
UNHELD = {
    's32': [('2.0', 'not an integer'), ('true', 'a pred literal')],
    's16': [('1e3', 'not an integer')],
    'u8': [('-nan', 'not an integer'), ('-1', 'out of its range')],
    's8': [('128', 'out of its range')],
    'u64': [('18446744073709551616', 'out of its range')],
    'pred': [('2', 'out of its range')],
    'f16': [('1e30', 'out of its range'), ('65520', 'out of its range')],
    'f32': [('3.4028236e38', 'out of its range'), ('false', 'a pred literal')],
    'bf16': [('-1e39', 'out of its range')],
    'f64': [('1e400', 'out of its range')],
    'f8e4m3fn': [('465', 'out of its range'), ('inf', 'out of its range')],
    'f8e5m2': [('61440', 'out of its range')],
}


def test_constant_unheld_refused():
    def check(t):
        for text, reason in UNHELD[t]:
            message = f'refused: constant {text} at {t} is {reason}'
            # A literal of the wrong kind raises TypeError
            error = TypeError if reason in ('not an integer', 'a pred literal') else ValueError
            with pytest.raises(error, match=re.escape(message)):
                refused(f'%y = {t}[2] constant({text});', t)

    each_type(UNHELD, check)


def test_iota_dimension_expression_refused():
    message = "refused: an expression in 'iota_dimension' is not supported by the oracle"
    with pytest.raises(NotImplementedError, match=message):
        refused('%y = s32[2] iota(), iota_dimension=`@c.k`;', 's32')
