"""Test compare, select and clamp at each element type, against NumPy"""

import re
import tempfile

import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    LOWERED,
    SIGNED,
    add_typed_instruction,
    assert_exact,
    each_type,
    import_oracle,
    run_typed,
)
from taidl import Accelerator

from taidl_to import generate_oracle

DIRECTIONS = {
    'EQ': np.equal,
    'NE': np.not_equal,
    'GE': np.greater_equal,
    'GT': np.greater,
    'LE': np.less_equal,
    'LT': np.less,
}
# The comparison types each element type takes
COMPARE_TYPES = {
    **dict.fromkeys(FLOATS, ('FLOAT', 'TOTALORDER')),
    **dict.fromkeys(SIGNED, ('SIGNED',)),
    **dict.fromkeys(['u8', 'u16', 'u32', 'u64', 'pred'], ('UNSIGNED',)),
}

N = 64


def values(t, rng):
    """Small values of t, often equal, none zero"""
    if t == 'pred':
        return rng.integers(0, 2, N).astype(np.bool_)
    low = -4 if t in FLOATS or t in SIGNED else 1
    x = rng.integers(low, 5, N)
    return (np.where(x == 0, 1, x) / (2 if t in FLOATS else 1)).astype(DTYPES[t])


def edges(t):
    """The values of t an order may read wrongly"""
    if t == 'pred':
        return [False, True]
    if t in FLOATS:
        # IEEE's own: zeros, infinities and nans of each sign
        return [0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 1.5]
    info = np.iinfo(DTYPES[t])
    # Signed -1 and the unsigned top bit: each flips sign when read with the other signedness
    return [info.min, info.max, 0, 1, -1 if info.min else 1 << (info.bits - 1)]


def operands(t, rng):
    """Two operands of t: every pair of t's edges, then small values"""
    a, b = values(t, rng), values(t, rng)
    pairs = len(edges(t)) ** 2
    a[:pairs] = np.repeat(np.array(edges(t), DTYPES[t]), len(edges(t)))
    b[:pairs] = np.tile(np.array(edges(t), DTYPES[t]), len(edges(t)))
    return a, b


def total_order(x):
    """x as integers in IEEE's total order: -nan, -inf, ..., -0, +0, ..., inf, nan"""
    bits = x.view(f'i{x.dtype.itemsize}').astype(np.int64)
    return np.where(bits < 0, bits ^ np.int64(2 ** (x.dtype.itemsize * 8 - 1) - 1), bits)


def compared(direction, t, kind, a, b):
    """direction's verdict on a and b, read as kind says"""
    if kind == 'TOTALORDER':
        a, b = total_order(a), total_order(b)
    elif t in FLOATS:
        a, b = a.astype(np.float64), b.astype(np.float64)
    return DIRECTIONS[direction](a, b)


def operand_shapes(word, rank0):
    """The shapes of word's three operands, select's predicate and clamp's bounds rank 0 if rank0"""
    scalar, full = (), (N,)
    if word == 'select':
        return [scalar if rank0 else full, full, full]
    return [scalar if rank0 else full, full, scalar if rank0 else full]


def add(acc, name, t, line, shapes, types):
    """An instruction reading values of shapes and types, computing line's %y, a t[N]"""
    add_typed_instruction(acc, name, list(zip(types, shapes)), line, (t, [N]))


ops = Accelerator('COMPARE', base_dir=tempfile.mkdtemp())
for t in LOWERED['compare']:
    for kind in [None, *COMPARE_TYPES[t]]:
        for direction in DIRECTIONS:
            given = f', type={kind}' if kind else ''
            line = f'%y = pred[{N}] compare(%x0, %x1), direction={direction}{given};'
            add(ops, f'compare_{direction}_{t}_{kind}', 'pred', line, [(N,), (N,)], [t, t])
for word in ('select', 'clamp'):
    for t in LOWERED[word]:
        for rank0 in (False, True):
            shapes = operand_shapes(word, rank0)
            types = ['pred', t, t] if word == 'select' else [t, t, t]
            line = f'%y = {t}[{N}] {word}(%x0, %x1, %x2);'
            add(ops, f'{word}_{t}_{rank0}', t, line, shapes, types)
generate_oracle(ops)
decorator, api = import_oracle(f'{ops.base_dir}/targets/COMPARE', 'oracle_COMPARE')


def run(name, t, xs):
    """Run name on xs and return its output, a t[N]"""
    return run_typed(decorator, api, name, xs, (t, [N]))


def test_compare(rng):
    def check(t):
        for kind in [None, *COMPARE_TYPES[t]]:
            for direction in DIRECTIONS:
                a, b = operands(t, rng)
                out = run(f'compare_{direction}_{t}_{kind}', 'pred', [a, b])
                expected = compared(direction, t, kind, a, b)
                np.testing.assert_array_equal(out, expected, err_msg=f'{direction} {kind}')

    each_type(LOWERED['compare'], check)


def test_select(rng):
    def check(t):
        for rank0 in (False, True):
            p = rng.integers(0, 2, operand_shapes('select', rank0)[0]).astype(np.bool_)
            a, b = operands(t, rng)
            # A select moves values, so byte for byte
            out = run(f'select_{t}_{rank0}', t, [p, a, b])
            np.testing.assert_array_equal(out.view(np.uint8), np.where(p, a, b).view(np.uint8))

    each_type(LOWERED['select'], check)


def test_clamp(rng):
    def check(t):
        for rank0 in (False, True):
            shape = operand_shapes('clamp', rank0)[0]
            x, _ = operands(t, rng)
            # Bounds either way round, nans at floats, none zero
            bounds = values(t, rng)
            if t in FLOATS:
                bounds[:4] = np.nan
            low, high = rng.choice(bounds, (2, *shape))
            expected = np.minimum(np.maximum(x, low), high)
            assert_exact(run(f'clamp_{t}_{rank0}', t, [low, x, high]), expected)

    each_type(LOWERED['clamp'], check)


def test_compare_type_refused():
    def check(t):
        for kind in ['FLOAT', 'TOTALORDER', 'SIGNED', 'UNSIGNED']:
            if kind in COMPARE_TYPES[t]:
                continue
            acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
            line = f'%y = pred[{N}] compare(%x0, %x1), direction=LT, type={kind};'
            add(acc, 'refused', 'pred', line, [(N,), (N,)], [t, t])
            fitting = ' or '.join(COMPARE_TYPES[t])
            message = f"refused: compare of '%y' takes type {fitting} at {t}, found {kind}"
            with pytest.raises(ValueError, match=re.escape(message)):
                generate_oracle(acc)

    each_type(LOWERED['compare'], check)
