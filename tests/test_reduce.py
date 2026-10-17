"""Test each reduce at every element type it takes"""

import os
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
    run_typed,
)
from taidl import Accelerator

from taidl_to import generate_oracle

# Each reduce: its NumPy reducer and the one over pred
REDUCES = {
    'reduce_add': (np.add, np.logical_or),
    'reduce_multiply': (np.multiply, np.logical_and),
    'reduce_max': (np.maximum, np.logical_or),
    'reduce_min': (np.minimum, np.logical_and),
    'reduce_and': (np.bitwise_and, np.logical_and),
    'reduce_or': (np.bitwise_or, np.logical_or),
    'reduce_xor': (np.bitwise_xor, np.logical_xor),
}
CASES = [(word, t) for word in REDUCES for t in LOWERED[word]]

# Dims listed out of order
SHAPE, DIMS, AXES = (3, 4, 5), '{2,0}', (0, 2)

acc = Accelerator('REDUCE', base_dir=tempfile.mkdtemp())
for word, t in CASES:
    add_typed_instruction(
        acc,
        f'{word}_{t}',
        [(t, SHAPE)],
        f'%y = {t}[4] {word}(%x0), dimensions={DIMS};',
        (t, [4]),
    )
    # Over an empty dimension a reduce gives its identity
    add_typed_instruction(
        acc,
        f'{word}_{t}_empty',
        [(t, [1])],
        f"""%s = {t}[] reshape(%x0);
    %e = {t}[4,0] broadcast(%s), dimensions={{}};
    %y = {t}[4] {word}(%e), dimensions={{1}};""",
        (t, [4]),
    )
    # Over every dimension, to a rank-0 result
    if t == 's32':
        line = f'%y = {t}[] {word}(%x0), dimensions={{0,1,2}};'
        add_typed_instruction(acc, f'{word}_{t}_all', [(t, SHAPE)], line, (t, []))
generate_oracle(acc)
decorator, api = import_oracle(os.path.join(acc.base_dir, 'targets', 'REDUCE'), 'oracle_REDUCE')


def sample(rng, t):
    """Values of t whose sums and products every float type holds exactly"""
    if t in FLOATS:
        values = rng.choice([-2, -1, -0.5, 0.5, 1, 2], SHAPE)
        return values.astype(DTYPES[t])
    if t == 'pred':
        return rng.integers(0, 2, SHAPE).astype(np.bool_)
    info = np.iinfo(DTYPES[t])
    return rng.integers(info.min, info.max, SHAPE, dtype=DTYPES[t], endpoint=True)


def reference(word, t, x):
    """The reduce over AXES in NumPy: floats in f64, integers wrapping in their type"""
    reducer, logical = REDUCES[word]
    if t == 'pred':
        return logical.reduce(x, axis=AXES)
    if t in FLOATS:
        # inf - inf is nan, as IEEE has it
        with np.errstate(invalid='ignore'):
            return reducer.reduce(x.astype(np.float64), axis=AXES).astype(DTYPES[t])
    return reducer.reduce(x, axis=AXES, dtype=DTYPES[t])


@pytest.mark.parametrize('word', REDUCES)
def test_reduce(word, rng):
    def check(t):
        x = sample(rng, t)
        out = run_typed(decorator, api, f'{word}_{t}', [x], (t, [4]))
        np.testing.assert_array_equal(out, reference(word, t, x))

    each_type(LOWERED[word], check)


def identity(word, t):
    """A reducer's identity in t: NumPy's where its ufunc has one, else the extreme of t"""
    reducer, logical = REDUCES[word]
    if t == 'pred':
        return logical.identity
    if word not in ('reduce_max', 'reduce_min'):
        return reducer.identity
    if t in FLOATS:
        return -np.inf if word == 'reduce_max' else np.inf
    info = np.iinfo(DTYPES[t])
    return info.min if word == 'reduce_max' else info.max


@pytest.mark.parametrize('word', REDUCES)
def test_reduce_to_rank_0(word, rng):
    x = sample(rng, 's32')
    out = run_typed(decorator, api, f'{word}_s32_all', [x], ('s32', []))
    np.testing.assert_array_equal(out, REDUCES[word][0].reduce(x, axis=None, dtype=np.int32))


@pytest.mark.parametrize('word', REDUCES)
def test_reduce_identity(word):
    def check(t):
        x = np.ones(1, DTYPES[t])
        out = run_typed(decorator, api, f'{word}_{t}_empty', [x], (t, [4]))
        expected = np.full(4, identity(word, t)).astype(DTYPES[t])
        # Bit for bit, the sign of a zero included
        np.testing.assert_array_equal(out.view(np.uint8), expected.view(np.uint8))

    each_type(LOWERED[word], check)


@pytest.mark.parametrize('word', ['reduce_add', 'reduce_multiply', 'reduce_max', 'reduce_min'])
def test_reduce_non_finite(word, rng):
    def check(t):
        x = sample(rng, t)
        # +inf, both infinities and a nan in three outputs' operands
        x[0, 0, 0], x[1, 1, 1], x[2, 1, 2], x[0, 2, 3] = np.inf, -np.inf, np.inf, np.nan
        out = run_typed(decorator, api, f'{word}_{t}', [x], (t, [4]))
        # In f64, where a nan equals a nan
        expected = reference(word, t, x).astype(np.float64)
        np.testing.assert_array_equal(out.astype(np.float64), expected)

    each_type(FLOATS, check)
