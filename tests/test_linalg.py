"""Test cholesky, triangular_solve and batch_norm_inference at every float type"""

import itertools
import os
import tempfile

import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    LOWERED,
    PLATFORM,
    add_typed_instruction,
    each_type,
    import_oracle,
    run_typed,
)
from taidl import Accelerator

from taidl_to import generate_oracle

# cholesky's lower: given true, given false, or left out
LOWERS = {'lower': ', lower=true', 'upper': ', lower=false', 'default': ''}

# triangular_solve's flags and transpose_a, each case with every attribute written
SOLVES = list(
    itertools.product(
        ['true', 'false'],
        ['true', 'false'],
        ['true', 'false'],
        ['NO_TRANSPOSE', 'TRANSPOSE', 'ADJOINT'],
    )
)

# batch_norm_inference's operand shape, feature_index and epsilon
NORMS = {
    'middle': ([2, 3, 4], '1', '0.25'),
    'first': ([2, 3, 4], '0', '2.5e-1'),
    # A literal with a leading zero, which the grammar takes
    'last': ([2, 3, 4], '02', '0'),
    # A dotless float, too small to change a variance
    'tiny': ([2, 3, 4], '1', '1e-30'),
}


def solve_name(t, left_side, lower, unit_diagonal, transpose_a):
    return f'solve_{t}_{left_side}_{lower}_{unit_diagonal}_{transpose_a}'


def b_shape(left_side):
    return [4, 3] if left_side == 'true' else [3, 4]


acc = Accelerator('LINALG', base_dir=tempfile.mkdtemp())
for t in FLOATS:
    for name, given in LOWERS.items():
        line = f'%y = {t}[2,4,4] cholesky(%x0){given};'
        add_typed_instruction(acc, f'cholesky_{t}_{name}', [(t, [2, 4, 4])], line, (t, [2, 4, 4]))
    for case in SOLVES:
        left_side, lower, unit_diagonal, transpose_a = case
        b = ','.join(map(str, b_shape(left_side)))
        line = (
            f'%y = {t}[{b}] triangular_solve(%x0, %x1), left_side={left_side}, lower={lower}, '
            f'unit_diagonal={unit_diagonal}, transpose_a={transpose_a};'
        )
        inputs = [(t, [4, 4]), (t, b_shape(left_side))]
        add_typed_instruction(acc, solve_name(t, *case), inputs, line, (t, b_shape(left_side)))
    # The three flags left out, false as in XLA-HLO
    line = f'%y = {t}[3,4] triangular_solve(%x0, %x1), transpose_a=NO_TRANSPOSE;'
    add_typed_instruction(acc, f'solve_{t}_default', [(t, [4, 4]), (t, [3, 4])], line, (t, [3, 4]))
    for name, (shape, feature, epsilon) in NORMS.items():
        dims = ','.join(map(str, shape))
        line = (
            f'%y = {t}[{dims}] batch_norm_inference(%x0, %x1, %x2, %x3, %x4), '
            f'epsilon={epsilon}, feature_index={feature};'
        )
        inputs = [(t, shape)] + [(t, [shape[int(feature)]])] * 4
        add_typed_instruction(acc, f'norm_{t}_{name}', inputs, line, (t, shape))
generate_oracle(acc)
decorator, api = import_oracle(os.path.join(acc.base_dir, 'targets', 'LINALG'), 'oracle_LINALG')


# Small integers and powers of two keep every step exact
def triangle(rng, lower, diagonal, other):
    """A 4x4 triangle of small integers on a power-of-two diagonal, the other triangle filled with other"""
    a = np.full((4, 4), float(other))
    keep = np.tril if lower else np.triu
    a = np.where(keep(np.ones((4, 4))) > 0, rng.integers(-1, 2, (4, 4)), a)
    np.fill_diagonal(a, diagonal)
    return a


def test_cholesky(rng):
    def check(t):
        for name in LOWERS:
            factors = np.stack([triangle(rng, True, rng.choice([1, 2, 4], 4), 0) for _ in range(2)])
            a = factors @ factors.transpose(0, 2, 1)
            out = run_typed(
                decorator, api, f'cholesky_{t}_{name}', [a.astype(DTYPES[t])], (t, [2, 4, 4])
            )
            out = out.astype(np.float64)
            # The other triangle is the implementation's own
            if name == 'lower':
                np.testing.assert_array_equal(np.tril(out), factors, err_msg=name)
            else:
                np.testing.assert_array_equal(
                    np.triu(out), factors.transpose(0, 2, 1), err_msg=name
                )

    each_type(LOWERED['cholesky'], check)


def solve(rng, t):
    """triangular_solve at t, with every attribute written, then with the flags left out"""
    for case in SOLVES:
        left_side, lower, unit_diagonal, transpose_a = case
        a = triangle(rng, lower == 'true', rng.choice([1, 2, 4, 0.5], 4), 7)
        used = np.where((np.tril if lower == 'true' else np.triu)(np.ones((4, 4))) > 0, a, 0)
        if unit_diagonal == 'true':
            # The diagonal is read as ones, whatever it holds
            a[np.diag_indices(4)] = 3
            np.fill_diagonal(used, 1)
        op = used if transpose_a == 'NO_TRANSPOSE' else used.T
        x = rng.integers(-2, 3, b_shape(left_side)).astype(np.float64)
        b = op @ x if left_side == 'true' else x @ op
        args = [a.astype(DTYPES[t]), b.astype(DTYPES[t])]
        out = run_typed(decorator, api, solve_name(t, *case), args, (t, b_shape(left_side)))
        np.testing.assert_array_equal(out.astype(np.float64), x, err_msg='-'.join(case))
    # Upper, right side; no diagonal 1, so unit_diagonal would show
    a = triangle(rng, False, [2, 0.5, 4, 2], 7)
    x = rng.integers(-2, 3, (3, 4)).astype(np.float64)
    b = x @ np.triu(a)
    args = [a.astype(DTYPES[t]), b.astype(DTYPES[t])]
    out = run_typed(decorator, api, f'solve_{t}_default', args, (t, [3, 4]))
    np.testing.assert_array_equal(out.astype(np.float64), x, err_msg='default flags')


def test_triangular_solve(rng):
    each_type([t for t in LOWERED['triangular_solve'] if t != 'f16'], lambda t: solve(rng, t))


@pytest.mark.skipif(PLATFORM == 'GPU', reason="the GPU's triangular_solve takes no f16")
def test_triangular_solve_at_f16(rng):
    solve(rng, 'f16')


def test_batch_norm_inference(rng):
    def check(t):
        for name, (shape, feature, epsilon) in NORMS.items():
            feature = int(feature)
            n = shape[feature]
            x = rng.integers(-4, 5, shape).astype(np.float64)
            scale, offset = (
                rng.choice([1, 2, -1, 0.5], n),
                rng.integers(-3, 4, n).astype(np.float64),
            )
            mean = rng.integers(-2, 3, n).astype(np.float64)
            # variance + epsilon is a power of four: exact roots
            variance = rng.choice([4, 1, 16, 0.25], n) - float(epsilon)
            args = [v.astype(DTYPES[t]) for v in [x, scale, offset, mean, variance]]
            out = run_typed(decorator, api, f'norm_{t}_{name}', args, (t, shape))
            along = [n if k == feature else 1 for k in range(len(shape))]
            vectors = (v.reshape(along) for v in [scale, offset, mean, variance])
            scale, offset, mean, variance = vectors
            expected = (x - mean) * scale / np.sqrt(variance + float(epsilon)) + offset
            np.testing.assert_array_equal(out.astype(np.float64), expected, err_msg=name)

    each_type(LOWERED['batch_norm_inference'], check)


def test_epsilon_past_f32_refused():
    refused = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    line = (
        '%y = f32[4,4] batch_norm_inference(%x0, %x1, %x1, %x1, %x1), epsilon=1e39, '
        'feature_index=0;'
    )
    inputs = [('f32', [4, 4]), ('f32', [4])]
    add_typed_instruction(refused, 'refused', inputs, line, ('f32', [4, 4]))
    message = 'refused: epsilon 1e39 at f32 is out of its range'
    with pytest.raises(ValueError, match=message):
        generate_oracle(refused)
