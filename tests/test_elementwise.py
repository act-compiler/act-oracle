"""Test the elementwise operations at each element type, against NumPy"""

import math
import tempfile

import ml_dtypes
import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    LOWERED,
    PLATFORM,
    ULPS,
    add_typed_instruction,
    assert_exact,
    each_type,
    import_oracle,
    run_typed,
    ulps,
)
from taidl import Accelerator
from taidl.operations import OPERATIONS

from taidl_to import generate_oracle

# This file's words, each at the types the table gives
WORDS = [
    'abs',
    'sign',
    'negate',
    'not',
    'popcnt',
    'count_leading_zeros',
    'is_finite',
    'reduce_precision',
    'power',
    'remainder',
    'and',
    'or',
    'shift_left',
    'shift_right_logical',
    'shift_right_arithmetic',
    'mulhi',
    'atan2',
    'sqrt',
    'rsqrt',
    'cbrt',
    'exponential_minus_one',
    'log',
    'log_plus_one',
    'logistic',
    'erf',
    'sine',
    'cosine',
    'tan',
    'asin',
    'acos',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'floor',
    'ceil',
    'round_nearest_afz',
    'round_nearest_even',
]

# Transcendental words, held within ulps
TRANSCENDENTAL = [
    'rsqrt',
    'cbrt',
    'exponential_minus_one',
    'log',
    'log_plus_one',
    'logistic',
    'erf',
    'sine',
    'cosine',
    'tan',
    'asin',
    'acos',
    'atan2',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'power',
]
# A word past its bound: twice the most it reaches
WIDER = {
    'CPU': {('exponential_minus_one', 'f16'): 4.1},
    'GPU': {('sqrt', 'f32'): 2},
}[PLATFORM]

# Each reduce_precision's exponent and mantissa bits, by its type
PRECISIONS = {'f64': (8, 23), 'f32': (5, 10), 'f16': (4, 3), 'bf16': (5, 2)}

# Where a float word's operands lie, else (-4, 4)
DOMAINS = {
    'sqrt': [(1 / 16, 16)],
    'rsqrt': [(1 / 16, 16)],
    'log': [(1 / 16, 16)],
    'log_plus_one': [(-0.9, 8)],
    'asin': [(-1, 1)],
    'acos': [(-1, 1)],
    'atanh': [(-0.95, 0.95)],
    'acosh': [(1, 16)],
    'tan': [(-1.4, 1.4)],
    'power': [(1 / 4, 4), (-3, 3)],
}

# Operands a float word singles out: zeros, infinities, nans, poles
SIGNED_ZEROS = [0.0, -0.0, np.inf, -np.inf, np.nan]
SPECIALS = {
    'is_finite': [[np.inf, -np.inf, np.nan]],
    'sqrt': [[-0.0, 0.0, np.inf, -1.0]],
    'rsqrt': [[0.0, -0.0, np.inf, -1.0, np.nan]],
    'log': [[0.0, -0.0, np.inf, -1.0, 1.0, np.nan]],
    'log_plus_one': [[-1.0, 0.0, -0.0, np.inf, -2.0, np.nan]],
    'logistic': [[np.inf, -np.inf, 0.0, np.nan]],
    'asin': [[0.0, -0.0, 1.0, -1.0, 2.0, np.nan]],
    'acos': [[0.0, -0.0, 1.0, -1.0, 2.0, np.nan]],
    'acosh': [[1.0, np.inf, 0.5, np.nan]],
    'atanh': [[0.0, -0.0, 1.0, -1.0, 2.0, np.nan]],
    'atan2': [
        [0.0, -0.0, 0.0, -0.0, 1.0, np.inf, -np.inf, np.nan],
        [-1.0, -1.0, 1.0, 1.0, 0.0, np.inf, -np.inf, 1.0],
    ],
    'power': [
        [-2.0, -2.0, np.nan, 0.0, -0.0, 1.0, -1.0, 2.0, 0.5, np.inf],
        [3.0, 2.0, 0.0, -1.0, -1.0, np.nan, np.inf, -np.inf, np.inf, -1.0],
    ],
    **{
        word: [SIGNED_ZEROS]
        for word in [
            'cbrt',
            'exponential_minus_one',
            'erf',
            'sine',
            'cosine',
            'tan',
            'sinh',
            'cosh',
            'tanh',
            'asinh',
        ]
    },
}

N = 64


def result(word, t):
    return 'pred' if word == 'is_finite' else t


def operands(word, t, rng):
    """The operands of word at t: random, with the cases its definition singles out"""
    count = OPERATIONS[word][0]
    dtype = DTYPES[t]
    if t == 'pred':
        return [rng.integers(0, 2, N).astype(np.bool_) for _ in range(count)]
    if t in FLOATS:
        return float_operands(word, t, rng)
    info = np.iinfo(dtype)
    width = info.bits
    xs = [rng.integers(info.min, info.max, N, endpoint=True, dtype=dtype) for _ in range(count)]
    # Each type's edges, each against each for two operands
    edges = np.array([0, 1, info.min, info.max], dtype)
    if count == 1:
        xs[0][:4] = edges
    else:
        xs[0][:16], xs[1][:16] = np.repeat(edges, 4), np.tile(edges, 4)
    if word == 'power':
        # Small bases, exponents below zero to 63, XLA's six bits
        xs[0] = rng.integers(max(info.min, -5), 6, N).astype(dtype)
        xs[1] = rng.integers(-3 if info.min else 0, 64, N).astype(dtype)
        # Zero to a negative power is undefined, so left out
        xs[0][(xs[1] < 0) & (xs[0] == 0)] = 2
        if info.min:
            xs[0][:3], xs[1][:3] = [-1, -1, 1], [-1, -2, -3]
    elif word == 'remainder':
        sign = rng.choice([1, -1] if info.min else [1], N)
        xs[1] = (sign * rng.integers(2, 100, N)).astype(dtype)
    elif word.startswith('shift'):
        # Amounts past the width, and below zero where signed
        xs[1] = rng.integers(-2 if info.min else 0, width + 2, N).astype(dtype)
    return xs


def float_operands(word, t, rng):
    """Float operands, exact in t"""
    count = OPERATIONS[word][0]
    dtype = DTYPES[t]
    if word in (
        'floor',
        'ceil',
        'round_nearest_afz',
        'round_nearest_even',
        'abs',
        'sign',
        'negate',
    ):
        # Quarters, so halves tie, and the zeros, infinities and nans
        x = (rng.integers(-32, 33, N) / 4).astype(dtype)
        x[:5] = [-0.0, np.inf, -np.inf, np.nan, -np.nan]
        return [x]
    if word == 'remainder':
        x = rng.uniform(-32, 32, N).astype(dtype)
        y = (rng.choice([1, -1], N) * rng.uniform(1 / 2, 4, N)).astype(dtype)
        # Signed zeros, infinite divisor and dividend, a zero divisor
        x[:6], y[:6] = [-2.0, 2.0, -0.0, 5.5, np.inf, 1.0], [1.0, -1.0, 3.0, np.inf, 2.0, 0.0]
        return [x, y]
    if word == 'reduce_precision':
        return [reduced_operand(t, rng)]
    domains = DOMAINS.get(word, [(-4, 4)] * count)
    xs = [rng.uniform(low, high, N).astype(dtype) for low, high in domains]
    for x, values in zip(xs, SPECIALS.get(word, [])):
        x[: len(values)] = values
    return xs


def reduced_operand(t, rng):
    """Magnitudes past the reduced type's range at both ends, ties of both parities and a nan"""
    exponent_bits, mantissa_bits = PRECISIONS[t]
    top = 2 ** (exponent_bits - 1) + 2
    x = (rng.choice([1, -1], N) * np.exp2(rng.uniform(-top, top, N))).astype(DTYPES[t])
    unit = np.dtype(f'u{x.dtype.itemsize}').type
    drop = ml_dtypes.finfo(x.dtype).nmant - mantissa_bits
    bits = x.view(unit)
    # Half a reduced unit past even, then odd, kept bits
    odd = (np.arange(8) % 2).astype(unit) << unit(drop)
    bits[4:12] = (bits[4:12] & ~unit((2 << drop) - 1)) | unit(1 << (drop - 1)) | odd
    x[:3] = [np.nan, np.inf, -0.0]
    return x


def round_afz(x):
    """x to the nearest integer, a tie away from zero"""
    whole = np.trunc(x)
    return np.where(np.abs(x - whole) >= 0.5, whole + np.sign(x), whole)


def reduce_precision(x, exponent_bits, mantissa_bits):
    """x rounded to mantissa_bits, ties to even, then a zero below the reduced exponent range and an
    inf above it, a nan kept"""
    finfo = ml_dtypes.finfo(x.dtype)
    width = x.dtype.itemsize * 8
    mantissa, exponent = finfo.nmant, width - 1 - finfo.nmant
    bias, reduced_bias = (1 << (exponent - 1)) - 1, (1 << (exponent_bits - 1)) - 1
    exponent_mask = ((1 << exponent) - 1) << mantissa
    sign_mask = 1 << (width - 1)
    out = []
    for bits in x.view(f'u{x.dtype.itemsize}').tolist():
        rounded = bits
        if mantissa_bits < mantissa:
            drop = mantissa - mantissa_bits
            last = (bits >> drop) & 1
            rounded = (bits + (1 << (drop - 1)) - 1 + last) & ~((1 << drop) - 1)
        if exponent_bits < exponent:
            field = rounded & exponent_mask
            if field > (bias + reduced_bias) << mantissa:
                rounded = (bits & sign_mask) | exponent_mask
            elif field <= (bias - reduced_bias) << mantissa:
                rounded = bits & sign_mask
        nan = (bits & exponent_mask) == exponent_mask and bits & ((1 << mantissa) - 1)
        out.append(bits if nan else rounded)
    return np.array(out, f'u{x.dtype.itemsize}').view(x.dtype)


def bitwise(word, x, s):
    """A shift of x by s, an amount past the width filling with zeros or the sign"""
    width = x.dtype.itemsize * 8
    mask = (1 << width) - 1
    out = []
    for value, amount in zip(x.tolist(), s.tolist()):
        bits, amount = value & mask, amount & mask
        if word == 'shift_left':
            out.append((bits << amount) & mask if amount < width else 0)
        elif word == 'shift_right_logical':
            out.append(bits >> amount if amount < width else 0)
        else:
            signed = bits - (1 << width) if bits >> (width - 1) else bits
            out.append((signed >> min(amount, width - 1)) & mask)
    return np.array(out, f'u{x.dtype.itemsize}').view(x.dtype)


def power(b, e):
    """b to the e: by wrapping products where e is at least zero, and where e is below zero 1 / b^-e
    toward zero, 0 unless b is 1 or -1"""
    if b.dtype.kind != 'i':
        return np.power(b, e)
    below = np.where(b == -1, 1 - 2 * (e % 2), b == 1)
    return np.where(e >= 0, np.power(b, np.maximum(e, 0)), below).astype(b.dtype)


def mulhi(a, b):
    """The high half of each double-width product"""
    width = a.dtype.itemsize * 8
    return np.array([(x * y) >> width for x, y in zip(a.tolist(), b.tolist())], a.dtype)


def count_leading_zeros(x):
    width = x.dtype.itemsize * 8
    bits = x.view(f'u{x.dtype.itemsize}').tolist()
    return np.array([width - value.bit_length() for value in bits], x.dtype)


def remainder(a, b):
    """The remainder of a truncated division, its sign a's"""
    return np.fmod(a, b)


# Each word's reference: float64 for floats, else the type
REFERENCES = {
    'abs': np.abs,
    'sign': lambda x: np.where(x > 0, 1, np.where(x < 0, -1, x)).astype(x.dtype),
    'negate': np.negative,
    'not': np.invert,
    'popcnt': lambda x: np.bitwise_count(x.view(f'u{x.dtype.itemsize}')).astype(x.dtype),
    'count_leading_zeros': count_leading_zeros,
    'is_finite': np.isfinite,
    'power': power,
    'remainder': remainder,
    'and': np.bitwise_and,
    'or': np.bitwise_or,
    'shift_left': lambda x, s: bitwise('shift_left', x, s),
    'shift_right_logical': lambda x, s: bitwise('shift_right_logical', x, s),
    'shift_right_arithmetic': lambda x, s: bitwise('shift_right_arithmetic', x, s),
    'mulhi': mulhi,
    'sqrt': np.sqrt,
    'rsqrt': lambda x: 1 / np.sqrt(x),
    'cbrt': np.cbrt,
    'exponential_minus_one': np.expm1,
    'log': np.log,
    'log_plus_one': np.log1p,
    'logistic': lambda x: 1 / (1 + np.exp(-x)),
    'erf': np.vectorize(math.erf),
    'sine': np.sin,
    'cosine': np.cos,
    'tan': np.tan,
    'asin': np.arcsin,
    'acos': np.arccos,
    'atan2': np.arctan2,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'asinh': np.arcsinh,
    'acosh': np.arccosh,
    'atanh': np.arctanh,
    'floor': np.floor,
    'ceil': np.ceil,
    'round_nearest_afz': round_afz,
    'round_nearest_even': np.rint,
}


def reference(word, t, xs):
    """What word computes on xs, at t's precision where IEEE defines it, in float64 otherwise"""
    if word == 'reduce_precision':
        return reduce_precision(xs[0], *PRECISIONS[t])
    if t in FLOATS:
        # An infinity, a nan or a pole in the input may give one out
        with np.errstate(invalid='ignore', divide='ignore'):
            exact = REFERENCES[word](*(x.astype(np.float64) for x in xs))
        if word in TRANSCENDENTAL or word == 'is_finite':
            return exact
        return exact.astype(DTYPES[t])
    return REFERENCES[word](*xs)


def bound(word, t):
    """The ulps word at t is held within, or None where it is held exact"""
    if (word, t) in WIDER:
        return WIDER[word, t]
    if word in TRANSCENDENTAL and t in FLOATS:
        return ULPS[t] * (2 if PLATFORM == 'GPU' else 1)
    return None


def line(word, t):
    """word at t on its operands %x<k>, giving %y"""
    values = ', '.join(f'%x{k}' for k in range(OPERATIONS[word][0]))
    given = ''
    if word == 'reduce_precision':
        exponent_bits, mantissa_bits = PRECISIONS[t]
        given = f', exponent_bits={exponent_bits}, mantissa_bits={mantissa_bits}'
    return f'%y = {result(word, t)}[{N}] {word}({values}){given};'


ops = Accelerator('ELEMENTWISE', base_dir=tempfile.mkdtemp())
for word in WORDS:
    for t in LOWERED[word]:
        inputs = [(t, [N])] * OPERATIONS[word][0]
        add_typed_instruction(ops, f'{word}_{t}', inputs, line(word, t), (result(word, t), [N]))
generate_oracle(ops)
decorator, api = import_oracle(f'{ops.base_dir}/targets/ELEMENTWISE', 'oracle_ELEMENTWISE')


@pytest.mark.parametrize('word', WORDS)
def test_elementwise(word, rng):
    def check(t):
        xs = operands(word, t, rng)
        out = run_typed(decorator, api, f'{word}_{t}', xs, (result(word, t), [N]))
        expected = reference(word, t, xs)
        limit = bound(word, t)
        if limit is None:
            assert_exact(out, expected)
        else:
            # An infinity or a nan is held exactly
            exact = np.asarray(expected, np.float64)
            finite = np.isfinite(exact)
            assert_exact(out[~finite], exact[~finite])
            assert ulps(out[finite], exact[finite], t) <= limit

    each_type(LOWERED[word], check)
