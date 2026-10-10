"""Test each structural operation at every element type"""

import os
import tempfile

import numpy as np
import pytest
from conftest import (
    DTYPES,
    FLOATS,
    INTEGERS,
    LOWERED,
    add_typed_instruction,
    each_type,
    import_oracle,
    run_bytes,
    run_typed,
)
from taidl import Accelerator

from taidl_to import generate_oracle

# Each word's x and index i inputs, line and shape
LINES = {
    'slice': (
        [('x', [6, 8])],
        '%y = {t}[3,3] slice(%x0), slice={{[1:6:2], [0:8:3]}};',
        [3, 3],
    ),
    'dynamic_slice': (
        [('x', [6, 8]), ('i', []), ('i', [])],
        '%y = {t}[2,3] dynamic_slice(%x0, %x1, %x2), dynamic_slice_sizes={{2,3}};',
        [2, 3],
    ),
    # A zero-width operand between two others
    'concatenate': (
        [('x', [2, 3]), ('x', [2, 4])],
        """%z = {t}[2,0] slice(%x0), slice={{[0:2], [3:3]}};
    %y = {t}[2,7] concatenate(%x0, %z, %x1), dimensions={{1}};""",
        [2, 7],
    ),
    'pad': (
        [('x', [3, 4]), ('x', [])],
        (
            '%y = {t}[4,7] pad(%x0, %x1), edge_padding_low={{-1,1}}, edge_padding_high={{0,2}}, '
            'interior_padding={{1,0}};'
        ),
        [4, 7],
    ),
    # A literal with a leading zero, which the grammar takes
    'reverse': (
        [('x', [3, 4])],
        '%y = {t}[3,4] reverse(%x0), dimensions={{01,0}};',
        [3, 4],
    ),
    'dynamic_update_slice': (
        [('x', [6, 8]), ('x', [2, 3]), ('i', []), ('i', [])],
        '%y = {t}[6,8] dynamic_update_slice(%x0, %x1, %x2, %x3);',
        [6, 8],
    ),
    # Rows of x at the indices
    'gather': (
        [('x', [5, 4]), ('i', [3, 1])],
        (
            '%y = {t}[3,4] gather(%x0, %x1), offset_dims={{1}}, collapsed_slice_dims={{0}}, '
            'start_index_map={{0}}, index_vector_dim=1, slice_sizes={{1,4}};'
        ),
        [3, 4],
    ),
}
INDEXED = ['dynamic_slice', 'dynamic_update_slice', 'gather']


def type_cases(word):
    """Every type word takes with s32 indices, then f32 with every other index type"""
    return [(t, 's32') for t in LOWERED[word]] + [('f32', i) for i in INTEGERS if i != 's32']


# More gathers at f32: shapes, attributes, indices, NumPy reference
GATHERS = {
    'windows': (
        [[4, 5], [2, 2], [2, 2, 3]],
        (
            'offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0,1}, index_vector_dim=1, '
            'slice_sizes={2,3}'
        ),
        [[1, 4], [-2, 1]],
        lambda x, i: np.stack([window(x, row, [2, 3]) for row in i]),
    ),
    'batch': (
        [[2, 5, 4], [2, 3, 1], [2, 3, 4]],
        (
            'offset_dims={2}, collapsed_slice_dims={1}, operand_batching_dims={0}, '
            'start_indices_batching_dims={0}, start_index_map={1}, index_vector_dim=2, '
            'slice_sizes={1,1,4}'
        ),
        [[[4], [0], [9]], [[1], [1], [2]]],
        lambda x, i: np.stack([x[b, np.clip(i[b, :, 0], 0, 4)] for b in range(2)]),
    ),
    # index_vector_dim at the indices' rank, with a leading zero
    'implicit': (
        [[5, 4], [3], [3, 4]],
        (
            'offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=01, '
            'slice_sizes={1,4}'
        ),
        [3, 0, -1],
        lambda x, i: x[np.clip(i, 0, 4)],
    ),
}

acc = Accelerator('STRUCTURAL', base_dir=tempfile.mkdtemp())
for word, (inputs, line, shape) in LINES.items():
    cases = type_cases(word) if word in INDEXED else [(t, 's32') for t in LOWERED[word]]
    for t, index in cases:
        declared = [(t if kind == 'x' else index, dims) for kind, dims in inputs]
        add_typed_instruction(acc, f'{word}_{t}_{index}', declared, line.format(t=t), (t, shape))
for name, ([x, i, y], given, _, _) in GATHERS.items():
    line = f'%y = f32[{",".join(map(str, y))}] gather(%x0, %x1), {given};'
    add_typed_instruction(acc, f'gather_{name}', [('f32', x), ('s32', i)], line, ('f32', y))
# Slice ranges and brace lists read at each call
add_typed_instruction(
    acc,
    'expressions',
    [('s32', [6, 8]), ('s32', []), ('s32', [])],
    """%a = s32[3,3] slice(%x0), slice={[`@c.i`:`@c.i + 5`:2], [0:`@c.n`:`@c.n / 2 - 1`]};
    %b = s32[3,3] transpose(%a), dimensions={`@c.k`, `1 - @c.k`};
    %c = s32[3,3] reverse(%b), dimensions={`@c.k`};
    %d = s32[`@c.k + 1`,3] dynamic_slice(%c, %x1, %x1), dynamic_slice_sizes={`@c.k + 1`, 3};
    %y = s32[2,7] pad(%d, %x2), edge_padding_low={`1 - @c.k`, 1}, edge_padding_high={0, 3}, interior_padding={0, 0};""",
    ('s32', [2, 7]),
    comp=['i', 'n', 'k'],
)
# A concatenate's dimension and gather's lists, read at each call
add_typed_instruction(
    acc,
    'list_expressions',
    [('s32', [4, 4]), ('s32', [4, 4]), ('s32', [2, 1])],
    """%a = s32[`4 + 4 * (1 - @c.k)`, `4 + 4 * @c.k`] concatenate(%x0, %x1), dimensions={`@c.k`};
    %b = s32[8,4] reshape(%a);
    %y = s32[2,4] gather(%b, %x2), offset_dims={`@c.one`}, collapsed_slice_dims={`@c.one - 1`}, start_index_map={0}, index_vector_dim=1, slice_sizes={`@c.one`, `4 * @c.one`};""",
    ('s32', [2, 4]),
    comp=['k', 'one'],
)
# Tile load, store: 16 rows of stride bytes, then zeros
acc.add_data_model('tiles', [2], [16, 64], 's8')
load = acc.add_instruction('tile_load', ['stride'], ['dst', 'addr'])
load.set_inputs([['d0', ['@a.addr'], ['16 * @c.stride']]])
load.set_outputs([['tiles', ['@a.dst'], ['1']]])
load.add_semantics("""
ENTRY {
    %In0 = u8[`16 * @c.stride`] parameter(0);
    %a = s8[`16 * @c.stride`] bitcast_convert(%In0);
    %data = s8[1,16,`@c.stride`] reshape(%a);
    %zero = s8[] constant(0);
    %zeros = s8[1,16,`64 - @c.stride`] broadcast(%zero), dimensions={};
    ROOT %Out0 = s8[1,16,64] concatenate(%data, %zeros), dimensions={2};
}
""")
store = acc.add_instruction('tile_store', ['stride'], ['src', 'addr'])
store.set_inputs([['tiles', ['@a.src'], ['1']]])
store.set_outputs([['d0', ['@a.addr'], ['16 * @c.stride']]])
store.add_semantics("""
ENTRY {
    %In0 = s8[1,16,64] parameter(0);
    %tmp = s8[1,16,`@c.stride`] slice(%In0), slice={[0:1:1], [0:16:1], [0:`@c.stride`:1]};
    %data = s8[`16 * @c.stride`] reshape(%tmp);
    ROOT %Out0 = u8[`16 * @c.stride`] bitcast_convert(%data);
}
""")
# A row store: each row's first 16 bytes, constant indices
row_store = acc.add_instruction('row_store', ['rows', 'stride'], ['src', 'addr'])
row_store.set_inputs(
    [['d0', ['@a.src'], ['@c.rows * 16']], ['d0', ['@a.addr'], ['@c.rows * @c.stride']]]
)
row_store.set_outputs([['d0', ['@a.addr'], ['@c.rows * @c.stride']]])
row_store.add_semantics("""
ENTRY {
    %In0 = u8[`@c.rows * 16`] parameter(0);
    %In1 = u8[`@c.rows * @c.stride`] parameter(1);
    %orig = u8[`@c.rows`, `@c.stride / 16`, 16] reshape(%In1);
    %data = u8[`@c.rows`, 1, 16] reshape(%In0);
    %zero = s32[] constant(0);
    %fin = u8[`@c.rows`, `@c.stride / 16`, 16] dynamic_update_slice(%orig, %data, %zero, %zero, %zero);
    ROOT %Out0 = u8[`@c.rows * @c.stride`] reshape(%fin);
}
""")
generate_oracle(acc)
decorator, api = import_oracle(
    os.path.join(acc.base_dir, 'targets', 'STRUCTURAL'), 'oracle_STRUCTURAL'
)


def sample(rng, t, shape):
    """Values of t: random bits of an integer, normal values of a float, either of a pred"""
    if t in FLOATS:
        return rng.standard_normal(shape).astype(DTYPES[t])
    if t == 'pred':
        return rng.integers(0, 2, shape).astype(np.bool_)
    info = np.iinfo(DTYPES[t])
    return rng.integers(info.min, info.max, shape, dtype=DTYPES[t], endpoint=True)


def run(word, t, index, args):
    _, _, shape = LINES[word]
    return run_typed(decorator, api, f'{word}_{t}_{index}', args, (t, shape))


def assert_same_bits(out, expected):
    expected = np.ascontiguousarray(expected).astype(out.dtype)
    np.testing.assert_array_equal(out.view(np.uint8), expected.view(np.uint8))


def starts(indices, shape, sizes):
    """Start indices clamped so a window of sizes fits in shape, as XLA clamps them"""
    return [min(max(int(i), 0), n - k) for i, n, k in zip(indices, shape, sizes)]


def window(x, indices, sizes):
    return x[tuple(slice(s, s + k) for s, k in zip(starts(indices, x.shape, sizes), sizes))]


def pad(x, value, low, high, interior):
    """XLA's pad: value between elements and at the edges, a negative edge cropping"""
    dilated = np.full([n + (n - 1) * i for n, i in zip(x.shape, interior)], value, x.dtype)
    dilated[tuple(slice(None, None, i + 1) for i in interior)] = x
    edges = [(max(lo, 0), max(hi, 0)) for lo, hi in zip(low, high)]
    padded = np.pad(dilated, edges, constant_values=value)
    return padded[
        tuple(slice(max(-lo, 0), n - max(-hi, 0)) for n, lo, hi in zip(padded.shape, low, high))
    ]


def index_values(index):
    """Start indices in range, past the end, and before the start where index is signed"""
    return [(1, 2), (5, 7)] + ([(-3, -1)] if index.startswith('s') else [])


def test_slice(rng):
    def check(t):
        x = sample(rng, t, [6, 8])
        assert_same_bits(run('slice', t, 's32', [x]), x[1:6:2, 0:8:3])

    each_type(LOWERED['slice'], check)


def test_dynamic_slice(rng):
    def check(case):
        t, index = case
        x = sample(rng, t, [6, 8])
        for i, j in index_values(index):
            args = [x, np.asarray(i, DTYPES[index]), np.asarray(j, DTYPES[index])]
            assert_same_bits(run('dynamic_slice', t, index, args), window(x, [i, j], [2, 3]))

    each_type(type_cases('dynamic_slice'), check)


def test_concatenate(rng):
    def check(t):
        a, b = sample(rng, t, [2, 3]), sample(rng, t, [2, 4])
        assert_same_bits(run('concatenate', t, 's32', [a, b]), np.concatenate([a, b], axis=1))

    each_type(LOWERED['concatenate'], check)


def test_pad(rng):
    def check(t):
        x, value = sample(rng, t, [3, 4]), sample(rng, t, [])
        expected = pad(x, value, [-1, 1], [0, 2], [1, 0])
        assert_same_bits(run('pad', t, 's32', [x, value]), expected)

    each_type(LOWERED['pad'], check)


def test_reverse(rng):
    def check(t):
        x = sample(rng, t, [3, 4])
        assert_same_bits(run('reverse', t, 's32', [x]), x[::-1, ::-1])

    each_type(LOWERED['reverse'], check)


def test_dynamic_update_slice(rng):
    def check(case):
        t, index = case
        x, u = sample(rng, t, [6, 8]), sample(rng, t, [2, 3])
        for i, j in index_values(index):
            args = [x, u, np.asarray(i, DTYPES[index]), np.asarray(j, DTYPES[index])]
            expected = x.copy()
            window_starts = starts([i, j], x.shape, u.shape)
            expected[tuple(slice(s, s + k) for s, k in zip(window_starts, u.shape))] = u
            assert_same_bits(run('dynamic_update_slice', t, index, args), expected)

    each_type(type_cases('dynamic_update_slice'), check)


def test_gather(rng):
    def check(case):
        t, index = case
        x = sample(rng, t, [5, 4])
        # Rows in range and past the end, clamped
        rows = np.asarray([[4], [0], [7]], DTYPES[index])
        assert_same_bits(run('gather', t, index, [x, rows]), x[np.clip(rows[:, 0], 0, 4)])

    each_type(type_cases('gather'), check)


@pytest.mark.parametrize('name', GATHERS)
def test_gather_dimension_numbers(name, rng):
    [x, _, y], _, indices, reference = GATHERS[name]
    x, i = sample(rng, 'f32', x), np.asarray(indices, np.int32)
    out = run_typed(decorator, api, f'gather_{name}', [x, i], ('f32', y))
    assert_same_bits(out, reference(x, i))


@pytest.mark.parametrize('i, n, k', [(1, 8, 0), (0, 6, 1)])
def test_expressions_read_at_the_call(i, n, k, rng):
    x, start, value = sample(rng, 's32', [6, 8]), np.asarray(1, np.int32), np.asarray(-7, np.int32)
    a = x[i : i + 5 : 2, 0 : n : n // 2 - 1]
    b = a.T if k else a
    c = np.flip(b, k)
    d = window(c, [1, 1], [k + 1, 3])
    expected = pad(d, value, [1 - k, 1], [0, 3], [0, 0])
    out = run_typed(decorator, api, 'expressions', [x, start, value], ('s32', [2, 7]), (i, n, k))
    assert_same_bits(out, expected)


@pytest.mark.parametrize('k', [0, 1])
def test_list_expressions_read_at_the_call(k, rng):
    a, b = sample(rng, 's32', [4, 4]), sample(rng, 's32', [4, 4])
    rows = np.asarray([[5], [9]], np.int32)
    out = run_typed(decorator, api, 'list_expressions', [a, b, rows], ('s32', [2, 4]), (k, 1))
    joined = np.concatenate([a, b], axis=k).reshape(8, 4)
    assert_same_bits(out, joined[np.clip(rows[:, 0], 0, 7)])


# Each indexed word with a non-integer index
REFUSED_INDICES = {
    'dynamic_slice': (
        'f32',
        '%y = s32[2,3] dynamic_slice(%x0, %x1, %x1), dynamic_slice_sizes={2,3};',
        [2, 3],
    ),
    'dynamic_update_slice': (
        'pred',
        '%y = s32[6,8] dynamic_update_slice(%x0, %x0, %x1, %x1);',
        [6, 8],
    ),
    'gather': (
        'f16',
        (
            '%y = s32[1,8] gather(%x0, %x1), offset_dims={1}, collapsed_slice_dims={0}, '
            'start_index_map={0}, index_vector_dim=0, slice_sizes={1,8};'
        ),
        [1, 8],
    ),
}


@pytest.mark.parametrize('word', REFUSED_INDICES)
def test_index_of_another_type_refused(word):
    index, line, shape = REFUSED_INDICES[word]
    refused = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    inputs = [('s32', [6, 8]), (index, [])]
    add_typed_instruction(refused, 'refused', inputs, line, ('s32', shape))
    message = f"refused: operation '{word}' at an index of type '{index}'"
    with pytest.raises(NotImplementedError, match=message):
        generate_oracle(refused)


@pytest.mark.parametrize('stride', [16, 40, 64])
def test_tile_load_and_store(stride, rng):
    data = sample(rng, 'u8', [16 * stride])

    # Back at its stride, then at the full 64-byte rows
    def calls():
        api.tile_load(stride, 0, 0)
        api.tile_store(stride, 0, 4096)
        api.tile_store(64, 0, 8192)

    out = run_bytes(decorator, calls, [data], 4096, 8192)
    np.testing.assert_array_equal(out[: 16 * stride], data)
    rows = np.zeros((16, 64), np.uint8)
    rows[:, :stride] = data.reshape(16, stride)
    np.testing.assert_array_equal(out[4096 : 4096 + 1024], rows.reshape(-1))


@pytest.mark.parametrize('rows, stride', [(4, 64), (3, 16)])
def test_row_store(rows, stride, rng):
    src, dst = sample(rng, 'u8', [rows * 16]), sample(rng, 'u8', [rows * stride])
    out = run_bytes(
        decorator, lambda: api.row_store(rows, stride, 0, len(src)), [src, dst], len(src), len(dst)
    )
    expected = dst.reshape(rows, stride).copy()
    expected[:, :16] = src.reshape(rows, 16)
    np.testing.assert_array_equal(out, expected.reshape(-1))
