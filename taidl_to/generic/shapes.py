"""The shape and type each lowered word gives its result, checked at generation and at each call

A value is its element type and its dims, each dim an int or an expression's text.
A list attribute holds decimal text, which numbers reads as ints, or an expression's text.
"""

# Bits per element in memory, eight for a pred
WIDTHS = {
    'f64': 64,
    'f32': 32,
    'f16': 16,
    'bf16': 16,
    'f8e4m3fn': 8,
    'f8e5m2': 8,
    'u2': 2,
    'u4': 4,
    'u8': 8,
    'u16': 16,
    'u32': 32,
    'u64': 64,
    's2': 2,
    's4': 4,
    's8': 8,
    's16': 16,
    's32': 32,
    's64': 64,
    'pred': 8,
}


def declared(result_type) -> list:
    """A result type's dims, as the spec writes them."""
    shape = result_type.shape()
    dims = shape.dim() if shape else []
    return [int(dim.getText()) if dim.INT() else dim.getText() for dim in dims]


def numbers(entries) -> list:
    """An attribute list's entries, each an int or an expression's text."""
    return [int(entry) if entry.lstrip('-').isdigit() else entry for entry in entries]


def known(entries) -> bool:
    """Whether every entry is an integer, none an expression's."""
    return all(isinstance(entry, int) for entry in entries)


def text(value) -> str:
    """A value's type as a semantics writes it."""
    typ, dims = value
    return f'{typ}[{",".join(str(dim) for dim in dims)}]'


def braces(dims) -> str:
    return '{' + ','.join(str(dim) for dim in dims) + '}'


def agree(dims, others) -> bool:
    """Whether two shapes can be equal: one rank, and no two integers apart."""
    return len(dims) == len(others) and all(
        a == b or not (isinstance(a, int) and isinstance(b, int)) for a, b in zip(dims, others)
    )


def within(sizes, dims) -> bool:
    """Whether each size is at least 0 and at most its dim, where both are integers."""
    return len(sizes) == len(dims) and all(
        not known([size, dim]) or 0 <= size <= dim for size, dim in zip(sizes, dims)
    )


def count(dims):
    """A shape's element count, or None where a dim is an expression."""
    total = 1
    for dim in dims:
        if not isinstance(dim, int):
            return None
        total *= dim
    return total


def valid(dims, rank) -> bool:
    """Whether dimension numbers are distinct and below a rank, or one is an expression."""
    if not known(dims):
        return True
    return len(set(dims)) == len(dims) and all(0 <= dim < rank for dim in dims)


def misnumbered(dims, name, value):
    """Dimension numbers of an operand that repeat or reach past its rank."""
    rank = len(value[1])
    if valid(dims, rank):
        return None
    found = braces(dims)
    return f'takes distinct dimensions below the rank {rank} of {name} {text(value)}, found {found}'


def typed(result, operands):
    """Each operand of the result's element type."""
    for name, value in operands:
        if value[0] != result[0]:
            return f'takes operands of its type {result[0]}, found {name} {text(value)}'
    return None


def gives(dims, result, operands, typ=None):
    """The result's type and shape against the ones its operands give, its own type by default."""
    typ = typ or result[0]
    if typ == result[0] and agree(dims, result[1]):
        return None
    given = ' and '.join(f'{name} {text(value)}' for name, value in operands)
    return f'gives {text((typ, dims))} of {given}, not {text(result)}'


def square(name, value):
    """A value whose last two dimensions are not a square matrix's."""
    if len(value[1]) >= 2 and agree(value[1][-1:], value[1][-2:-1]):
        return None
    return f'takes square matrices in the last two dimensions, found {name} {text(value)}'


def indexes(name, value, indices):
    """One index of rank 0 per dimension of an operand, every index of one type."""
    if len(indices) != len(value[1]):
        return f'takes one index per dimension of {name} {text(value)}, found {len(indices)}'
    for index, (typ, dims) in indices:
        if dims or typ != indices[0][1][0]:
            return f'takes indices of rank 0 and one type, found {index} {text((typ, dims))}'
    return None


def free(result, operands, attributes):
    """A word whose result takes any shape: a constant's literal fills it."""


def elementwise(result, operands, attributes):
    for name, value in operands:
        if not agree(value[1], result[1]):
            return f'takes operands of its shape {text(result)}, found {name} {text(value)}'
    return typed(result, operands)


def iota(result, operands, attributes):
    rank, dim = len(result[1]), int(attributes['iota_dimension'])
    if 0 <= dim < rank:
        return None
    return f'takes an iota_dimension below the rank {rank} of its result, found {dim}'


def is_finite(result, operands, attributes):
    return gives(operands[0][1][1], result, operands, 'pred')


def reduce_precision(result, operands, attributes):
    exponent, mantissa = int(attributes['exponent_bits']), int(attributes['mantissa_bits'])
    if exponent < 1 or mantissa < 0:
        return f'takes at least 1 exponent bit and 0 mantissa bits, found {exponent} and {mantissa}'
    return elementwise(result, operands, attributes)


def compare(result, operands, attributes):
    (lhs, a), (rhs, b) = operands
    if a[0] != b[0] or not agree(a[1], b[1]):
        return f'takes operands alike, found {lhs} {text(a)} and {rhs} {text(b)}'
    return gives(a[1], result, operands, 'pred')


def select(result, operands, attributes):
    (name, pred), *branches = operands
    if pred[0] != 'pred' or (pred[1] and not agree(pred[1], result[1])):
        return f'takes a pred of its shape or of rank 0, found {name} {text(pred)}'
    return elementwise(result, branches, attributes)


def clamp(result, operands, attributes):
    low, operand, high = operands
    for name, value in (low, high):
        if value[1] and not agree(value[1], result[1]):
            return f'takes bounds of its shape or of rank 0, found {name} {text(value)}'
    return elementwise(result, [operand], attributes) or typed(result, [low, high])


def convert(result, operands, attributes):
    return gives(operands[0][1][1], result, operands)


def bitcast_convert(result, operands, attributes):
    name, (typ, dims) = operands[0]
    width, result_width = WIDTHS[typ], WIDTHS[result[0]]
    # Narrowing adds a last dimension; widening packs it away
    if width > result_width:
        dims = dims + [width // result_width]
    elif width < result_width:
        last = result_width // width
        if not agree(dims[-1:], [last]):
            return (
                f'takes an operand whose last dimension is {last}, found {name} {text((typ, dims))}'
            )
        dims = dims[:-1]
    return gives(dims, result, operands)


def reshape(result, operands, attributes):
    name, value = operands[0]
    size, result_size = count(value[1]), count(result[1])
    if None not in (size, result_size) and size != result_size:
        return f'takes an operand of {result_size} elements, found {name} {text(value)}'
    return typed(result, operands)


def transpose(result, operands, attributes):
    name, value = operands[0]
    permutation = numbers(attributes['dimensions'])
    if not known(permutation):
        return typed(result, operands)
    if sorted(permutation) != list(range(len(value[1]))):
        rank = len(value[1])
        return (
            f'takes a permutation of the {rank} dimensions of {name} {text(value)}, '
            f'found {braces(permutation)}'
        )
    return typed(result, operands) or gives([value[1][d] for d in permutation], result, operands)


def broadcast(result, operands, attributes):
    name, value = operands[0]
    mapped = numbers(attributes['dimensions'])
    if not known(mapped):
        return typed(result, operands)
    if len(mapped) != len(value[1]) or not valid(mapped, len(result[1])):
        return (
            f'takes one distinct dimension below {len(result[1])} per dimension of '
            f'{name} {text(value)}, found {braces(mapped)}'
        )
    # A dimension of 1 spreads, any other keeps its size
    for k, (dim, size) in enumerate(zip(mapped, value[1])):
        if size != 1 and not agree([size], [result[1][dim]]):
            return f'maps dimension {k} of {name} {text(value)} to one of {result[1][dim]}'
    return typed(result, operands)


def concatenate(result, operands, attributes):
    (first, value), *_ = operands
    (dim,) = numbers(attributes['dimensions'])
    rank = len(value[1])
    for name, other in operands:
        if len(other[1]) != rank:
            return (
                f'takes operands of one rank, found {first} {text(value)} and {name} {text(other)}'
            )
    if not isinstance(dim, int):
        return typed(result, operands)
    if not 0 <= dim < rank:
        return f'takes a dimension below the rank {rank} of {first} {text(value)}, found {dim}'
    # Alike but along the dimension, whose sizes add up
    for name, other in operands:
        if not agree(other[1][:dim] + other[1][dim + 1 :], value[1][:dim] + value[1][dim + 1 :]):
            return f'takes operands alike off dimension {dim}, found {name} {text(other)}'
    sizes = [other[1][dim] for _, other in operands]
    joined = value[1][:dim] + [sum(sizes) if known(sizes) else '?'] + value[1][dim + 1 :]
    return typed(result, operands) or gives(joined, result, operands)


def pad(result, operands, attributes):
    (name, value), (padding, given) = operands
    if given[1]:
        return f'takes a padding value of rank 0, found {padding} {text(given)}'
    lists = ['edge_padding_low', 'edge_padding_high', 'interior_padding']
    low, high, interior = (numbers(attributes[key]) for key in lists)
    rank = len(value[1])
    if not len(low) == len(high) == len(interior) == rank:
        return f'takes edge and interior padding per dimension of {name} {text(value)}'
    if any(isinstance(gap, int) and gap < 0 for gap in interior):
        return f'takes interior padding of 0 or more, found {braces(interior)}'
    # Each dimension with its edges and its gaps between elements
    dims = [
        lo + size + hi + max(size - 1, 0) * gap if known([lo, size, hi, gap]) else '?'
        for lo, size, hi, gap in zip(low, value[1], high, interior)
    ]
    return typed(result, operands) or gives(dims, result, operands)


def reverse(result, operands, attributes):
    name, value = operands[0]
    dims = numbers(attributes['dimensions'])
    return misnumbered(dims, name, value) or elementwise(result, operands, attributes)


def slice_ranges(result, operands, attributes):
    name, value = operands[0]
    ranges = [numbers(bounds) for bounds in attributes['slice']]
    if len(ranges) != len(value[1]):
        return f'takes one range per dimension of {name} {text(value)}, found {len(ranges)}'
    dims = []
    for (start, limit, stride), size in zip(ranges, value[1]):
        if not known([start, limit, stride]):
            dims.append('?')
        elif stride < 1 or not within([start, limit], [limit, size]):
            return f'takes ranges within {name} {text(value)}, found [{start}:{limit}:{stride}]'
        else:
            dims.append(-(-(limit - start) // stride))
    return typed(result, operands) or gives(dims, result, operands)


def dynamic_slice(result, operands, attributes):
    (name, value), *indices = operands
    sizes = numbers(attributes['dynamic_slice_sizes'])
    if not within(sizes, value[1]):
        return f'takes slice sizes within {name} {text(value)}, found {braces(sizes)}'
    reason = indexes(name, value, indices)
    return reason or typed(result, operands[:1]) or gives(sizes, result, operands[:1])


def dynamic_update_slice(result, operands, attributes):
    (name, value), (update, given), *indices = operands
    if not within(given[1], value[1]):
        return f'takes an update within {name} {text(value)}, found {update} {text(given)}'
    reason = indexes(name, value, indices)
    return reason or typed(result, operands[:2]) or gives(value[1], result, operands[:1])


# A gather's dimension lists, in StableHLO's order
GATHER_LISTS = [
    'offset_dims',
    'collapsed_slice_dims',
    'operand_batching_dims',
    'start_indices_batching_dims',
    'start_index_map',
]
# The two a spec may leave out, empty then
GATHER_DEFAULTS = {'operand_batching_dims': [], 'start_indices_batching_dims': []}


def gather(result, operands, attributes):
    (name, value), (index, indices) = operands
    lists = {**GATHER_DEFAULTS, **attributes}
    offset, collapsed, batching, index_batching, start = (
        numbers(lists[key]) for key in GATHER_LISTS
    )
    sizes = numbers(attributes['slice_sizes'])
    vector = int(attributes['index_vector_dim'])
    if not known(offset + collapsed + batching + index_batching + start + sizes):
        return typed(result, operands[:1])
    if not within(sizes, value[1]):
        return f'takes slice sizes within {name} {text(value)}, found {braces(sizes)}'
    rank = len(indices[1])
    if not 0 <= vector <= rank:
        return f'takes an index_vector_dim at most the rank {rank} of {index} {text(indices)}'
    # The index vector's length, 1 at the indices' rank
    length = indices[1][vector] if vector < rank else 1
    if isinstance(length, int) and len(start) != length:
        return f'takes a start_index_map of {length} dimensions, found {braces(start)}'
    reason = misnumbered(collapsed + batching, name, value) or misnumbered(start, name, value)
    if reason:
        return reason
    # Each list of operand dimensions in order
    for key, dims in zip(GATHER_LISTS, (offset, collapsed, batching)):
        if dims != sorted(dims):
            return f'takes {key} in order, found {braces(dims)}'
    if any(sizes[dim] > 1 for dim in collapsed + batching):
        found = braces(sizes)
        return f'takes slice sizes of 0 or 1 in collapsed and batching dimensions, found {found}'
    if len(index_batching) != len(batching) or vector in index_batching:
        return (
            'takes one indices batching dimension per operand batching dimension, '
            'none equal to index_vector_dim'
        )
    reason = misnumbered(index_batching, index, indices)
    if reason:
        return reason
    # The indices' batch dimensions, the slice's own at offset_dims
    batch = [size for k, size in enumerate(indices[1]) if k != vector]
    kept = [size for k, size in enumerate(sizes) if k not in collapsed + batching]
    if len(offset) != len(kept) or not valid(offset, len(batch) + len(kept)):
        found = braces(offset)
        return f'takes {len(kept)} distinct offset_dims below the rank of its result, found {found}'
    slices, batches = iter(kept), iter(batch)
    dims = [next(slices) if k in offset else next(batches) for k in range(len(batch) + len(kept))]
    return typed(result, operands[:1]) or gives(dims, result, operands)


def reduce(result, operands, attributes):
    name, value = operands[0]
    dims = numbers(attributes['dimensions'])
    if not known(dims):
        return typed(result, operands)
    kept = [size for k, size in enumerate(value[1]) if k not in dims]
    return (
        misnumbered(dims, name, value) or typed(result, operands) or gives(kept, result, operands)
    )


def dot(result, operands, attributes):
    (lhs, a), (rhs, b) = operands
    # The batch lists alone may be left out
    lb, rb = (numbers(attributes.get(f'{side}_batch_dims', [])) for side in ('lhs', 'rhs'))
    lc, rc = (numbers(attributes[f'{side}_contracting_dims']) for side in ('lhs', 'rhs'))
    if not known(lb + lc + rb + rc):
        return None
    if len(lb) != len(rb) or len(lc) != len(rc):
        return 'takes as many lhs as rhs batch and contracting dimensions'
    reason = misnumbered(lb + lc, lhs, a) or misnumbered(rb + rc, rhs, b)
    if reason:
        return reason
    for x, y in zip(lb + lc, rb + rc):
        if not agree([a[1][x]], [b[1][y]]):
            return f'pairs dimension {x} of {lhs} {text(a)} with dimension {y} of {rhs} {text(b)}'
    # The batch, then each operand's other dimensions; any element types
    others = [size for k, size in enumerate(a[1]) if k not in lb + lc]
    others += [size for k, size in enumerate(b[1]) if k not in rb + rc]
    return gives([a[1][k] for k in lb] + others, result, operands)


def cholesky(result, operands, attributes):
    name, value = operands[0]
    return square(name, value) or elementwise(result, operands, attributes)


def triangular_solve(result, operands, attributes):
    (lhs, a), (rhs, b) = operands
    reason = square(lhs, a)
    if reason:
        return reason
    # From the left, b's rows meet a; else its columns
    left = attributes.get('left_side', 'false') == 'true'
    met = b[1][-2:-1] if left else b[1][-1:]
    if len(b[1]) != len(a[1]) or not agree(b[1][:-2] + met, a[1][:-2] + a[1][-1:]):
        side = 'rows' if left else 'columns'
        return f'takes a right side whose {side} meet {lhs} {text(a)}, found {rhs} {text(b)}'
    return typed(result, operands) or gives(b[1], result, operands)


def batch_norm_inference(result, operands, attributes):
    (name, value), *vectors = operands
    rank, feature = len(value[1]), int(attributes['feature_index'])
    if not 0 <= feature < rank:
        return (
            f'takes a feature_index below the rank {rank} of {name} {text(value)}, found {feature}'
        )
    for vector, given in vectors:
        if not agree(given[1], [value[1][feature]]):
            return (
                f'takes vectors of the {value[1][feature]} features of {name} {text(value)}, '
                f'found {vector} {text(given)}'
            )
    return typed(result, operands) or gives(value[1], result, operands[:1])


# The check of each word, by name
RULES = {
    'constant': free,
    'iota': iota,
    **dict.fromkeys(
        [
            'abs',
            'negate',
            'sign',
            'add',
            'subtract',
            'multiply',
            'mulhi',
            'divide',
            'remainder',
            'maximum',
            'minimum',
            'power',
            'sqrt',
            'rsqrt',
            'cbrt',
            'exponential',
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
            'floor',
            'ceil',
            'round_nearest_afz',
            'round_nearest_even',
            'and',
            'or',
            'xor',
            'not',
            'shift_left',
            'shift_right_logical',
            'shift_right_arithmetic',
            'popcnt',
            'count_leading_zeros',
            'copy',
        ],
        elementwise,
    ),
    'compare': compare,
    'select': select,
    'clamp': clamp,
    'is_finite': is_finite,
    'convert': convert,
    'bitcast_convert': bitcast_convert,
    'reduce_precision': reduce_precision,
    'reshape': reshape,
    'transpose': transpose,
    'broadcast': broadcast,
    'concatenate': concatenate,
    'pad': pad,
    'reverse': reverse,
    'slice': slice_ranges,
    'dynamic_slice': dynamic_slice,
    'dynamic_update_slice': dynamic_update_slice,
    'gather': gather,
    'dot': dot,
    'cholesky': cholesky,
    'triangular_solve': triangular_solve,
    'batch_norm_inference': batch_norm_inference,
    **dict.fromkeys(
        [
            'reduce_add',
            'reduce_multiply',
            'reduce_max',
            'reduce_min',
            'reduce_and',
            'reduce_or',
            'reduce_xor',
        ],
        reduce,
    ),
}


def disagreement(word, result, operands, attributes):
    """Why a line's shapes or attributes disagree, or None."""
    return RULES[word](result, operands, attributes)
