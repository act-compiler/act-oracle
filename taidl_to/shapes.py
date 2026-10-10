"""The shape and type each lowered word gives its result, checked where the spec writes integers

A value is its element type and its dims, each dim an int or an expression's text.
A list attribute holds decimal text, which numbers reads as ints.
"""

# Bytes per element in memory, one for a pred
WIDTHS = {
    'f64': 8,
    'f32': 4,
    'f16': 2,
    'bf16': 2,
    'u8': 1,
    'u16': 2,
    'u32': 4,
    'u64': 8,
    's8': 1,
    's16': 2,
    's32': 4,
    's64': 8,
    'pred': 1,
}


def declared(result_type) -> list:
    """A result type's dims, as the spec writes them."""
    shape = result_type.shape()
    dims = shape.dim() if shape else []
    return [int(dim.getText()) if dim.INT() else dim.getText() for dim in dims]


def numbers(entries) -> list:
    """An attribute list's entries, as ints."""
    return [int(entry) for entry in entries]


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


def count(dims):
    """A shape's element count, or None where a dim is an expression."""
    total = 1
    for dim in dims:
        if not isinstance(dim, int):
            return None
        total *= dim
    return total


def valid(dims, rank) -> bool:
    """Whether dimension numbers are distinct and below a rank."""
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


def elementwise(result, operands, attributes):
    for name, value in operands:
        if not agree(value[1], result[1]):
            return f'takes operands of its shape {text(result)}, found {name} {text(value)}'
    return typed(result, operands)


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


def reduce(result, operands, attributes):
    name, value = operands[0]
    dims = numbers(attributes['dimensions'])
    kept = [size for k, size in enumerate(value[1]) if k not in dims]
    return (
        misnumbered(dims, name, value) or typed(result, operands) or gives(kept, result, operands)
    )


def dot(result, operands, attributes):
    (lhs, a), (rhs, b) = operands
    # The batch lists alone may be left out
    lb, rb = (numbers(attributes.get(f'{side}_batch_dims', [])) for side in ('lhs', 'rhs'))
    lc, rc = (numbers(attributes[f'{side}_contracting_dims']) for side in ('lhs', 'rhs'))
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


# The check of each word, by name
RULES = {
    **dict.fromkeys(
        [
            'add',
            'subtract',
            'multiply',
            'divide',
            'maximum',
            'minimum',
            'exponential',
            'xor',
            'copy',
        ],
        elementwise,
    ),
    'convert': convert,
    'bitcast_convert': bitcast_convert,
    'reshape': reshape,
    'transpose': transpose,
    'broadcast': broadcast,
    'dot': dot,
    'reduce_add': reduce,
}


def disagreement(word, result, operands, attributes):
    """Why a line's shapes or attributes disagree, or None."""
    return RULES[word](result, operands, attributes)
