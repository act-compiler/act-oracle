"""What a generated api.py traces and checks with:
loop expressions, checks, SSA names, types, emitters"""

import itertools
import math
import numbers
import re

import numpy as np

# A loop expression's tokens, any other character last
LOOP_TOKEN = re.compile(r'0|[1-9][0-9]*|%[A-Za-z_][A-Za-z0-9_]*|[-+*/%()]|\S')


def merge(terms):
    """A value's terms, like terms summed and zero terms dropped"""
    sums = {}
    for atoms, coeff in terms:
        sums[atoms] = sums.get(atoms, 0) + coeff
    return tuple((atoms, coeff) for atoms, coeff in sums.items() if coeff)


def literal(number):
    """A value of one integer"""
    return merge([((), number)])


def multiply(a, b):
    """a * b, each term of a times each of b"""
    return merge(
        (atoms_a + atoms_b, coeff_a * coeff_b) for atoms_a, coeff_a in a for atoms_b, coeff_b in b
    )


def constant(value):
    """A value's integer, or None if it reads a loop variable"""
    if any(atoms for atoms, _ in value):
        return None
    return sum(coeff for _, coeff in value)


def variables(value):
    """The loop variables a value reads"""
    found = set()
    for atoms, _ in value:
        for atom in atoms:
            found |= {atom} if isinstance(atom, str) else variables(atom[1]) | variables(atom[2])
    return found


def evaluate(value, env):
    """A value's integer at env's loop variable values"""
    total = 0
    for atoms, coeff in value:
        for atom in atoms:
            if isinstance(atom, str):
                coeff *= env[atom]
            else:
                a, b = evaluate(atom[1], env), evaluate(atom[2], env)
                coeff *= a // b if atom[0] == '/' else a % b
        total += coeff
    return total


def iterations(value, loops):
    """Each binding of a value's loop variables over its loops, outer first"""
    names = [var for var in loops if var in variables(value)]
    for point in itertools.product(*(loops[var] for var in names)):
        yield dict(zip(names, point))


def divide(operator, a, b, loops):
    """a / b or a % b, floored as the language floors"""
    # Refuse a divisor zero at any iteration, naming it
    for env in iterations(b, loops):
        if evaluate(b, env) == 0:
            raise ZeroDivisionError(env)
    divisor = constant(b)
    if divisor is not None and constant(a) is not None:
        return literal(constant(a) // divisor if operator == '/' else constant(a) % divisor)
    # Every coefficient a multiple of the divisor: an exact quotient
    if divisor is not None and all(coeff % divisor == 0 for _, coeff in a):
        return merge((atoms, coeff // divisor) for atoms, coeff in a) if operator == '/' else ()
    return ((((operator, a, b),), 1),)


# Each operator of the language on two values
OPERATIONS = {
    '+': lambda a, b, loops: merge(a + b),
    '-': lambda a, b, loops: merge(a + multiply(literal(-1), b)),
    '*': lambda a, b, loops: multiply(a, b),
    '/': lambda a, b, loops: divide('/', a, b, loops),
    '%': lambda a, b, loops: divide('%', a, b, loops),
}


class LoopValue:
    """A loop expression's value, which the language's arithmetic operators combine"""

    def __init__(self, value, loops, name=None):
        # Its terms, its loops' ranges, its SSA name once emitted
        self.value, self.loops, self.name = value, loops, name

    def combine(self, operator, other, reflected=False):
        """self operator other, or other operator self if reflected"""
        if isinstance(other, LoopValue):
            other_value, loops = other.value, self.loops | other.loops
        else:
            other_value, loops = literal(other), self.loops
        a, b = (other_value, self.value) if reflected else (self.value, other_value)
        return LoopValue(OPERATIONS[operator](a, b, loops), loops)

    def __add__(self, other):
        return self.combine('+', other)

    def __radd__(self, other):
        return self.combine('+', other, True)

    def __sub__(self, other):
        return self.combine('-', other)

    def __rsub__(self, other):
        return self.combine('-', other, True)

    def __mul__(self, other):
        return self.combine('*', other)

    def __rmul__(self, other):
        return self.combine('*', other, True)

    def __floordiv__(self, other):
        return self.combine('/', other)

    def __rfloordiv__(self, other):
        return self.combine('/', other, True)

    def __mod__(self, other):
        return self.combine('%', other)

    def __rmod__(self, other):
        return self.combine('%', other, True)

    def __neg__(self):
        return self.combine('*', -1)


def read_loop_expression(instruction, name, expression, loop_stack):
    """A call's loop expression as a LoopValue of terms, each a coefficient and its atoms.

    An atom is a loop variable or a quotient (operator, a, b) of two values.
    """
    tokens = LOOP_TOKEN.findall(expression) + ['']
    loops = {loop['var']: range(loop['start'], loop['end']) for loop in loop_stack}
    unreadable = ValueError(f'{instruction}: {name} {expression!r} is not a loop expression')
    at = 0

    def take():
        nonlocal at
        at += 1
        return tokens[at - 1]

    # The language's precedence, each operator grouping from the left
    def additive():
        value = multiplicative()
        while tokens[at] in ('+', '-'):
            value = OPERATIONS[take()](value, multiplicative(), loops)
        return value

    def multiplicative():
        value = unary()
        while tokens[at] in ('*', '/', '%'):
            value = OPERATIONS[take()](value, unary(), loops)
        return value

    def unary():
        if tokens[at] == '-':
            take()
            return multiply(literal(-1), unary())
        return primary()

    def primary():
        token = take()
        if token == '(':
            value = additive()
            if take() != ')':
                raise unreadable
            return value
        if token.isascii() and token.isdigit():
            return literal(int(token))
        if token in loops:
            return (((token,), 1),)
        # A % followed by a name is a loop variable
        if len(token) > 1 and token[0] == '%':
            raise ValueError(
                f'{instruction}: {name} {expression!r} reads {token}, which no enclosing loop holds'
            )
        raise unreadable

    value = additive()
    if tokens[at]:
        raise unreadable
    return LoopValue(value, loops)


def value_lines(value, final_name):
    """StableHLO lines computing a loop expression's value as final_name"""
    lines = []

    def emit(value, prefix):
        # Each term its coefficient times its atoms, then their sum
        operands = []
        for k, (atoms, coeff) in enumerate(value):
            result = f'{prefix}.c{k}'
            lines.append(f'{result} = stablehlo.constant dense<{coeff}> : tensor<i64>')
            for n, atom in enumerate(atoms):
                factor = atom if isinstance(atom, str) else emit_quotient(*atom, f'{result}.q{n}')
                lines.append(f'{result}.m{n} = stablehlo.multiply {result}, {factor} : tensor<i64>')
                result = f'{result}.m{n}'
            operands.append(result)
        if not operands:
            lines.append(f'{prefix} = stablehlo.constant dense<0> : tensor<i64>')
            return prefix
        total = operands[0]
        for n, operand in enumerate(operands[1:]):
            lines.append(f'{prefix}.a{n} = stablehlo.add {total}, {operand} : tensor<i64>')
            total = f'{prefix}.a{n}'
        return total

    def emit_quotient(operator, a, b, prefix):
        # Floored from StableHLO's truncating remainder and divide
        a, b = emit(a, f'{prefix}.n'), emit(b, f'{prefix}.d')
        lines.append(f'{prefix}.t = stablehlo.remainder {a}, {b} : tensor<i64>')
        lines.append(f'{prefix}.u = stablehlo.add {prefix}.t, {b} : tensor<i64>')
        lines.append(f'{prefix}.r = stablehlo.remainder {prefix}.u, {b} : tensor<i64>')
        if operator == '%':
            return f'{prefix}.r'
        lines.append(f'{prefix}.s = stablehlo.subtract {a}, {prefix}.r : tensor<i64>')
        lines.append(f'{prefix} = stablehlo.divide {prefix}.s, {b} : tensor<i64>')
        return prefix

    total = emit(value, final_name)
    if total != final_name:
        lines.append(f'{final_name} = stablehlo.reshape {total} : (tensor<i64>) -> tensor<i64>')
    return lines


def generate_loop_eq_vars(instruction, name, expression, loop_stack, global_counters):
    """StableHLO lines computing a call's loop expression, and its LoopValue, named"""
    read = read_loop_expression(instruction, name, expression, loop_stack)
    read.name = f'%loop_final.{global_counters["loop_var"]}'
    return '\n'.join(value_lines(read.value, read.name)) + '\n', read


def integral(value):
    """Whether a value is a Python or NumPy integer, never a bool"""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def check_values(instruction, values, computational=False):
    """A call's values, each an int or a loop expression, refused by name if neither"""
    for name, value in values.items():
        if not integral(value) and not isinstance(value, str):
            raise TypeError(f'{instruction}: {name} {value!r} is not an integer')
        # A computational attribute takes an int alone
        if computational and isinstance(value, str):
            raise TypeError(
                f'{instruction}: computational attribute {name} takes an integer, '
                f'not the loop expression {value!r}'
            )
    # A NumPy integer as an int
    return {name: value if isinstance(value, str) else int(value) for name, value in values.items()}


def division_by_zero(instruction, values, error):
    """The refusal of error, a call's division by zero, its values and iteration named"""
    shown = ', '.join(f'{name}={value!r}' for name, value in values.items())
    # A loop expression's divisor gives the iteration
    env = error.args[0] if error.args and isinstance(error.args[0], dict) else {}
    iteration = ', '.join(f'{var} = {at}' for var, at in env.items())
    return ValueError(
        f'{instruction}: an expression divides by zero with {shown}'
        + (f' at {iteration}' if env else '')
    )


def lhs_util(global_counters, lhs_name, lhs_vartype='', lhs_dim=()):
    if lhs_name in global_counters:
        global_counters[lhs_name]['counter'] += 1
        lhs_type = global_counters[lhs_name]['type']
    else:
        dim_sizes = ','.join(str(dim) for dim in lhs_dim)
        lhs_type = f'{lhs_vartype}[{dim_sizes}]'
    lhs_name = parameter_util(global_counters, lhs_name)
    return lhs_name, lhs_type


def parameter_util(global_counters, lhs_name):
    if lhs_name in global_counters:
        lhs_name += f'.{global_counters[lhs_name]["counter"]}'
    else:
        lhs_name += f'.{global_counters["instruction_counter"]}'
    lhs_name = '%' + lhs_name
    return lhs_name


# TAIDL element type -> MLIR element type
MLIR_ELEM_TYPE = {
    'f64': 'f64',
    'f32': 'f32',
    'f16': 'f16',
    'bf16': 'bf16',
    'f8e4m3fn': 'f8E4M3FN',
    'f8e5m2': 'f8E5M2',
    'u2': 'ui2',
    'u4': 'ui4',
    'u8': 'ui8',
    'u16': 'ui16',
    'u32': 'ui32',
    'u64': 'ui64',
    's2': 'i2',
    's4': 'i4',
    's8': 'i8',
    's16': 'i16',
    's32': 'i32',
    's64': 'i64',
    'pred': 'i1',
}


def mlir_type(xla_type):
    """bf16[64,64] -> tensor<64x64xbf16>, f32[] -> tensor<f32>"""
    base, dims = xla_type.split('[', 1)
    dims = dims.split(']', 1)[0].replace(' ', '')
    return (
        'tensor<'
        + ''.join(f'{d}x' for d in dims.split(',') if d)
        + MLIR_ELEM_TYPE[base.strip()]
        + '>'
    )


def mlir_elem(mlir_t):
    return mlir_t[len('tensor<') : -1].rsplit('x', 1)[-1]


def mlir_zero(mlir_t):
    floats = ('f64', 'f32', 'f16', 'bf16', 'f8E4M3FN', 'f8E5M2')
    return '0.0' if mlir_elem(mlir_t) in floats else '0'


# Exponent and mantissa bits of each float type
FLOAT_BITS = {
    'f64': (11, 52),
    'f32': (8, 23),
    'f16': (5, 10),
    'bf16': (8, 7),
    'f8e4m3fn': (4, 3),
    'f8e5m2': (5, 2),
}

# Float types with no infinity, their top exponent finite
NO_INFINITY = ('f8e4m3fn',)


def float_largest(typ):
    """The largest finite value of a float type, and half its unit in the last place"""
    exponent, mantissa = FLOAT_BITS[typ]
    top = (1 << (exponent - 1)) - 1 - mantissa
    if typ in NO_INFINITY:
        # The top exponent's all-ones mantissa is its nan
        return ((1 << (mantissa + 1)) - 2) << (top + 1), 1 << top
    return ((1 << (mantissa + 1)) - 1) << top, 1 << (top - 1)


def mlir_literal(value, typ, shown=None):
    """A Python scalar as an MLIR literal of element type typ, refusing one typ cannot hold"""
    shown = shown or (str(value).lower() if isinstance(value, bool) else str(value))
    if isinstance(value, bool) and typ != 'pred':
        raise TypeError(f'{shown} at {typ} is a pred literal')
    if typ in FLOAT_BITS:
        exponent, mantissa = FLOAT_BITS[typ]
        if isinstance(value, int) or math.isfinite(value):
            # Out of range: a finite value rounding past typ's largest
            largest, half = float_largest(typ)
            # At the tie, rounding to even keeps a no-infinity type's largest
            bound = largest + half
            if abs(value) > bound or (abs(value) == bound and typ not in NO_INFINITY):
                raise ValueError(f'{shown} at {typ} is out of its range')
            # MLIR reads a float with a dot before its exponent
            text = repr(float(value))
            return text if '.' in text else text.replace('e', '.0e')
        # A decimal past a double's range reads as an infinity, which some types lack
        if math.isinf(value) and ('inf' not in shown or typ in NO_INFINITY):
            raise ValueError(f'{shown} at {typ} is out of its range')
        # MLIR reads a non-finite float by its bits
        bits = ((1 << exponent) - 1) << mantissa
        if math.isnan(value):
            # A type with no infinity sets every mantissa bit
            bits |= (1 << mantissa) - 1 if typ in NO_INFINITY else 1 << (mantissa - 1)
        if math.copysign(1, value) < 0:
            bits |= 1 << (exponent + mantissa)
        return f'0x{bits:X}'
    if isinstance(value, float):
        raise TypeError(f'{shown} at {typ} is not an integer')
    low, high = integer_range(typ)
    if not low <= value <= high:
        raise ValueError(f'{shown} at {typ} is out of its range')
    return str(int(value))


def integer_range(typ):
    """The least and the greatest value of an integer type or pred"""
    if typ == 'pred':
        return 0, 1
    width = int(typ[1:])
    if typ[0] == 's':
        return -(1 << (width - 1)), (1 << (width - 1)) - 1
    return 0, (1 << width) - 1


def reduce_init(op, typ):
    """The identity of reducer op at element type typ, as an MLIR literal; a float sum starts
    from +0.0, as XLA's does"""
    least, greatest = (-math.inf, math.inf) if typ in FLOAT_BITS else integer_range(typ)
    if typ in NO_INFINITY:
        # A float with no infinity is bounded by its largest
        greatest = float_largest(typ)[0]
        least = -greatest
    identities = {'add': 0, 'multiply': 1, 'maximum': least, 'minimum': greatest, 'or': 0, 'xor': 0}
    if typ not in FLOAT_BITS:
        # and's identity has every bit set
        identities['and'] = -1 if typ[0] == 's' else greatest
    return mlir_literal(identities[op], typ)


def dims_list(dims):
    """A list of dimensions as StableHLO writes it, in its order"""
    return '[' + ', '.join(str(d) for d in dims) + ']'


# Each sh_* helper records its result's MLIR type in ssa_types
def sh_zeros(lhs, xla_type, ssa_types):
    t = mlir_type(xla_type)
    ssa_types[lhs] = t
    return f'{lhs} = stablehlo.constant dense<{mlir_zero(t)}> : {t}'


def sh_constant(lhs, lhs_type, value, instruction, ssa_types):
    """value at every element of lhs_type, refused by name where lhs_type cannot hold it"""
    t = mlir_type(lhs_type)
    try:
        literal = mlir_literal(value, lhs_type.split('[')[0])
    except (TypeError, ValueError) as error:
        raise type(error)(f'{instruction}: constant {error}') from None
    ssa_types[lhs] = t
    return f'{lhs} = stablehlo.constant dense<{literal}> : {t}'


def sh_iota(lhs, lhs_type, dimension, ssa_types):
    t = mlir_type(lhs_type)
    ssa_types[lhs] = t
    return f'{lhs} = stablehlo.iota dim = {dimension} : {t}'


def sh_generic(op, lhs, lhs_type, operands, ssa_types, properties):
    """op on operands in MLIR's generic form, which every dialect reads"""
    t = mlir_type(lhs_type)
    types = ', '.join(ssa_types[x] for x in operands)
    ssa_types[lhs] = t
    return f'{lhs} = "{op}"({", ".join(operands)}){properties} : ({types}) -> {t}'


def sh_triangular_solve(lhs, lhs_type, operands, ssa_types, properties):
    """triangular_solve on operands; at f16 solved at f32"""
    op = 'stablehlo.triangular_solve'
    if not lhs_type.startswith('f16['):
        return sh_generic(op, lhs, lhs_type, operands, ssa_types, properties)
    # Both operands convert up, the result down
    wide = [f'{lhs}.a', f'{lhs}.b']
    lines = []
    for w, x in zip(wide, operands):
        ssa_types[w] = ssa_types[x][: -len('f16>')] + 'f32>'
        lines.append(f'{w} = stablehlo.convert {x} : ({ssa_types[x]}) -> {ssa_types[w]}')
    wide_type = 'f32' + lhs_type[len('f16') :]
    lines.append(sh_generic(op, f'{lhs}.f32', wide_type, wide, ssa_types, properties))
    lines.append(sh_cast('convert', lhs, lhs_type, f'{lhs}.f32', ssa_types))
    return '\n\t'.join(lines)


def mlir_attribute(value):
    """An attribute's value as MLIR writes it: text, a list of integers or an integer"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return f'array<i64: {", ".join(str(v) for v in value)}>' if value else 'array<i64>'
    return f'{value} : i64'


def mlir_properties(attributes):
    """A call's attributes, by name, as an op's properties"""
    given = ', '.join(f'{name} = {mlir_attribute(value)}' for name, value in attributes.items())
    return f' <{{{given}}}>' if given else ''


def gather_numbers(index_vector_dim, **dims):
    """A gather's dimension numbers as StableHLO writes them"""
    given = ', '.join(f'{name} = {dims_list(value)}' for name, value in dims.items())
    return f'#stablehlo.gather<{given}, index_vector_dim = {index_vector_dim}>'


def sh_cast(op, lhs, lhs_type, x, ssa_types):
    t, x_t = mlir_type(lhs_type), ssa_types[x]
    ssa_types[lhs] = t
    if op != 'bitcast_convert' or (mlir_elem(x_t) == 'i1') == (mlir_elem(t) == 'i1'):
        return f'{lhs} = stablehlo.{op} {x} : ({x_t}) -> {t}'
    # A pred is one byte: convert it through u8
    from_pred = mlir_elem(x_t) == 'i1'
    byte_t = (x_t if from_pred else t)[: -len('i1>')] + 'ui8>'
    first, second = ('convert', 'bitcast_convert') if from_pred else ('bitcast_convert', 'convert')
    return (
        f'{lhs}.byte = stablehlo.{first} {x} : ({x_t}) -> {byte_t}\n\t'
        f'{lhs} = stablehlo.{second} {lhs}.byte : ({byte_t}) -> {t}'
    )


def sh_copy(lhs, lhs_type, x, ssa_types):
    """copy keeps its operand's type: an identity reshape"""
    t = mlir_type(lhs_type)
    if ssa_types[x] != t:
        raise ValueError(f'copy of {x} : {ssa_types[x]} does not have the result type {t}')
    ssa_types[lhs] = t
    return f'{lhs} = stablehlo.reshape {x} : ({t}) -> {t}'


def sh_transpose(lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    line = f'{lhs} = stablehlo.transpose {x}, dims = {dims_list(dims)} : ({ssa_types[x]}) -> {t}'
    ssa_types[lhs] = t
    return line


def sh_broadcast(lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    line = f'{lhs} = stablehlo.broadcast_in_dim {x}, dims = {dims_list(dims)} : ({ssa_types[x]}) -> {t}'
    ssa_types[lhs] = t
    return line


def sh_dot(lhs, lhs_type, a, b, lhs_batch, lhs_contracting, rhs_batch, rhs_contracting, ssa_types):
    t = mlir_type(lhs_type)
    lb, rb = dims_list(lhs_batch), dims_list(rhs_batch)
    batching = f'batching_dims = {lb} x {rb}, ' if lb != '[]' or rb != '[]' else ''
    line = (
        f'{lhs} = stablehlo.dot_general {a}, {b}, {batching}'
        f'contracting_dims = {dims_list(lhs_contracting)} x {dims_list(rhs_contracting)}, '
        # The highest precision on every platform
        'precision = [HIGHEST, HIGHEST] : '
        f'({ssa_types[a]}, {ssa_types[b]}) -> {t}'
    )
    ssa_types[lhs] = t
    return line


def sh_reduce(op, lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    scalar_t = f'tensor<{mlir_elem(t)}>'
    init = reduce_init(op, lhs_type.split('[')[0])
    lines = [
        f'{lhs}.init = stablehlo.constant dense<{init}> : {scalar_t}',
        (
            f'{lhs} = stablehlo.reduce({x} init: {lhs}.init) applies stablehlo.{op} '
            f'across dimensions = {dims_list(dims)} : ({ssa_types[x]}, {scalar_t}) -> {t}'
        ),
    ]
    ssa_types[lhs] = t
    return '\n\t'.join(lines)


def index_constants(lhs, starts):
    """One tensor<i64> per start: a literal's constant, or a loop value's SSA name"""
    lines, locs = [], []
    for k, start in enumerate(starts):
        if not isinstance(start, LoopValue):
            locs.append(f'{lhs}.i{k}')
            lines.append(f'{locs[-1]} = stablehlo.constant dense<{start}> : tensor<i64>')
        elif start.name:
            # An attribute's value, emitted at the call
            locs.append(start.name)
        else:
            # A start computed over an attribute's value
            locs.append(f'{lhs}.i{k}')
            lines += value_lines(start.value, locs[-1])
    return lines, locs


def tensor_dims(mlir_t):
    """tensor<64x64xbf16> -> [64, 64]"""
    return [int(d) for d in mlir_t[len('tensor<') : -1].split('x')[:-1]]


def check_sizes(instruction, sizes, dims):
    """Refuse a region size or a semantics dim that is not positive, at any iteration"""
    for kind, given in (('size', sizes), ('dimension', dims)):
        for text, size in given.items():
            # A plain size as a LoopValue over no loops
            read = size if isinstance(size, LoopValue) else LoopValue(literal(size), {})
            for env in iterations(read.value, read.loops):
                at = evaluate(read.value, env)
                if at < 1:
                    iteration = ', '.join(f'{var} = {value}' for var, value in env.items())
                    raise ValueError(
                        f'{instruction}: {kind} {text} is {at}'
                        + (f' at {iteration}' if env else '')
                        + ', not positive'
                    )


def debug_index(data, global_counters):
    """The buffer a debug call names and its index, an integer per leading dim, none for ':'"""
    name, bracket, index = data.partition('[')
    # Each buffer's counter a dict, each trace counter an int
    buffers = [key for key, counter in global_counters.items() if isinstance(counter, dict)]
    if name not in buffers:
        raise ValueError(f'debug: {data!r} names no buffer')
    if not bracket or not index.endswith(']') or not index[:-1].strip():
        raise ValueError(
            f'debug: {data!r} gives no index; {name + "[:]"!r} prints the whole buffer'
        )
    # A bare ':' the whole buffer, each entry a Python integer literal
    entries = (
        [] if index[:-1].strip() == ':' else [entry.strip() for entry in index[:-1].split(',')]
    )
    indices = []
    for entry in entries:
        try:
            indices.append(int(entry, 0))
        except ValueError:
            raise TypeError(f'debug: {data!r} has index {entry!r}, not an integer') from None
    dims = tensor_dims(mlir_type(global_counters[name]['type']))
    if len(entries) > len(dims):
        raise ValueError(
            f'debug: {data!r} gives {len(entries)} indices for the {len(dims)} dimensions of {name}'
        )
    return name, indices


def check_window(instruction, buffer, starts, window_t, buffer_t):
    """Refuse a window outside its buffer at any iteration of its loops"""
    dims = zip(starts, tensor_dims(window_t), tensor_dims(buffer_t))
    for k, (start, size, extent) in enumerate(dims):
        # A literal start as a LoopValue over no loops
        read = start if isinstance(start, LoopValue) else LoopValue(literal(start), {})
        for env in iterations(read.value, read.loops):
            at = evaluate(read.value, env)
            if at < 0 or at + size > extent:
                iteration = ', '.join(f'{var} = {value}' for var, value in env.items())
                raise ValueError(
                    f'{instruction}: window [{at}:{at + size}] on dimension {k} of {buffer} '
                    f'runs outside its extent {extent}' + (f' at {iteration}' if env else '')
                )


def sh_slice_load(lhs, lhs_type, rhs_loc, starts, ssa_types):
    lines, locs = index_constants(lhs, starts)
    t = mlir_type(lhs_type)
    sizes = lhs_type.split('[', 1)[1].split(']')[0]
    lines.append(
        f'{lhs} = stablehlo.dynamic_slice {", ".join([rhs_loc] + locs)}, sizes = [{sizes}] : '
        f'({", ".join([ssa_types[rhs_loc]] + ["tensor<i64>"] * len(locs))}) -> {t}'
    )
    ssa_types[lhs] = t
    return '\n\t'.join(lines)


def sh_slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, starts, ssa_types):
    lines, locs = index_constants(lhs_loc, starts)
    t = mlir_type(lhs_type)
    lines.append(
        f'{lhs_loc} = stablehlo.dynamic_update_slice {", ".join([rhs_loc, rhs_update] + locs)} : '
        f'({", ".join([ssa_types[rhs_loc], ssa_types[rhs_update]] + ["tensor<i64>"] * len(locs))}) -> {t}'
    )
    ssa_types[lhs_loc] = t
    return lines


# NumPy dtype name -> TAIDL element type
XLA_ELEM_TYPE = {
    'bfloat16': 'bf16',
    'float16': 'f16',
    'float32': 'f32',
    'float64': 'f64',
    'float8_e4m3fn': 'f8e4m3fn',
    'float8_e5m2': 'f8e5m2',
    'uint2': 'u2',
    'uint4': 'u4',
    'uint8': 'u8',
    'uint16': 'u16',
    'uint32': 'u32',
    'uint64': 'u64',
    'int2': 's2',
    'int4': 's4',
    'int8': 's8',
    'int16': 's16',
    'int32': 's32',
    'int64': 's64',
    'bool': 'pred',
}

# Elements per byte of each sub-byte type
PACKED = {'s2': 4, 'u2': 4, 's4': 2, 'u4': 2}


def spec_mlir_type(dims, dtype):
    return mlir_type(f'{XLA_ELEM_TYPE[np.dtype(dtype).name]}[{",".join(str(d) for d in dims)}]')


def packing(dtype):
    """Elements per byte of a dtype in HBM, one unless it is sub-byte"""
    return PACKED.get(XLA_ELEM_TYPE[np.dtype(dtype).name], 1)


def spec_bytes(spec):
    """Byte count of a spec and its u8 view shape; a multi-byte dtype gains a trailing dim, a
    sub-byte one packs into whole bytes"""
    width = int(np.dtype(spec['dtype']).itemsize)
    dims = [int(d) for d in spec['shape']]
    count, pack = int(np.prod(dims, dtype=np.int64)), packing(spec['dtype'])
    if pack > 1:
        return count // pack, [count // pack]
    return width * count, dims + ([width] if width > 1 else [])


def packed(name, value, value_t, dtype, nbytes):
    """The lines packing value, a sub-byte value_t, into name, its nbytes bytes"""
    packed_t = spec_mlir_type([nbytes, packing(dtype)], dtype)
    return [
        f'{name}.p = stablehlo.reshape {value} : ({value_t}) -> {packed_t}',
        f'{name} = stablehlo.bitcast_convert {name}.p : ({packed_t}) -> {mlir_type(f"u8[{nbytes}]")}',
    ]


def byte_cast(dtype):
    """A bool moves to and from its bytes by a convert, any other dtype by a bitcast"""
    return 'convert' if np.dtype(dtype) == np.bool_ else 'bitcast_convert'


# Kernel inputs in, outputs out; HBM stays inside the module
def kernel_prologue(hbm_size, inputs, constants):
    """The module's parameters, and the lines laying them and the constants into %d0.0"""
    params, lines, pieces = [], [], []
    for j, const in enumerate(constants):
        if packing(const['dtype']) > 1:
            # A sub-byte constant, by its values, packs as an input does
            values = np.asarray(const['value']).reshape(-1)
            nbytes, value_t = spec_bytes(const)[0], spec_mlir_type([values.size], const['dtype'])
            given = ', '.join(str(int(v)) for v in values)
            lines.append(f'%hbm.v{j} = stablehlo.constant dense<[{given}]> : {value_t}')
            lines += packed(f'%hbm.c{j}', f'%hbm.v{j}', value_t, const['dtype'], nbytes)
            pieces.append((int(const['addr']), nbytes, f'%hbm.c{j}', mlir_type(f'u8[{nbytes}]')))
            continue
        flat = np.ascontiguousarray(np.asarray(const['value'])).reshape(-1).view(np.uint8)
        flat_t = mlir_type(f'u8[{flat.size}]')
        lines.append(f'%hbm.c{j} = stablehlo.constant dense<"0x{flat.tobytes().hex()}"> : {flat_t}')
        pieces.append((int(const['addr']), int(flat.size), f'%hbm.c{j}', flat_t))
    for i, spec in enumerate(inputs):
        arg_t = spec_mlir_type(spec['shape'], spec['dtype'])
        nbytes, byte_dims = spec_bytes(spec)
        byte_t, flat_t = spec_mlir_type(byte_dims, np.uint8), mlir_type(f'u8[{nbytes}]')
        params.append(f'%arg{i}: {arg_t}')
        if packing(spec['dtype']) > 1:
            lines += packed(f'%hbm.f{i}', f'%arg{i}', arg_t, spec['dtype'], nbytes)
        else:
            cast = byte_cast(spec['dtype'])
            lines.append(f'%hbm.b{i} = stablehlo.{cast} %arg{i} : ({arg_t}) -> {byte_t}')
            lines.append(f'%hbm.f{i} = stablehlo.reshape %hbm.b{i} : ({byte_t}) -> {flat_t}')
        pieces.append((int(spec['addr']), nbytes, f'%hbm.f{i}', flat_t))

    parts, cursor = [], 0

    def gap(nbytes):
        gap_t = mlir_type(f'u8[{nbytes}]')
        lines.append(f'%hbm.g{len(parts)} = stablehlo.constant dense<0> : {gap_t}')
        parts.append((f'%hbm.g{len(parts)}', gap_t))

    for addr, nbytes, loc, loc_t in sorted(pieces):
        if addr > cursor:
            gap(addr - cursor)
        parts.append((loc, loc_t))
        cursor = addr + nbytes
    if cursor < hbm_size:
        gap(hbm_size - cursor)
    hbm_t = mlir_type(f'u8[{hbm_size}]')
    lines.append(
        f'%d0.0 = stablehlo.concatenate {", ".join(p for p, _ in parts)}, dim = 0 : '
        f'({", ".join(t for _, t in parts)}) -> {hbm_t}'
    )
    return params, lines


def kernel_epilogue(hbm_loc, outputs, ssa_types):
    """Each output sliced out of the final HBM: the lines, the names and the types"""
    lines, names, types = [], [], []
    for i, spec in enumerate(outputs):
        addr = int(spec['addr'])
        nbytes, byte_dims = spec_bytes(spec)
        byte_t, flat_t = spec_mlir_type(byte_dims, np.uint8), mlir_type(f'u8[{nbytes}]')
        out_t = spec_mlir_type(spec['shape'], spec['dtype'])
        lines.append(
            f'%out.s{i} = stablehlo.slice {hbm_loc} [{addr}:{addr + nbytes}:1] : ({ssa_types[hbm_loc]}) -> {flat_t}'
        )
        pack = packing(spec['dtype'])
        if pack > 1:
            # A sub-byte output unpacks from its bytes
            packed_t = spec_mlir_type([nbytes, pack], spec['dtype'])
            lines.append(
                f'%out.p{i} = stablehlo.bitcast_convert %out.s{i} : ({flat_t}) -> {packed_t}'
            )
            lines.append(f'%out.o{i} = stablehlo.reshape %out.p{i} : ({packed_t}) -> {out_t}')
        else:
            lines.append(f'%out.r{i} = stablehlo.reshape %out.s{i} : ({flat_t}) -> {byte_t}')
            cast = byte_cast(spec['dtype'])
            lines.append(f'%out.o{i} = stablehlo.{cast} %out.r{i} : ({byte_t}) -> {out_t}')
        names.append(f'%out.o{i}')
        types.append(out_t)
    return lines, names, types
