"""What a generated api.py traces with: loop expressions, SSA names, types, emitters"""

import math
import re

import numpy as np


def parse_simple_expression(expression):
    expression = expression.replace(' ', '')
    if not expression:
        return [(0, [], [])]

    terms = []

    parts = re.split(r'([+-])', expression)

    current_sign = 1

    for part in parts:
        if part == '+':
            current_sign = 1
        elif part == '-':
            current_sign = -1
        elif part:
            # The term only has mult and division
            coeff, mult_vars, div_vars = parse_term(part)
            terms.append((current_sign * coeff, mult_vars, div_vars))
    return terms if terms else [(0, [], [])]


def parse_term(term):
    parts = re.split(r'([*/])', term)

    coeff = 1
    mult_vars = []
    div_vars = []

    i = 0
    while i < len(parts):
        if i % 2 == 0:
            val = parts[i]
            if val.isdigit():
                num = int(val)
                if i == 0:
                    coeff = num
                else:
                    op = parts[i - 1]
                    if op == '*':
                        coeff *= num
                    elif op == '/':
                        coeff = int(coeff / num)
            elif val.startswith('%'):
                if i == 0:
                    mult_vars.append(val)
                else:
                    op = parts[i - 1]
                    if op == '*':
                        mult_vars.append(val)
                    elif op == '/':
                        div_vars.append(val)
            else:
                raise ValueError(f'unknown token {val!r} in a loop expression')
        i += 1
    return coeff, mult_vars, div_vars


def generate_loop_eq_vars(expression, global_counters):
    terms = parse_simple_expression(expression)
    final_name = f'%loop_final.{global_counters["loop_var"]}'
    lines = []

    def make_operand(k, coeff, mult_vars, div_vars):
        result = f'{final_name}.c{k}'
        lines.append(f'{result} = stablehlo.constant dense<{int(coeff)}> : tensor<i64>')
        for n, var in enumerate(mult_vars):
            lines.append(f'{result}.m{n} = stablehlo.multiply {result}, {var} : tensor<i64>')
            result = f'{result}.m{n}'
        for n, var in enumerate(div_vars):
            lines.append(f'{result}.d{n} = stablehlo.divide {result}, {var} : tensor<i64>')
            result = f'{result}.d{n}'
        return result

    operands = [
        make_operand(k, coeff, mult_vars, div_vars)
        for k, (coeff, mult_vars, div_vars) in enumerate(terms)
        if coeff != 0
    ]

    if not operands:
        return f'{final_name} = stablehlo.constant dense<0> : tensor<i64>\n', final_name

    nested_adds = operands[0]
    for n, operand in enumerate(operands[1:]):
        lines.append(f'{final_name}.a{n} = stablehlo.add {nested_adds}, {operand} : tensor<i64>')
        nested_adds = f'{final_name}.a{n}'
    lines.append(f'{final_name} = stablehlo.reshape {nested_adds} : (tensor<i64>) -> tensor<i64>')
    return '\n'.join(lines) + '\n', final_name


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
    'u8': 'ui8',
    'u16': 'ui16',
    'u32': 'ui32',
    'u64': 'ui64',
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
    return '0.0' if mlir_elem(mlir_t) in ('f64', 'f32', 'f16', 'bf16') else '0'


# Exponent and mantissa bits of each float type
FLOAT_BITS = {'f64': (11, 52), 'f32': (8, 23), 'f16': (5, 10), 'bf16': (8, 7)}


def mlir_literal(value, typ, shown=None):
    """A Python scalar as an MLIR literal of element type typ, refusing one typ cannot hold"""
    shown = shown or (str(value).lower() if isinstance(value, bool) else str(value))
    if isinstance(value, bool) and typ != 'pred':
        raise TypeError(f'{shown} at {typ} is a pred literal')
    if typ in FLOAT_BITS:
        exponent, mantissa = FLOAT_BITS[typ]
        if isinstance(value, int) or math.isfinite(value):
            # Out of range: a finite value rounding past typ's largest
            past = ((1 << (mantissa + 2)) - 1) << ((1 << (exponent - 1)) - mantissa - 2)
            if abs(value) >= past:
                raise ValueError(f'{shown} at {typ} is out of its range')
            # MLIR reads a float with a dot before its exponent
            text = repr(float(value))
            return text if '.' in text else text.replace('e', '.0e')
        # A decimal past a double's range reads as an infinity
        if math.isinf(value) and 'inf' not in shown:
            raise ValueError(f'{shown} at {typ} is out of its range')
        # MLIR reads a non-finite float by its bits
        bits = ((1 << exponent) - 1) << mantissa
        if math.isnan(value):
            bits |= 1 << (mantissa - 1)
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
    """One tensor<i64> constant per literal index; a loop expression passes through"""
    lines, locs = [], []
    for k, s in enumerate(starts):
        if '%' in s:
            locs.append(s)
        else:
            locs.append(f'{lhs}.i{k}')
            lines.append(f'{locs[-1]} = stablehlo.constant dense<{int(s)}> : tensor<i64>')
    return lines, locs


def sh_slice_load(lhs, lhs_type, rhs_loc, starts, ssa_types):
    lines, locs = index_constants(lhs, [str(start) for start in starts])
    t = mlir_type(lhs_type)
    sizes = lhs_type.split('[', 1)[1].split(']')[0]
    lines.append(
        f'{lhs} = stablehlo.dynamic_slice {", ".join([rhs_loc] + locs)}, sizes = [{sizes}] : '
        f'({", ".join([ssa_types[rhs_loc]] + ["tensor<i64>"] * len(locs))}) -> {t}'
    )
    ssa_types[lhs] = t
    return '\n\t'.join(lines)


def sh_slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, starts, ssa_types):
    lines, locs = index_constants(lhs_loc, [str(start) for start in starts])
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
    'uint8': 'u8',
    'uint16': 'u16',
    'uint32': 'u32',
    'uint64': 'u64',
    'int8': 's8',
    'int16': 's16',
    'int32': 's32',
    'int64': 's64',
    'bool': 'pred',
}


def spec_mlir_type(dims, dtype):
    return mlir_type(f'{XLA_ELEM_TYPE[np.dtype(dtype).name]}[{",".join(str(d) for d in dims)}]')


def spec_bytes(spec):
    """Byte count of a spec and its u8 view shape; a multi-byte dtype gains a trailing dim"""
    width = int(np.dtype(spec['dtype']).itemsize)
    dims = [int(d) for d in spec['shape']]
    return width * int(np.prod(dims, dtype=np.int64)), dims + ([width] if width > 1 else [])


def byte_cast(dtype):
    """A bool moves to and from its bytes by a convert, any other dtype by a bitcast"""
    return 'convert' if np.dtype(dtype) == np.bool_ else 'bitcast_convert'


# Kernel inputs in, outputs out; HBM stays inside the module
def kernel_prologue(hbm_size, inputs, constants):
    """The module's parameters, and the lines laying them and the constants into %d0.0"""
    params, lines, pieces = [], [], []
    for j, const in enumerate(constants):
        flat = np.ascontiguousarray(np.asarray(const['value'])).reshape(-1).view(np.uint8)
        flat_t = mlir_type(f'u8[{flat.size}]')
        lines.append(f'%hbm.c{j} = stablehlo.constant dense<"0x{flat.tobytes().hex()}"> : {flat_t}')
        pieces.append((int(const['addr']), int(flat.size), f'%hbm.c{j}', flat_t))
    for i, spec in enumerate(inputs):
        arg_t = spec_mlir_type(spec['shape'], spec['dtype'])
        nbytes, byte_dims = spec_bytes(spec)
        byte_t, flat_t = spec_mlir_type(byte_dims, np.uint8), mlir_type(f'u8[{nbytes}]')
        params.append(f'%arg{i}: {arg_t}')
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
        lines.append(f'%out.r{i} = stablehlo.reshape %out.s{i} : ({flat_t}) -> {byte_t}')
        cast = byte_cast(spec['dtype'])
        lines.append(f'%out.o{i} = stablehlo.{cast} %out.r{i} : ({byte_t}) -> {out_t}')
        names.append(f'%out.o{i}')
        types.append(out_t)
    return lines, names, types
