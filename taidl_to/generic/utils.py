import re

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
        elif part and part not in ['+', '-']:
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
                    op = parts[i-1]
                    if op == '*':
                        coeff *= num
                    elif op == '/':
                        coeff = int(coeff / num)
            elif val.startswith('%'):
                if i == 0:
                    mult_vars.append(val)
                else:
                    op = parts[i-1]
                    if op == '*':
                        mult_vars.append(val)
                    elif op == '/':
                        div_vars.append(val)
        i += 1
    return coeff, mult_vars, div_vars


def generate_loop_eq_vars(expression, global_counters):
    terms = parse_simple_expression(expression)
    final_name = f"%loop_final.{global_counters['loop_var']}"
    lines = []

    def make_operand(k, coeff, mult_vars, div_vars):
        result = f"{final_name}.c{k}"
        lines.append(f"{result} = stablehlo.constant dense<{int(coeff)}> : tensor<i64>")
        for n, var in enumerate(mult_vars):
            lines.append(f"{result}.m{n} = stablehlo.multiply {result}, {var} : tensor<i64>")
            result = f"{result}.m{n}"
        for n, var in enumerate(div_vars):
            lines.append(f"{result}.d{n} = stablehlo.divide {result}, {var} : tensor<i64>")
            result = f"{result}.d{n}"
        return result

    operands = [make_operand(k, coeff, mult_vars, div_vars) for k, (coeff, mult_vars, div_vars) in enumerate(terms) if coeff != 0]

    if not operands:
        return f"{final_name} = stablehlo.constant dense<0> : tensor<i64>\n", final_name

    nested_adds = operands[0]
    for n, operand in enumerate(operands[1:]):
        lines.append(f"{final_name}.a{n} = stablehlo.add {nested_adds}, {operand} : tensor<i64>")
        nested_adds = f"{final_name}.a{n}"
    lines.append(f"{final_name} = stablehlo.reshape {nested_adds} : (tensor<i64>) -> tensor<i64>")
    return "\n".join(lines) + "\n", final_name


def substitute(template, attrs, state, lvars, consts):
    if(type(template) == int):
        return str(template)
    for key, value in attrs.items():
        template = template.replace('@a.' + key, str(value))
    for key, value in state.items():
        template = template.replace('@s.' + key, str(value))
    for key, value in lvars.items():
        template = template.replace('@l.' + key, str(value))
    for key, value in consts.items():
        template = template.replace('@c.' + key, str(value))
    return template


def compute(exp_: str) -> int | float:
    """
    Compute the value of an expression
    """
    import ast

    try:
        if(f"%loop_final" in exp_):
            return exp_
        output = eval(compile(ast.parse(exp_, mode='eval'), '', 'eval'))
        if(type(output) is float):
            output = int(output)
    except Exception:
        print("Expression: ")
        print(exp_)
        output = "Failed"
        assert(0)
    return output

def lhs_util(attrs, state, global_counters, lvars, consts, lhs_name, lhs_vartype="", lhs_dim=[]):
    if(lhs_name in global_counters):
        global_counters[lhs_name]["counter"] += 1
        lhs_type = global_counters[lhs_name]["type"]
    else:
        dim_sizes = simplify_vals(lhs_dim, attrs, state, lvars, consts)
        dim_sizes = ','.join(dim_sizes)
        lhs_type = f'{lhs_vartype}[{dim_sizes}]'
    lhs_name = parameter_util(global_counters, attrs, state, lvars, consts, lhs_name)
    return lhs_name, lhs_type

def parameter_util(global_counters, attrs, state, lvars, consts, lhs_name):
    lhs_name = substitute(lhs_name, attrs, state, lvars, consts)
    if(lhs_name in global_counters):
        lhs_name += f'.{global_counters[lhs_name]["counter"]}'
    else:
        lhs_name += f'.{global_counters["instruction_counter"]}'
    lhs_name = "%" + lhs_name
    return lhs_name


def simplify_vals(vals, attrs, state, lvars, consts):
    simplify = lambda x: str((compute(substitute(x, attrs, state, lvars, consts))))
    vals = list(map(simplify, vals))
    return vals

# Each sh_* helper records its result's MLIR type in ssa_types
MLIR_ELEM_TYPE = {"f64": "f64", "f32": "f32", "f16": "f16", "bf16": "bf16",
              "u8": "ui8", "u16": "ui16", "u32": "ui32", "u64": "ui64",
              "s8": "i8", "s16": "i16", "s32": "i32", "s64": "i64", "pred": "i1"}


def mlir_type(xla_type):
    """bf16[64,64] -> tensor<64x64xbf16>, f32[] -> tensor<f32>"""
    base, dims = xla_type.split('[', 1)
    dims = dims.split(']', 1)[0].replace(' ', '')
    return "tensor<" + "".join(f"{d}x" for d in dims.split(',') if d) + MLIR_ELEM_TYPE[base.strip()] + ">"


def mlir_elem(mlir_t):
    return mlir_t[len("tensor<"):-1].rsplit('x', 1)[-1]


def mlir_zero(mlir_t):
    return "0.0" if mlir_elem(mlir_t) in ("f64", "f32", "f16", "bf16") else "0"


def dims_list(dims):
    """{1, 0} as text, a set or {} -> [1, 0], keeping a text's order"""
    if isinstance(dims, (set, dict)):
        return str(sorted(dims))
    return "[" + ", ".join(d.strip() for d in dims.strip("{} ").split(',') if d.strip()) + "]"


def sh_zeros(lhs, xla_type, ssa_types):
    t = mlir_type(xla_type)
    ssa_types[lhs] = t
    return f"{lhs} = stablehlo.constant dense<{mlir_zero(t)}> : {t}"


def sh_unary(op, lhs, lhs_type, x, ssa_types):
    t = mlir_type(lhs_type)
    ssa_types[lhs] = t
    return f"{lhs} = stablehlo.{op} {x} : {t}"


def sh_binary(op, lhs, lhs_type, a, b, ssa_types):
    t = mlir_type(lhs_type)
    ssa_types[lhs] = t
    return f"{lhs} = stablehlo.{op} {a}, {b} : {t}"


def sh_cast(op, lhs, lhs_type, x, ssa_types):
    t = mlir_type(lhs_type)
    line = f"{lhs} = stablehlo.{op} {x} : ({ssa_types[x]}) -> {t}"
    ssa_types[lhs] = t
    return line


def sh_copy(lhs, lhs_type, x, ssa_types):
    """copy keeps its operand's type: an identity reshape"""
    t = mlir_type(lhs_type)
    if ssa_types[x] != t:
        raise ValueError(f"copy of {x} : {ssa_types[x]} does not have the result type {t}")
    ssa_types[lhs] = t
    return f"{lhs} = stablehlo.reshape {x} : ({t}) -> {t}"


def sh_transpose(lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    line = f"{lhs} = stablehlo.transpose {x}, dims = {dims_list(dims)} : ({ssa_types[x]}) -> {t}"
    ssa_types[lhs] = t
    return line


def sh_broadcast(lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    line = f"{lhs} = stablehlo.broadcast_in_dim {x}, dims = {dims_list(dims)} : ({ssa_types[x]}) -> {t}"
    ssa_types[lhs] = t
    return line


def sh_dot(lhs, lhs_type, a, b, lhs_batch, lhs_contracting, rhs_batch, rhs_contracting, ssa_types):
    t = mlir_type(lhs_type)
    lb, rb = dims_list(lhs_batch), dims_list(rhs_batch)
    batching = f"batching_dims = {lb} x {rb}, " if lb != "[]" or rb != "[]" else ""
    line = (f"{lhs} = stablehlo.dot_general {a}, {b}, {batching}"
            f"contracting_dims = {dims_list(lhs_contracting)} x {dims_list(rhs_contracting)} : "
            f"({ssa_types[a]}, {ssa_types[b]}) -> {t}")
    ssa_types[lhs] = t
    return line


def sh_reduce(op, lhs, lhs_type, x, dims, ssa_types):
    t = mlir_type(lhs_type)
    scalar_t = f"tensor<{mlir_elem(t)}>"
    lines = [f"{lhs}.init = stablehlo.constant dense<{mlir_zero(scalar_t)}> : {scalar_t}",
             f"{lhs} = stablehlo.reduce({x} init: {lhs}.init) applies stablehlo.{op} "
             f"across dimensions = {dims_list(dims)} : ({ssa_types[x]}, {scalar_t}) -> {t}"]
    ssa_types[lhs] = t
    return "\n\t".join(lines)


def index_constants(lhs, starts):
    """One tensor<i64> constant per literal index; a loop expression passes through"""
    lines, locs = [], []
    for k, s in enumerate(starts):
        if '%' in s:
            locs.append(s)
        else:
            locs.append(f"{lhs}.i{k}")
            lines.append(f"{locs[-1]} = stablehlo.constant dense<{int(s)}> : tensor<i64>")
    return lines, locs


def sh_slice_load(lhs, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, consts, ssa_types):
    simplify = lambda x: str(compute(substitute(x, attrs, state, lvars, consts)))
    lines, locs = index_constants(lhs, [simplify(sc) for sc in slice_configs])
    t = mlir_type(lhs_type)
    sizes = lhs_type.split('[', 1)[1].split(']')[0]
    lines.append(f"{lhs} = stablehlo.dynamic_slice {', '.join([rhs_loc] + locs)}, sizes = [{sizes}] : "
                 f"({', '.join([ssa_types[rhs_loc]] + ['tensor<i64>'] * len(locs))}) -> {t}")
    ssa_types[lhs] = t
    return "\n\t".join(lines)


def sh_slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices, attrs, state, lvars, consts, ssa_types):
    simplify = lambda x: str(compute(substitute(x, attrs, state, lvars, consts)))
    lines, locs = index_constants(lhs_loc, list(map(simplify, start_indices)))
    t = mlir_type(lhs_type)
    lines.append(f"{lhs_loc} = stablehlo.dynamic_update_slice {', '.join([rhs_loc, rhs_update] + locs)} : "
                 f"({', '.join([ssa_types[rhs_loc], ssa_types[rhs_update]] + ['tensor<i64>'] * len(locs))}) -> {t}")
    ssa_types[lhs_loc] = t
    return lines
