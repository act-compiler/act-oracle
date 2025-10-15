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

    def make_operand(coeff, mult_vars, div_vars):
        result = f"s32[] constant({int(coeff)})"

        if not mult_vars and not div_vars:
            return result

        for var in mult_vars:
            result = f"s32[] multiply({result}, {var})"

        for var in div_vars:
            result = f"s32[] divide({result}, {var})"

        return result

    operands = [make_operand(coeff, mult_vars, div_vars) for coeff, mult_vars, div_vars in terms if coeff != 0]
    
    final_name = f"%loop_final.{global_counters['loop_var']}"
    
    if not operands:
        hlo_line = f"{final_name} = s32[] constant(0)\n"
        return hlo_line, final_name

    nested_adds = operands[0]
    for operand in operands[1:]:
        nested_adds = f"s32[] add({nested_adds}, {operand})"
    hlo_line = f"{final_name} = {nested_adds}\n"

    return hlo_line, final_name


def replace_values(string, symbol, list):
    pattern = rf'@{symbol}\.(\w+)'
    replaced_string = re.sub(pattern, fr'{list}["\1"]', string)
    return replaced_string
    
def replace_values(string, symbol, list):
    pattern = rf'@{symbol}\.(\w+)'
    replaced_string = re.sub(pattern, fr'{list}["\1"]', string)
    return replaced_string

def blind_substitute(template):
    template = replace_values(template, "a", "attrs")
    template = replace_values(template, "s", "state")
    template = replace_values(template, "c", "comp_attrs")
    template = replace_values(template, "l", "lvars")
    return template

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

def create_num_str(var_dim: list)->str:
        output = '{'
        for i in range(len(var_dim) - 1, -1, -1):
            if(i == 0):
                output += str(i)
            else:
                output += str(i) + ","
        output += "}"
        return output

def get_dim(lhs_dim, attrs, state, lvars, consts):
    dim_sizes = simplify_vals(lhs_dim, attrs, state, lvars, consts)
    dim_sizes = ','.join(dim_sizes)
    return "[" + dim_sizes + "]"

def lhs_util(attrs, state, global_counters, lvars, consts, lhs_name, lhs_vartype="", lhs_dim=[]):
    if(lhs_name in global_counters):
        global_counters[lhs_name]["counter"] += 1
        lhs_type = f'{global_counters[lhs_name]["type"]}{global_counters[lhs_name]["dim"]}'
    else:
        num = create_num_str(lhs_dim)
        dim_sizes = simplify_vals(lhs_dim, attrs, state, lvars, consts)
        dim_sizes = ','.join(dim_sizes)
        lhs_type = f'{lhs_vartype}[{dim_sizes}]{num}'
        if(len(dim_sizes) == 0):
            lhs_type = f"{lhs_vartype}[]"
    lhs_name = parameter_util(global_counters, attrs, state, lvars, consts, lhs_name)
    return lhs_name, lhs_type

def parameter_util(global_counters, attrs, state, lvars, consts, lhs_name):
    lhs_name = substitute(lhs_name, attrs, state, lvars, consts)
    if(lhs_name in global_counters):
        lhs_name += f'.{global_counters[lhs_name]["counter"]}'
    else:
        lhs_name += f'.{global_counters["loop_counter"]}'
        lhs_name += f'.{global_counters["instruction_counter"]}'
    lhs_name = "%" + lhs_name
    return lhs_name


def simplify_vals(vals, attrs, state, lvars, consts):
    simplify = lambda x: str((compute(substitute(x, attrs, state, lvars, consts))))
    vals = list(map(simplify, vals))
    return vals

def non_dynamic_slice(slice_configs, attrs, state, lvars, consts):
    slice_configs = [sc.split(':') for sc in slice_configs]
    simplify = lambda x: str(int(compute(substitute(x, attrs, state, lvars, consts))))
    slice_configs = [list(map(simplify, sc)) for sc in slice_configs]
    slice_configs = [('[' + ':'.join(sc) + ']') for sc in slice_configs]
    slice_configs = ', '.join(slice_configs)

    line = f'slice={{{slice_configs}}}'
    return line

def slice_load(lhs, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, consts):
    slice_configs = [sc.split(':') for sc in slice_configs]
    simplify = lambda x: str((compute(substitute(x, attrs, state, lvars, consts))))
    slice_configs = [list(map(simplify, sc)) for sc in slice_configs]
    start_indices = ""
    for sc in slice_configs:
        if("%" not in sc[0]):
            start_indices += f", s32[] constant({sc[0]})"
        else:
            start_indices += f", {sc[0]}"
    slice_sizes= lhs_type.split('[', 1)[1].split(']')[0]
    line = f'{lhs} = {lhs_type} dynamic-slice({rhs_loc}{start_indices}), '
    line += f'dynamic_slice_sizes={{{slice_sizes}}}'
    return line


def slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices, attrs, state, lvars, consts, prefix):
    simplify = lambda x: str((compute(substitute(x, attrs, state, lvars, consts))))
    start_indices = list(map(simplify, start_indices))
    lines = []
    #for i in range(len(start_indices)):
        #lines.append(f'%start_indices.{i}.{prefix} = s32[] constant({start_indices[i]})')
    update_slice = f'{lhs_loc} = {lhs_type} dynamic-update-slice({rhs_loc}, {rhs_update}'
    for i in range(len(start_indices)):
        if("%" not in start_indices[i]):
            update_slice += f', s32[] constant({start_indices[i]})'
        else:
            update_slice += f', {start_indices[i]}'
    update_slice += f')'
    lines.append(update_slice)
    return lines


def reshape_helper(lhs_loc, lhs_type, rhs_loc):
    return f'{lhs_loc} = {lhs_type} reshape({rhs_loc})'

def convert_helper(lhs_loc, lhs_type, rhs_loc):
    return f'{lhs_loc} = {lhs_type} convert({rhs_loc})'


def dot_helper(lhs_loc, lhs_type, rhs_loc_A, rhs_loc_B, lhs_batch, lhs_contracting, rhs_batch, rhs_contracting):
    line = f'{lhs_loc} = {lhs_type} dot({rhs_loc_A}, {rhs_loc_B}), '
    line += f'lhs_batch_dims={lhs_batch}, lhs_contracting_dims={lhs_contracting}, '
    line += f'rhs_batch_dims={rhs_batch}, rhs_contracting_dims={rhs_contracting}'
    return line
