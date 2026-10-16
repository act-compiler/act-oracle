"""Code generation of each instruction's API function, which traces its StableHLO"""

import ast
from collections import Counter

from taidl import Instruction

from .expression_writer import condition, held, integer, python, reference
from .semantic_generator import generate_semantic_code
from .template import dict_text, generate_code, indent_code, layout, templates

# Module state an API function reads or sets
SHARED_NAMES = {
    'comp_attrs',
    'cur_scope',
    'global_counters',
    'instruction_scopes',
    'ssa_types',
    'state',
}


def tile_dims(instruction: Instruction, value: str, holding=False) -> list:
    """A tile's dims as Python, as its semantics declares them, each looped one held if holding"""
    lines = instruction.instr_semantics_ast.instruction()
    line = next(line for line in lines if line.IDENTIFIER().getText() == value)
    shape = line.result_type().shape()
    expressions = instruction.instr_expression_asts
    return [integer(dim, expressions, holding) for dim in shape.dim()] if shape else []


def generate_constraints(instruction: Instruction) -> str:
    """Generate a check of each constraint, at each iteration, against the state before the call"""
    output = ''

    for text, tree in zip(instruction.constraints, instruction.constraint_asts):
        holds, read = condition(tree)
        mapping = {
            'instruction': f"'{instruction.instruction}'",
            'text': repr(text),
            'read': read,
            'condition': holds,
        }
        output += generate_code(templates['CHECK_CONSTRAINT'], mapping)

    return output


def generate_updates(instruction: Instruction) -> str:
    """Generate each state update in order, its value an integer"""
    output = ''

    for tree in instruction.update_asts:
        mapping = {'target': reference(tree.STATE().getText()), 'value': python(tree.expression())}
        output += generate_code(templates['UPDATE_STATE'], mapping)

    return output


def generate_inputs(instruction: Instruction) -> str:
    """Generate input loading code from instruction's instr_input_asts"""
    output = ''

    for counter, (buffer_name, starts, sizes) in enumerate(instruction.instr_input_asts):
        model = instruction.find_data_model(buffer_name)

        # Starts and sizes as Python evaluated at each call
        size_list = [python(size) for size in sizes] + [str(dim) for dim in model.unit_dim]
        start_list = [python(start) for start in starts] + ['0'] * len(model.unit_dim)
        # The tile's dims, else its region's sizes, and a copy held to one value
        if instruction.instr_semantics_ast:
            tile_list = tile_dims(instruction, f'%In{counter}')
            dims_list = tile_dims(instruction, f'%In{counter}', holding=True)
        else:
            tile_list = size_list
            dims_list = [held(size) for size in sizes] + [str(dim) for dim in model.unit_dim]

        mapping = {
            'instruction': f"'{instruction.instruction}'",
            'rhs_name': f"'{buffer_name}'",
            'lhs': f"'In{counter}'",
            'type': f"'{model.var_type}'",
            'size': '[' + ', '.join(size_list) + ']',
            'tile': '[' + ', '.join(tile_list) + ']',
            'dims': '[' + ', '.join(dims_list) + ']',
            'starts': '[' + ', '.join(start_list) + ']',
        }
        output += generate_code(templates['SLICE_LOAD'], mapping)

    return output


def generate_outputs(instruction: Instruction) -> str:
    """Generate output storing code from instruction's instr_output_asts"""
    code_output = ''

    for counter, (buffer_name, starts, sizes) in enumerate(instruction.instr_output_asts):
        model = instruction.find_data_model(buffer_name)

        size_list = [python(size) for size in sizes] + [str(dim) for dim in model.unit_dim]
        start_list = [python(start) for start in starts] + ['0'] * len(model.unit_dim)

        mapping = {
            'instruction': f"'{instruction.instruction}'",
            'lhs': f"'Out{counter}'",
            'rhs_name': f"'{buffer_name}'",
            'type': f"'{model.var_type}'",
            'size': '[' + ', '.join(size_list) + ']',
            'starts': '[' + ', '.join(start_list) + ']',
        }
        code_output += generate_code(templates['SLICE_STORE'], mapping)

    return code_output


def generate_sizes(instruction: Instruction) -> str:
    """Generate a check of each size and dimension an expression gives, at the call"""
    sizes, dims = {}, {}

    # Each region's sizes, then each dim its semantics declares
    for regions, trees in (
        (instruction.instr_inputs, instruction.instr_input_asts),
        (instruction.instr_outputs, instruction.instr_output_asts),
    ):
        for (_, _, texts), (_, _, size_trees) in zip(regions, trees):
            for text, tree in zip(texts, size_trees):
                sizes[f'`{text}`'] = python(tree)
    lines = instruction.instr_semantics_ast.instruction() if instruction.instr_semantics_ast else []
    for line in lines:
        shape = line.result_type().shape()
        for dim in shape.dim() if shape else []:
            if dim.EXPRESSION() and dim.getText() not in sizes:
                dims[dim.getText()] = python(instruction.instr_expression_asts[dim.EXPRESSION()])

    # A literal needs no check
    sizes = {text: source for text, source in sizes.items() if not source.isdigit()}
    if not sizes and not dims:
        return ''
    mapping = {
        'instruction': f"'{instruction.instruction}'",
        'sizes': '{' + ', '.join(f'{text!r}: {source}' for text, source in sizes.items()) + '}',
        'dims': '{' + ', '.join(f'{text!r}: {source}' for text, source in dims.items()) + '}',
    }
    return generate_code(templates['CHECK_SIZES'], mapping)


def generate_api_function(instruction: Instruction) -> str:
    """Generate the API function for an instruction"""
    attr_list = ', '.join(instruction.comp_attr + instruction.parameters)
    func_name = instruction.instruction

    set_attributes = ''
    for attr in instruction.parameters:
        set_attributes += f"'{attr}': {attr},\n"
    set_attributes = dict_text(set_attributes)

    set_comp_attr = ''
    for attr in instruction.comp_attr:
        set_comp_attr += f"'{attr}': {attr},\n"
    set_comp_attr = dict_text(set_comp_attr)

    constraints = generate_constraints(instruction)
    updates = generate_updates(instruction)
    sizes = generate_sizes(instruction)
    inputs = generate_inputs(instruction)
    semantics = generate_semantic_code(instruction)
    outputs = generate_outputs(instruction)

    parts = []
    if constraints:
        parts.append(constraints)
    if updates:
        parts.append(updates)
    if sizes:
        parts.append(sizes)
    if inputs:
        parts.append(inputs)
    if semantics:
        parts.append(semantics)
    if outputs:
        parts.append(outputs)
    # Its body indented under the template's try
    fsim_compile = indent_code('\n'.join(parts))

    output = generate_code(
        templates['API_FUNC'],
        {
            'name': f"'{func_name}'",
            'fsim_compile': fsim_compile,
            'set_attributes': set_attributes,
            'set_comp_attr': set_comp_attr,
            'updates': str(bool(instruction.update)),
        },
    )

    func_def = f'\n\ndef {func_name}({attr_list}) -> None:\n'
    output = layout(func_def + indent_code(output))

    # Refuse an attribute named as one the oracle reserves
    clash = clashing(output, instruction.comp_attr + instruction.parameters)
    if clash:
        raise NotImplementedError(
            f"{func_name}: attribute name '{clash[0]}' is reserved by the oracle"
        )

    return output


def python_name(name: str) -> bool:
    """Whether a name is one of Python's own, as __debug__ is."""
    return name.startswith('__') and name.endswith('__')


def clashing(function: str, attributes: list[str]) -> list[str]:
    """The attributes named as Python's own, a shared name or a name the function reads."""
    names = [node for node in ast.walk(ast.parse(function)) if isinstance(node, ast.Name)]
    loads = Counter(node.id for node in names if isinstance(node.ctx, ast.Load))
    stores = {node.id for node in names if isinstance(node.ctx, ast.Store)}
    # An attribute is read once; a local is written first
    return [
        attr
        for attr in attributes
        if python_name(attr) or attr in SHARED_NAMES or (loads[attr] > 1 and attr not in stores)
    ]
