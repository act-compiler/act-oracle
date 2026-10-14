"""Code generation of each instruction's API function, which traces its StableHLO"""

from taidl import Instruction

from .expression_writer import python
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


def generate_inputs(instruction: Instruction) -> str:
    """Generate input loading code from instruction's instr_input_asts"""
    output = ''

    for counter, (buffer_name, starts, sizes) in enumerate(instruction.instr_input_asts):
        model = instruction.find_data_model(buffer_name)

        # Starts and sizes as Python evaluated at each call
        size_list = [python(size) for size in sizes] + [str(dim) for dim in model.unit_dim]
        start_list = [python(start) for start in starts] + ['0'] * len(model.unit_dim)

        mapping = {
            'rhs_name': f"'{buffer_name}'",
            'lhs': f"'In{counter}'",
            'type': f"'{model.var_type}'",
            'size': '[' + ', '.join(size_list) + ']',
            'starts': '[' + ', '.join(start_list) + ']',
        }
        output += generate_code(templates['SLICE_LOAD'], mapping)

    return output


def generate_outputs(instruction: Instruction) -> str:
    """Generate output storing code from instruction's instr_output_asts"""
    code_output = ''

    for counter, (buffer_name, starts, _) in enumerate(instruction.instr_output_asts):
        model = instruction.find_data_model(buffer_name)

        start_list = [python(start) for start in starts] + ['0'] * len(model.unit_dim)

        mapping = {
            'lhs': f"'Out{counter}'",
            'rhs_name': f"'{buffer_name}'",
            'starts': '[' + ', '.join(start_list) + ']',
        }
        code_output += generate_code(templates['SLICE_STORE'], mapping)

    return code_output


def generate_api_function(instruction: Instruction) -> str:
    """Generate the API function for an instruction"""
    attr_list = ', '.join(instruction.comp_attr + instruction.parameters)
    func_name = instruction.instruction

    # Refuse an attribute shadowing one
    for attr in instruction.comp_attr + instruction.parameters:
        if attr in SHARED_NAMES:
            raise NotImplementedError(
                f"{func_name}: attribute name '{attr}' is reserved by the oracle"
            )

    set_attributes = ''
    for attr in instruction.parameters:
        set_attributes += f"'{attr}': {attr},\n"
    set_attributes = dict_text(set_attributes)

    set_comp_attr = ''
    for attr in instruction.comp_attr:
        set_comp_attr += f"'{attr}': {attr},\n"
    set_comp_attr = dict_text(set_comp_attr)

    constraints = ''
    for idx, line in enumerate(instruction.constraints):
        constraints += f'# f{idx} = (' + line + ')\n'
    for idx, line in enumerate(instruction.constraints):
        if idx == 0:
            constraints += f'\n# flag = f{idx}'
        else:
            constraints += f' and f{idx}'

    update = ''
    for line in instruction.update:
        update += '\n' + line

    inputs = generate_inputs(instruction)
    semantics = generate_semantic_code(instruction)
    outputs = generate_outputs(instruction)

    parts = []
    if inputs:
        parts.append(inputs)
    if semantics:
        parts.append(semantics)
    if outputs:
        parts.append(outputs)
    fsim_compile = '\n'.join(parts)

    output = generate_code(
        templates['API_FUNC'],
        {
            'update': update,
            'constraints': constraints,
            'fsim_compile': fsim_compile,
            'set_attributes': set_attributes,
            'set_comp_attr': set_comp_attr,
        },
    )

    func_def = f'\n\ndef {func_name}({attr_list}) -> None:\n'
    output = layout(func_def + indent_code(output))

    return output
