"""Code generation of each instruction's API function, which traces its StableHLO"""

from taidl import Instruction

from .semantic_generator import generate_semantic_code
from .template import dict_text, generate_code, indent_code, init_templates, layout, templates

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
    """Generate input loading code from instruction's instr_inputs"""
    init_templates()
    output = ''

    for counter, slice_spec in enumerate(instruction.instr_inputs):
        assert len(slice_spec) == 3, 'Wrong input slice formatting.'

        buffer_name = slice_spec[0]
        start_indices = slice_spec[1]
        shapes = slice_spec[2]

        model = instruction.find_data_model(buffer_name)

        size_list = [f"'{s}'" for s in shapes]
        for unit_dim in model.unit_dim:
            size_list.append(f"'{unit_dim}'")

        slice_configs = [f"'{start}'" for start in start_indices]
        for unit_dim in model.unit_dim:
            slice_configs.append("'0'")

        mapping = {
            'rhs_name': f"'{buffer_name}'",
            'lhs': f"'In{counter}'",
            'type': f"'{model.var_type}'",
            'size': '[' + ', '.join(size_list) + ']',
            'slice': '[' + ', '.join(slice_configs) + ']',
        }
        output += generate_code(templates['SLICE_LOAD'], mapping)

    return output


def generate_outputs(instruction: Instruction) -> str:
    """Generate output storing code from instruction's instr_outputs"""
    init_templates()
    code_output = ''

    for counter, slice_spec in enumerate(instruction.instr_outputs):
        assert len(slice_spec) == 3, 'Wrong output slice formatting.'

        buffer_name = slice_spec[0]
        start_indices = slice_spec[1]

        model = instruction.find_data_model(buffer_name)

        start_list = [f"'{idx}'" for idx in start_indices]
        for _ in model.unit_dim:
            start_list.append("'0'")

        mapping = {
            'lhs': f"'Out{counter}'",
            'rhs_name': f"'{buffer_name}'",
            'slice': '[' + ', '.join(start_list) + ']',
        }
        code_output += generate_code(templates['SLICE_STORE'], mapping)

    return code_output


def generate_api_function(instruction: Instruction) -> str:
    """Generate the API function for an instruction"""
    init_templates()

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
    semantics = generate_semantic_code(instruction.instr_semantics_ast, instruction.instruction)
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
