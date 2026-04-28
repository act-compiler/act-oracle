"""Code generation methods for Instruction objects.

These were originally on the Instruction class in TAIDL but belong to the oracle
since they produce oracle-specific output (HLO simulation code).
"""

from typing import List
from taidl import Instruction

from .template import init_templates, templates, generate_code, indent_code
from .semantic_generator import generate_semantic_code


def generate_inputs(instruction: Instruction) -> str:
    """Generate input loading code from instruction's instr_inputs"""
    init_templates()
    output = ""

    for counter, slice_spec in enumerate(instruction.instr_inputs, start=1):
        assert len(slice_spec) == 3, "Wrong input slice formatting."

        buffer_name = slice_spec[0]
        start_indices = slice_spec[1]
        shapes = slice_spec[2]

        model = instruction.find_data_model(buffer_name)

        size_list = [f"'{s}'" for s in shapes]
        for unit_dim in model.unit_dim:
            size_list.append(f"'{unit_dim}'")

        slice_configs = []
        for start, shape in zip(start_indices, shapes):
            slice_configs.append(f"'{start}:{start}+{shape}'")
        for unit_dim in model.unit_dim:
            slice_configs.append(f"'0:{unit_dim}'")

        mapping = {
            "rhs_name": f"'{buffer_name}'",
            "lhs": f"'In{counter}'",
            "type": f"'{model.var_type}'",
            "size": "[" + ",".join(size_list) + "]",
            "slice": "[" + ",".join(slice_configs) + "]"
        }
        output += generate_code(templates["SLICE_LOAD"], mapping)

    return indent_code(output) if output else ""


def generate_outputs(instruction: Instruction) -> str:
    """Generate output storing code from instruction's instr_outputs"""
    init_templates()
    code_output = ""

    for counter, slice_spec in enumerate(instruction.instr_outputs):
        assert len(slice_spec) == 3, "Wrong output slice formatting."

        buffer_name = slice_spec[0]
        start_indices = slice_spec[1]

        model = instruction.find_data_model(buffer_name)

        start_list = [f"'{idx}'" for idx in start_indices]
        for _ in model.unit_dim:
            start_list.append("'0'")

        mapping = {
            "lhs": f"'Out{counter}'",
            "rhs_name": f"'{buffer_name}'",
            "slice": "[" + ",".join(start_list) + "]"
        }
        code_output += generate_code(templates["SLICE_STORE"], mapping)

    return indent_code(code_output) if code_output else ""


def generate_api_function(instruction: Instruction) -> str:
    """Generate the API function for an instruction"""
    init_templates()

    attr_list = ",".join(instruction.comp_attr + instruction.parameters)
    func_name = instruction.instruction

    set_attributes = ""
    for attr in instruction.parameters:
        set_attributes += f'\t"{attr}": {attr},\n'
    set_attributes = indent_code(set_attributes)

    set_comp_attr = ""
    for attr in instruction.comp_attr:
        set_comp_attr += f'\t"{attr}": {attr},\n'
    set_comp_attr = indent_code(set_comp_attr)

    constraints = ""
    for idx, line in enumerate(instruction.constraints):
        constraints += f'#f{idx} = (' + line + ')\n'
    for idx, line in enumerate(instruction.constraints):
        if idx == 0:
            constraints += f'\n#flag = f{idx} '
        else:
            constraints += f'and f{idx} '

    update = ""
    for line in instruction.update:
        update += '\n' + line
    cost = instruction.cost
    fsim = "pass"

    inputs = generate_inputs(instruction)
    semantics = generate_semantic_code(instruction.instr_semantics)
    outputs = generate_outputs(instruction)

    parts = []
    if inputs:
        parts.append(inputs)
    if semantics:
        parts.append(semantics)
    if outputs:
        parts.append(outputs)
    fsim_compile = '\n'.join(parts)

    output = generate_code(templates["API_FUNC"], {
        "attributes": attr_list,
        "func_name": func_name,
        "update": update,
        "cost": cost,
        "constraints": constraints,
        "fsim": fsim,
        "fsim_compile": fsim_compile,
        "set_attributes": set_attributes,
        "set_comp_attr": set_comp_attr
    })

    func_def = f'def {func_name}({attr_list}) -> None:\n'
    output = func_def + indent_code(output)

    return output


def generate_semantic_function(instruction: Instruction) -> str:
    """Generate the semantic function for an instruction"""
    inputs = generate_inputs(instruction)
    semantics = generate_semantic_code(instruction.instr_semantics)
    outputs = generate_outputs(instruction)

    output = f'\ndef {instruction.instruction}_semantics(attrs, state, global_counters):\n'
    output += '\toutput = []\n\tlvars={}\n'
    output += inputs
    output += semantics
    output += outputs
    output += '\treturn output\n'
    return output
