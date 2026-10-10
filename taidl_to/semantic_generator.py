"""Semantic code generator using ANTLR4 parsing"""

import re

from .oracle_visitor import OracleVisitor

# The trace's counters, and the value the loop counter numbers
TRACE_NAMES = {'instruction_counter', 'total_loop_counter', 'loop_var', 'loop_final'}


def generate_semantic_code(instruction) -> str:
    if instruction.instr_semantics_ast is None:
        # Nothing computes an output
        if instruction.instr_output_asts:
            raise ValueError(
                f"{instruction.instruction}: no semantics gives '%Out0' for output region 0"
            )
        return ''

    # Each value's type, as its line declares it
    lines = instruction.instr_semantics_ast.instruction()
    types = {line.IDENTIFIER().getText(): line.result_type().TYPE().getText() for line in lines}
    # A value takes no data model's or trace's name
    reserved = TRACE_NAMES | {model.var_name for model in instruction.data_models}
    for value in types:
        if value[1:] in reserved:
            raise NotImplementedError(
                f"{instruction.instruction}: value name '{value}' is reserved by the oracle"
            )
    regions = [
        ('In', 'input', instruction.instr_input_asts),
        ('Out', 'output', instruction.instr_output_asts),
    ]
    for kind, where, given in regions:
        # A %In<k> or %Out<k> with no region of its kind
        for value in types:
            if not given and re.fullmatch(rf'%{kind}\d+', value):
                raise ValueError(f"{instruction.instruction}: '{value}' has no {where} region")
        # Each tile has its data model's type
        for k, (buffer, _, _) in enumerate(given):
            value, model = f'%{kind}{k}', instruction.find_data_model(buffer)
            if types[value] != model.var_type:
                raise ValueError(
                    f"{instruction.instruction}: '{value}' is {types[value]}, "
                    f'its data model {buffer} holds {model.var_type}'
                )

    visitor = OracleVisitor(instruction.instruction, instruction.instr_expression_asts)
    visitor.visit(instruction.instr_semantics_ast)

    code = '\n'.join(visitor.instruction_lines)

    return code
