"""Semantic code generator using ANTLR4 parsing"""

from .oracle_visitor import OracleVisitor


def generate_semantic_code(instruction) -> str:
    if instruction.instr_semantics_ast is None:
        return ''

    # Each value's type, as its line declares it
    lines = instruction.instr_semantics_ast.instruction()
    types = {line.IDENTIFIER().getText(): line.result_type().TYPE().getText() for line in lines}
    regions = [('In', instruction.instr_input_asts), ('Out', instruction.instr_output_asts)]
    for kind, given in regions:
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
