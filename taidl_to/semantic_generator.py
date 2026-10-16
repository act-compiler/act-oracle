"""Semantic code generator using ANTLR4 parsing"""

from .oracle_visitor import OracleVisitor


def generate_semantic_code(instruction) -> str:
    if instruction.instr_semantics_ast is None:
        return ''

    visitor = OracleVisitor(instruction.instruction, instruction.instr_expression_asts)
    visitor.visit(instruction.instr_semantics_ast)

    code = '\n'.join(visitor.instruction_lines)

    return code
