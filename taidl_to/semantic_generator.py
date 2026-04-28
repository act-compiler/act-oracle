"""Semantic code generator using ANTLR4 parsing"""

from .oracle_visitor import OracleVisitor


def generate_semantic_code(ast) -> str:
    if ast is None:
        return "\tpass"

    visitor = OracleVisitor()
    visitor.visit(ast)

    code_lines = [line for line in visitor.instruction_lines if line != "Module:"]

    indented_lines = ["\t" + line for line in code_lines]

    code = "\n".join(indented_lines)

    return code
