"""Semantic code generator using ANTLR4 parsing"""

from .oracle_visitor import OracleVisitor


def generate_semantic_code(ast) -> str:
    if ast is None:
        return ""

    visitor = OracleVisitor()
    visitor.visit(ast)

    code = "\n".join(visitor.instruction_lines)

    return code
