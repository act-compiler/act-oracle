"""Semantic code generator using ANTLR4 parsing"""

from .idl_visitor import parse_idl


def generate_semantic_code(semantics_text: str) -> str:
    if not semantics_text or not semantics_text.strip():
        return "\tpass"

    instruction_lines = parse_idl(semantics_text)

    code_lines = [line for line in instruction_lines if line != "Module:"]

    indented_lines = ["\t" + line for line in code_lines]

    code = "\n".join(indented_lines)

    return code
