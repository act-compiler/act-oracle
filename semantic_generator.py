"""Semantic code generator using ANTLR4 parsing

This module provides a clean interface for parsing instruction semantics
and generating Python code for HLO oracle generation.
"""

from idl_visitor import parse_idl


def generate_semantic_code(semantics_text: str) -> str:
    """
    Generate Python code for instruction semantics using ANTLR4 parsing.

    Args:
        semantics_text: HLO-like IR text describing instruction semantics

    Returns:
        Indented Python code string that generates HLO when executed

    Example:
        Input:
            ENTRY load_rm {
                %In1 = u8[`@c.n * 128`] parameter(0);
                ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%In1);
            }

        Output: Indented Python code lines for generating the HLO
    """
    if not semantics_text or not semantics_text.strip():
        return "\tpass"

    # Parse using ANTLR4 visitor
    instruction_lines = parse_idl(semantics_text)

    # Filter out module header line if present
    code_lines = [line for line in instruction_lines if line != "Module:"]

    # Indent each line with a tab
    indented_lines = ["\t" + line for line in code_lines]

    # Join into Python code block
    code = "\n".join(indented_lines)

    return code
