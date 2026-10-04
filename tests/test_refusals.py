"""Test what generation refuses"""

import os
import re
import tempfile

import pytest
import taidl.antlr4
from taidl import Accelerator
from taidl.operations import BOOL, INT, INTS, LITERAL, NUMBER, ONE_INT, OPERATIONS, RANGES

from taidl_to import generate_oracle
from taidl_to.oracle_visitor import OracleVisitor


def unmapped():
    """The words the grammar lexes that no template lowers."""
    with open(os.path.join(os.path.dirname(taidl.antlr4.__file__), 'IDLV2.g4')) as file:
        grammar = re.sub(r'//.*', '', file.read())
    rule = re.search(r'^OPERATION\s*:(.*?);', grammar, re.MULTILINE | re.DOTALL).group(1)
    # parameter declares an input, with no template of its own
    lowered = {'parameter', *OracleVisitor('t').op_to_template}
    return [word for word in re.findall(r"'([^']+)'", rule) if word not in lowered]


# A value of each form an attribute takes
SAMPLES = {INT: '0', INTS: '{0}', ONE_INT: '{0}', RANGES: '{[0:2]}', BOOL: 'true', NUMBER: '0.5'}


def written(op):
    """A line that writes op with the operands and the required attributes of its row."""
    takes, attributes = OPERATIONS[op]
    if takes == LITERAL:
        operands = '0'
    else:
        count = takes[0] if isinstance(takes, tuple) else takes
        operands = ', '.join(['%In0'] * count)
    given = ''.join(
        f', {name}={form[0] if isinstance(form, tuple) else SAMPLES[form]}'
        for name, (form, required) in attributes.items()
        if required
    )
    return f'ROOT %Out0 = u8[2] {op}({operands}){given};'


@pytest.mark.parametrize('op', unmapped())
def test_unmapped_operation_refused(op):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[2] parameter(0);
    {written(op)}
}}
""")
    with pytest.raises(NotImplementedError, match=f"refused: operation '{op}'"):
        generate_oracle(acc)


# Module state a generated function reads or sets
SHARED_NAMES = [
    'comp_attrs',
    'cur_scope',
    'global_counters',
    'instruction_scopes',
    'ssa_types',
    'state',
]


@pytest.mark.parametrize('addressing', [False, True])
@pytest.mark.parametrize('name', SHARED_NAMES)
def test_attribute_named_as_a_shared_name_refused(name, addressing):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    # The name as a computation or an addressing attribute
    attributes = ([], [name, 'addr']) if addressing else ([name], ['addr'])
    instr = acc.add_instruction('refused', *attributes)
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[2] parameter(0);
    ROOT %Out0 = u8[2] copy(%In0);
}
""")
    with pytest.raises(NotImplementedError, match=f"refused: attribute name '{name}' is reserved"):
        generate_oracle(acc)
