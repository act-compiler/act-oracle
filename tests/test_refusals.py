"""Test what generation refuses, and the oracle it leaves"""

import os
import re
import subprocess
import sys
import tempfile

import pytest
import taidl.antlr4
from conftest import EVERY, LOWERED, each_type, import_oracle
from taidl import Accelerator
from taidl.operations import BOOL, INT, INTS, LITERAL, NUMBER, ONE_INT, OPERATIONS, RANGES

from taidl_to import generate_oracle


def unmapped():
    """The words the grammar lexes that the table of lowered words leaves out."""
    with open(os.path.join(os.path.dirname(taidl.antlr4.__file__), 'IDLV2.g4')) as file:
        grammar = re.sub(r'//.*', '', file.read())
    rule = re.search(r'^OPERATION\s*:(.*?);', grammar, re.MULTILINE | re.DOTALL).group(1)
    # parameter declares an input, with no template of its own
    lowered = {'parameter', *LOWERED}
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


@pytest.mark.parametrize('word', [word for word, types in LOWERED.items() if types != EVERY])
def test_operation_at_a_type_it_does_not_take_refused(word):
    operands = ', '.join(['%a'] * OPERATIONS[word][0])

    def check(t):
        acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
        instr = acc.add_instruction('refused', [], ['addr'])
        instr.set_inputs([['d0', ['@a.addr'], ['2']]])
        instr.set_outputs([['d0', ['@a.addr'], ['2']]])
        instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[2] parameter(0);
    %a = {t}[2] convert(%In0);
    %b = {t}[2] {word}({operands});
    ROOT %Out0 = u8[2] convert(%b);
}}
""")
        with pytest.raises(NotImplementedError, match=f"refused: operation '{word}' at type '{t}'"):
            generate_oracle(acc)

    each_type([t for t in EVERY if t not in LOWERED[word]], check)


def test_state_update_refused():
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    acc.add_initial_state('count', 0)
    instr = acc.add_instruction('refused', [], ['addr'], update=['@s.count = @s.count + 1'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[2] parameter(0);
    ROOT %Out0 = u8[2] copy(%In0);
}
""")
    message = "refused: state update '@s.count = @s.count [+] 1' is not supported by the oracle"
    with pytest.raises(NotImplementedError, match=message):
        generate_oracle(acc)


# Types the grammar lexes that the oracle does not lower
UNLOWERED_TYPES = ['s2', 'u2', 's4', 'u4', 'f8e4m3fn', 'f8e5m2']


@pytest.mark.parametrize('t', UNLOWERED_TYPES)
def test_unlowered_type_in_semantics_refused(t):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[2] parameter(0);
    %a = {t}[2] convert(%In0);
    ROOT %Out0 = u8[2] convert(%a);
}}
""")
    with pytest.raises(NotImplementedError, match=f"refused: type '{t}' is not supported"):
        generate_oracle(acc)


@pytest.mark.parametrize('t', UNLOWERED_TYPES)
def test_unlowered_type_in_data_model_refused(t):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    acc.add_data_model('d1', [4], [4], t)
    with pytest.raises(NotImplementedError, match=f"d1: type '{t}' is not supported"):
        generate_oracle(acc)


def two_bytes(base_dir, name, line):
    """Accelerator REGENERATED, its one instruction ending in line"""
    acc = Accelerator('REGENERATED', base_dir=base_dir)
    instr = acc.add_instruction(name, [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[2] parameter(0);
    {line}
}}
""")
    return acc


COPY = 'ROOT %Out0 = u8[2] copy(%In0);'


def test_refused_generation_leaves_no_oracle():
    base_dir = tempfile.mkdtemp()
    generate_oracle(two_bytes(base_dir, 'first', COPY))
    op = unmapped()[0]
    with pytest.raises(NotImplementedError, match=f"refused: operation '{op}'"):
        generate_oracle(two_bytes(base_dir, 'refused', written(op)))
    # Neither the previous oracle nor a part of the new one
    assert os.listdir(os.path.join(base_dir, 'targets', 'REGENERATED')) == []


def test_generation_replaces_the_previous_oracle():
    base_dir = tempfile.mkdtemp()
    generate_oracle(two_bytes(base_dir, 'first', COPY))
    generate_oracle(two_bytes(base_dir, 'second', COPY))
    target_dir = os.path.join(base_dir, 'targets', 'REGENERATED')
    # Nothing left beside the new oracle
    assert os.listdir(target_dir) == ['oracle']
    _, replaced = import_oracle(target_dir, 'oracle_REPLACED')
    assert hasattr(replaced, 'second') and not hasattr(replaced, 'first')


def test_failed_build_leaves_no_oracle(monkeypatch):
    base_dir = tempfile.mkdtemp()
    generate_oracle(two_bytes(base_dir, 'first', COPY))
    # The handler build fails without a Python
    monkeypatch.setattr(sys, 'executable', os.path.join(base_dir, 'no-python'))
    with pytest.raises(subprocess.CalledProcessError):
        generate_oracle(two_bytes(base_dir, 'second', COPY))
    assert os.listdir(os.path.join(base_dir, 'targets', 'REGENERATED')) == []


# An input and an output typed unlike d0's u8
UNLIKE = {
    '%In0': """
    %In0 = bf16[1] parameter(0);
    %a = u8[1,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[2] reshape(%a);""",
    '%Out0': """
    %In0 = u8[2] parameter(0);
    ROOT %Out0 = s8[2] convert(%In0);""",
}


@pytest.mark.parametrize('value', UNLIKE)
def test_tile_typed_unlike_its_data_model_refused(value):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    instr.add_semantics(f'ENTRY {{{UNLIKE[value]}\n}}')
    typ = 'bf16' if value == '%In0' else 's8'
    with pytest.raises(
        ValueError, match=f"refused: '{value}' is {typ}, its data model d0 holds u8"
    ):
        generate_oracle(acc)


def test_expression_in_a_list_attribute_refused():
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', ['k'], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['4']]])
    instr.set_outputs([['d0', ['@a.addr'], ['4']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2,2] transpose(%a), dimensions={`@c.k`, 0};
    ROOT %Out0 = u8[4] reshape(%b);
}
""")
    message = "refused: an expression in 'dimensions' is not supported by the oracle"
    with pytest.raises(NotImplementedError, match=message):
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
