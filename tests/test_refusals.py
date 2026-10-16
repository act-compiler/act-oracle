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


def test_outputs_with_no_semantics_refused():
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    with pytest.raises(ValueError, match="refused: no semantics gives '%Out0' for output region 0"):
        generate_oracle(acc)


@pytest.mark.parametrize('kind', ['input', 'output'])
def test_value_with_no_region_refused(kind):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    # The other kind's region alone
    region = [['d0', ['@a.addr'], ['2']]]
    if kind == 'input':
        instr.set_outputs(region)
    else:
        instr.set_inputs(region)
    instr.add_semantics("""
ENTRY {
    %In0 = u8[2] parameter(0);
    ROOT %Out0 = u8[2] copy(%In0);
}
""")
    value = '%In0' if kind == 'input' else '%Out0'
    with pytest.raises(ValueError, match=f"refused: '{value}' has no {kind} region"):
        generate_oracle(acc)


# Disagreeing semantics between 4 bytes, and their refusal
DISAGREE = {
    'operand shape': (
        """%In0 = u8[4] parameter(0);
    %a = u8[3] add(%In0, %In0);
    ROOT %Out0 = u8[4] reshape(%a);""",
        "add of '%a' takes operands of its shape u8[3], found '%In0' u8[4]",
    ),
    'operand type': (
        """%In0 = u8[4] parameter(0);
    %a = s8[4] convert(%In0);
    ROOT %Out0 = u8[4] maximum(%In0, %a);""",
        "maximum of '%Out0' takes operands of its type u8, found '%a' s8[4]",
    ),
    'copy type': (
        """%In0 = u8[4] parameter(0);
    %a = s8[4] copy(%In0);
    ROOT %Out0 = u8[4] convert(%a);""",
        "copy of '%a' takes operands of its type s8, found '%In0' u8[4]",
    ),
    'convert shape': (
        """%In0 = u8[4] parameter(0);
    %a = f32[2] convert(%In0);
    ROOT %Out0 = u8[4] bitcast_convert(%a);""",
        "convert of '%a' gives f32[4] of '%In0' u8[4], not f32[2]",
    ),
    'reshape count': (
        """%In0 = u8[4] parameter(0);
    %a = u8[3] reshape(%In0);
    ROOT %Out0 = u8[4] reshape(%a);""",
        "reshape of '%a' takes an operand of 3 elements, found '%In0' u8[4]",
    ),
    'bitcast shape': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u16[1] bitcast_convert(%a);
    ROOT %Out0 = u8[4] reshape(%b);""",
        "bitcast_convert of '%b' gives u16[2] of '%a' u8[2,2], not u16[1]",
    ),
    'bitcast width': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = f32[2] bitcast_convert(%a);
    ROOT %Out0 = u8[4] reshape(%b);""",
        "bitcast_convert of '%b' takes an operand whose last dimension is 4, found '%a' u8[2,2]",
    ),
    'transpose permutation': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2,2] transpose(%a), dimensions={0};
    ROOT %Out0 = u8[4] reshape(%b);""",
        "transpose of '%b' takes a permutation of the 2 dimensions of '%a' u8[2,2], found {0}",
    ),
    'transpose shape': (
        """%In0 = u8[4] parameter(0);
    %a = u8[1,4] reshape(%In0);
    %b = u8[1,4] transpose(%a), dimensions={1,0};
    ROOT %Out0 = u8[4] reshape(%b);""",
        "transpose of '%b' gives u8[4,1] of '%a' u8[1,4], not u8[1,4]",
    ),
    'broadcast dimensions': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2,2,2] broadcast(%a), dimensions={0};
    ROOT %Out0 = u8[4] reshape(%b);""",
        (
            "broadcast of '%b' takes one distinct dimension below 3 per dimension of '%a' u8[2,2], "
            'found {0}'
        ),
    ),
    'broadcast size': (
        """%In0 = u8[4] parameter(0);
    %a = u8[3,4] broadcast(%In0), dimensions={0};
    ROOT %Out0 = u8[4] reshape(%a);""",
        "broadcast of '%a' maps dimension 0 of '%In0' u8[4] to one of 3",
    ),
    'reduce dimensions': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2] reduce_add(%a), dimensions={2};
    ROOT %Out0 = u8[4] reshape(%b);""",
        "reduce_add of '%b' takes distinct dimensions below the rank 2 of '%a' u8[2,2], found {2}",
    ),
    'reduce shape': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[4] reduce_add(%a), dimensions={1};
    ROOT %Out0 = u8[4] copy(%b);""",
        "reduce_add of '%b' gives u8[2] of '%a' u8[2,2], not u8[4]",
    ),
    'dot pairs': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[1,4] reshape(%In0);
    %c = u8[2,4] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = u8[4] reshape(%c);""",
        "dot of '%c' pairs dimension 1 of '%a' u8[2,2] with dimension 0 of '%b' u8[1,4]",
    ),
    'dot lists': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2] dot(%a, %a), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = u8[4] reshape(%b);""",
        "dot of '%b' takes as many lhs as rhs batch and contracting dimensions",
    ),
    'dot dimensions': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[2] dot(%a, %a), lhs_contracting_dims={2}, rhs_contracting_dims={0};
    ROOT %Out0 = u8[4] reshape(%b);""",
        "dot of '%b' takes distinct dimensions below the rank 2 of '%a' u8[2,2], found {2}",
    ),
    'dot shape': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,2] reshape(%In0);
    %b = u8[4] dot(%a, %a), lhs_contracting_dims={1}, rhs_contracting_dims={0};
    ROOT %Out0 = u8[4] copy(%b);""",
        "dot of '%b' gives u8[2,2] of '%a' u8[2,2] and '%a' u8[2,2], not u8[4]",
    ),
    'input tile': (
        """%In0 = u8[8] parameter(0);
    ROOT %Out0 = u8[4] reshape(%In0);""",
        "'%In0' is u8[8], more elements than the 4 of its input region in d0",
    ),
    'output tile': (
        """%In0 = u8[4] parameter(0);
    %a = u8[2,4] broadcast(%In0), dimensions={1};
    ROOT %Out0 = u8[8] reshape(%a);""",
        "'%Out0' is u8[8], more elements than the 4 of its output region in d0",
    ),
}


@pytest.mark.parametrize('case', DISAGREE)
def test_disagreeing_shapes_and_attributes_refused(case):
    lines, message = DISAGREE[case]
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    instr = acc.add_instruction('refused', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], ['4']]])
    instr.set_outputs([['d0', ['@a.addr'], ['4']]])
    instr.add_semantics(f'ENTRY {{\n    {lines}\n}}')
    with pytest.raises(ValueError, match=re.escape(f'refused: {message}')):
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

# Names the generated oracle keeps, by what the spec names
RESERVED = [(kind, name) for kind in ('attribute', 'addressing') for name in SHARED_NAMES] + [
    ('instruction', 'debug'),
    ('instruction', 'state'),
    ('instruction', 'str'),
    ('instruction', 'compile_time'),
    ('instruction', '__debug__'),
    ('attribute', 'str'),
    ('attribute', '__debug__'),
    ('data model', 'loop_var'),
    ('data model', 'debug_slice'),
    ('data model', 'In0'),
    ('value', 'd0'),
    ('value', 'instruction_counter'),
]


@pytest.mark.parametrize('kind, name', RESERVED)
def test_reserved_name_refused(kind, name):
    acc = Accelerator('REFUSED', base_dir=tempfile.mkdtemp())
    if kind == 'data model':
        acc.add_data_model(name, [4], [2], 'u8')
    instruction = name if kind == 'instruction' else 'refused'
    # An attribute's name as a computation or an addressing attribute
    computation = [name] if kind == 'attribute' else []
    addressing = [name, 'addr'] if kind == 'addressing' else ['addr']
    instr = acc.add_instruction(instruction, computation, addressing)
    instr.set_inputs([['d0', ['@a.addr'], ['2']]])
    instr.set_outputs([['d0', ['@a.addr'], ['2']]])
    value = f'%{name}' if kind == 'value' else '%a'
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[2] parameter(0);
    {value} = u8[2] copy(%In0);
    ROOT %Out0 = u8[2] copy({value});
}}
""")
    shown = value if kind == 'value' else name
    # A data model owns its refusal, else the instruction
    owner = name if kind == 'data model' else instruction
    named = 'attribute' if kind == 'addressing' else kind
    message = f"^{owner}: {named} name '{shown}' is reserved by the oracle$"
    with pytest.raises(NotImplementedError, match=message):
        generate_oracle(acc)
