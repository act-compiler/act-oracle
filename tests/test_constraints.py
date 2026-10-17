"""Test that each call is checked against its instruction's constraints"""

import os
import tempfile

import numpy as np
import pytest
from conftest import check_oracle, import_oracle
from taidl import Accelerator

from taidl_to import generate_oracle

# Rows of eight bytes, a state limit short of its end
acc = Accelerator('CONSTRAINED', base_dir=tempfile.mkdtemp())
acc.add_data_model('d1', [128], [8], 'u8')
acc.add_initial_state('limit', 64)
instr = acc.add_instruction(
    'load',
    ['n'],
    ['src', 'dst'],
    constraints=['@c.n <= 32', '@a.dst % 8 == 0', '@a.dst + @c.n <= @s.limit'],
)
instr.set_inputs([['d0', ['@a.src'], ['@c.n * 8']]])
instr.set_outputs([['d1', ['@a.dst'], ['@c.n']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 8`] parameter(0);
    ROOT %Out0 = u8[`@c.n`,8] reshape(%In0);
}
""")
instr = acc.add_instruction('store', ['n'], ['src', 'dst'])
instr.set_inputs([['d1', ['@a.src'], ['@c.n']]])
instr.set_outputs([['d0', ['@a.dst'], ['@c.n * 8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n`,8] parameter(0);
    ROOT %Out0 = u8[`@c.n * 8`] reshape(%In0);
}
""")
# A copy within d1 under each non-overlap mark
for name, mark in [('move', 'set_apart'), ('move_whole', 'set_no_partial_overlap')]:
    instr = acc.add_instruction(name, ['n'], ['src', 'dst'])
    instr.set_inputs([['d1', ['@a.src'], ['@c.n']]])
    instr.set_outputs([['d1', ['@a.dst'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n`,8] parameter(0);
    ROOT %Out0 = u8[`@c.n`,8] copy(%In0);
}
""")
    getattr(instr, mark)([True])
# A constraint that divides by an attribute
acc.add_instruction('every', [], ['row', 'step'], constraints=['@a.row % @a.step == 0'])
generate_oracle(acc)
# Its oracle laid out as ruff lays out the tree
check_oracle(os.path.join(acc.base_dir, 'targets', 'CONSTRAINED', 'oracle'))
decorator, api = import_oracle(
    os.path.join(acc.base_dir, 'targets', 'CONSTRAINED'), 'oracle_CONSTRAINED'
)
kernel = decorator.kernel

INPUT = [{'addr': 0, 'shape': (256,), 'dtype': np.uint8}]
OUTPUT = [{'addr': 256, 'shape': (256,), 'dtype': np.uint8}]
APART = '@a.dst + @c.n <= @a.src or @a.src + @c.n <= @a.dst'


def test_constraints_kept_runs():
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 4)
        api.load(n=4, src='32 * %i', dst='8 * %i')
        api.end_loop()
        api.move(n=16, src=0, dst=16)
        api.move_whole(n=16, src=16, dst=16)
        api.store(n=32, src=0, dst=256)

    x = np.arange(256, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    # Four rows at every eighth row, then rows 0 to 15 copied to 16 to 31
    expected = np.zeros((32, 8), np.uint8)
    for i in range(4):
        expected[8 * i : 8 * i + 4] = x[32 * i : 32 * i + 32].reshape(4, 8)
    expected[16:32] = expected[0:16]
    np.testing.assert_array_equal(np.asarray(outputs[0]), expected.reshape(-1))


# A computational attribute, an alignment, a state value and each mark broken
@pytest.mark.parametrize(
    'name, values, constraint, shown',
    [
        ('load', {'n': 64, 'src': 0, 'dst': 0}, '@c.n <= 32', '@c.n=64'),
        ('load', {'n': 8, 'src': 0, 'dst': 4}, '@a.dst % 8 == 0', '@a.dst=4'),
        (
            'load',
            {'n': 32, 'src': 0, 'dst': 48},
            '@a.dst + @c.n <= @s.limit',
            '@a.dst=48, @c.n=32, @s.limit=64',
        ),
        ('move', {'n': 16, 'src': 0, 'dst': 8}, APART, '@a.dst=8, @c.n=16, @a.src=0'),
        (
            'move_whole',
            {'n': 16, 'src': 0, 'dst': 8},
            f'@a.dst == @a.src or {APART}',
            '@a.dst=8, @a.src=0, @c.n=16',
        ),
    ],
    ids=['computational', 'alignment', 'state', 'apart', 'no_partial_overlap'],
)
def test_broken_constraint_refused(name, values, constraint, shown):
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        getattr(api, name)(**values)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f"{name}: constraint '{constraint}' does not hold with {shown}"


def test_looped_broken_constraint_refused():
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 4)
        api.load(n=4, src=0, dst='12 * %i')
        api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    message = "load: constraint '@a.dst % 8 == 0' does not hold with @a.dst=12 at %i = 1"
    assert str(refusal.value) == message


# Each value bound at the first failing iteration, outer loop first
def test_nested_looped_broken_constraint_refused():
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 2)
        api.start_loop('j', 0, 3)
        api.move(n=8, src='8 * %i', dst='8 * %j + 8')
        api.end_loop()
        api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    message = f"move: constraint '{APART}' does not hold with @a.dst=8, @c.n=8, @a.src=8"
    assert str(refusal.value) == f'{message} at %i = 1, %j = 0'


def test_looped_divisor_zero_refused():
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 2)
        api.every(row=8, step='%i')
        api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    message = "every: an expression divides by zero with row=8, step='%i' at %i = 0"
    assert str(refusal.value) == message
