"""Test that each call runs its instruction's state updates in the order the spec gives them"""

import os
import tempfile

import numpy as np
import pytest
from conftest import check_oracle, import_oracle
from taidl import Accelerator

from taidl_to import generate_oracle

# Rows of eight bytes, each load written a step past the last
acc = Accelerator('STATEFUL', base_dir=tempfile.mkdtemp())
acc.add_data_model('d1', [64], [8], 'u8')
acc.add_initial_state('row', 0)
acc.add_initial_state('step', 1)
acc.add_initial_state('count', 0)
acc.add_instruction('config', ['step'], [], update=['@s.step = @c.step', '@s.row = @s.step'])
instr = acc.add_instruction(
    'load',
    ['n'],
    ['src'],
    constraints=['@s.row + @c.n <= 32'],
    update=['@s.row = @s.row + @s.step'],
)
instr.set_inputs([['d0', ['@a.src'], ['@c.n * 8']]])
instr.set_outputs([['d1', ['@s.row'], ['@c.n']]])
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
# Each call writes the count after its own update
instr = acc.add_instruction('count', [], ['dst'], update=['@s.count = @s.count + 1'])
instr.set_outputs([['d0', ['@a.dst'], ['1']]])
instr.add_semantics("""
ENTRY {
    ROOT %Out0 = u8[1] constant(`@s.count`);
}
""")
# A row read back once a load has passed it
instr = acc.add_instruction('peek', [], ['row', 'dst'], constraints=['@a.row < @s.row'])
instr.set_inputs([['d1', ['@a.row'], ['1']]])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[1,8] parameter(0);
    ROOT %Out0 = u8[8] reshape(%In0);
}
""")
generate_oracle(acc)
# Its oracle laid out as ruff lays out the tree
check_oracle(os.path.join(acc.base_dir, 'targets', 'STATEFUL', 'oracle'))
decorator, api = import_oracle(os.path.join(acc.base_dir, 'targets', 'STATEFUL'), 'oracle_STATEFUL')
kernel = decorator.kernel

INPUT = [{'addr': 0, 'shape': (256,), 'dtype': np.uint8}]
OUTPUT = [{'addr': 256, 'shape': (256,), 'dtype': np.uint8}]
X = np.arange(256, dtype=np.uint8)


def run(body):
    """The output of a kernel whose body is body, compiled and run on X"""
    k = kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)(body)
    k('fsim-compile')()
    outputs, _ = k('fsim')(X)
    return np.asarray(outputs[0]).reshape(32, 8)


def test_updates_run_in_order():
    def body():
        api.config(step=2)
        api.load(n=1, src=0)
        api.load(n=1, src=8)
        api.store(n=32, src=0, dst=256)

    # The second update reads the first, so the loads write rows 4 and 6
    expected = np.zeros((32, 8), np.uint8)
    expected[4], expected[6] = X[0:8], X[8:16]
    np.testing.assert_array_equal(run(body), expected)


def test_constraint_reads_state_before_update():
    @kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.config(step=16)
        api.load(n=16, src=0)
        api.load(n=1, src=0)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    message = "load: constraint '@s.row + @c.n <= 32' does not hold with @s.row=32, @c.n=1"
    assert str(refusal.value) == message


def test_state_writing_loop_runs_as_unrolled():
    # A loop around an update, a loop that updates nothing within it
    def looped():
        api.start_loop('i', 0, 2)
        api.load(n=1, src='16 * %i')
        api.start_loop('j', 0, 2)
        api.store(n=1, src='%i + 1', dst='256 + 8 * (2 * %i + %j)')
        api.end_loop()
        api.end_loop()

    def unrolled():
        for i in range(2):
            api.load(n=1, src=16 * i)
            for j in range(2):
                api.store(n=1, src=i + 1, dst=256 + 8 * (2 * i + j))

    expected = np.zeros((32, 8), np.uint8)
    for i in range(2):
        expected[2 * i : 2 * i + 2] = X[16 * i : 16 * i + 8]
    np.testing.assert_array_equal(run(looped), expected)
    np.testing.assert_array_equal(run(unrolled), expected)


def test_semantics_reads_state_after_its_update():
    def looped():
        api.start_loop('i', 0, 4)
        api.count(dst='256 + %i')
        api.end_loop()

    def unrolled():
        for i in range(4):
            api.count(dst=256 + i)

    expected = np.zeros((32, 8), np.uint8)
    expected[0, :4] = [1, 2, 3, 4]
    np.testing.assert_array_equal(run(looped), expected)
    np.testing.assert_array_equal(run(unrolled), expected)


def past_d1():
    api.start_loop('i', 0, 5)
    api.load(n=1, src=0)
    api.store(n=1, src='16 * %i', dst=256)
    api.end_loop()


def past_limit():
    api.config(step=8)
    api.start_loop('i', 0, 4)
    api.load(n=1, src=0)
    api.end_loop()


# A loop that updates nothing within an unrolled one
def peek_unloaded():
    api.start_loop('i', 0, 2)
    api.load(n=1, src=0)
    api.start_loop('j', 0, 3)
    api.peek(row='%j', dst=256)
    api.end_loop()
    api.end_loop()


# A refusal in an unrolled loop names its iteration, outer first
@pytest.mark.parametrize(
    'body, message',
    [
        (
            past_d1,
            'store: window [64:65] on dimension 0 of d1 runs outside its extent 64 at %i = 4',
        ),
        (
            past_limit,
            "load: constraint '@s.row + @c.n <= 32' does not hold with @s.row=32, @c.n=1 at %i = 3",
        ),
        (
            peek_unloaded,
            (
                "peek: constraint '@a.row < @s.row' does not hold with @a.row=1, @s.row=1 "
                'at %i = 0, %j = 1'
            ),
        ),
    ],
    ids=['window', 'constraint', 'nested'],
)
def test_unrolled_refusal_names_iteration(body, message):
    k = kernel(hbm=512, input=INPUT, constant=[], output=OUTPUT)(body)
    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message
