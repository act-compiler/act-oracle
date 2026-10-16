"""Test the kernel API of the generated oracle"""

import glob
import os
import subprocess
import sys
import tempfile
import time
import types

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from conftest import (
    DTYPES,
    EXAMPLES_DIR,
    NARROW,
    PLATFORM,
    attention_reference,
    generate_and_import_oracle,
    generate_example,
    import_oracle,
    load_bf16_matrix,
    run_kernel,
)
from taidl import Accelerator

from taidl_to import generate_oracle

MINI_TRN_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn')
kernel, api = generate_and_import_oracle(MINI_TRN_DIR)

# Q, K and V in, one output after them
INPUT = [{'addr': a, 'shape': (64, 64), 'dtype': jnp.bfloat16} for a in (0, 8192, 16384)]
OUTPUT = [{'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16}]


def load_qkv():
    return [
        load_bf16_matrix(os.path.join(MINI_TRN_DIR, 'data', f'{n}.dat'), (64, 64)) for n in 'QKV'
    ]


@pytest.mark.parametrize('mode', ['simulate', 'FSIM'])
def test_unknown_mode_refused(mode):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError, match=f"unknown oracle mode '{mode}'"):
        k(mode)


@pytest.mark.parametrize('platform', ['cpu', 'gpu', 'TPU'])
def test_unknown_platform_refused(platform):
    decorator = sys.modules[kernel.__module__]
    with pytest.raises(ValueError, match=f"unknown simulation platform '{platform}'"):
        decorator.set_simulation_backend(platform)
    # The refused name leaves the backend as it was
    assert decorator.backend.platform == PLATFORM.lower()


# A fresh process: a CPU oracle's first launch, then the GPU
CPU_THEN_GPU = """
import importlib
import sys

import jax
import jax.numpy as jnp
from jax.extend.backend import backends

sys.path.insert(0, sys.argv[1])
decorator = importlib.import_module('oracle.decorator')
api = importlib.import_module('oracle.api')
if sys.argv[2] == 'set':
    decorator.set_simulation_backend('CPU')


@decorator.kernel(
    hbm=32768,
    input=[{'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16}],
    output=[{'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16}],
)
def copy():
    api.load_rm(n=64, addr_in=0, addr_out=0)
    api.store_rm(n=64, addr_in=0, addr_out=24576)


copy('fsim-compile')()
print(jax.config.jax_platforms, sorted(backends()))
try:
    decorator.set_simulation_backend('GPU')
except ValueError as refusal:
    print(refusal)
"""


@pytest.mark.parametrize('first', ['launch', 'set'])
def test_cpu_oracle_starts_cpu_alone(first):
    target_dir = os.path.dirname(os.path.dirname(api.__file__))
    run = subprocess.run(
        [sys.executable, '-c', CPU_THEN_GPU, target_dir, first],
        capture_output=True,
        text=True,
        check=False,
    )
    # jax starts its CPU platform alone, then refuses the GPU
    assert run.stdout.splitlines() == [
        "cpu ['cpu']",
        "simulation platform 'GPU' cannot be set: jax has started only its CPU platform",
    ], run.stderr


def test_compile_runs_the_body_once():
    calls = []

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        calls.append(1)
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    assert len(calls) == 1


def test_debug_prints_values(capfd):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.debug(prefix='d1', data='d1[0]')
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    Q, K, V = load_qkv()
    k('fsim')(Q, K, V)
    # Row 0 of d1, which holds Q
    lines = capfd.readouterr().out.splitlines()
    shape = lines.index('shape: (1, 64)')
    assert lines[shape - 1] == 'd1: '
    values = lines[shape + 1].strip('[] ').split(',')
    np.testing.assert_allclose([float(v) for v in values], np.asarray(Q[0], np.float64), rtol=1e-5)


# A debug call's buffer or index refused by name
@pytest.mark.parametrize(
    'data, error, message',
    [
        (
            'd1[200]',
            ValueError,
            'window [200:201] on dimension 0 of d1 runs outside its extent 128',
        ),
        ('d1[-3]', ValueError, 'window [-3:-2] on dimension 0 of d1 runs outside its extent 128'),
        ('d9[0]', ValueError, "'d9[0]' names no buffer"),
        ('d1', ValueError, "'d1' gives no index; 'd1[:]' prints the whole buffer"),
        ('d1[]', ValueError, "'d1[]' gives no index; 'd1[:]' prints the whole buffer"),
        ('d1[0.5]', TypeError, "'d1[0.5]' has index '0.5', not an integer"),
        ('d1[0, 0, 0]', ValueError, "'d1[0, 0, 0]' gives 3 indices for the 2 dimensions of d1"),
    ],
    ids=['end', 'start', 'buffer', 'missing', 'empty', 'fraction', 'extra'],
)
def test_debug_index_refused(data, error, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.debug(prefix='d1', data=data)

    with pytest.raises(error) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'debug: {message}'


# A bare ':' prints the whole buffer, any integer literal a row
@pytest.mark.parametrize('data, shape', [('d1[:]', '(128, 64)'), ('d1[0x10]', '(1, 64)')])
def test_debug_index_forms_print(capfd, data, shape):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.debug(prefix='d1', data=data)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    k('fsim')(*load_qkv())
    assert f'shape: {shape}' in capfd.readouterr().out.splitlines()


def test_generation_rebuilds_the_debug_handler():
    start = time.time()
    build_dir = os.path.join(generate_example(MINI_TRN_DIR), 'oracle', 'build')
    handler = glob.glob(os.path.join(build_dir, 'print_handler_ext*.so'))
    # Built by this generation, not one before it
    assert handler and os.path.getmtime(handler[0]) >= start


def test_debug_before_any_instruction(capfd):
    # A fresh import, so debug is its first traced call
    decorator, fresh = import_oracle(os.path.dirname(os.path.dirname(api.__file__)), 'oracle_fresh')

    @decorator.kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        fresh.debug(prefix='d1', data='d1[0]')
        fresh.load_rm(n=64, addr_in=0, addr_out=0)
        fresh.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    k('fsim')(*load_qkv())
    # Nothing is loaded into d1 yet
    lines = capfd.readouterr().out.splitlines()
    values = lines[lines.index('d1: ') + 2].strip('[] ').split(',')
    assert [float(v) for v in values] == [0.0] * 64


def test_unended_loop_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 4)
        api.load_rm(n=16, addr_in='2048 * %i', addr_out='16 * %i')

    with pytest.raises(ValueError, match='loop %i has no end_loop'):
        k('fsim-compile')()


def test_unstarted_loop_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.end_loop()

    with pytest.raises(ValueError, match='end_loop has no start_loop'):
        k('fsim-compile')()


# Each loop bound that is not an integer
@pytest.mark.parametrize(
    'start, end, message',
    [
        (0, 4.5, "start_loop 'i': end 4.5 is not an integer"),
        (0.0, 4, "start_loop 'i': start 0.0 is not an integer"),
        (0, True, "start_loop 'i': end True is not an integer"),
    ],
    ids=['end', 'start', 'bool'],
)
def test_non_integer_loop_bound_refused(start, end, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', start, end)
        api.load_rm(n=16, addr_in='2048 * %i', addr_out='16 * %i')
        api.end_loop()

    with pytest.raises(TypeError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


def test_loop_variable_in_use_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 2)
        api.start_loop('i', 0, 2)
        api.load_rm(n=8, addr_in='1024 * %i', addr_out='8 * %i')
        api.end_loop()
        api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == "start_loop 'i': loop variable %i is already in use"


# Each loop variable that is not a name or is one the module keeps
@pytest.mark.parametrize(
    'var, error, message',
    [
        (1, TypeError, 'start_loop 1: loop variable 1 is not a string'),
        ('i j', ValueError, "start_loop 'i j': loop variable 'i j' is not a name"),
        (
            'arg0',
            ValueError,
            "start_loop 'arg0': %arg0 is a name the module keeps for its parameters",
        ),
    ],
    ids=['number', 'space', 'parameter'],
)
def test_loop_variable_name_refused(var, error, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop(var, 0, 2)
        api.load_rm(n=8, addr_in=0, addr_out=0)
        api.end_loop()

    with pytest.raises(error) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


# NumPy integer bounds run as the ints they hold
def test_numpy_integer_loop_bounds_run():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', np.int64(0), np.int32(4))
        api.load_rm(n=16, addr_in='2048 * %i', addr_out='16 * %i')
        api.end_loop()
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    Q, K, V = load_qkv()
    k('fsim-compile')()
    outputs, _ = k('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


def test_loop_expression_as_computation_attribute_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 4)
        api.load_rm(n='%i', addr_in=0, addr_out=0)
        api.end_loop()

    with pytest.raises(TypeError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == (
        "load_rm: computational attribute n takes an integer, not the loop expression '%i'"
    )


# Each value a call may pass that is not an integer
@pytest.mark.parametrize(
    'values, message',
    [
        ({'addr_in': 0.5}, 'load_rm: addr_in 0.5 is not an integer'),
        ({'addr_in': 8192.0}, 'load_rm: addr_in 8192.0 is not an integer'),
        ({'addr_in': True}, 'load_rm: addr_in True is not an integer'),
        ({'addr_in': np.bool_(True)}, 'load_rm: addr_in np.True_ is not an integer'),
        ({'addr_out': [0]}, 'load_rm: addr_out [0] is not an integer'),
        ({'n': 64.0}, 'load_rm: n 64.0 is not an integer'),
    ],
    ids=['fraction', 'float', 'bool', 'numpy_bool', 'list', 'computation'],
)
def test_non_integer_value_refused(values, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(**({'n': 64, 'addr_in': 0, 'addr_out': 0} | values))
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(TypeError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


# NumPy integers run as the ints they hold
def test_numpy_integer_values_run():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=np.int64(64), addr_in=np.int32(8192), addr_out=np.uint8(0))
        api.store_rm(n=np.int64(64), addr_in=np.int64(0), addr_out=np.int64(24576))

    Q, K, V = load_qkv()
    k('fsim-compile')()
    outputs, _ = k('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(K).tobytes()


def run_printing(kern, a, prefix, inputs):
    """Compile and run on inputs a kernel of oracle a that prints after prefix"""

    @kern(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        a.load_rm(n=64, addr_in=0, addr_out=0)
        a.debug(prefix=prefix, data='d1[0]')
        a.store_rm(n=64, addr_in=0, addr_out=24576)

    k('fsim-compile')()
    k('fsim')(*inputs)


def test_two_oracles_print(capfd):
    other_decorator, other_api = import_oracle(generate_example(MINI_TRN_DIR), 'oracle_other')
    # Each oracle registers its own debug handler
    inputs = load_qkv()
    for kern, a, prefix in ((kernel, api, 'first'), (other_decorator.kernel, other_api, 'second')):
        run_printing(kern, a, prefix, inputs)
    lines = capfd.readouterr().out.splitlines()
    assert 'first: ' in lines and 'second: ' in lines


def looped():
    """The attention kernel with its Q and V loads and its store in loops"""

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 4)
        api.load_rm(n=16, addr_in='2048 * %i', addr_out='16 * %i')
        api.end_loop()
        api.load_cm(n=64, addr_in=8192, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.softmax(n=64, addr=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop('i', 0, 4)
        api.load_rm(n=16, addr_in='16384 + 2048 * %i', addr_out='64 + 16 * %i')
        api.end_loop()
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop('j', 0, 8)
        api.store_rm(n=8, addr_in='8 * %j', addr_out='24576 + 1024 * %j')
        api.end_loop()

    return k


# Each loop expression the oracle cannot read
@pytest.mark.parametrize(
    'expression, message',
    [
        ('(%i + 1 * 1024', "load_rm: addr_in '(%i + 1 * 1024' is not a loop expression"),
        ('0.5 * %i * 4096', "load_rm: addr_in '0.5 * %i * 4096' is not a loop expression"),
        ('abc', "load_rm: addr_in 'abc' is not a loop expression"),
        ('', "load_rm: addr_in '' is not a loop expression"),
        ('1024 * %k', "load_rm: addr_in '1024 * %k' reads %k, which no enclosing loop holds"),
    ],
    ids=['bracket', 'fraction', 'name', 'empty', 'variable'],
)
def test_unreadable_loop_expression_refused(expression, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 2)
        api.load_rm(n=8, addr_in=expression, addr_out='8 * %i')
        api.end_loop()
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


# A divisor zero as written or at an iteration, that one named
@pytest.mark.parametrize(
    'expression, iteration', [('%i / 0', ''), ('%i % (2 - 2)', ''), ('4096 / %i', ' at %i = 0')]
)
def test_loop_expression_division_by_zero_refused(expression, iteration):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.start_loop('i', 0, 2)
        api.load_rm(n=8, addr_in=expression, addr_out='8 * %i')
        api.end_loop()
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    shown = f"n=8, addr_in={expression!r}, addr_out='8 * %i'"
    assert str(refusal.value) == f'load_rm: an expression divides by zero with {shown}{iteration}'


def nested():
    """The attention kernel with its Q and V loads in two-deep loops"""

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        # Q's halves in two inner loops of eight rows
        api.start_loop('io', 0, 2)
        api.start_loop('ii', 0, 2)
        api.load_rm(n=8, addr_in='4096 * %io + 1024 * %ii', addr_out='32 * %io + 8 * %ii')
        api.end_loop()
        api.start_loop('jj', 0, 2)
        api.load_rm(
            n=8, addr_in='4096 * %io + 2048 + 1024 * %jj', addr_out='32 * %io + 16 + 8 * %jj'
        )
        api.end_loop()
        api.end_loop()
        api.load_cm(n=64, addr_in=8192, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.softmax(n=64, addr=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.start_loop('io', 0, 2)
        api.start_loop('ii', 0, 2)
        api.load_rm(
            n=16, addr_in='16384 + 4096 * %io + 2048 * %ii', addr_out='64 + 32 * %io + 16 * %ii'
        )
        api.end_loop()
        api.end_loop()
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    return k


def q_looped(load_q):
    """The attention kernel with its Q load in load_q's loops"""

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        load_q()
        api.load_cm(n=64, addr_in=8192, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.softmax(n=64, addr=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.load_rm(n=64, addr_in=16384, addr_out=64)
        api.gemm(addr_1=0, addr_2=64, addr_out=0)
        api.mov(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    return k


def bracketed():
    """Q's quarters at bracketed addresses"""

    def load_q():
        api.start_loop('i', 0, 4)
        api.load_rm(n=16, addr_in='2048 * (%i + 1) - 2048', addr_out='16 * (3 - (3 - %i))')
        api.end_loop()

    return q_looped(load_q)


def floored():
    """Q's quarters at floor quotients of negative numerators"""

    def load_q():
        api.start_loop('i', 0, 4)
        api.load_rm(n=16, addr_in='(2 * %i - 7) / 2 * 2048 + 8192', addr_out='16 * %i')
        api.end_loop()

    return q_looped(load_q)


def remaindered():
    """Q's eighths at quotients and remainders of the loop variable"""

    def load_q():
        api.start_loop('i', 0, 8)
        api.load_rm(
            n=8, addr_in='%i / 2 * 2048 + (%i - 8) % 2 * 1024', addr_out='%i / 2 * 16 + %i % 2 * 8'
        )
        api.end_loop()

    return q_looped(load_q)


def nested_divided():
    """Q's eighths at expressions over both of two loops' variables"""

    def load_q():
        api.start_loop('j', 0, 2)
        api.start_loop('i', 0, 4)
        api.load_rm(
            n=8,
            addr_in='(%i + 4 * %j) * (%j + 1) / (%j + 1) * 1024',
            addr_out='(%i + 4 * %j + 8) % 8 * 8',
        )
        api.end_loop()
        api.end_loop()

    return q_looped(load_q)


@pytest.mark.parametrize('twin', [looped, nested, bracketed, floored, remaindered, nested_divided])
def test_loop_matches_unrolled(twin):
    k = twin()
    k('fsim-compile')()
    outputs, _ = k('fsim')(*load_qkv())
    if PLATFORM == 'GPU':
        # On the GPU, the twin is held to the reference
        reference, tol = attention_reference(os.path.join(MINI_TRN_DIR, 'data'))
        assert np.max(np.abs(np.asarray(outputs[0], np.float64) - reference)) <= tol
        return
    unrolled = run_kernel(
        kernel,
        api,
        'attention',
        os.path.join(MINI_TRN_DIR, 'assembly'),
        os.path.join(MINI_TRN_DIR, 'data'),
    )
    assert np.asarray(outputs[0]).tobytes() == np.asarray(unrolled).tobytes()


# A window past either end of d1 or of HBM
@pytest.mark.parametrize(
    'name, values, message',
    [
        ('load_rm', {'addr_in': 0, 'addr_out': 100}, '[100:164] on dimension 0 of d1'),
        ('load_rm', {'addr_in': 0, 'addr_out': -8}, '[-8:56] on dimension 0 of d1'),
        ('load_rm', {'addr_in': 28672, 'addr_out': 0}, '[28672:36864] on dimension 0 of d0'),
        ('store_rm', {'addr_in': 0, 'addr_out': 28672}, '[28672:36864] on dimension 0 of d0'),
        ('store_rm', {'addr_in': 0, 'addr_out': -8}, '[-8:8184] on dimension 0 of d0'),
    ],
    ids=['end', 'start', 'hbm_read', 'hbm_write', 'hbm_start'],
)
def test_window_outside_buffer_refused(name, values, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        getattr(api, name)(n=64, **values)

    extent = 128 if message.endswith('d1') else 32768
    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'{name}: window {message} runs outside its extent {extent}'


def past_last():
    """Rows past d1 at the last iteration alone"""
    api.start_loop('i', 0, 3)
    api.load_rm(n=32, addr_in=0, addr_out='50 * %i')
    api.end_loop()


def past_nested():
    """Rows past d1 at one iteration of an inner loop"""
    api.start_loop('i', 0, 2)
    api.start_loop('j', 0, 4)
    api.load_rm(n=16, addr_in=0, addr_out='96 * %i + 16 * %j')
    api.end_loop()
    api.end_loop()


# A looped window past d1, its iteration named
@pytest.mark.parametrize(
    'body, message',
    [
        (past_last, '[100:132] on dimension 0 of d1 runs outside its extent 128 at %i = 2'),
        (
            past_nested,
            '[128:144] on dimension 0 of d1 runs outside its extent 128 at %i = 1, %j = 2',
        ),
    ],
    ids=['last', 'nested'],
)
def test_looped_window_outside_buffer_refused(body, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        body()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'load_rm: window {message}'


# A size the call makes zero or negative, before any window
@pytest.mark.parametrize('n, size', [(0, 0), (-1, -128)], ids=['zero', 'negative'])
def test_size_not_positive_refused(n, size):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=n, addr_in=0, addr_out=0)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'load_rm: size `@c.n * 128` is {size}, not positive'


# Calls in a loop with no iterations, never checked
def test_loop_of_no_iteration_traces_nothing():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.start_loop('i', 0, 0)
        api.load_rm(n=64, addr_in=8192, addr_out=100)
        api.load_rm(n=64, addr_in=8192, addr_out='100 + %i')
        api.debug(prefix='d1', data='d1[200]')
        api.end_loop()
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    Q, K, V = load_qkv()
    k('fsim-compile')()
    outputs, _ = k('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


# Two 8192-byte inputs that overlap or run past HBM
@pytest.mark.parametrize('second, error', [(4096, 'overlap'), (12288, 'runs outside HBM')])
def test_input_layout_refused(second, error):
    @kernel(
        hbm=16384,
        input=[INPUT[0], dict(INPUT[1], addr=second)],
        constant=[],
        output=[dict(OUTPUT[0], addr=0)],
    )
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)

    with pytest.raises(ValueError, match=error):
        k('fsim-compile')()


# Each kind of piece, past HBM or before it
@pytest.mark.parametrize(
    'pieces, message',
    [
        (
            {'output': [dict(OUTPUT[0], addr=28672)]},
            "output 0 of kernel 'copy_q', 8192 bytes at byte 28672, runs outside HBM of 32768 bytes",
        ),
        (
            {'output': [dict(OUTPUT[0], addr=-8)]},
            "output 0 of kernel 'copy_q', 8192 bytes at byte -8, runs outside HBM of 32768 bytes",
        ),
        (
            {'input': [*INPUT[:2], dict(INPUT[2], addr=28672)]},
            "input 2 of kernel 'copy_q', 8192 bytes at byte 28672, runs outside HBM of 32768 bytes",
        ),
        (
            {'input': [dict(INPUT[0], addr=-8), *INPUT[1:]]},
            "input 0 of kernel 'copy_q', 8192 bytes at byte -8, runs outside HBM of 32768 bytes",
        ),
        (
            {'constant': [dict(INPUT[0], addr=28672, value=np.zeros((64, 64), jnp.bfloat16))]},
            "constant 0 of kernel 'copy_q', 8192 bytes at byte 28672, runs outside HBM of 32768 bytes",
        ),
    ],
    ids=['output', 'output-before', 'input', 'input-before', 'constant'],
)
def test_piece_outside_hbm_refused(pieces, message):
    @kernel(hbm=32768, **({'input': INPUT, 'constant': [], 'output': OUTPUT} | pieces))
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError) as refusal:
        copy_q('fsim-compile')()
    assert str(refusal.value) == message


# Each pair of laid pieces, either one first
@pytest.mark.parametrize(
    'pieces, message',
    [
        (
            {'input': [INPUT[0], dict(INPUT[1], addr=4096), INPUT[2]]},
            "input 1 of kernel 'copy_q', 8192 bytes at byte 4096, overlaps input 0, 8192 bytes at byte 0",
        ),
        (
            {'input': [dict(INPUT[0], addr=4096), dict(INPUT[1], addr=0), INPUT[2]]},
            "input 0 of kernel 'copy_q', 8192 bytes at byte 4096, overlaps input 1, 8192 bytes at byte 0",
        ),
        (
            {'constant': [dict(INPUT[0], addr=4096, value=np.zeros((64, 64), jnp.bfloat16))]},
            "constant 0 of kernel 'copy_q', 8192 bytes at byte 4096, overlaps input 0, 8192 bytes at byte 0",
        ),
        (
            {
                'input': [*INPUT[:2], dict(INPUT[2], addr=20480)],
                'constant': [dict(INPUT[0], addr=16384, value=np.zeros((64, 64), jnp.bfloat16))],
            },
            "input 2 of kernel 'copy_q', 8192 bytes at byte 20480, overlaps constant 0, 8192 bytes at byte 16384",
        ),
    ],
    ids=['input', 'input-first', 'constant', 'constant-first'],
)
def test_pieces_overlap_refused(pieces, message):
    @kernel(hbm=32768, **({'input': INPUT, 'constant': [], 'output': OUTPUT} | pieces))
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError) as refusal:
        copy_q('fsim-compile')()
    assert str(refusal.value) == message


# An output may lie over an input
def test_output_over_input_runs():
    @kernel(hbm=32768, input=INPUT, constant=[], output=[dict(OUTPUT[0], addr=0)])
    def k():
        api.load_rm(n=64, addr_in=8192, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=0)

    k('fsim-compile')()
    Q, K, V = load_qkv()
    outputs, _ = k('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(K).tobytes()


def test_constant_reaches_hbm():
    V = load_qkv()[2]

    @kernel(
        hbm=16384, input=[], constant=[dict(INPUT[0], value=V)], output=[dict(OUTPUT[0], addr=8192)]
    )
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=8192)

    k('fsim-compile')()
    outputs, _ = k('fsim')()
    assert np.asarray(outputs[0]).tobytes() == np.asarray(V).tobytes()


# A constant whose value is unlike its declaration
@pytest.mark.parametrize(
    'value, message',
    [
        (
            np.zeros((32, 64), jnp.bfloat16),
            "constant 0 of kernel 'k' has shape (32, 64), declared (64, 64)",
        ),
        (
            np.zeros((64, 64), np.float32),
            "constant 0 of kernel 'k' has dtype float32, declared bfloat16",
        ),
        (
            np.zeros((64, 64), jnp.bfloat16).tolist(),
            "constant 0 of kernel 'k' has dtype float64, declared bfloat16",
        ),
    ],
    ids=['shape', 'dtype', 'list'],
)
def test_unlike_constant_refused(value, message):
    @kernel(
        hbm=16384,
        input=[],
        constant=[dict(INPUT[0], value=value)],
        output=[dict(OUTPUT[0], addr=8192)],
    )
    def k():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=8192)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')()
    assert str(refusal.value) == message


# Each non-integer a declaration may take
@pytest.mark.parametrize(
    'pieces, message',
    [
        ({'hbm': 32768.0}, "kernel 'copy_q': hbm 32768.0 is not an integer"),
        (
            {'input': [INPUT[0], dict(INPUT[1], addr=8192.0), INPUT[2]]},
            "input 1 of kernel 'copy_q': addr 8192.0 is not an integer",
        ),
        (
            {'input': [dict(INPUT[0], addr=True), *INPUT[1:]]},
            "input 0 of kernel 'copy_q': addr True is not an integer",
        ),
        (
            {'input': [*INPUT[:2], dict(INPUT[2], addr=np.bool_(True))]},
            "input 2 of kernel 'copy_q': addr np.True_ is not an integer",
        ),
        (
            {'input': [dict(INPUT[0], shape=(64.0, 64)), *INPUT[1:]]},
            "input 0 of kernel 'copy_q': shape (64.0, 64) has dimension 64.0, not an integer",
        ),
        (
            {'output': [dict(OUTPUT[0], addr=8192.5)]},
            "output 0 of kernel 'copy_q': addr 8192.5 is not an integer",
        ),
        (
            {'constant': [dict(INPUT[0], addr=0.7, value=np.zeros((64, 64), jnp.bfloat16))]},
            "constant 0 of kernel 'copy_q': addr 0.7 is not an integer",
        ),
        (
            {
                'constant': [
                    dict(INPUT[0], shape=(64, 64.0), value=np.zeros((64, 64), jnp.bfloat16))
                ]
            },
            "constant 0 of kernel 'copy_q': shape (64, 64.0) has dimension 64.0, not an integer",
        ),
    ],
    ids=['hbm', 'addr', 'bool', 'numpy-bool', 'shape', 'output', 'constant', 'constant-shape'],
)
def test_non_integer_declaration_refused(pieces, message):
    @kernel(**({'hbm': 32768, 'input': INPUT, 'constant': [], 'output': OUTPUT} | pieces))
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(TypeError) as refusal:
        copy_q('fsim-compile')()
    assert str(refusal.value) == message


def test_numpy_integer_declaration_runs():
    # NumPy integers as the HBM size, an addr and a dimension
    numpy_input = [*INPUT[:2], dict(INPUT[2], addr=np.int64(16384), shape=(np.int32(64), 64))]

    @kernel(hbm=np.int64(32768), input=numpy_input, constant=[], output=OUTPUT)
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    copy_q('fsim-compile')()
    Q, K, V = load_qkv()
    outputs, _ = copy_q('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


# Each dtype a declaration may take that the oracle lacks
@pytest.mark.parametrize(
    'pieces, message',
    [
        (
            {'input': [dict(INPUT[0], shape=(16, 64), dtype=np.complex64), *INPUT[1:]]},
            "input 0 of kernel 'copy_q': dtype complex64 is not supported by the oracle",
        ),
        (
            {'input': [*INPUT[:2], dict(INPUT[2], dtype='bf16')]},
            "input 2 of kernel 'copy_q': dtype 'bf16' is not supported by the oracle",
        ),
        (
            {'output': [dict(OUTPUT[0], dtype=jnp.float8_e4m3b11fnuz)]},
            "output 0 of kernel 'copy_q': dtype float8_e4m3b11fnuz is not supported by the oracle",
        ),
        (
            {
                'input': INPUT[1:],
                'constant': [
                    dict(
                        INPUT[0],
                        shape=(16, 64),
                        dtype=np.complex64,
                        value=np.zeros((16, 64), np.complex64),
                    )
                ],
            },
            "constant 0 of kernel 'copy_q': dtype complex64 is not supported by the oracle",
        ),
    ],
    ids=['input', 'string', 'output', 'constant'],
)
def test_unsupported_dtype_refused(pieces, message):
    @kernel(**({'hbm': 32768, 'input': INPUT, 'constant': [], 'output': OUTPUT} | pieces))
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(TypeError) as refusal:
        copy_q('fsim-compile')()
    assert str(refusal.value) == message


# A dtype as a string or a NumPy dtype runs
@pytest.mark.parametrize('dtype', ['bfloat16', np.dtype(jnp.bfloat16)], ids=['string', 'numpy'])
def test_dtype_forms_run(dtype):
    @kernel(
        hbm=32768,
        input=[dict(spec, dtype=dtype) for spec in INPUT],
        constant=[],
        output=[dict(OUTPUT[0], dtype=dtype)],
    )
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    copy_q('fsim-compile')()
    Q, K, V = load_qkv()
    outputs, _ = copy_q('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


def test_kernel_keeps_its_executable():
    def copy(addr):
        """A kernel that copies the input at addr to the output"""

        @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
        def k():
            api.load_rm(n=64, addr_in=addr, addr_out=0)
            api.store_rm(n=64, addr_in=0, addr_out=24576)

        return k

    copy_q, copy_k = copy(0), copy(8192)
    copy_q('fsim-compile')()
    copy_k('fsim-compile')()
    # Each runs its own executable, not the last one compiled
    Q, K, V = load_qkv()
    for k, expected in ((copy_q, Q), (copy_k, K)):
        outputs, _ = k('fsim')(Q, K, V)
        assert np.asarray(outputs[0]).tobytes() == np.asarray(expected).tobytes()


def test_kernel_keeps_its_compile_time(monkeypatch):
    decorator = sys.modules[kernel.__module__]
    # Each compile reads the clock twice
    clock = iter([0.0, 0.002, 1.0, 1.5])
    monkeypatch.setattr(decorator, 'time', types.SimpleNamespace(time=lambda: next(clock)))

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def first():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def second():
        api.load_rm(n=64, addr_in=8192, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    compiled_first = first('fsim-compile')
    compiled_second = second('fsim-compile')
    # Each reports its own time, not the last one compiled
    assert compiled_first() == (INPUT, 2.0)
    assert compiled_second() == (INPUT, 500.0)


def test_run_before_compile_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    Q, K, V = load_qkv()
    with pytest.raises(
        ValueError, match="kernel 'copy_q' is not compiled; run it in 'fsim-compile' first"
    ):
        copy_q('fsim')(Q, K, V)
    # Once compiled, the same kernel runs
    copy_q('fsim-compile')()
    outputs, _ = copy_q('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


# Q, K and V as plain NumPy arrays
QKV = [np.full((64, 64), v, jnp.bfloat16) for v in (1, 2, 3)]


@pytest.mark.parametrize(
    'args, error, message',
    [
        (QKV[:2], ValueError, "kernel 'copy_q' declares 3 inputs, got 2 arguments"),
        (QKV + QKV[:1], ValueError, "kernel 'copy_q' declares 3 inputs, got 4 arguments"),
        (
            [QKV[0].tolist(), *QKV[1:]],
            TypeError,
            "argument 0 of kernel 'copy_q' is a list, not an array",
        ),
        (
            [QKV[0], QKV[1][:, :32], QKV[2]],
            ValueError,
            "argument 1 of kernel 'copy_q' has shape (64, 32), declared (64, 64)",
        ),
        (
            [*QKV[:2], QKV[2].astype(np.float32)],
            ValueError,
            "argument 2 of kernel 'copy_q' has dtype float32, declared bfloat16",
        ),
    ],
    ids=['two', 'four', 'list', 'shape', 'dtype'],
)
def test_unlike_arguments_refused(args, error, message):
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    copy_q('fsim-compile')()
    with pytest.raises(error) as refusal:
        copy_q('fsim')(*args)
    assert str(refusal.value) == message
    # Arguments like the declared inputs run
    outputs, _ = copy_q('fsim')(*QKV)
    assert np.asarray(outputs[0]).tobytes() == QKV[0].tobytes()


def test_list_shape_runs():
    # A shape declared as a list
    listed = [{**spec, 'shape': list(spec['shape'])} for spec in INPUT]

    @kernel(hbm=32768, input=listed, constant=[], output=OUTPUT)
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    copy_q('fsim-compile')()
    outputs, _ = copy_q('fsim')(*QKV)
    assert np.asarray(outputs[0]).tobytes() == QKV[0].tobytes()


def test_compile_arguments_refused():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def copy_q():
        api.load_rm(n=64, addr_in=0, addr_out=0)
        api.store_rm(n=64, addr_in=0, addr_out=24576)

    with pytest.raises(ValueError) as refusal:
        copy_q('fsim-compile')(QKV[0], 3)
    assert str(refusal.value) == "kernel 'copy_q' takes no arguments in 'fsim-compile'"


# Each kind of API call, outside a kernel body
OUTSIDE = {
    'load_rm': lambda a: a.load_rm(n=64, addr_in=0, addr_out=0),
    'debug': lambda a: a.debug(prefix='d1', data='d1[0]'),
    'start_loop': lambda a: a.start_loop('i', 0, 4),
    'end_loop': lambda a: a.end_loop(),
}


# Before any compile, then after a compile, a run or a refused one
@pytest.mark.parametrize('when', ['fresh', 'compiled', 'run', 'refused'])
def test_call_outside_kernel_refused(when):
    # A fresh import, so no kernel is compiled yet
    decorator, fresh = import_oracle(
        os.path.dirname(os.path.dirname(api.__file__)), f'oracle_outside_{when}'
    )

    @decorator.kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def copy_q():
        fresh.load_rm(n=64, addr_in=0, addr_out=0)
        fresh.store_rm(n=64, addr_in=0, addr_out=24576)

    @decorator.kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def unstarted():
        fresh.end_loop()

    Q, K, V = load_qkv()
    if when in ('compiled', 'run'):
        copy_q('fsim-compile')()
    if when == 'run':
        copy_q('fsim')(Q, K, V)
    if when == 'refused':
        with pytest.raises(ValueError, match='end_loop has no start_loop'):
            unstarted('fsim-compile')()
    for name, call in OUTSIDE.items():
        with pytest.raises(ValueError) as refusal:
            call(fresh)
        assert str(refusal.value) == f"'{name}' must be called inside a kernel"
    # Inside a kernel each call records as before
    copy_q('fsim-compile')()
    outputs, _ = copy_q('fsim')(Q, K, V)
    assert np.asarray(outputs[0]).tobytes() == np.asarray(Q).tobytes()


# Elements per byte of each sub-byte type
PACK = {'s2': 4, 'u2': 4, 's4': 2, 'u4': 2}

# A buffer of each narrow type, 32 elements, loaded from and stored to its bytes
narrow = Accelerator('NARROW', base_dir=tempfile.mkdtemp())
for k, t in enumerate(NARROW):
    pack = PACK.get(t, 1)
    nbytes, packed = 32 // pack, f'{t}[{32 // pack}{f",{pack}" if pack > 1 else ""}]'
    narrow.add_data_model(f'd{k + 1}', [4], [8], t)
    instr = narrow.add_instruction(f'load_{t}', [], ['addr'])
    instr.set_inputs([['d0', ['@a.addr'], [str(nbytes)]]])
    instr.set_outputs([[f'd{k + 1}', ['0'], ['4']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[{nbytes}] parameter(0);
    %a = {packed} bitcast_convert(%In0);
    ROOT %Out0 = {t}[4,8] reshape(%a);
}}
""")
    instr = narrow.add_instruction(f'store_{t}', [], ['addr'])
    instr.set_inputs([[f'd{k + 1}', ['0'], ['4']]])
    instr.set_outputs([['d0', ['@a.addr'], [str(nbytes)]]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = {t}[4,8] parameter(0);
    %a = {packed} reshape(%In0);
    ROOT %Out0 = u8[{nbytes}] bitcast_convert(%a);
}}
""")
generate_oracle(narrow)
narrow_kernel, narrow_api = import_oracle(
    os.path.join(narrow.base_dir, 'targets', 'NARROW'), 'oracle_NARROW'
)


@pytest.mark.parametrize('t', NARROW)
def test_narrow_values_through_hbm(t):
    dtype, rng = DTYPES[t], np.random.default_rng(len(t))
    # Small integers each type holds exactly
    low, high = (-8, 8) if t[0] == 'f' else (ml_dtypes.iinfo(dtype).min, ml_dtypes.iinfo(dtype).max)
    x, c = (rng.integers(low, high + 1, (4, 8)).astype(dtype) for _ in range(2))
    nbytes = 32 // PACK.get(t, 1)
    spec = {'shape': (4, 8), 'dtype': dtype}
    load, store = getattr(narrow_api, f'load_{t}'), getattr(narrow_api, f'store_{t}')

    # The input and the constant, each through its buffer to an output
    @narrow_kernel.kernel(
        hbm=4 * nbytes,
        input=[{'addr': 0, **spec}],
        constant=[{'addr': nbytes, 'value': c, **spec}],
        output=[{'addr': 2 * nbytes, **spec}, {'addr': 3 * nbytes, **spec}],
    )
    def k():
        load(addr=0)
        store(addr=2 * nbytes)
        load(addr=nbytes)
        store(addr=3 * nbytes)

    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    for out, expected in zip(outputs, (x, c)):
        assert np.asarray(out).dtype == dtype
        np.testing.assert_array_equal(
            np.asarray(out).astype(np.float64), expected.astype(np.float64)
        )


@pytest.mark.parametrize(('t', 'shape'), [('s2', (6,)), ('u4', (3, 3))])
def test_narrow_value_of_part_bytes_refused(t, shape):
    spec = {'addr': 0, 'shape': shape, 'dtype': DTYPES[t]}

    @narrow_kernel.kernel(hbm=64, input=[spec], constant=[], output=[])
    def k():
        pass

    count, name = int(np.prod(shape)), np.dtype(DTYPES[t]).name
    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == (
        f"input 0 of kernel 'k': {count} elements of {name} do not fill whole bytes"
    )


# Loads whose start divides by a computational attribute
divide = Accelerator('DIVIDE', base_dir=tempfile.mkdtemp())
divide.add_data_model('d1', [1], [8], 'u8')
for name, operator in (('load_quotient', '/'), ('load_remainder', '%')):
    instr = divide.add_instruction(name, ['n'], ['src'])
    instr.set_inputs([['d0', [f'@a.src {operator} @c.n'], ['8']]])
    instr.set_outputs([['d1', ['0'], ['1']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[8] parameter(0);
    ROOT %Out0 = u8[1,8] reshape(%In0);
}
""")
# An instruction with no region and no semantics
divide.add_instruction('fence', [], [])
generate_oracle(divide)
divide_kernel, divide_api = import_oracle(
    os.path.join(divide.base_dir, 'targets', 'DIVIDE'), 'oracle_DIVIDE'
)


@pytest.mark.parametrize('name', ['load_quotient', 'load_remainder'])
def test_division_by_zero_refused(name):
    @divide_kernel.kernel(
        hbm=16, input=[{'addr': 0, 'shape': (16,), 'dtype': np.uint8}], constant=[], output=[]
    )
    def k():
        getattr(divide_api, name)(n=0, src=8)

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'{name}: an expression divides by zero with n=0, src=8'


def test_instruction_without_regions_runs():
    hbm = [{'addr': 0, 'shape': (16,), 'dtype': np.uint8}]

    @divide_kernel.kernel(hbm=16, input=hbm, constant=[], output=hbm)
    def k():
        divide_api.fence()

    x = np.arange(16, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x)


# Starts computed over an attribute, in HBM and in a buffer
shift = Accelerator('SHIFT', base_dir=tempfile.mkdtemp())
shift.add_data_model('d1', [8], [8], 'u8')
instr = shift.add_instruction('load_past', [], ['src', 'row'])
instr.set_inputs([['d0', ['@a.src + 8'], ['8']]])
instr.set_outputs([['d1', ['@a.row * 2'], ['1']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[8] parameter(0);
    ROOT %Out0 = u8[1,8] reshape(%In0);
}
""")
instr = shift.add_instruction('touch_rows', [], ['src', 'rows'])
instr.set_inputs([['d0', ['@a.src'], ['@a.rows - 8']]])
instr = shift.add_instruction('store_row', [], ['row', 'dst'])
instr.set_inputs([['d1', ['@a.row'], ['1']]])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[1,8] parameter(0);
    ROOT %Out0 = u8[8] reshape(%In0);
}
""")
generate_oracle(shift)
shift_kernel, shift_api = import_oracle(
    os.path.join(shift.base_dir, 'targets', 'SHIFT'), 'oracle_SHIFT'
)
SHIFT_INPUT = [{'addr': 0, 'shape': (40,), 'dtype': np.uint8}]
SHIFT_OUTPUT = [{'addr': 40, 'shape': (32,), 'dtype': np.uint8}]


# HBM's bytes 8 to 39, through every other row of d1
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
def test_start_over_looped_attribute_runs(looped):
    @shift_kernel.kernel(hbm=72, input=SHIFT_INPUT, constant=[], output=SHIFT_OUTPUT)
    def k():
        if looped:
            shift_api.start_loop('i', 0, 4)
            shift_api.load_past(src='8 * %i', row='%i')
            shift_api.end_loop()
            shift_api.start_loop('i', 0, 4)
            shift_api.store_row(row='2 * %i', dst='40 + 8 * %i')
            shift_api.end_loop()
        else:
            for i in range(4):
                shift_api.load_past(src=8 * i, row=i)
            for i in range(4):
                shift_api.store_row(row=2 * i, dst=40 + 8 * i)

    x = np.arange(1, 41, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x[8:40])


# A start over a looped attribute past HBM or past d1
@pytest.mark.parametrize(
    'src, message',
    [
        ('32 * %i', '[72:80] on dimension 0 of d0 runs outside its extent 72 at %i = 2'),
        ('8 * %i', '[8:9] on dimension 0 of d1 runs outside its extent 8 at %i = 4'),
    ],
    ids=['hbm', 'buffer'],
)
def test_start_over_looped_attribute_outside_buffer_refused(src, message):
    @shift_kernel.kernel(hbm=72, input=SHIFT_INPUT, constant=[], output=SHIFT_OUTPUT)
    def k():
        shift_api.start_loop('i', 0, 5)
        shift_api.load_past(src=src, row='%i')
        shift_api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == f'load_past: window {message}'


# A size over a looped attribute, zero at one iteration
def test_size_over_looped_attribute_not_positive_refused():
    @shift_kernel.kernel(hbm=72, input=SHIFT_INPUT, constant=[], output=SHIFT_OUTPUT)
    def k():
        shift_api.start_loop('i', 0, 2)
        shift_api.touch_rows(src=0, rows='9 - %i')
        shift_api.end_loop()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == 'touch_rows: size `@a.rows - 8` is 0 at %i = 1, not positive'


# Tiles over regions sized at the call, in HBM
sized = Accelerator('SIZED', base_dir=tempfile.mkdtemp())
sized.add_data_model('d1', [4], [8], 'u8')
for name, comp, size in (('load_n', ['n'], '@c.n'), ('load_rows', [], '@a.rows')):
    instr = sized.add_instruction(name, comp, ['src'] + (['rows'] if not comp else []))
    instr.set_inputs([['d0', ['@a.src'], [size]]])
    instr.set_outputs([['d1', ['0'], ['1']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[8] parameter(0);
    ROOT %Out0 = u8[1,8] reshape(%In0);
}
""")
instr = sized.add_instruction('take_rows', [], ['src', 'rows'])
instr.set_inputs([['d0', ['@a.src'], ['8']]])
instr.set_outputs([['d1', ['0'], ['1']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@a.rows`] parameter(0);
    ROOT %Out0 = u8[1,8] reshape(%In0);
}
""")
instr = sized.add_instruction('store_n', ['n'], ['dst'])
instr.set_inputs([['d1', ['0'], ['1']]])
instr.set_outputs([['d0', ['@a.dst'], ['@c.n']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[1,8] parameter(0);
    ROOT %Out0 = u8[8] reshape(%In0);
}
""")
generate_oracle(sized)
sized_kernel, sized_api = import_oracle(
    os.path.join(sized.base_dir, 'targets', 'SIZED'), 'oracle_SIZED'
)
SIZED_INPUT = [{'addr': 0, 'shape': (16,), 'dtype': np.uint8}]
SIZED_OUTPUT = [{'addr': 16, 'shape': (8,), 'dtype': np.uint8}]


def sized_at_call():
    sized_api.load_n(n=8, src=8)


def sized_in_loop():
    sized_api.start_loop('i', 0, 1)
    sized_api.load_rows(src=8, rows='8 + %i')
    sized_api.end_loop()


def tile_in_loop():
    sized_api.start_loop('i', 0, 2)
    sized_api.take_rows(src=8, rows='8 + %i / 2')
    sized_api.end_loop()


# A tile whose region has its shape at the call
@pytest.mark.parametrize(
    'load', [sized_at_call, sized_in_loop, tile_in_loop], ids=['call', 'looped', 'looped_tile']
)
def test_tile_of_its_region_shape_at_the_call_runs(load):
    x = np.arange(16, dtype=np.uint8)

    @sized_kernel.kernel(hbm=24, input=SIZED_INPUT, constant=[], output=SIZED_OUTPUT)
    def k():
        load()
        sized_api.store_n(n=8, dst=16)

    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x[8:16])


def short_load():
    sized_api.load_n(n=4, src=0)


def short_store():
    sized_api.load_n(n=8, src=0)
    sized_api.store_n(n=4, dst=16)


def grown_in_loop():
    sized_api.start_loop('i', 0, 2)
    sized_api.load_rows(src=0, rows='8 + %i')
    sized_api.end_loop()


def tile_grown_in_loop():
    sized_api.start_loop('i', 0, 2)
    sized_api.take_rows(src=0, rows='8 + %i')
    sized_api.end_loop()


# A tile unlike its region's shape at the call, its iteration named
@pytest.mark.parametrize(
    'body, message',
    [
        (short_load, "load_n: '%In0' is u8[8], not the shape [4] of its input region in d0"),
        (short_store, "store_n: '%Out0' is u8[8], not the shape [4] of its output region in d0"),
        (
            grown_in_loop,
            "load_rows: '%In0' is u8[8], not the shape [9] of its input region in d0 at %i = 1",
        ),
        (
            tile_grown_in_loop,
            "take_rows: '%In0' is u8[9], not the shape [8] of its input region in d0 at %i = 1",
        ),
    ],
    ids=['input', 'output', 'looped', 'looped_tile'],
)
def test_tile_unlike_its_region_shape_at_the_call_refused(body, message):
    @sized_kernel.kernel(hbm=24, input=SIZED_INPUT, constant=[], output=SIZED_OUTPUT)
    def k():
        body()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


# A tile of its region's element count in another shape, read and written row-major
def test_tile_reshaped_to_its_region_runs():
    acc = Accelerator('RESHAPED', base_dir=tempfile.mkdtemp())
    acc.add_data_model('d1', [4], [8], 'u8')
    instr = acc.add_instruction('load_t', [], ['src'])
    instr.set_inputs([['d0', ['@a.src'], ['8']]])
    instr.set_outputs([['d1', ['0'], ['1']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[2,4] parameter(0);
    ROOT %Out0 = u8[4,2] transpose(%In0), dimensions={1,0};
}
""")
    instr = acc.add_instruction('store_t', [], ['dst'])
    instr.set_inputs([['d1', ['0'], ['1']]])
    instr.set_outputs([['d0', ['@a.dst'], ['8']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[8] parameter(0);
    ROOT %Out0 = u8[2,4] reshape(%In0);
}
""")
    generate_oracle(acc)
    decorator, reshaped = import_oracle(
        os.path.join(acc.base_dir, 'targets', 'RESHAPED'), 'oracle_RESHAPED'
    )
    x = np.arange(8, dtype=np.uint8)

    @decorator.kernel(
        hbm=16,
        input=[{'addr': 0, 'shape': (8,), 'dtype': np.uint8}],
        constant=[],
        output=[{'addr': 8, 'shape': (8,), 'dtype': np.uint8}],
    )
    def k():
        reshaped.load_t(src=0)
        reshaped.store_t(dst=8)

    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x.reshape(2, 4).T.reshape(-1))


# load_cm at an n other than 64, its [64, n] tile laid row-major into n rows
def test_load_cm_at_any_n_runs():
    @kernel(hbm=32768, input=INPUT, constant=[], output=OUTPUT)
    def k():
        api.load_cm(n=32, addr_in=0, addr_out=0)
        api.store_rm(n=32, addr_in=0, addr_out=24576)

    Q, K, V = load_qkv()
    k('fsim-compile')()
    outputs, _ = k('fsim')(Q, K, V)
    expected = np.asarray(Q)[:32].T.reshape(32, 64)
    assert np.asarray(outputs[0])[:32].tobytes() == expected.tobytes()


# Lines whose shapes the call's values give
called = Accelerator('CALLED', base_dir=tempfile.mkdtemp())
called.add_data_model('d1', [8], [8], 'u8')
instr = called.add_instruction('add_n', ['n'], ['src', 'dst'])
instr.set_inputs([['d0', ['@a.src'], ['@c.n']], ['d0', ['@a.src'], ['8']]])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n`] parameter(0);
    %In1 = u8[8] parameter(1);
    ROOT %Out0 = u8[8] add(%In0, %In1);
}
""")
instr = called.add_instruction('dot_k', ['k'], [])
instr.set_inputs([['d1', ['0'], ['@c.k']], ['d1', ['0'], ['8']]])
instr.set_outputs([['d1', ['0'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.k`,8] parameter(0);
    %In1 = u8[8,8] parameter(1);
    ROOT %Out0 = u8[8,8] dot(%In0, %In1), lhs_contracting_dims={0}, rhs_contracting_dims={0};
}
""")
instr = called.add_instruction('slide', [], ['src', 'off', 'dst'])
instr.set_inputs([['d0', ['@a.src'], ['16']]])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[16] parameter(0);
    ROOT %Out0 = u8[8] slice(%In0), slice={[`@a.off`:`@a.off + 8`]};
}
""")
instr = called.add_instruction('fill', [], ['v', 'dst'])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    ROOT %Out0 = u8[8] constant(`@a.v`);
}
""")
instr = called.add_instruction('take', [], ['src', 'rows', 'dst'])
instr.set_inputs([['d0', ['@a.src'], ['@a.rows']]])
instr.set_outputs([['d0', ['@a.dst'], ['8']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@a.rows`] parameter(0);
    ROOT %Out0 = u8[8] copy(%In0);
}
""")
instr = called.add_instruction('touch', [], ['src', 'rows'])
instr.set_inputs([['d0', ['@a.src'], ['@a.rows']]])
instr = called.add_instruction('head', [], ['src', 'rows', 'dst'])
instr.set_inputs([['d0', ['@a.src'], ['@a.rows']]])
instr.set_outputs([['d0', ['@a.dst'], ['@a.rows']]])
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@a.rows`] parameter(0);
    ROOT %Out0 = u8[`@a.rows`] copy(%In0);
}
""")
generate_oracle(called)
called_kernel, called_api = import_oracle(
    os.path.join(called.base_dir, 'targets', 'CALLED'), 'oracle_CALLED'
)
CALLED_INPUT = [{'addr': 0, 'shape': (32,), 'dtype': np.uint8}]
CALLED_OUTPUT = [{'addr': 32, 'shape': (16,), 'dtype': np.uint8}]


# Twice HBM's first sixteen bytes, at each call or in a loop
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
def test_shapes_agreeing_at_the_call_run(looped):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        if looped:
            called_api.start_loop('i', 0, 2)
            called_api.add_n(n=8, src='8 * %i', dst='32 + 8 * %i')
            called_api.end_loop()
        else:
            for i in range(2):
                called_api.add_n(n=8, src=8 * i, dst=32 + 8 * i)

    x = np.arange(32, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), 2 * x[:16])


def short_add():
    called_api.add_n(n=4, src=0, dst=32)


def short_dot():
    called_api.dot_k(k=4)


def slid_past():
    called_api.start_loop('i', 0, 4)
    called_api.slide(src=0, off='4 * %i', dst=32)
    called_api.end_loop()


# A line whose shapes disagree at the call, its iteration named
@pytest.mark.parametrize(
    'body, message',
    [
        (short_add, "add_n: add of '%Out0' takes operands of its shape u8[8], found '%In0' u8[4]"),
        (
            short_dot,
            (
                "dot_k: dot of '%Out0' pairs dimension 0 of '%In0' u8[4,8] "
                "with dimension 0 of '%In1' u8[8,8]"
            ),
        ),
        (
            slid_past,
            "slide: slice of '%Out0' takes ranges within '%In0' u8[16], found [12:20:1] at %i = 3",
        ),
    ],
    ids=['elementwise', 'dot', 'looped'],
)
def test_shapes_disagreeing_at_the_call_refused(body, message):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        body()

    with pytest.raises(ValueError) as refusal:
        k('fsim-compile')
    assert str(refusal.value) == message


# A slice at one offset over its loop, against its unrolled twin
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
def test_looped_value_of_one_integer_in_a_semantics_runs(looped):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        if looped:
            called_api.start_loop('i', 0, 2)
            called_api.slide(src=0, off='4 + %i / 2', dst='32 + 8 * %i')
            called_api.end_loop()
        else:
            for i in range(2):
                called_api.slide(src=0, off=4, dst=32 + 8 * i)

    x = np.arange(32, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), np.tile(x[4:12], 2))


def slid(looped):
    if looped:
        called_api.start_loop('i', 0, 2)
        called_api.slide(src=0, off='4 * %i', dst='32 + 8 * %i')
        called_api.end_loop()
    else:
        for i in range(2):
            called_api.slide(src=0, off=4 * i, dst=32 + 8 * i)


def filled(looped):
    if looped:
        called_api.start_loop('i', 0, 2)
        called_api.fill(v='%i', dst='32 + 8 * %i')
        called_api.end_loop()
    else:
        for i in range(2):
            called_api.fill(v=i, dst=32 + 8 * i)


# A looped value in a semantics that changes over its loop, against its unrolled twin
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
@pytest.mark.parametrize(
    'body, expected',
    [(slid, np.r_[0:8, 4:12]), (filled, np.repeat([0, 1], 8))],
    ids=['attribute', 'constant'],
)
def test_looped_value_changing_in_a_semantics_runs(body, expected, looped):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        body(looped)

    k('fsim-compile')()
    outputs, _ = k('fsim')(np.arange(32, dtype=np.uint8))
    np.testing.assert_array_equal(np.asarray(outputs[0]), expected)


# A tile of one shape over its loop, against its unrolled twin
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
def test_looped_tile_of_one_shape_runs(looped):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        if looped:
            called_api.start_loop('i', 0, 2)
            called_api.take(src='8 * %i', rows='8 + %i / 2', dst='32 + 8 * %i')
            called_api.touch(src=0, rows='8 + %i / 2')
            called_api.end_loop()
        else:
            for i in range(2):
                called_api.take(src=8 * i, rows=8, dst=32 + 8 * i)
                called_api.touch(src=0, rows=8)

    x = np.arange(32, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x[:16])


# A tile and a region whose looped shape changes over its loop, against its unrolled twin
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'unrolled'])
def test_looped_tile_shape_changing_runs(looped):
    @called_kernel.kernel(hbm=48, input=CALLED_INPUT, constant=[], output=CALLED_OUTPUT)
    def k():
        if looped:
            called_api.start_loop('i', 0, 2)
            called_api.head(src='4 * %i', rows='4 + 8 * %i', dst='32 + 4 * %i')
            called_api.touch(src=0, rows='8 + %i')
            called_api.end_loop()
        else:
            for i in range(2):
                called_api.head(src=4 * i, rows=4 + 8 * i, dst=32 + 4 * i)
                called_api.touch(src=0, rows=8 + i)

    x = np.arange(32, dtype=np.uint8)
    k('fsim-compile')()
    outputs, _ = k('fsim')(x)
    np.testing.assert_array_equal(np.asarray(outputs[0]), x[:16])
