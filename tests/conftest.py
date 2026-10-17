"""Shared test utilities for TAIDL-TO oracle tests"""

import importlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
# The platform every oracle simulates on, set by --platform
PLATFORM = 'CPU'
# Each imported oracle gets a package name of its own
oracle_ids = itertools.count()
# The tree's ruff settings, which the examples' generated oracles meet
PYPROJECT = os.path.join(os.path.dirname(EXAMPLES_DIR), 'pyproject.toml')


def pytest_addoption(parser):
    parser.addoption(
        '--platform',
        choices=['CPU', 'GPU'],
        default='CPU',
        help='simulate the oracles on the CPU or the GPU',
    )


def pytest_configure(config):
    global PLATFORM
    PLATFORM = config.getoption('--platform')


# Element types as a semantics spells them
FLOATS = ['f64', 'f32', 'f16', 'bf16']
SIGNED = ['s8', 's16', 's32', 's64']
INTEGERS = ['u8', 'u16', 'u32', 'u64', *SIGNED]
EVERY = FLOATS + INTEGERS + ['pred']

# Each element type and the dtype that holds it
DTYPES = {
    'f64': np.float64,
    'f32': np.float32,
    'f16': np.float16,
    'bf16': ml_dtypes.bfloat16,
    'u8': np.uint8,
    'u16': np.uint16,
    'u32': np.uint32,
    'u64': np.uint64,
    's8': np.int8,
    's16': np.int16,
    's32': np.int32,
    's64': np.int64,
    'pred': np.bool_,
}

# Each lowered word and the element types it lowers at
LOWERED = {
    'reshape': EVERY,
    'convert': EVERY,
    'copy': EVERY,
    'bitcast_convert': EVERY,
    'exponential': FLOATS,
    'transpose': EVERY,
    'add': EVERY,
    'subtract': FLOATS + INTEGERS,
    'multiply': EVERY,
    'divide': FLOATS + INTEGERS,
    'broadcast': EVERY,
    'maximum': EVERY,
    'minimum': EVERY,
    'xor': INTEGERS + ['pred'],
    'dot': EVERY,
    'reduce_add': EVERY,
    'constant': EVERY,
    'iota': FLOATS + INTEGERS,
    'abs': FLOATS + SIGNED,
    'negate': FLOATS + INTEGERS,
    'sign': FLOATS + SIGNED,
    'mulhi': INTEGERS,
    'remainder': FLOATS + INTEGERS,
    'power': FLOATS + INTEGERS,
    'sqrt': FLOATS,
    'rsqrt': FLOATS,
    'cbrt': FLOATS,
    'exponential_minus_one': FLOATS,
    'log': FLOATS,
    'log_plus_one': FLOATS,
    'logistic': FLOATS,
    'erf': FLOATS,
    'sine': FLOATS,
    'cosine': FLOATS,
    'tan': FLOATS,
    'asin': FLOATS,
    'acos': FLOATS,
    'atan2': FLOATS,
    'sinh': FLOATS,
    'cosh': FLOATS,
    'tanh': FLOATS,
    'asinh': FLOATS,
    'acosh': FLOATS,
    'atanh': FLOATS,
    'floor': FLOATS,
    'ceil': FLOATS,
    'round_nearest_afz': FLOATS,
    'round_nearest_even': FLOATS,
    'and': INTEGERS + ['pred'],
    'or': INTEGERS + ['pred'],
    'not': INTEGERS + ['pred'],
    'shift_left': INTEGERS,
    'shift_right_logical': INTEGERS,
    'shift_right_arithmetic': INTEGERS,
    'popcnt': INTEGERS,
    'count_leading_zeros': INTEGERS,
    'is_finite': FLOATS,
    'reduce_precision': FLOATS,
}


def byte_shape(t, dims):
    """The shape of a t[dims]'s bytes, a multi-byte type gaining a trailing dim"""
    width = np.dtype(DTYPES[t]).itemsize
    return ','.join(str(d) for d in (*dims, width)[: len(dims) + (width > 1)])


def byte_count(t, dims):
    """The bytes a t[dims] takes"""
    return np.dtype(DTYPES[t]).itemsize * int(np.prod(dims))


def loaded(k, t, dims):
    """The lines that read input k as %x<k>, a t[dims]"""
    # A pred converts from its byte
    cast = 'convert' if t == 'pred' else 'bitcast_convert'
    return f"""%In{k} = u8[{byte_count(t, dims)}] parameter({k});
    %b{k} = u8[{byte_shape(t, dims)}] reshape(%In{k});
    %x{k} = {t}[{','.join(map(str, dims))}] {cast}(%b{k});"""


def stored(value, t, dims):
    """The lines that write value, a t[dims], to %Out0 as bytes"""
    # A pred converts to its byte
    cast = 'convert' if t == 'pred' else 'bitcast_convert'
    return f"""%bytes = u8[{byte_shape(t, dims)}] {cast}({value});
    ROOT %Out0 = u8[{byte_count(t, dims)}] reshape(%bytes);"""


def add_typed_instruction(acc, name, inputs, body, output, comp=()):
    """Add an instruction reading each (type, dims) input from HBM as %x<k>, running body and
    storing its %y, of output's (type, dims), to HBM; comp names its computational attributes"""
    instr = acc.add_instruction(name, list(comp), [f'a{k}' for k in range(len(inputs))] + ['out'])
    regions = [[f'@a.a{k}', byte_count(*value)] for k, value in enumerate(inputs)]
    instr.set_inputs([['d0', [start], [str(size)]] for start, size in regions])
    instr.set_outputs([['d0', ['@a.out'], [str(byte_count(*output))]]])
    lines = [loaded(k, t, dims) for k, (t, dims) in enumerate(inputs)]
    instr.add_semantics('ENTRY {\n' + '\n'.join([*lines, body, stored('%y', *output)]) + '\n}')


def run_bytes(decorator, calls, inputs, out_addr, out_bytes):
    """Run calls, a kernel's body, on byte inputs laid in turn from address 0; return the bytes
    it leaves at out_addr"""
    addrs = itertools.accumulate((len(x) for x in inputs), initial=0)

    @decorator.kernel(
        hbm=out_addr + out_bytes,
        input=[{'addr': a, 'shape': x.shape, 'dtype': np.uint8} for a, x in zip(addrs, inputs)],
        constant=[],
        output=[{'addr': out_addr, 'shape': (out_bytes,), 'dtype': np.uint8}],
    )
    def k():
        calls()

    k('fsim-compile')()
    outputs, _ = k('fsim')(*inputs)
    return np.asarray(outputs[0])


def run_typed(decorator, api, name, args, output, comp=()):
    """Run instruction name, given comp, on args laid in turn in HBM as bytes; return its
    output of (type, dims)"""
    data = [np.ascontiguousarray(x).reshape(-1).view(np.uint8) for x in args]
    addrs = list(itertools.accumulate((len(d) for d in data), initial=0))
    t, dims = output
    out = run_bytes(
        decorator, lambda: getattr(api, name)(*comp, *addrs), data, addrs[-1], byte_count(t, dims)
    )
    return out.view(DTYPES[t]).reshape(dims)


# A transcendental word's CPU bound in ulps, per float type
ULPS = {'bf16': 2, 'f16': 2, 'f32': 8, 'f64': 8}


def ulps(out, exact, t):
    """The largest distance of out from exact, in units in the last place of t at exact"""
    finfo = ml_dtypes.finfo(DTYPES[t])
    scale = np.floor(np.log2(np.maximum(np.abs(exact), finfo.smallest_normal)))
    return float(np.max(np.abs(out.astype(np.float64) - exact) / np.exp2(scale - finfo.nmant)))


def assert_exact(out, expected):
    """out equals expected: a nan matches a nan of either sign, a zero only a zero of its sign"""
    if out.dtype in [DTYPES[t] for t in FLOATS]:
        out, expected = out.astype(np.float64), np.asarray(expected).astype(np.float64)
        np.testing.assert_array_equal(out, expected)
        # A nan's sign is the platform's
        signed = ~np.isnan(expected)
        np.testing.assert_array_equal(np.signbit(out[signed]), np.signbit(expected[signed]))
    else:
        np.testing.assert_array_equal(out, expected)


@pytest.fixture
def rng(request):
    """A generator seeded by the test's name, so its data do not depend on the tests before it"""
    return np.random.default_rng(zlib.crc32(request.node.name.encode()))


def each_type(types, check):
    """Run check(t) at every type, failing with each type it fails at named"""
    failed = {}
    for t in types:
        try:
            check(t)
        # A failure at one type leaves the others to run
        except (AssertionError, pytest.fail.Exception) as error:
            failed[t] = error
    if failed:
        named = '\n'.join(f'{t}: {error}' for t, error in failed.items())
        first = next(iter(failed.values()))
        raise AssertionError(f'fails at {", ".join(failed)}\n{named}') from first


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def check_oracle(oracle_dir):
    """Fail unless the oracle's Python passes ruff's format check and lint"""
    for check in (['format', '--check', '--diff'], ['check']):
        command = ['ruff', *check, '--no-cache', '--config', PYPROJECT, oracle_dir]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        assert run.returncode == 0, f'ruff {check[0]} of {oracle_dir}:\n{run.stdout}{run.stderr}'


def import_oracle(target_dir, name):
    """Import the oracle generated under target_dir as package `name` and return (decorator, api)"""
    oracle_dir = os.path.join(target_dir, 'oracle')
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(oracle_dir, '__init__.py'), submodule_search_locations=[oracle_dir]
    )
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    decorator = importlib.import_module(f'{name}.decorator')
    decorator.set_simulation_backend(PLATFORM)
    return decorator, importlib.import_module(f'{name}.api')


def generate_example(example_dir):
    """Generate the example's oracle into a temp dir and return its target dir"""
    work_dir = tempfile.mkdtemp()
    # The spec's copy generates its oracle beside it
    name = os.path.basename(example_dir)
    path = shutil.copy(os.path.join(example_dir, f'{name}.py'), work_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    (accelerator,) = os.listdir(os.path.join(work_dir, 'targets'))
    target_dir = os.path.join(work_dir, 'targets', accelerator)
    # Its oracle laid out as ruff lays out the tree
    check_oracle(os.path.join(target_dir, 'oracle'))
    return target_dir


def generate_and_import_oracle(example_dir):
    """Generate oracle into a temp dir and return (kernel, api)"""
    oracle_decorator, oracle_api = import_oracle(
        generate_example(example_dir), f'oracle_{next(oracle_ids)}'
    )
    return oracle_decorator.kernel, oracle_api


def run_kernel(kernel, api, kernel_module_name, assembly_dir, data_dir):
    """Run a kernel on the example's Q, K, V and return its output"""
    sys.path.insert(0, assembly_dir)
    kernel_module = importlib.import_module(kernel_module_name)
    attention = kernel_module.attention(kernel, api)
    attention('fsim-compile')()

    Q = load_bf16_matrix(os.path.join(data_dir, 'Q.dat'), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, 'K.dat'), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, 'V.dat'), (64, 64))

    outputs, _ = attention('fsim')(Q, K, V)
    return outputs[0]


def load_golden(data_dir, kernel_name):
    """Return a kernel's golden and the record of where it came from"""
    with open(os.path.join(data_dir, 'goldens.json')) as f:
        goldens = json.load(f)
    name, record = next((n, r) for n, r in goldens.items() if kernel_name in r['kernels'])
    return load_bf16_matrix(os.path.join(data_dir, name), (64, 64)), record


def bf16(x):
    """Round float64 values to bf16, kept in float64"""
    return x.astype(ml_dtypes.bfloat16).astype(np.float64)


def attention_reference(data_dir):
    """Return softmax(Q K^T) V in numpy, rounded to bf16 where the spec rounds, and its tolerance"""
    Q, K, V = (
        np.fromfile(os.path.join(data_dir, f'{n}.dat'), dtype=ml_dtypes.bfloat16)
        .reshape(64, 64)
        .astype(np.float64)
        for n in 'QKV'
    )
    S = bf16(Q @ K.T)
    E = bf16(np.exp(S))
    P = bf16(E / bf16(E.sum(axis=1, keepdims=True)))
    # On the CPU, two bf16 ulps at max|V|
    ulp = 2.0 ** (np.floor(np.log2(np.abs(V).max())) - 7)
    # On the GPU, twice the two ulps an H100 reaches
    tol = (2 if PLATFORM == 'CPU' else 4) * ulp
    return bf16(P @ V), tol
