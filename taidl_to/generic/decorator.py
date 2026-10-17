import copy
import time
from collections import defaultdict

import jax
import numpy as np
from jax import ffi
from jax.extend.backend import get_backend, get_compile_options

from . import api
from .build import print_handler_ext

backend = None


def set_simulation_backend(platform='CPU'):
    global backend
    if platform not in ('CPU', 'GPU'):
        raise ValueError(f'unknown simulation platform {platform!r}')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if platform == 'CPU':
        backend = get_backend('cpu')
    else:
        backend = get_backend('gpu')


def reset_api():
    api.ssa_types = {}
    api.loop_stack = []
    api.cur_scope = 'main'
    api.state = copy.deepcopy(api.default_state)
    api.instruction_scopes = defaultdict(list)


def compile_module(module: str):
    # The debug handler, on each platform it was built for
    for platform, targets in print_handler_ext.registrations().items():
        for name, target in targets.items():
            # Named after this package, so each oracle registers its own
            ffi.register_ffi_target(f'{__package__}.{name}', target, platform=platform)

    options = get_compile_options(num_replicas=1, num_partitions=1)
    device = backend.devices()[0]

    start = time.time()
    executable = backend.compile_and_load(module, [device], options)
    elapsed = round((time.time() - start) * 1000, 3)

    return executable, device, elapsed


def kernel(hbm: int, input: list, output: list, constant=()):
    def decorator(func):
        # Each decorated kernel's own executable, device and compile time
        executable, device, compile_time = None, None, None

        def launchable(mode):
            nonlocal executable, device, compile_time
            if mode not in ('fsim', 'fsim-compile'):
                raise ValueError(f'unknown oracle mode {mode!r}')

            def wrapper(*args):
                if mode == 'fsim-compile':
                    # A compiled kernel takes its inputs only in 'fsim'
                    if args:
                        raise ValueError(
                            f"kernel {func.__name__!r} takes no arguments in 'fsim-compile'"
                        )
                    return input, compile_time

                # Only a compiled kernel runs
                if executable is None:
                    raise ValueError(
                        f"kernel {func.__name__!r} is not compiled; run it in 'fsim-compile' first"
                    )

                # One argument per declared input
                if len(args) != len(input):
                    raise ValueError(
                        f'kernel {func.__name__!r} declares {len(input)} inputs, '
                        f'got {len(args)} arguments'
                    )

                # The module lays the inputs into HBM
                operands = []
                for i, (arg, spec) in enumerate(zip(args, input)):
                    # Each an array of its input's shape and dtype
                    what = f'argument {i} of kernel {func.__name__!r}'
                    if not isinstance(arg, (np.ndarray, np.generic, jax.Array)):
                        raise TypeError(f'{what} is a {type(arg).__name__}, not an array')
                    shape = tuple(spec['shape'])
                    if arg.shape != shape:
                        raise ValueError(f'{what} has shape {arg.shape}, declared {shape}')
                    if arg.dtype != spec['dtype']:
                        raise ValueError(
                            f'{what} has dtype {arg.dtype}, declared {np.dtype(spec["dtype"])}'
                        )
                    operands.append(jax.device_put(arg, device=device))

                start = time.time()
                outputs = executable.execute(operands)
                jax.block_until_ready(outputs)
                elapsed = round((time.time() - start) * 1000, 3)

                return outputs, elapsed

            api.mode = mode
            if backend is None:
                set_simulation_backend()

            if mode == 'fsim-compile':
                reset_api()
                counters, _ = api.semantic_init(hbm)
                api.global_counters = counters
                func()
                module = api.generate_module(hbm, input, constant, output)
                executable, device, compile_time = compile_module(module)

            return wrapper

        return launchable

    return decorator
