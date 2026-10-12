import time
import copy
from collections import defaultdict

import jax
from jax.extend.backend import get_backend, get_compile_options
from jax import ffi


from .build import print_handler_ext
from . import api


backend = None


def set_simulation_backend(platform="CPU"):
    global backend
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if platform == "CPU":
        backend = get_backend("cpu")
    elif platform == "GPU":
        backend = get_backend("gpu")


def reset_api():
    api.ssa_types = {}
    api.loop_stack = []
    api.cur_scope = "main"
    api.state = copy.deepcopy(api.default_state)
    api.instruction_scopes = defaultdict(list)


def compile_module(module: str):
    # The debug handler, on each platform it was built for
    for platform, targets in print_handler_ext.registrations().items():
        for name, target in targets.items():
            ffi.register_ffi_target(name, target, platform=platform)

    options = get_compile_options(num_replicas=1, num_partitions=1)

    start = time.time()
    executable = backend.compile_and_load(module, [backend.devices()[0]], options)
    elapsed = round((time.time() - start) * 1000, 3)

    api.executable = executable
    return elapsed


def kernel(hbm: int, input: list, output: list, constant=[]):
    def decorator(func):
        def launchable(mode):
            if mode not in ('fsim', 'fsim-compile'):
                raise ValueError(f"unknown oracle mode {mode!r}")

            def wrapper(f, *args, **kwargs):
                if mode != 'fsim':
                    f()

                if mode == 'fsim-compile':
                    return input, api.compile_time

                if mode == 'fsim':
                    # The module lays the inputs into HBM
                    assert len(args) == len(input), "args != input"
                    operands = []
                    for arg, spec in zip(args, input):
                        assert arg.shape == spec['shape'], "arg.shape != input.shape"
                        assert arg.dtype == spec['dtype'], "arg.dtype != input.dtype"
                        operands.append(jax.device_put(arg, device=backend.devices()[0]))

                    start = time.time()
                    outputs = api.executable.execute(operands)
                    jax.block_until_ready(outputs)
                    elapsed = round((time.time() - start) * 1000, 3)

                    return outputs, elapsed

            api.mode = mode
            if backend == None:
                set_simulation_backend()

            if mode == 'fsim-compile':
                reset_api()
                counters, _ = api.semantic_init(hbm)
                api.global_counters = counters
                func()
                module = api.generate_module(hbm, input, constant, output)
                api.compile_time = compile_module(module)

            return lambda *args, **kwargs: wrapper(func, *args, **kwargs)
        return launchable
    return decorator
