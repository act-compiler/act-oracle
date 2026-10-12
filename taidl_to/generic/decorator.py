import time
import copy
from collections import defaultdict

import jax
import jax.numpy as jnp
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
                    input_hbm = jnp.zeros(hbm, dtype=jnp.uint8)

                    for const in constant:
                        const_array = jnp.array(const['value'])
                        flat = const_array.flatten().view(jnp.uint8)
                        input_hbm = input_hbm.at[const['addr']: const['addr'] + flat.size].set(flat)

                    assert len(args) == len(input), "args != input"
                    for arg, spec in zip(args, input):
                        assert arg.shape == spec['shape'], "arg.shape != input.shape"
                        assert arg.dtype == spec['dtype'], "arg.dtype != input.dtype"
                        arg_array = jnp.array(arg)
                        flat = arg_array.flatten().view(jnp.uint8)
                        input_hbm = input_hbm.at[spec['addr']: spec['addr'] + flat.size].set(flat)

                    arg = jax.device_put(input_hbm, device=backend.devices()[0])

                    start = time.time()
                    result = api.executable.execute([arg])
                    jax.block_until_ready(result)
                    elapsed = round((time.time() - start) * 1000, 3)

                    output_hbm = result[0]
                    outputs = []

                    for out in output:
                        addr, shape, dtype = out['addr'], out['shape'], out['dtype']
                        size = int(jnp.prod(jnp.array(shape)) * jnp.dtype(dtype).itemsize)
                        data = jnp.array(output_hbm[addr:addr + size]).view(dtype).reshape(shape)
                        outputs.append(jnp.array(data))

                    return outputs, elapsed

            api.mode = mode
            if backend == None:
                set_simulation_backend()

            if mode == 'fsim-compile':
                reset_api()
                counters, _ = api.semantic_init(hbm)
                api.global_counters = counters
                func()
                module = api.generate_module(hbm)
                api.compile_time = compile_module(module)

            return lambda *args, **kwargs: wrapper(func, *args, **kwargs)
        return launchable
    return decorator
