import os
import sys
import time
import importlib
import copy
from collections import defaultdict

import jax
import jax.numpy as jnp
from jax.lib import xla_bridge, xla_client
from jax.extend import ffi


from .build import print_handler_ext
from . import api


backend = None


def set_simulation_backend(platform="CPU"):
    global backend
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if platform == "CPU":
        backend = xla_bridge.get_backend("cpu")
    elif platform == "GPU":
        backend = xla_bridge.get_backend("gpu")


def reset_api():
    api.loop_variables = ["main"]
    api.conditional_clauses = []
    api.cur_scope = "main"
    api.instruction_store = []
    api.compare_function = ""
    api.call_statement = ""
    api.state = copy.deepcopy(api.default_state)
    api.total_cost = 0
    api.instruction_scopes = defaultdict(list)
    api.scope_tuple_count = defaultdict(int)
    api.instructions = []


def generate_hlo(hbm: int) -> str:
    return api.generate_full_hlotext(hbm)


def compile_or_load_executable(hlo_str: str):
    for name, target in print_handler_ext.registrations().items():
        ffi.register_ffi_target(name, target)

    hlo_module = xla_client._xla.hlo_module_from_text(hlo_str)
    hlo_proto = hlo_module.as_serialized_hlo_module_proto()
    options = xla_client.CompileOptions()
    computation = xla_client.XlaComputation(hlo_proto)
    mlir = xla_client._xla.mlir.xla_computation_to_mlir_module(computation)

    start = time.time()
    executable = backend.compile(mlir, compile_options=options)
    elapsed = round((time.time() - start) * 1000, 3)

    api.executable = executable
    return elapsed


def kernel(hbm: int, input: list, output: list, constant=[]):
    def decorator(func):
        def launchable(mode):
            def wrapper(f, *args, **kwargs):
                if mode != 'fsim':
                    f()

                if mode == 'fsim-compile':
                    return input, api.compile_time

                if mode == 'cost':
                    return api.total_cost

                if mode == 'fsim':
                    global backend
                    input_hbm = jnp.zeros(hbm, dtype=jnp.int8)

                    for const in constant:
                        const_array = jnp.array(const['value'])
                        flat = const_array.flatten().view(jnp.int8)
                        input_hbm = input_hbm.at[const['addr']: const['addr'] + flat.size].set(flat)

                    assert len(args) == len(input), "args != input"
                    for arg, spec in zip(args, input):
                        assert arg.shape == spec['shape'], "arg.shape != input.shape"
                        assert arg.dtype == spec['dtype'], "arg.dtype != input.dtype"
                        arg_array = jnp.array(arg)
                        flat = arg_array.flatten().view(jnp.int8)
                        input_hbm = input_hbm.at[spec['addr']: spec['addr'] + flat.size].set(flat)

                    arg = jax.device_put(input_hbm.reshape((1, hbm)),
                                         device=api.executable.local_devices()[0])

                    start = time.time()
                    result = api.executable.execute(arg)
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
                api.instructions = []
                counters, _ = api.semantic_init(hbm)
                api.global_counters = counters
                func()
                hlo = generate_hlo(hbm)
                api.compile_time = compile_or_load_executable(hlo)

            return lambda *args, **kwargs: wrapper(func, *args, **kwargs)
        return launchable
    return decorator


def launch(func, mode):
    return func(mode)
