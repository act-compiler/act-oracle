"""Oracle code generator for TAIDL-TO

Generates the test oracle (functional simulator) from a TAIDL Accelerator spec.
"""

import os
from pathlib import Path
import shutil
import subprocess
from typing import List

from taidl import Accelerator

from .template import init_templates, templates, generate_code, write_file, indent_code
from .instruction_codegen import generate_api_function


def generate_semantic_init(data_models):
    """Generate semantic initialization code"""
    init_templates()
    template_semantic_init = templates["SEMANTIC_INIT"]
    template_semantic_counter = templates["SEMANTIC_COUNTER"]
    template_prologue_init = templates["PROLOGUE_INIT"]

    counters = ""
    prologue = ""
    for model in data_models:
        if (model.var_name == 'd0'):
            continue
        dimensions = model.array_dim_str.replace("'", "")
        mapping = {
            "var_name": model.var_name,
            "var_type": model.var_type,
            "var_dim": dimensions,
            "var_num": model.num_dim_str
        }
        counters += generate_code(template_semantic_counter, mapping)
        prologue += generate_code(template_prologue_init, mapping)

    counters = indent_code(counters, level=2)

    output = generate_code(template_semantic_init, {
        "custom_counters": counters,
        "custom_prologue": prologue
    })
    return output


def generate_oracle(accelerator: Accelerator, output_dir: str = None) -> None:
    """
    Generate Oracle API code for an accelerator.

    Args:
        accelerator: TAIDL Accelerator object
        output_dir: Output directory. Defaults to targets/<name>/oracle/ relative to cwd.
    """
    init_templates()

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'targets', accelerator.name, 'oracle')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy generic oracle runtime
    generic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic')
    for item in os.listdir(generic_dir):
        src = os.path.join(generic_dir, item)
        dst = os.path.join(output_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"Copied generic oracle structure to {output_dir}")

    # Generate API file
    semantic_init = generate_semantic_init(accelerator.data_model)

    consts = ""
    for constant in accelerator.constants:
        consts += "'" + constant.const_name + "': " + str(constant.value) + ",\n"

    state_str = ""
    for constant in accelerator.state:
        state_str += "'" + constant.const_name + "': " + str(constant.value) + ",\n"

    output = generate_code(templates["API_FILE"], {
        "constants": consts,
        "state": state_str,
        "semantic_init": semantic_init,
        "API_NAME": accelerator.name
    })

    for instruction in accelerator.instructions:
        output += generate_api_function(instruction)

    write_file(output, os.path.join(output_dir, "api.py"))
    print(f"Generated api.py")

    # Build xla-debug
    xla_debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xla-debug')
    xla_build_dir = os.path.join(xla_debug_dir, 'build')
    if not os.path.exists(xla_build_dir):
        subprocess.run(["./build.sh"], cwd=xla_debug_dir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)

    # Copy built xla-debug to final destination
    if os.path.exists(xla_build_dir):
        dest_dir = os.path.join(output_dir, 'build')
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(xla_build_dir, dest_dir)

    print(f"Oracle generation complete for {accelerator.name}")
    print(f"Output: {output_dir}/")
