"""Oracle code generator for TAIDLv2

This module provides the main interface for generating oracle/API code.
It coordinates template-based generation of Python API files for accelerators.
"""

import os
from pathlib import Path
import shutil
from typing import List
import subprocess

from .api_generator import generate_api_file


def generate_oracle(accelerator_name: str, instructions: List, constants: List,
                    state: List, data_models: List, base_dir: str) -> None:
    """
    Generate Oracle API code for an accelerator.

    Args:
        accelerator_name: Name of the accelerator
        instructions: List of Instruction objects
        constants: List of Constant objects
        state: List of state Constant objects
        data_models: List of DataModel objects
        base_dir: Base directory of the project
    """
    # Setup directories
    top_gen_directory = os.path.join(base_dir, 'targets', accelerator_name)
    Path(top_gen_directory).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(base_dir, 'generators', 'oracle', 'generic')
    oracle_gen_dir = os.path.join(top_gen_directory, 'oracle')

    # Copy generic oracle structure
    if os.path.exists(oracle_gen_dir):
        shutil.rmtree(oracle_gen_dir)
    shutil.copytree(generic_dir, oracle_gen_dir)

    print(f"Copied generic oracle structure to {oracle_gen_dir}")

    # Generate API file
    generate_api_file(oracle_gen_dir, accelerator_name, instructions,
                      constants, state, data_models)

    print(f"Generated api.py")

    print(f"Oracle API generation complete for {accelerator_name}")

    # Build the xla-debug
    print(f"Building oracle for {accelerator_name}")
    xla_debug_dir = os.path.join(base_dir, 'xla-debug')
    xla_build_dir = os.path.join(xla_debug_dir, 'build')
    if os.path.exists(xla_build_dir):
        shutil.rmtree(xla_build_dir)
    subprocess.run(["./build.sh"], cwd=xla_debug_dir, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    print(f"Oracle build complete for {accelerator_name}")

    # Copy built xla-debug to final destination
    if os.path.exists(xla_build_dir):
        dest_dir = os.path.join(oracle_gen_dir, 'build')
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(xla_build_dir, dest_dir)
        print(f"The oracle API is located at {oracle_gen_dir}/")
    else:
        raise RuntimeError("xla-debug build failed. Please check the build logs.")
