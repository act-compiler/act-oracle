"""Oracle code generator for TAIDL-TO

This module provides the main interface for generating oracle/API code.
It coordinates template-based generation of Python API files for accelerators.
"""

import os
from pathlib import Path
import shutil
from typing import List
import subprocess

from taidl import Accelerator

from .api_generator import generate_api_file


def generate_oracle(accelerator: Accelerator, base_dir: str = None) -> None:
    """
    Generate Oracle API code for an accelerator.

    Args:
        accelerator: TAIDL Accelerator object
        base_dir: Base directory of the project. Defaults to accelerator.base_dir.
    """
    if base_dir is None:
        base_dir = accelerator.base_dir

    # Setup directories
    top_gen_directory = os.path.join(base_dir, 'targets', accelerator.name)
    Path(top_gen_directory).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic')
    oracle_gen_dir = os.path.join(top_gen_directory, 'oracle')

    # Copy generic oracle structure
    if os.path.exists(oracle_gen_dir):
        shutil.rmtree(oracle_gen_dir)
    shutil.copytree(generic_dir, oracle_gen_dir)

    print(f"Copied generic oracle structure to {oracle_gen_dir}")

    # Generate API file
    generate_api_file(oracle_gen_dir, accelerator.name, accelerator.instructions,
                      accelerator.constants, accelerator.state, accelerator.data_model)

    print(f"Generated api.py")

    print(f"Oracle API generation complete for {accelerator.name}")

    # Build the xla-debug
    print(f"Building oracle for {accelerator.name}")
    xla_debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xla-debug')
    xla_build_dir = os.path.join(xla_debug_dir, 'build')
    if os.path.exists(xla_build_dir):
        # shutil.rmtree(xla_build_dir)
        pass  # TODO: Workaround for now, remove when arm64 docker is fixed
    else:  # TODO: Workaround for now, remove when arm64 docker is fixed
        subprocess.run(["./build.sh"], cwd=xla_debug_dir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    print(f"Oracle build complete for {accelerator.name}")

    # Copy built xla-debug to final destination
    if os.path.exists(xla_build_dir):
        dest_dir = os.path.join(oracle_gen_dir, 'build')
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(xla_build_dir, dest_dir)
        print(f"The oracle API is located at {oracle_gen_dir}/")
    else:
        raise RuntimeError("xla-debug build failed. Please check the build logs.")
