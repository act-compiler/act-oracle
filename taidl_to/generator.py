"""generate_oracle, which writes an accelerator's oracle"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from taidl import Accelerator

from .api_generator import generate_api_file
from .generic.utils import MLIR_ELEM_TYPE
from .template import write_file


def generate_oracle(accelerator: Accelerator, base_dir: str | None = None) -> None:
    """Write the accelerator's oracle to <base_dir>/targets/<name>/oracle.

    base_dir is accelerator.base_dir unless given; the previous oracle
    is removed first, so a refused or failed generation leaves none.
    """
    if base_dir is None:
        base_dir = accelerator.base_dir

    # Remove the previous oracle before any refusal
    top_gen_directory = os.path.join(base_dir, 'targets', accelerator.name)
    oracle_gen_dir = os.path.join(top_gen_directory, 'oracle')
    if os.path.exists(oracle_gen_dir):
        shutil.rmtree(oracle_gen_dir)

    # Refuse a type the oracle does not lower, writing nothing
    for model in accelerator.data_model:
        if model.var_type not in MLIR_ELEM_TYPE:
            raise NotImplementedError(
                f"{model.var_name}: type '{model.var_type}' is not supported by the oracle"
            )

    # Build the API's text first, so a refusal writes nothing
    api_text = generate_api_file(
        accelerator.instructions,
        accelerator.state,
        accelerator.data_model,
    )

    # Setup directories
    Path(top_gen_directory).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic')
    xla_debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xla-debug')

    # Assemble the new oracle aside, so a failed build leaves none
    with tempfile.TemporaryDirectory(dir=top_gen_directory) as staging_dir:
        new_oracle_dir = os.path.join(staging_dir, 'oracle')
        shutil.copytree(generic_dir, new_oracle_dir)
        write_file(api_text, os.path.join(new_oracle_dir, 'api.py'))
        # Build the debug handler into it, afresh at every generation
        build_dir = os.path.join(new_oracle_dir, 'build')
        subprocess.run(
            [
                'cmake',
                '-S',
                xla_debug_dir,
                '-B',
                build_dir,
                f'-DPython_EXECUTABLE={sys.executable}',
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(['cmake', '--build', build_dir], check=True, stdout=subprocess.DEVNULL)
        # Move it into place, complete
        os.rename(new_oracle_dir, oracle_gen_dir)

    print(f'Generated the oracle for {accelerator.name} at {oracle_gen_dir}/')
