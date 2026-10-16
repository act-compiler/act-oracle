"""Test the examples side by side"""

import glob
import os
import runpy
import shutil
import tempfile

from conftest import EXAMPLES_DIR, import_oracle, oracle_ids
from taidl import Accelerator


def test_examples_generate_apart():
    # Every example's spec run in one directory
    work_dir = tempfile.mkdtemp()
    accelerators = []
    for spec in sorted(glob.glob(os.path.join(EXAMPLES_DIR, '*', '*.py'))):
        shutil.copy(spec, work_dir)
        defined = runpy.run_path(os.path.join(work_dir, os.path.basename(spec))).values()
        accelerators.append(next(value for value in defined if isinstance(value, Accelerator)))
    # Each keeps an oracle of its own
    for accelerator in accelerators:
        _, api = import_oracle(
            os.path.join(work_dir, 'targets', accelerator.name), f'oracle_{next(oracle_ids)}'
        )
        missing = [
            i.instruction for i in accelerator.instructions if not hasattr(api, i.instruction)
        ]
        assert not missing, f'{accelerator.name} lacks {missing}'
