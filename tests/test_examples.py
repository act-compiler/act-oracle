"""Test the examples side by side"""

import compileall
import glob
import os
import runpy
import shutil
import tempfile

from conftest import EXAMPLES_DIR, generate_example, import_oracle, oracle_ids
from taidl import Accelerator

from taidl_to import generic


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


def test_oracle_carries_no_bytecode_cache():
    # The generic sources hold a bytecode cache
    generic_dir = os.path.dirname(generic.__file__)
    cache_dir = os.path.join(generic_dir, '__pycache__')
    made = not os.path.isdir(cache_dir)
    if made:
        compileall.compile_dir(generic_dir, quiet=1)
    try:
        oracle_dir = os.path.join(
            generate_example(os.path.join(EXAMPLES_DIR, 'mini_trn')), 'oracle'
        )
    finally:
        # Removed again when made here
        if made:
            shutil.rmtree(cache_dir)
    # The oracle carries none of it
    cached = [
        os.path.relpath(path, oracle_dir)
        for pattern in ('__pycache__', '*.pyc')
        for path in glob.glob(os.path.join(oracle_dir, '**', pattern), recursive=True)
    ]
    assert not cached, f'the oracle carries {cached}'
