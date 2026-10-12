"""Regenerate an example's goldens from its oracle

Usage: python tests/regenerate.py <example_dir>, one example per process; tests/regenerate.sh
runs it for every example in the CI image; each golden comes from the first kernel it lists.
"""

import os
import sys
import json
import platform
import importlib

import jax
import jaxlib
import numpy as np

from conftest import generate_and_import_oracle, run_kernel, attention_reference

if __name__ == "__main__":
    example_dir = os.path.abspath(sys.argv[1])
    assembly_dir = os.path.join(example_dir, "assembly")
    data_dir = os.path.join(example_dir, "data")
    with open(os.path.join(data_dir, "goldens.json")) as f:
        goldens = json.load(f)

    kernel, api = generate_and_import_oracle(example_dir)
    backend = importlib.import_module(kernel.__module__).backend
    reference, tol = attention_reference(data_dir)

    for name, record in goldens.items():
        outputs = [np.asarray(run_kernel(kernel, api, k, assembly_dir, data_dir))
                   for k in record["kernels"]]
        # Kernels sharing a golden must agree bit for bit
        for k, output in zip(record["kernels"], outputs):
            assert output.tobytes() == outputs[0].tobytes(), \
                f"{k} differs from {record['kernels'][0]}; give it a golden of its own"
        # Never write a golden the reference rejects
        max_diff = np.max(np.abs(outputs[0].astype(np.float64) - reference))
        assert max_diff <= tol, f"{name} is {max_diff} from the reference, beyond {tol}"
        outputs[0].tofile(os.path.join(data_dir, name))
        record.update(jax=jax.__version__, jaxlib=jaxlib.__version__,
                      platform=backend.platform, machine=platform.machine())
        print(f"Wrote {name} for {', '.join(record['kernels'])}, {max_diff} from the reference")

    with open(os.path.join(data_dir, "goldens.json"), "w") as f:
        json.dump(goldens, f, indent=2)
        f.write("\n")
