"""Opt-in exact comparisons with the unmodified local research code.

Set ``--legacy-root /path/to/research-archive`` or ``CIVOL_LEGACY_ROOT``
to enable these checks. Input panels and outputs are temporary; no licensed
data is needed.
Timing metadata is the only omitted estimator result. Exact numeric equality
is required within the same interpreter, dependency versions, and backend.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


CORE_MODULES = (
    "common_space", "private_dynamics", "horizon", "horizon_gpu",
    "benchmarks", "outcomes", "universe", "estimator", "adaptive_garch",
)


def calculation_functions(path, excluded=()):
    """Compare function bodies and numerical defaults, excluding module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [part.id for target in node.targets for part in ast.walk(target)
                     if isinstance(part, ast.Name)]
            if names and all(name.isupper() for name in names) and "SOURCE_FILES" not in names:
                result["constants:" + ",".join(names)] = ast.dump(node, include_attributes=False)
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in excluded:
            continue
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        result[node.name] = ast.dump(node, include_attributes=False)
    return result


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_calculation_source_unchanged(legacy_root, name):
    old = legacy_root / "thesis_method" / "core" / f"{name}.py"
    if name == "adaptive_garch":
        old = legacy_root / "src" / f"{name}.py"
    new = Path(__file__).resolve().parents[1] / "src" / "civol" / "core" / f"{name}.py"
    # Provenance hashes and subprocess transport are infrastructure, not formulas.
    excluded = {"source_hashes", "_gpu_batches"}
    original = calculation_functions(old, excluded)
    refactored = calculation_functions(new, excluded)
    assert original.keys() <= refactored.keys()
    for function, source in original.items():
        assert refactored[function] == source, f"Calculation changed: {name}.{function}"


def test_rank_calculation_source_unchanged(legacy_root):
    old = legacy_root / "thesis_method" / "rank_rule" / "rank_rules.py"
    new = Path(__file__).resolve().parents[1] / "src" / "civol" / "rank_rule" / "rank_rules.py"
    assert calculation_functions(old) == calculation_functions(new)


def assert_exact(actual, expected, location="result"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), location
        for key in expected:
            assert_exact(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), location
        try:
            expected_array, actual_array = np.asarray(expected), np.asarray(actual)
        except ValueError:
            expected_array = np.asarray(expected, dtype=object)
        if expected_array.dtype.kind in "biufc":
            np.testing.assert_array_equal(actual_array, expected_array, err_msg=location)
        else:
            for index, (a, e) in enumerate(zip(actual, expected)):
                assert_exact(a, e, f"{location}[{index}]")
    elif isinstance(expected, float) and np.isnan(expected):
        assert isinstance(actual, float) and np.isnan(actual), location
    else:
        assert actual == expected, f"{location}: {actual!r} != {expected!r}"


def run_pair(legacy_root, tmp_path, mode, gpu_python=None):
    probe = Path(__file__).with_name("_equivalence_probe.py")
    env = dict(os.environ)
    # Fix process-level thread counts and keep all JIT caches out of original code.
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"),
               PYTHONDONTWRITEBYTECODE="1")
    results = {}
    for implementation in ("legacy", "package"):
        output = tmp_path / f"{implementation}.json"
        cmd = [sys.executable, str(probe), "--implementation", implementation,
               "--legacy-root", str(legacy_root), "--output", str(output), "--mode", mode]
        if gpu_python is not None:
            cmd += ["--gpu-python", str(gpu_python)]
        process = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True,
                                 text=True, timeout=600)
        assert process.returncode == 0, f"{implementation} failed:\n{process.stdout}\n{process.stderr}"
        results[implementation] = json.loads(output.read_text(encoding="utf-8"))
    assert_exact(results["package"], results["legacy"])


def test_numpy_end_to_end_results_identical(legacy_root, tmp_path):
    run_pair(legacy_root, tmp_path, mode="panel")


def test_gpu_results_identical(legacy_root, tmp_path, gpu_python):
    run_pair(legacy_root, tmp_path, mode="gpu", gpu_python=gpu_python)
