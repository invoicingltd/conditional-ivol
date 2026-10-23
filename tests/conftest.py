"""Optional checks against a local, unmodified research checkout."""

import os
from pathlib import Path
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--legacy-root",
        default=os.environ.get("CIVOL_LEGACY_ROOT"),
        help="Research archive directory containing thesis_method/ (optional).",
    )


@pytest.fixture(scope="session")
def legacy_root(request):
    value = request.config.getoption("--legacy-root")
    if not value:
        pytest.skip("Supply --legacy-root or CIVOL_LEGACY_ROOT for original-code comparisons")
    root = Path(value).expanduser().resolve()
    if not (root / "thesis_method" / "core" / "estimator.py").is_file():
        pytest.fail(f"No original thesis_method/core/estimator.py under {root}")
    return root


@pytest.fixture(scope="session")
def gpu_python():
    """Skip unavailable CUDA; never let a numpy fallback satisfy a GPU test."""
    from civol.core.paths import GPU_PYTHON

    if not Path(GPU_PYTHON).is_file():
        pytest.skip(f"GPU Python interpreter unavailable: {GPU_PYTHON}")
    try:
        probe = subprocess.run(
            [str(GPU_PYTHON), "-I", "-c",
             "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"GPU runtime unavailable: {exc}")
    if probe.returncode != 0:
        pytest.skip("CUDA-enabled PyTorch unavailable in CIVOL_GPU_PYTHON")
    return Path(GPU_PYTHON)
