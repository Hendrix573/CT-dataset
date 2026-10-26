"""Small local end-to-end training check; command-line arguments override defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# This Anaconda installation loads two Intel OpenMP runtimes. Keep the workaround
# local to this smoke-test entry point; cloud training should use a clean environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", str(Path("outputs/local_cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path("outputs/local_cache/torch").resolve()))

from train import main  # noqa: E402


if __name__ == "__main__":
    defaults = [
        "--data-root", "cxr_first",
        "--output-dir", "outputs/dev_multiscale_smoke",
        "--epochs", "1",
        "--image-size", "64",
        "--local-canvas-size", "128",
        "--local-token-grid", "2",
        "--progressive-augment",
        "--batch-size", "1",
        "--accum-studies", "2",
        "--train-eval-interval", "1",
        "--workers", "0",
        "--device", "cpu",
        "--max-train-studies", "2",
        "--max-valid-studies", "2",
        "--max-images-per-study", "1",
    ]
    main(defaults + sys.argv[1:])
