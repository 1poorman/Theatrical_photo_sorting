# Regression scripts

Run scripts from the project root, using the visual service environment for tests
that import Torch, OpenCV or model adapters. Each tracked script retains its
existing path; `tests/.gitignore` explicitly selects files shipped with Git.
Add a new regression to that allowlist before committing it.

`local/` holds historical `tmp_*.py` experiments and patch scripts. Their bytes
are preserved for reference; embedded historical paths are not migration commands.
Other local acceptance scripts and real-data tests are also excluded unless
explicitly selected. Generated test files belong in `outputs/` or temporary directories.

SwinFace format/preprocessing checks run with `python tests/test_swinface.py` and
`python tests/test_swinface_compact.py`. With the local official checkpoint and
aligned faces installed, set `RUN_SWINFACE_REAL=1` or `RUN_SWINFACE_COMPACT_REAL=1`
respectively for numerical and export/reload checks. `SWINFACE_TEST_DEVICE=cuda:1`
selects a CUDA device; dynamic INT8 is checked only on CPU. The full 109-face
compression comparison is reproduced by `tools/face/benchmark_swinface_quantization.py`;
see the [report](../docs/reports/swinface_quantization_20261111/README.md) for commands.
