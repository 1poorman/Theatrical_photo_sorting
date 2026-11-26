# Regression scripts

Run scripts from the project root, using the visual service environment for tests
that import Torch, OpenCV or model adapters. Each tracked script retains its
existing path; `tests/.gitignore` explicitly selects files shipped with Git.
Add a new regression to that allowlist before committing it.

Agent graph checks run with `python tests/test_langchain_agent.py`. Shared
contracts, durable task integration, project memory and budgeted image previews
run with `python tests/test_agent_optimization.py` in an environment containing
both `requirements/agent.txt` and the visual/API dependencies. Live model smoke
checks use `tools/validation/evaluate_photo_agent.py` with a fresh output directory;
they do not establish scene or identity accuracy.

`python tests/test_agent_storage.py` checks one-time legacy imports, incremental
checkpoints, stable cursors and runtime idempotency using temporary databases.
`node tests/test_workspace_chat_browser.cjs` runs the actual workspace page in
Chromium against local HTTP fixtures, including project switching during a reply.
Use `PLAYWRIGHT_PATH` / `CHROMIUM_PATH` to select local browser dependencies.

The agent API integration check keeps real chat routes and middleware, isolating
only unrelated GPU-server, image-search and age-model imports. It sends in-process
ASGI requests without model initialization. The normal visual `test_workspace.py`
separately covers the full main-module import. If Python 3.11 thread wakeups stall
in a restricted sandbox, repeat the bounded offline check on the host; do not
count an incomplete sandbox run as passed.

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
