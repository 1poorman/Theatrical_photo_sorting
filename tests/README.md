# Regression scripts

Run scripts from the project root, using the visual service environment for tests
that import Torch, OpenCV or model adapters. Each tracked script retains its
existing path; `tests/.gitignore` explicitly selects files shipped with Git.
Add a new regression to that allowlist before committing it.

`local/` holds historical `tmp_*.py` experiments and patch scripts. Their bytes
are preserved for reference; embedded historical paths are not migration commands.
Other local acceptance scripts and real-data tests are also excluded unless
explicitly selected. Generated test files belong in `outputs/` or temporary directories.
