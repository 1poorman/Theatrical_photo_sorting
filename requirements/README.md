# Dependency sets

Run commands from the project root. `vision.txt` installs the visual API;
`environment.yml` defines its Conda environment. Optional components use
`agent.txt` (conversation tools), `embeddinggemma.txt` (experimental text model),
and `dali.txt` (isolated preprocessing evaluation).

```bash
conda env create -f requirements/environment.yml
python -m pip install -r requirements/vision.txt
python -m pip install -r requirements/agent.txt
```

Conda resolves the `vision.txt` reference relative to `environment.yml`.
DALI remains an optional experiment; see the preprocessing guide before installing it.

The visual API includes `jsonschema` for both agent entry points' shared argument
validation. Advanced LangChain chat additionally requires `agent.txt`. Install
these dependencies in one Python environment; do not combine site-packages from
different Python minor versions.
