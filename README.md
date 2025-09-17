# Jan Incubator Audio

This repository is for POC of `audio` and `realtime` endpoints.

## Development Setup

Make sure `uv` is installed

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment

```bash
uv venv --python=3.11 --managed-python
source .venv/bin/activate

```

Setup pre-commit hooks

```bash
uv pip install pre-commit
pre-commit install  # install pre-commit hooks

pre-commit  # manually run pre-commit
pre-commit run --all-files  # if you forgot to install pre-commit previously
```
