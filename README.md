# Ramulator2-ext

Extensions and modifications to [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).

> **Note for reviewers:** The `ramulator2/` directory contains a vendored copy of Ramulator 2.0 (v2.1 branch from `CMU-SAFARI/ramulator2`) with our modifications applied. For anonymous submission, we have flattened the git submodule and included the modified code directly in this repository. The original Ramulator 2.0 is available at [https://github.com/CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2).

# Quick Start: Build and Reproduce Results

Use the project-root virtual environment for builds, tests, and paper artifact
reproduction.

## Ubuntu Setup

The following commands assume Ubuntu 24.04 or a similar recent Ubuntu release
where `python3` is Python 3.11 or newer:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3 python3-venv python3-pip
```

Create or refresh the project-root environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e .
```

## Compile the Modified Ramulator2 Backend

Build the vendored simulator with CMake, but point CMake at the project-root
Python interpreter. This keeps the compiled Python extension and the
reproduction scripts on the same environment:

```bash
cd ramulator2
mkdir -p build
cd build
cmake -DPython_EXECUTABLE=../../.venv/bin/python ..
make -j"$(nproc)"
cd ..
../.venv/bin/python -m pip install --no-build-isolation -e .
PYTHONPATH=python ../.venv/bin/python -m ramulator codegen
cd ..
```

Sanity check the build with Ramulator's example trace:

```bash
cd ramulator2
PYTHONPATH=python ../.venv/bin/python examples/example_config.py
cd ..
```

## Reproduce the Paper Results

After the build succeeds, follow [`scripts/README.md`](scripts/README.md) to
reproduce the generated data, figures, and tables under `results/`.

## Virtual Environment Policy

`.venv/` at the repository root is the canonical environment for this project.
If you recreate it or switch Python versions, remove or recreate
`ramulator2/build/` before running CMake again so the cached
`Python_EXECUTABLE` points at the current environment.

## Docker

```bash
docker pull ubuntu:24.04
docker compose build --no-cache
docker compose up -d
docker compose exec ramulator2 bash
```
