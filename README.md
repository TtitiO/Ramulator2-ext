# Ramulator2-ext

Extensions and modifications to [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).

> **Note for reviewers:** The `ramulator2/` directory contains a vendored copy of Ramulator 2.0 (v2.1 branch from `CMU-SAFARI/ramulator2`) with our modifications applied. For anonymous submission, we have flattened the git submodule and included the modified code directly in this repository. The original Ramulator 2.0 is available at [https://github.com/CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2).

# Installation for Ramulator v2.1
- ubuntu:
```
cd ramulator2
mkdir -p build
cd build
cmake -DPython_EXECUTABLE=<YOUR-Ramulator2-ext-path>/.venv/bin/python ..
make -j$(nproc)
cd ..
uv pip install setuptools
uv pip install --no-build-isolation -e .
```
- docker:
```
docker pull ubuntu:24.04
docker compose build --no-cache
docker compose up -d
docker compose exec ramulator2 bash
```
