# Ramulator2-ext

# Installation for Ramulator v2.1
- ubuntu:
```
cd ramulator2
git switch v2.1
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
```
