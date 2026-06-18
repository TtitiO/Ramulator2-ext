# Reproducing the paper artifacts

The paper uses generated artifacts from `scripts/gen_figures.py`.

## Setup

Build Ramulator2 and make sure the local virtual environment exists, then run
commands from the repository root:

```bash
export PYTHONPATH="ramulator2/python:ramulator2"
```

## Reproduce Everything

```bash
.venv/bin/python scripts/gen_figures.py --all --workers 8
```

This collects the simulation data and renders the generated figure. Results are
written under `results/`.

## Reproduce Separately

Collect data only:

```bash
.venv/bin/python scripts/gen_figures.py --collect cross-model --workers 8
.venv/bin/python scripts/gen_figures.py --collect pim-sharing --workers 8
```

Render the figure from existing data:

```bash
.venv/bin/python scripts/gen_figures.py --render cross-model
```

Use `--force` with a collect command to regenerate cached simulation results.
