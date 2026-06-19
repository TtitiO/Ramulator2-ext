# Reproducing the paper artifacts

The paper uses generated artifacts from `scripts/gen_figures.py`.

## Setup

First follow the Ubuntu build flow in the repository-level
[`README.md`](../README.md). In short, use the project-root `.venv/`, build
`ramulator2/` with `-DPython_EXECUTABLE=../../.venv/bin/python`, and install the
vendored Ramulator package into that same environment.

Run all reproduction commands from the repository root with the vendored
Ramulator Python sources on `PYTHONPATH`:

```bash
export PYTHONPATH="ramulator2/python:ramulator2"
```

This keeps the figure-generation dependencies and the compiled Ramulator
extension in the same environment.

## Reproduce Everything

```bash
.venv/bin/python scripts/gen_figures.py --all --workers 8
```

This collects the simulation data and renders the generated figure. Results are
written under `results/`. Cached simulation parts are reused unless a collect
command is run with `--force`.

## Reproduce Separately

Collect data only:

```bash
.venv/bin/python scripts/gen_figures.py --collect cross-model --workers 8
.venv/bin/python scripts/gen_figures.py --collect pim-sharing --workers 8
```

Render the cross-model figure from existing data:

```bash
.venv/bin/python scripts/gen_figures.py --render cross-model
```

`pim-sharing` is a table/data target, so it is produced by `--collect
pim-sharing` or `--all`; there is no separate render step for it.

Use `--force` with a collect command to regenerate cached simulation results.
