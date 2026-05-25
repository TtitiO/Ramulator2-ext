# LPDDR5-PIM figure generation

Collect and render all 4 paper-facing figures (F2–F5).

## Quick start

```bash
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --all --workers 8
```

All commands require `PYTHONPATH="ramulator2/python:ramulator2"` (or export it once).

## Common workflows

```bash
# F2/F3 — shared-MPU sweep
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --collect f2 --workers 4
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --render  f2

# F4 — cross-model decode+prefill cycles
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --collect f4 --workers 8
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --render  f4

# F5 — prefill attention scaling
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --collect f5 --workers 8
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --render  f5

# Collect everything, then render
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --collect all --workers 8
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --render  all
```

## Output files

| Figure | Collection output | Rendered to |
|--------|-------------------|-------------|
| F2 | `results/f2_f3_lpddr5_pim_sweep.json` | `results/figures/f2_shared_mpu_throughput.{pdf,png}` |
| F3 | (same sweep JSON) | `results/figures/f3_pim_latency_vs_nop.{pdf,png}` |
| F4 | `results/f4_decode_cycles.json` + `results/f4_prefill_cycles.json` | `results/figures/f4_cross_model_cycles.{pdf,png}` |
| F5 | `results/f5_prefill_sweep.json` | `results/figures/f5_prefill_attention_scaling.{pdf,png}` |

All outputs go under `results/`, which is gitignored.

## Incremental caching (resumable)

F4 and F5 use **per-task part files** for incremental resumability:

```
results/
  f4_parts/           # one JSON per (model, phase, mode)
    opt_125m__decode__steady_state.json
    llama2_7b__prefill__cold_start.json
    mixtral_8x7b__decode__steady_state.json
    ...
  f5_parts/           # one JSON per (prompt_len, mode)
    llama2_7b__P2__steady_state.json
    llama2_7b__P128__cold_start.json
    ...
```

- Re-running `--collect` skips completed parts automatically.
- Use `--force` to regenerate everything from scratch.
- Kill and restart safely — no progress is lost.

## How many simulation points?

| Figure | Models | Modes | Total tasks |
|--------|--------|-------|-------------|
| F2/F3 | 4 bank counts × 3 BPM × 12 NOP | — | 144 (F2 uses 12 NOP=1 rows) |
| F4 decode | 15 models (Llama2, OPT, Qwen2.5, Gemma, Mixtral) | steady + cold | 30 |
| F4 prefill | 14 models (dense only, no Mixtral) | steady + cold | 28 |
| F5 | 10 prompt lengths (2,4,8,…,128) | steady + cold | 20 |

With 8 workers, F4 completes in 30–60 min and F5 in 15–30 min. Large models
(Llama2-70B, Qwen2.5-72B, prefill P=128) dominate runtime.

## Verification without simulation

Compile-check and render from existing paper caches:

```bash
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python -m py_compile scripts/gen_figures.py scripts/lib/*.py
PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python scripts/gen_figures.py --help
```

## Dependencies

The script depends only on:

- `ramulator2/python/ramulator` — Ramulator Python bindings (auto-generated from C++)
- `scripts/lib/` — local helpers (`runner.py`, `backend_replay.py`, etc.)
- `matplotlib`, `numpy` — rendering

