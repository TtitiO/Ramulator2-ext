# Figure generation

```bash
export PYTHONPATH="ramulator2/python:ramulator2"

# All figures (collect + render)
.venv/bin/python scripts/gen_figures.py --all --workers 8

# Collect only
.venv/bin/python scripts/gen_figures.py --collect f2 --workers 4
.venv/bin/python scripts/gen_figures.py --collect f4 --workers 8
.venv/bin/python scripts/gen_figures.py --collect f5 --workers 8

# Render from cached data
.venv/bin/python scripts/gen_figures.py --render f2
.venv/bin/python scripts/gen_figures.py --render all
```

## Output files

| Figure | Data file | Rendered output |
|--------|----------|-----------------|
| F2 | `results/f2_f3_lpddr5_pim_sweep.json` | `results/figures/f2_pim_block_throughput.{pdf,png}` |
| F3 | (same as F2) | `results/figures/f3_pim_latency_vs_nop.{pdf,png}` |
| F4 | `results/f4_decode_cycles.json` + `f4_prefill_cycles.json` | `results/figures/f4_cross_model_cycles.{pdf,png}` |
| F5 | `results/f5_prefill_sweep.json` | `results/figures/f5_prefill_attention_scaling.{pdf,png}` |
| F6 | `results/f6_parameter_sensitivity.json` | `results/figures/f6_parameter_sensitivity.{pdf,png}` |
| ftable | `results/ftable_pim_comparison.json` | (table, no render) |

## Incremental caching

F4, F5, and ftable use per-task part files. Re-running `--collect` skips completed parts. Use `--force` to regenerate everything.

```
results/
  f4_parts/      # one JSON per (model, phase, mode)
  f5_parts/      # one JSON per (prompt_len, mode)
  ftable_parts/  # one JSON per (model, pim_config)
```

## Simulation point counts

| Figure | Data points |
|--------|-------------|
| F2/F3 | 4 bank counts × 3 BPM × 12 NOP = 144 (F2 uses 12 NOP=1 rows) |
| F4 decode | 15 models × 2 modes = 30 |
| F4 prefill | 14 models × 2 modes = 28 |
| F5 | 10 prompt lengths × 2 modes = 20 |
| F6 | 4 timing + 5 energy + 3 bank_mapping = 12 |
| ftable | 15 models × 2 PIM configs = 30 |

With 8 workers, F4 ~30–60 min, F5 ~15–30 min (large models dominate).

## Dependencies

- `ramulator2/python/ramulator` (auto-generated Python bindings from C++ build)
- `scripts/lib/` (local helpers: `runner.py`, `backend_replay.py`, `energy.py`, `lpddr5_pim_cfg.py`)
- `matplotlib`, `numpy`
