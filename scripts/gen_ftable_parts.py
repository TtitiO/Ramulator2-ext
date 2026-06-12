#!/usr/bin/env python3
"""Generate results/ftable_parts/ — the k1-vs-k2 PIM MAC comparison.

Standalone regenerator for the ftable simulation parts.  Each (model, config)
point is simulated independently and cached as its own JSON part, so the run is
resumable: re-running only re-simulates missing parts unless --force is given.

The k1/k2 contrast is the CD-PIM vs LP-Spec comparison:
  k1 = pim_banks_per_mpu=1  -> dedicated per-bank compute unit
  k2 = pim_banks_per_mpu=2  -> one MPU time-shared across 2 banks

Lowering is per-kind (mac_mode="per_kind"), the physically faithful model:
  - weight-stationary ops (FFN/projection/MoE: stationary weight shards, broadcast
    activation) -> all-bank PIM_MAC_AB.  k2 pays 2x the AB MAC latency because each
    shared MPU walks its 2 banks serially -- this is the k1/k2 discriminator.
  - data-stationary ops (AttentionScore/Context: each bank holds a distinct
    KV-cache slice) -> per-bank PIM_MAC, identical under k1 and k2.

All-bank MAC ops are strictly serialized in the controller, so generate_and_replay
forces max_inflight_requests=1 for the per_kind path; a wider window only piles
blocked AB requests into the read buffer and slows simulation ~65x, no result change.

Usage:
  scripts/gen_ftable_parts.py                 # fill in missing parts, 6 workers
  scripts/gen_ftable_parts.py --force         # re-simulate everything
  scripts/gen_ftable_parts.py --workers 3     # cap parallelism
  scripts/gen_ftable_parts.py --aggregate     # also write ftable_pim_comparison.json

Parts land in results/ftable_parts/<model>__<k1|k2>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ramulator's frontend refuses to expand traces beyond a record cap unless this
# is lifted; the all-bank gemma prefill point alone issues ~195M MAC ops.
os.environ.setdefault("RAMULATOR_MAX_EXPANDED_RECORDS", "1000000000000")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ramulator2" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

# Kept identical to gen_figures.py FTABLE_WORKLOADS / PIM_CONFIGS so parts are
# interchangeable with the --collect ftable path.
FTABLE_WORKLOADS = (
    {"model_key": "llama2-7b",    "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "llama2-13b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "llama2-70b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "opt-125m",     "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "opt-350m",     "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "opt-1.3b",     "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "qwen25-7b",    "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "qwen25-14b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "qwen25-32b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "qwen25-72b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "gemma-2b",     "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "gemma-7b",     "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "gemma2-9b",    "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "gemma2-27b",   "phase": "decode", "past_len": 1024, "prompt_len": None},
    {"model_key": "mixtral-8x7b", "phase": "decode", "past_len": 1024, "prompt_len": None},
)

PIM_CONFIGS = {
    "k1": {"pim_banks_per_mpu": 1, "pim_mac_execution_model": "shared_mpu_serial"},
    "k2": {"pim_banks_per_mpu": 2, "pim_mac_execution_model": "shared_mpu_serial"},
}


def _part_path(parts_dir: Path, model: str, pim_label: str) -> Path:
    safe = model.replace("-", "_").replace(".", "_")
    return parts_dir / f"{safe}__{pim_label}.json"


def _run_task(task: dict) -> dict:
    """Simulate one (model, config) point and write its part atomically.

    Module-level so ProcessPoolExecutor can pickle it.
    """
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len") or 1024,
        prompt_len=task.get("prompt_len") or 12,
        materialize_weights=False,
        pim_cfg_override=task["pim_cfg_override"],
        max_inflight_requests=16,   # all-bank/per_kind path internally clamps to 1
        mac_mode="per_kind",        # FFN/proj broadcast (PIM_MAC_AB); attention per-bank (PIM_MAC)
    )
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def build_tasks(parts_dir: Path, *, force: bool) -> list[dict]:
    tasks: list[dict] = []
    for wl in FTABLE_WORKLOADS:
        for label, cfg in PIM_CONFIGS.items():
            part = _part_path(parts_dir, wl["model_key"], label)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": wl["model_key"],
                "phase": wl["phase"],
                "past_len": wl["past_len"],
                "prompt_len": wl["prompt_len"],
                "pim_cfg_override": cfg,
                "pim_label": label,
                "part_path": str(part),
            })
    return tasks


def aggregate(output_dir: Path, parts_dir: Path) -> None:
    """Fold the parts into results/ftable_pim_comparison.json (same schema as gen_figures)."""
    from lib.backend_replay import _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    rows: list[dict] = []
    for wl in FTABLE_WORKLOADS:
        model_key, phase = wl["model_key"], wl["phase"]
        # Mixtral-8x7B is generated via a dedicated function, not the model
        # registry, so get_model_spec raises for it — fall back to fixed metadata.
        try:
            spec = get_model_spec(model_key)
        except (KeyError, ValueError):
            spec = None
        if spec is not None:
            model_name = spec.name
            hidden_size = int(spec.hidden_size)
            num_layers = int(spec.num_layers)
        elif model_key == "mixtral-8x7b":
            model_name = "Mixtral-8x7B"
            hidden_size = 4096
            num_layers = 32
        else:
            model_name = model_key
            hidden_size = 0
            num_layers = 0
        k1_part = _part_path(parts_dir, model_key, "k1")
        k2_part = _part_path(parts_dir, model_key, "k2")
        if not (k1_part.exists() and k2_part.exists()):
            print(f"  WARNING: missing part(s) for {model_key}; skipping row")
            continue
        k1 = json.loads(k1_part.read_text(encoding="utf-8"))
        k2 = json.loads(k2_part.read_text(encoding="utf-8"))
        cyc1, cyc2 = int(k1["cycles"]), int(k2["cycles"])
        slowdown = (cyc2 / cyc1) if cyc1 > 0 else 0.0
        mpu_stalls_k2 = int(k2.get("pim_mpu_group_stalls", 0) or 0)
        label = f"{model_name} {phase}"
        if phase == "decode" and wl.get("past_len"):
            label += f" (past={wl['past_len']})"
        elif phase == "prefill" and wl.get("prompt_len"):
            label += f" (P={wl['prompt_len']})"
        rows.append({
            "workload": label,
            "model_key": model_key,
            "model_family": _infer_model_family(model_name),
            "phase": phase,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "cycles_k1": cyc1,
            "cycles_k2": cyc2,
            "runtime_ns_k1": float(k1["runtime_ns"]),
            "runtime_ns_k2": float(k2["runtime_ns"]),
            "slowdown": round(slowdown, 4),
            "pim_simultaneous_active_banks_peak_k1": int(k1.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_simultaneous_active_banks_peak_k2": int(k2.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_ab_mac_latency_cycles_k1": int(k1.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_ab_mac_latency_cycles_k2": int(k2.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_mpu_group_stalls_k2": mpu_stalls_k2,
            "pim_dependency_stalls_k2": int(k2.get("pim_dependency_stalls", 0) or 0),
            "pim_capacity_stalls_k2": int(k2.get("pim_capacity_stalls", 0) or 0),
            "num_bank_timing_blocked_k2": int(k2.get("num_bank_timing_blocked_cycles", 0) or 0),
            "shared_block_stall_pct": round((mpu_stalls_k2 / cyc2 * 100.0) if cyc2 > 0 else 0.0, 2),
            "pim_banks_per_mpu_k1": int(k1.get("pim_banks_per_mpu", 1) or 1),
            "pim_banks_per_mpu_k2": int(k2.get("pim_banks_per_mpu", 2) or 2),
            "replay_ok_k1": bool(k1.get("replay_ok")),
            "replay_ok_k2": bool(k2.get("replay_ok")),
        })

    out = {
        "description": "Transformer-trace PIM comparison: CD-PIM dedicated per-bank CU (k=1) vs LP-Spec 2-banks/MPU shared (k=2). Per-kind lowering: weight-stationary FFN/projection/MoE → all-bank broadcast PIM_MAC_AB (k2 pays 2x AB latency); data-stationary attention (KV per-bank slice) → per-bank PIM_MAC (k-invariant).",
        "provenance": {"date": time.strftime("%Y-%m-%d"), "generator": "scripts/gen_ftable_parts.py"},
        "rows": rows,
        "schema_version": 1,
    }
    table_path = output_dir / "ftable_pim_comparison.json"
    table_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {table_path} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate results/ftable_parts/ (k1-vs-k2 all-bank PIM MAC).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="results directory (default: <repo>/results)")
    parser.add_argument("--force", action="store_true", help="re-simulate even if a part already exists")
    parser.add_argument("--workers", type=int, default=6, help="parallel simulation workers (default 6)")
    parser.add_argument("--aggregate", action="store_true",
                        help="also fold parts into ftable_pim_comparison.json after collecting")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    output_dir = args.output_dir
    parts_dir = output_dir / "ftable_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(parts_dir, force=args.force)
    expected = len(FTABLE_WORKLOADS) * len(PIM_CONFIGS)
    cached = expected - len(tasks)
    if cached > 0:
        print(f"[ftable] {cached}/{expected} parts already cached, {len(tasks)} to simulate", flush=True)

    if tasks:
        print(f"[ftable] simulating {len(tasks)} point(s) with {args.workers} worker(s)", flush=True)
        t0 = time.time()
        if args.workers <= 1:
            for i, task in enumerate(tasks, 1):
                _run_task(task)
                print(f"[ftable] {i}/{len(tasks)}: {task['model_key']} {task['pim_label']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                future_map = {pool.submit(_run_task, t): t for t in tasks}
                for i, fut in enumerate(as_completed(future_map), 1):
                    task = future_map[fut]
                    try:
                        fut.result()
                        print(f"[ftable] {i}/{len(tasks)}: {task['model_key']} {task['pim_label']}", flush=True)
                    except Exception as exc:  # surface which point failed, keep the rest
                        print(f"[ftable] FAILED {i}/{len(tasks)}: {task['model_key']} {task['pim_label']}: {exc}", flush=True)
        print(f"[ftable] done in {time.time() - t0:.1f}s", flush=True)
    else:
        print("[ftable] nothing to do (all parts cached); use --force to re-simulate", flush=True)

    if args.aggregate:
        aggregate(output_dir, parts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
