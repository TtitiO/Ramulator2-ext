#!/usr/bin/env python3
"""Collect and render LPDDR5-PIM paper figures F2-F5.

Examples:
    python scripts/gen_figures.py --collect f2
    python scripts/gen_figures.py --render f2
    python scripts/gen_figures.py --all f2
    python scripts/gen_figures.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Iterable


# ── Path setup: make the script work from a clean checkout ────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAMULATOR2_DIR = PROJECT_ROOT / "ramulator2"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(RAMULATOR2_DIR / "python"))
sys.path.insert(0, str(RAMULATOR2_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.energy import extract_pim_stats
from lib.lpddr5_pim_cfg import make_rr_cfg
from lib.runner import run_single


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_DIRNAME = "figures"
SWEEP_JSON = "f2_f3_lpddr5_pim_sweep.json"
F4_DECODE_JSON = "f4_decode_cycles.json"
F4_PREFILL_JSON = "f4_prefill_cycles.json"
F5_JSON = "f5_prefill_sweep.json"

BANK_COUNTS = (4, 8, 16, 32)
BANKS_PER_MPU = (1, 2, 4)
NOP_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20)
MODES = ("steady_state", "cold_start")
F5_PROMPT_LENGTHS = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

C_BLUE = "#6a8caf"
C_COPPER = "#c07850"
C_GREEN = "#7aa870"
C_EDGE = "#555555"
C_GRID = "0.78"
C_BPM = {1: C_BLUE, 2: C_COPPER, 4: C_GREEN}
M_BPM = {1: "o", 2: "s", 4: "^"}
L_BPM = {1: "-", 2: "--", 4: ":"}
LBL_BPM = {1: "banks/MPU = 1", 2: "banks/MPU = 2", 4: "banks/MPU = 4"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def _grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.5, alpha=0.3, color=C_GRID)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        print(f"saved {path}")
    plt.close(fig)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a rows list")
    return [row for row in rows if isinstance(row, dict)]


def _write_sweep_json(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SWEEP_JSON
    _write_json(json_path, {
        "schema_version": 1,
        "description": "LPDDR5-PIM shared-MPU sweep: throughput and latency vs bank count, banks/MPU, and NOP",
        "date": date.today().isoformat(),
        "rows": rows,
    })
    print(f"wrote {json_path}")


def _read_sweep_rows(output_dir: Path) -> list[dict]:
    json_path = output_dir / SWEEP_JSON
    if not json_path.exists():
        raise FileNotFoundError(f"missing {json_path}; run --collect f2 first")
    return _load_json_rows(json_path)


def _load_paper_figure_module():
    module_path = PROJECT_ROOT / "paper" / "scripts" / "gen_paper_figures.py"
    spec = importlib.util.spec_from_file_location("paper_gen_figures", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _time_unit_ns() -> float:
    import ramulator

    dram = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400",
        pim_enabled=True,
        pim_mode="bank",
        pim_datatype="int8",
    )
    _, timing = dram.resolve()
    return float(timing["tCK_ps"]) / 1000.0


def _row_from_stats(
    *,
    figure: str,
    active_banks: int,
    banks_per_mpu: int,
    nop: int,
    stats: dict,
    time_unit_ns: float,
) -> dict:
    extracted = extract_pim_stats(stats, time_unit_ns)
    command_counts = extracted.pop("command_counts")
    row = {
        "figure": figure,
        "active_banks": int(active_banks),
        "pim_banks_per_mpu": int(banks_per_mpu),
        "nop": int(nop),
        **extracted,
    }
    for command, count in command_counts.items():
        row[f"count_{command}"] = int(count)
    return row


def _collect_sweep_point(task: dict) -> dict:
    active_banks = int(task["active_banks"])
    bpm = int(task["banks_per_mpu"])
    nop = int(task["nop"])
    time_unit_ns = float(task["time_unit_ns"])
    stats = run_single(cfg_override=make_rr_cfg(active_banks, bpm), nop=nop, num_probes=4096, warmup=10000)
    return _row_from_stats(
        figure="f2_f3",
        active_banks=active_banks,
        banks_per_mpu=bpm,
        nop=nop,
        stats=stats,
        time_unit_ns=time_unit_ns,
    )


def collect_f2(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    """Collect the shared F2/F3 LPDDR5-PIM sweep as JSON."""
    if not force and (output_dir / SWEEP_JSON).exists():
        print(f"using existing {(output_dir / SWEEP_JSON)}")
        return

    time_unit_ns = _time_unit_ns()
    tasks = [
        {"active_banks": active_banks, "banks_per_mpu": bpm, "nop": nop, "time_unit_ns": time_unit_ns}
        for active_banks in BANK_COUNTS
        for bpm in BANKS_PER_MPU
        for nop in NOP_VALUES
    ]
    rows: list[dict] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_collect_sweep_point(task))
            print(
                f"[f2/f3] {index}/{len(tasks)}: banks={task['active_banks']} bpm={task['banks_per_mpu']} nop={task['nop']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {pool.submit(_collect_sweep_point, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                rows.append(future.result())
                print(
                    f"[f2/f3] {index}/{len(tasks)}: banks={task['active_banks']} bpm={task['banks_per_mpu']} nop={task['nop']}",
                    flush=True,
                )
    rows.sort(key=lambda row: (int(row["active_banks"]), int(row["pim_banks_per_mpu"]), int(row["nop"])))
    _write_sweep_json(rows, output_dir)


def collect_f3(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    collect_f2(output_dir, force=force, workers=workers)


def render_f2(output_dir: Path) -> None:
    rows = [row for row in _read_sweep_rows(output_dir) if int(row["nop"]) == 1]
    data: dict[int, list[tuple[int, float]]] = {bpm: [] for bpm in BANKS_PER_MPU}
    for row in rows:
        bpm = int(row["pim_banks_per_mpu"])
        if bpm in data:
            data[bpm].append((int(row["active_banks"]), float(row["request_throughput"])))
    for values in data.values():
        values.sort()

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for bpm in BANKS_PER_MPU:
        if not data[bpm]:
            continue
        xs = [point[0] for point in data[bpm]]
        ys = [point[1] for point in data[bpm]]
        ax.plot(xs, ys, marker=M_BPM[bpm], color=C_BPM[bpm], linestyle=L_BPM[bpm], lw=1.8, ms=5, label=LBL_BPM[bpm])
    ax.set_xlabel("Active banks")
    ax.set_ylabel("Throughput (requests/ns)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(BANK_COUNTS), [str(v) for v in BANK_COUNTS])
    ax.set_ylim(bottom=0)
    _grid(ax, "y")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=C_EDGE)
    fig.tight_layout()
    _save(fig, output_dir / FIGURE_DIRNAME, "f2_shared_mpu_throughput")


def render_f3(output_dir: Path) -> None:
    rows = _read_sweep_rows(output_dir)
    by_case: dict[tuple[int, int], dict[int, float]] = {}
    for row in rows:
        key = (int(row["active_banks"]), int(row["pim_banks_per_mpu"]))
        by_case.setdefault(key, {})[int(row["nop"])] = float(row["avg_pim_latency_ns"])

    banks = [bank for bank in BANK_COUNTS if any((bank, bpm) in by_case for bpm in BANKS_PER_MPU)]
    if not banks:
        raise ValueError("sweep has no F3 rows")
    fig, axes = plt.subplots(1, len(banks), figsize=(3.0 * len(banks), 3.0), sharey=False)
    if len(banks) == 1:
        axes = [axes]
    for ax, active_banks in zip(axes, banks):
        values_all: list[float] = []
        for bpm in BANKS_PER_MPU:
            series = by_case.get((active_banks, bpm), {})
            if not series:
                continue
            xs = sorted(series)
            ys = [series[nop] for nop in xs]
            values_all.extend(ys)
            ax.plot(xs, ys, marker=M_BPM[bpm], color=C_BPM[bpm], linestyle=L_BPM[bpm], lw=1.8, ms=4, label=LBL_BPM[bpm])
        ax.set_title(f"{active_banks} banks", pad=6)
        ax.set_xlabel("NOP (outstanding PIM requests)")
        if values_all:
            ylo, yhi = min(values_all), max(values_all)
            margin = max((yhi - ylo) * 0.12, yhi * 0.02, 0.2)
            ax.set_ylim(ylo - margin, yhi + margin)
        _grid(ax, "both")
    axes[0].set_ylabel("Average PIM latency (ns)")
    axes[-1].legend(loc="best", frameon=True, framealpha=0.95, edgecolor=C_EDGE)
    fig.tight_layout(pad=0.8)
    _save(fig, output_dir / FIGURE_DIRNAME, "f3_pim_latency_vs_nop")


def _run_f4_task(task: dict) -> dict:
    """Module-level worker for F4 parallel collection (must be picklable)."""
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len", 1024),
        prompt_len=task.get("prompt_len", 12),
        materialize_weights=task["materialize_weights"],
    )
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def _run_f5_task(task: dict) -> dict:
    """Module-level worker for F5 parallel collection (must be picklable)."""
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        "prefill", "llama2-7b",
        prompt_len=task["prompt_len"],
        materialize_weights=task["materialize_weights"],
    )
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def collect_f4(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    """Collect F4 cross-model decode+prefill caches with full parallelism.

    Each (model, phase, mode) is saved as a separate part file for incremental
    resumability.  Only missing parts are re-simulated on restart.
    """
    from lib.backend_replay import generate_and_replay, prefill_formula, _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    decode_path = output_dir / F4_DECODE_JSON
    prefill_path = output_dir / F4_PREFILL_JSON
    if not force and decode_path.exists() and prefill_path.exists():
        print(f"using existing {decode_path} and {prefill_path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models for decode (includes Mixtral)
    F4_DECODE_MODELS = (
        "llama2-7b", "llama2-13b", "llama2-70b",
        "opt-125m", "opt-350m", "opt-1.3b",
        "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
        "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b",
        "mixtral-8x7b",
    )
    # Models for prefill (no Mixtral — no MoE prefill path)
    F4_PREFILL_MODELS = (
        "llama2-7b", "llama2-13b", "llama2-70b",
        "opt-125m", "opt-350m", "opt-1.3b",
        "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
        "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b",
    )
    F4_PREFILL_PROMPT_LEN = 12
    DECODE_PAST_LEN = 1024  # all models use past_len=1024 for the cross-model comparison

    parts_dir = output_dir / "f4_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(model: str, phase: str, mode: str) -> Path:
        safe = model.replace("-", "_").replace(".", "_")
        return parts_dir / f"{safe}__{phase}__{mode}.json"

    # Build task list for all (model, phase, mode) combinations
    tasks: list[dict] = []
    for model in F4_DECODE_MODELS:
        for mode in MODES:
            part = _part_path(model, "decode", mode)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": model, "phase": "decode", "mode": mode,
                "past_len": DECODE_PAST_LEN,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part),
            })
    for model in F4_PREFILL_MODELS:
        for mode in MODES:
            part = _part_path(model, "prefill", mode)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": model, "phase": "prefill", "mode": mode,
                "prompt_len": F4_PREFILL_PROMPT_LEN,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part),
            })

    total = len(tasks)
    cached = (len(F4_DECODE_MODELS) * 2 + len(F4_PREFILL_MODELS) * 2) - total
    if cached > 0:
        print(f"[f4] {cached} parts already cached, {total} remaining", flush=True)
    if total > 0:
        print(f"[f4] collecting {total} simulation points with {workers} workers", flush=True)

    if total > 0:
        if workers <= 1:
            for idx, task in enumerate(tasks, 1):
                _run_f4_task(task)
                print(f"[f4] {idx}/{total}: {task['model_key']} {task['phase']} {task['mode']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(_run_f4_task, t): t for t in tasks}
                for idx, future in enumerate(as_completed(future_map), 1):
                    task = future_map[future]
                    try:
                        future.result()
                        print(f"[f4] {idx}/{total}: {task['model_key']} {task['phase']} {task['mode']}", flush=True)
                    except Exception as exc:
                        print(f"[f4] FAILED {idx}/{total}: {task['model_key']} {task['phase']} {task['mode']}: {exc}", flush=True)

    # Assemble decode cache
    decode_rows: list[dict] = []
    for model in F4_DECODE_MODELS:
        spec = get_model_spec(model) if model != "mixtral-8x7b" else None
        for mode in MODES:
            part = _part_path(model, "decode", mode)
            if not part.exists():
                print(f"  WARNING: missing decode part {part}")
                continue
            data = json.loads(part.read_text(encoding="utf-8"))
            model_name = spec.name if spec else "Mixtral-8x7B"
            decode_rows.append({
                "model_name": model_name,
                "model_family": _infer_model_family(model_name),
                "mode": mode,
                "cycles": int(data["cycles"]),
                "runtime_ns": float(data["runtime_ns"]),
                "runtime_s": float(data["runtime_ns"]) / 1e9,
                "pim_mac_issued": int(data["pim_mac_issued"]),
                "hidden_size": int(spec.hidden_size) if spec else 4096,
                "ffn_hidden_size": int(spec.ffn_hidden_size) if spec else 14336,
                "num_layers": int(spec.num_layers) if spec else 32,
                "replay_status": "PASS" if data.get("replay_ok") else "FAIL",
                "data_source": "backend_replay",
                "dimension_scope": "real",
                "source_cache": str(part),
            })
    decode_payload = {
        "figure_id": "fig18_cross_model_decode_cycles",
        "description": "Cross-model dense decode backend replay cycles",
        "phase": "decode",
        "metric_units": {"cycles": "cycles", "runtime_ns": "ns"},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": decode_rows,
    }
    _write_json(decode_path, decode_payload)
    print(f"wrote {decode_path} ({len(decode_rows)} rows)")

    # Assemble prefill cache
    prefill_rows: list[dict] = []
    for model in F4_PREFILL_MODELS:
        formula = prefill_formula(model, prompt_len=F4_PREFILL_PROMPT_LEN)
        spec = get_model_spec(model)
        for mode in MODES:
            part = _part_path(model, "prefill", mode)
            if not part.exists():
                print(f"  WARNING: missing prefill part {part}")
                continue
            data = json.loads(part.read_text(encoding="utf-8"))
            prefill_rows.append({
                "model_name": spec.name,
                "model_family": _infer_model_family(spec.name),
                "model_key": model,
                "mode": mode,
                "cycles": int(data["cycles"]),
                "runtime_ns": float(data["runtime_ns"]),
                "runtime_s": float(data["runtime_ns"]) / 1e9,
                "pim_mac_issued": int(data["pim_mac_issued"]),
                "pim_bcast_issued": int(data.get("pim_bcast_issued", 0)),
                "prompt_len": F4_PREFILL_PROMPT_LEN,
                "replay_layers": int(spec.num_layers),
                "replay_status": "PASS" if data.get("replay_ok") else "FAIL",
                "data_source": "backend_replay",
                "dimension_scope": "real",
                **{k: formula[k] for k in (
                    "hidden_size", "ffn_hidden_size", "ffn_variant", "activation",
                    "num_heads", "num_kv_heads", "head_dim", "datatype", "citation",
                    "seq_len", "prefill_causal_pairs", "valid_attention_pairs_per_layer",
                    "attention_issued_work_elements_per_layer", "score_tile_tokens",
                    "context_tile_tokens", "pim_mac_lanes", "primitive_ops_per_mac",
                    "per_layer_pim_mac_buckets", "kv_residency_policy",
                    "model_total_layers",
                )},
                "phase": "prefill",
                "materialize_weights": mode == "cold_start",
                "trace_name": f"{model}_prefill_P{F4_PREFILL_PROMPT_LEN}_{mode}",
                "command_counts": data.get("opcode_counts", {}),
                "pim_mac_density": 0.0,  # placeholder
            })
    prefill_payload = {
        "schema_version": 1,
        "figure_id": "fig22_cross_model_prefill_cycles",
        "description": "Cross-model dense prefill backend replay cycles",
        "phase": "prefill",
        "metric_units": {"cycles": "cycles", "runtime_ns": "ns"},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py",
                       "prompt_len": F4_PREFILL_PROMPT_LEN},
        "rows": prefill_rows,
        "caveats": ["Simulator-diagnostic cycles, not silicon-calibrated"],
    }
    _write_json(prefill_path, prefill_payload)
    print(f"wrote {prefill_path} ({len(prefill_rows)} rows)")


def render_f4(output_dir: Path) -> None:
    """Render F4 with paper/scripts/gen_paper_figures.py's renderer."""
    paper_figures = _load_paper_figure_module()
    paper_figures.CROSS_MODEL_DECODE_CACHE = output_dir / F4_DECODE_JSON
    paper_figures.CROSS_MODEL_PREFILL_CACHE = output_dir / F4_PREFILL_JSON
    paper_figures.gen_f4(output_dir / FIGURE_DIRNAME)


def collect_f5(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    """Collect F5 Llama2-7B prefill prompt-sweep with incremental caching.

    Each (prompt_len, mode) is saved as a part file for resumability.
    """
    from lib.backend_replay import generate_and_replay, prefill_formula

    path = output_dir / F5_JSON
    if not force and path.exists():
        print(f"using existing {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    parts_dir = output_dir / "f5_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(pl: int, mode: str) -> Path:
        return parts_dir / f"llama2_7b__P{pl}__{mode}.json"

    # Build task list
    tasks: list[dict] = []
    for pl in F5_PROMPT_LENGTHS:
        for mode in MODES:
            part = _part_path(pl, mode)
            if not force and part.exists():
                continue
            tasks.append({
                "prompt_len": pl, "mode": mode,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part),
            })

    total = len(tasks)
    cached = len(F5_PROMPT_LENGTHS) * len(MODES) - total
    if cached > 0:
        print(f"[f5] {cached} parts already cached, {total} remaining", flush=True)
    if total > 0:
        print(f"[f5] collecting {total} simulation points with {workers} workers", flush=True)

    if total > 0:
        if workers <= 1:
            for idx, task in enumerate(tasks, 1):
                _run_f5_task(task)
                print(f"[f5] {idx}/{total}: P={task['prompt_len']} {task['mode']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(_run_f5_task, t): t for t in tasks}
                for idx, future in enumerate(as_completed(future_map), 1):
                    task = future_map[future]
                    try:
                        future.result()
                        print(f"[f5] {idx}/{total}: P={task['prompt_len']} {task['mode']}", flush=True)
                    except Exception as exc:
                        print(f"[f5] FAILED {idx}/{total}: P={task['prompt_len']} {task['mode']}: {exc}", flush=True)

    # Assemble F5 cache
    rows: list[dict] = []
    for pl in F5_PROMPT_LENGTHS:
        formula = prefill_formula("llama2-7b", prompt_len=pl)
        buckets = formula["per_layer_pim_mac_buckets"]
        nl = formula["model_total_layers"]
        total_pim_mac_per_layer = sum(buckets.values())

        for mode in MODES:
            part = _part_path(pl, mode)
            if not part.exists():
                print(f"  WARNING: missing F5 part {part}")
                continue
            data = json.loads(part.read_text(encoding="utf-8"))
            pim_mac = int(data["pim_mac_issued"])
            pim_bcast = int(data.get("pim_bcast_issued", 0))
            cycles = int(data["cycles"])
            runtime_ns = float(data["runtime_ns"])
            # pim_mac_density = pim_mac_cycles / total_cycles (approximate)
            density = (pim_mac * 4) / cycles if cycles > 0 else 0.0  # 4 = mac_interval

            rows.append({
                "prompt_len": pl,
                "mode": mode,
                "status": "PASS" if data.get("replay_ok") else "FAIL",
                "runtime_ns": runtime_ns,
                "cycles": cycles,
                "pim_mac": total_pim_mac_per_layer * nl,  # analytical total
                "pim_bcast": pim_bcast,
                "pim_mac_density": density,
                "prefill_causal_pairs": formula["prefill_causal_pairs"],
                "valid_attention_pairs_per_layer": formula["valid_attention_pairs_per_layer"],
                "attention_issued_work_elements_per_layer": formula["attention_issued_work_elements_per_layer"],
                "per_layer_pim_mac_buckets": buckets,
                "num_layers": nl,
                "model": "Llama2-7B",
                "phase": "prefill",
                "seq_len": pl,
                "datatype": "int8",
                "materialize_weights": mode == "cold_start",
                "controller_pim_mac_issued": pim_mac,
                "concrete_opcode_counts_replay_input": data.get("opcode_counts", {}),
                "avg_pim_latency_cycles": 0,  # not tracked in this replay mode
            })

    payload = {
        "schema_version": 1,
        "figure_id": "fig21_llama2_7b_prefill_prompt_sweep",
        "description": "Llama2-7B full-depth prefill backend replay across prompt lengths",
        "model": "Llama2-7B",
        "phase": "prefill",
        "sweep": {"prompt_len_values": list(F5_PROMPT_LENGTHS), "modes": list(MODES)},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": rows,
        "caveats": ["Simulator-diagnostic cycles, not silicon-calibrated"],
    }
    _write_json(path, payload)
    print(f"wrote {path} ({len(rows)} rows)")


def render_f5(output_dir: Path) -> None:
    """Render F5 with paper/scripts/gen_paper_figures.py's renderer."""
    paper_figures = _load_paper_figure_module()
    paper_figures.PREFILL_PROMPT_SWEEP_CACHE = output_dir / F5_JSON
    paper_figures.gen_f5(output_dir / FIGURE_DIRNAME)


COLLECTORS = {"f2": collect_f2, "f3": collect_f3, "f4": collect_f4, "f5": collect_f5}
RENDERERS = {"f2": render_f2, "f3": render_f3, "f4": render_f4, "f5": render_f5}


def _expand_target(target: str | None) -> list[str]:
    if target in (None, "all"):
        return ["f2", "f3", "f4", "f5"]
    target = target.lower()
    if target not in COLLECTORS:
        raise ValueError(f"unknown figure target: {target}")
    return [target]


def _dedup_collect_targets(targets: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen_sweep = False
    for target in targets:
        if target in {"f2", "f3"}:
            if seen_sweep:
                continue
            out.append("f2")
            seen_sweep = True
        else:
            out.append(target)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and render LPDDR5-PIM F2-F5 artifacts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--collect", nargs="?", const="all", choices=["f2", "f3", "f4", "f5", "all"], help="collect data for a figure")
    group.add_argument("--render", nargs="?", const="all", choices=["f2", "f3", "f4", "f5", "all"], help="render a figure from cached data")
    group.add_argument("--all", nargs="?", const="all", choices=["f2", "f3", "f4", "f5", "all"], help="collect and render")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="recompute existing data artifacts")
    parser.add_argument("--workers", type=int, default=4, help="parallel collection workers")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    _apply_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.collect is not None:
        for target in _dedup_collect_targets(_expand_target(args.collect)):
            COLLECTORS[target](args.output_dir, force=args.force, workers=args.workers)
        return 0

    if args.render is not None:
        for target in _expand_target(args.render):
            RENDERERS[target](args.output_dir)
        return 0

    for target in _dedup_collect_targets(_expand_target(args.all)):
        COLLECTORS[target](args.output_dir, force=args.force, workers=args.workers)
    for target in _expand_target(args.all):
        RENDERERS[target](args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
