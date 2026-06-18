#!/usr/bin/env python3
"""Collect and render LPDDR5-PIM paper figures F2-F6 and the ftable comparison.

Usage:
    python scripts/gen_figures.py --collect f2
    python scripts/gen_figures.py --render f4
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAMULATOR2_DIR = PROJECT_ROOT / "ramulator2"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(RAMULATOR2_DIR / "python"))
sys.path.insert(0, str(RAMULATOR2_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from lib.energy import extract_pim_stats, _nested
from lib.lpddr5_pim_cfg import make_rr_cfg
from lib.runner import run_single


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_DIRNAME = "figures"
SWEEP_JSON = "f2_f3_lpddr5_pim_sweep.json"
F4_DECODE_JSON = "f4_decode_cycles.json"
F4_PREFILL_JSON = "f4_prefill_cycles.json"
F5_JSON = "f5_prefill_sweep.json"
F6_JSON = "f6_parameter_sensitivity.json"
FTABLE_JSON = "ftable_pim_comparison.json"

BANK_COUNTS = (4, 8, 16, 32)
BANKS_PER_MPU = (1, 2, 4)
NOP_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20)
MODES = ("steady_state", "cold_start")
F5_PROMPT_LENGTHS = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
F6_TIMING_VALUES = (4, 8, 12, 16)
F6_TIMING_BASE_VALUE = 8
F6_TIMING_ISOLATION_LAT = 1
F6_E_COMP_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
F6_BANKS_PER_MPU = (1, 2, 4)
F6_ACTIVE_BANKS = 16
F6_TIMING_ACTIVE_BANKS = 1
F6_BASE_BANKS_PER_MPU = 1
F6_NUM_PIM_REQUESTS = 4096
F6_BASE_NOP = 1
F6_ROOFLINE_READS = 200_000

C_BLUE = "#6a8caf"
C_COPPER = "#c07850"
C_GREEN = "#7aa870"
C_EDGE = "#555555"
C_GRID = "0.78"
C_ANNOT = "0.35"
C_REF = "#888888"
C_BPM = {1: C_BLUE, 2: C_COPPER, 4: C_GREEN}
M_BPM = {1: "o", 2: "s", 4: "^"}
L_BPM = {1: "-", 2: "--", 4: ":"}
LBL_BPM = {1: "1 bank / PIM block", 2: "2 banks / PIM block", 4: "4 banks / PIM block"}


def _apply_style() -> None:
    plt.rcParams.update({
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
    })


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
        "description": "LPDDR5-PIM PIM-block mapping sweep",
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
        pim_enabled=True, pim_mode="bank", pim_datatype="int8")
    _, timing = dram.resolve()
    return float(timing["tCK_ps"]) / 1000.0


def _row_from_stats(*, figure: str, active_banks: int, banks_per_mpu: int,
                    nop: int, stats: dict, time_unit_ns: float) -> dict:
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
    stats = run_single(
        cfg_override=make_rr_cfg(active_banks, bpm),
        nop=nop, num_probes=4096, warmup=10000)
    return _row_from_stats(
        figure="f2_f3", active_banks=active_banks, banks_per_mpu=bpm,
        nop=nop, stats=stats, time_unit_ns=time_unit_ns)


# F2 / F3 — shared-MPU sweep -------------------------------------------------

def collect_f2(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    if not force and (output_dir / SWEEP_JSON).exists():
        print(f"using existing {output_dir / SWEEP_JSON}")
        return

    time_unit_ns = _time_unit_ns()
    tasks = [
        {"active_banks": ab, "banks_per_mpu": bpm, "nop": nop, "time_unit_ns": time_unit_ns}
        for ab in BANK_COUNTS for bpm in BANKS_PER_MPU for nop in NOP_VALUES]
    rows: list[dict] = []

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_collect_sweep_point(task))
            print(f"[f2/f3] {index}/{len(tasks)}: banks={task['active_banks']} "
                  f"bpm={task['banks_per_mpu']} nop={task['nop']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {pool.submit(_collect_sweep_point, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                rows.append(future.result())
                print(f"[f2/f3] {index}/{len(tasks)}: banks={task['active_banks']} "
                      f"bpm={task['banks_per_mpu']} nop={task['nop']}", flush=True)

    rows.sort(key=lambda r: (int(r["active_banks"]), int(r["pim_banks_per_mpu"]), int(r["nop"])))
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
        xs = [p[0] for p in data[bpm]]
        ys = [p[1] for p in data[bpm]]
        ax.plot(xs, ys, marker=M_BPM[bpm], color=C_BPM[bpm], linestyle=L_BPM[bpm],
                lw=1.8, ms=5, label=LBL_BPM[bpm])
    ax.set_xlabel("Active banks")
    ax.set_ylabel("Throughput (requests/ns)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(BANK_COUNTS), [str(v) for v in BANK_COUNTS])
    ax.set_ylim(bottom=0)
    _grid(ax, "y")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=C_EDGE)
    fig.tight_layout()
    _save(fig, output_dir / FIGURE_DIRNAME, "f2_pim_block_throughput")


def render_f3(output_dir: Path) -> None:
    rows = _read_sweep_rows(output_dir)
    by_case: dict[tuple[int, int], dict[int, float]] = {}
    for row in rows:
        key = (int(row["active_banks"]), int(row["pim_banks_per_mpu"]))
        by_case.setdefault(key, {})[int(row["nop"])] = float(row["avg_pim_latency_ns"])

    banks = [b for b in BANK_COUNTS if any((b, bpm) in by_case for bpm in BANKS_PER_MPU)]
    if not banks:
        raise ValueError("sweep has no F3 rows")
    ncols = 2 if len(banks) > 1 else 1
    nrows = (len(banks) + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.8 * nrows), sharey=False)
    axes_flat = list(getattr(axes_grid, "flat", [axes_grid]))

    for ax, active_banks in zip(axes_flat, banks):
        values_all: list[float] = []
        for bpm in BANKS_PER_MPU:
            series = by_case.get((active_banks, bpm), {})
            if not series:
                continue
            xs = sorted(series)
            ys = [series[nop] for nop in xs]
            values_all.extend(ys)
            ax.plot(xs, ys, marker=M_BPM[bpm], color=C_BPM[bpm], linestyle=L_BPM[bpm],
                    lw=1.8, ms=4, label=LBL_BPM[bpm])
        ax.set_title(f"{active_banks} banks", pad=6)
        ax.set_xlabel("NOP (outstanding PIM requests)")
        if values_all:
            ylo, yhi = min(values_all), max(values_all)
            margin = max((yhi - ylo) * 0.12, yhi * 0.02, 0.2)
            ax.set_ylim(ylo - margin, yhi + margin)
        _grid(ax, "both")

    axes_flat[0].set_ylabel("Average PIM latency (ns)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    for ax in axes_flat[len(banks):]:
        ax.axis("off")
    if len(axes_flat) > len(banks):
        axes_flat[len(banks)].legend(handles, labels, loc="center", frameon=True,
                                      framealpha=0.95, edgecolor=C_EDGE)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, framealpha=0.95, edgecolor=C_EDGE)
    fig.tight_layout(pad=0.9)
    _save(fig, output_dir / FIGURE_DIRNAME, "f3_pim_latency_vs_nop")


# ── F4 — cross-model decode + prefill cycles ────────────────────────────

def _run_f4_task(task: dict) -> dict:
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len", 1024),
        prompt_len=task.get("prompt_len", 12),
        materialize_weights=task["materialize_weights"],
        pim_cfg_override=task.get("pim_cfg_override"),
        max_inflight_requests=task.get("max_inflight_requests", 1),
        mac_mode=task.get("mac_mode", "per_kind"))
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def collect_f4(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    from lib.backend_replay import generate_and_replay, prefill_formula, _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    decode_path = output_dir / F4_DECODE_JSON
    prefill_path = output_dir / F4_PREFILL_JSON
    if not force and decode_path.exists() and prefill_path.exists():
        print(f"using existing {decode_path} and {prefill_path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    F4_DECODE_MODELS = (
        "llama2-7b", "llama2-13b", "llama2-70b",
        "opt-125m", "opt-350m", "opt-1.3b",
        "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
        "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b",
        "mixtral-8x7b")
    F4_PREFILL_MODELS = (
        "llama2-7b", "llama2-13b", "llama2-70b",
        "opt-125m", "opt-350m", "opt-1.3b",
        "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
        "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b")
    F4_PREFILL_PROMPT_LEN = 12
    DECODE_PAST_LEN = 1024

    parts_dir = output_dir / "f4_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(model: str, phase: str, mode: str) -> Path:
        safe = model.replace("-", "_").replace(".", "_")
        return parts_dir / f"{safe}__{phase}__{mode}.json"

    from lib.backend_replay import pim_cfg_shared
    f4_pim_cfg = pim_cfg_shared()
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
                "part_path": str(part), "pim_cfg_override": f4_pim_cfg,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})

    for model in F4_PREFILL_MODELS:
        for mode in MODES:
            part = _part_path(model, "prefill", mode)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": model, "phase": "prefill", "mode": mode,
                "prompt_len": F4_PREFILL_PROMPT_LEN,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part), "pim_cfg_override": f4_pim_cfg,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})

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

    # Assemble decode rows
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
                "source_cache": str(part)})
    _write_json(decode_path, {
        "figure_id": "fig18_cross_model_decode_cycles",
        "description": "Cross-model dense decode backend replay cycles",
        "phase": "decode",
        "metric_units": {"cycles": "cycles", "runtime_ns": "ns"},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": decode_rows})
    print(f"wrote {decode_path} ({len(decode_rows)} rows)")

    # Assemble prefill rows
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
                    "model_total_layers")},
                "phase": "prefill",
                "materialize_weights": mode == "cold_start",
                "trace_name": f"{model}_prefill_P{F4_PREFILL_PROMPT_LEN}_{mode}",
                "command_counts": data.get("opcode_counts", {}),
                "pim_mac_density": 0.0})
    _write_json(prefill_path, {
        "schema_version": 1,
        "figure_id": "fig22_cross_model_prefill_cycles",
        "description": "Cross-model dense prefill backend replay cycles",
        "phase": "prefill",
        "metric_units": {"cycles": "cycles", "runtime_ns": "ns"},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py",
                       "prompt_len": F4_PREFILL_PROMPT_LEN},
        "rows": prefill_rows,
        "caveats": ["Simulator-diagnostic cycles, not silicon-calibrated"]})
    print(f"wrote {prefill_path} ({len(prefill_rows)} rows)")


def render_f4(output_dir: Path) -> None:
    """Render F4 cross-model cycles. Falls back to standalone renderer when
    paper/scripts/gen_paper_figures.py is unavailable."""
    try:
        paper_figures = _load_paper_figure_module()
        paper_figures.CROSS_MODEL_DECODE_CACHE = output_dir / F4_DECODE_JSON
        paper_figures.CROSS_MODEL_PREFILL_CACHE = output_dir / F4_PREFILL_JSON
        paper_figures.gen_f4(output_dir / FIGURE_DIRNAME)
        return
    except (ImportError, ModuleNotFoundError):
        pass

    # Standalone renderer
    C_BAR_A = "#b8c8dc"
    C_BAR_B = "#d8c0b0"

    def _cycles_label(v: float) -> str:
        if v >= 1e9: return f"{v/1e9:.1f}B"
        if v >= 1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"

    def _f4_bar_panel(ax, rows: list[dict], *, title: str, ylabel: bool = True) -> None:
        order: list[str] = []
        for r in rows:
            n = str(r.get("model_name", "?"))
            if n not in order:
                order.append(n)
        by = {(str(r.get("model_name", "?")), str(r.get("mode", "steady_state"))): r for r in rows}
        modes = [("steady_state", "Steady", C_BAR_A), ("cold_start", "Cold", C_BAR_B)]
        x = list(range(len(order)))
        w = 0.34
        vals_all: list[float] = []
        for idx, (mode, label, color) in enumerate(modes):
            off = [p + (idx - 0.5) * w for p in x]
            vals = [float(by.get((n, mode), {}).get("cycles", 0) or 0) for n in order]
            vals_all.extend(v for v in vals if v > 0)
            bars = ax.bar(off, vals, w, label=label, color=color,
                          edgecolor=C_EDGE, linewidth=0.3, alpha=0.88)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.06,
                            _cycles_label(v), ha="center", va="bottom",
                            fontsize=5.0, rotation=90, color=C_ANNOT)
        ax.set_xticks(x, order, rotation=40, ha="right")
        ax.set_yscale("log")
        if vals_all:
            ax.set_ylim(min(vals_all) * 0.4, max(vals_all) * 3.5)
        if ylabel:
            ax.set_ylabel("Backend cycles")
        ax.set_title(title, pad=8)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), frameon=False, handlelength=1.0)
        _grid(ax, "y")

    d_rows = json.loads((output_dir / F4_DECODE_JSON).read_text("utf-8"))["rows"]
    p_rows = json.loads((output_dir / F4_PREFILL_JSON).read_text("utf-8"))["rows"]
    fig = plt.figure(figsize=(10.5, 3.2))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.32, top=0.88, wspace=0.15)
    _f4_bar_panel(fig.add_subplot(1, 2, 1), d_rows, title="(a) Decode backend cycles")
    _f4_bar_panel(fig.add_subplot(1, 2, 2), p_rows, title="(b) Prefill backend cycles", ylabel=False)
    _save(fig, output_dir / FIGURE_DIRNAME, "f4_cross_model_cycles")


# ── F5 — prefill prompt-length sweep ────────────────────────────────────

def _run_f5_task(task: dict) -> dict:
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        "prefill", "llama2-7b",
        prompt_len=task["prompt_len"],
        materialize_weights=task["materialize_weights"])
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def collect_f5(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
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

    tasks: list[dict] = []
    for pl in F5_PROMPT_LENGTHS:
        for mode in MODES:
            part = _part_path(pl, mode)
            if not force and part.exists():
                continue
            tasks.append({
                "prompt_len": pl, "mode": mode,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part)})

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

    # Assemble rows
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
            density = (pim_mac * 4) / cycles if cycles > 0 else 0.0

            rows.append({
                "prompt_len": pl, "mode": mode,
                "status": "PASS" if data.get("replay_ok") else "FAIL",
                "runtime_ns": runtime_ns,
                "cycles": cycles,
                "pim_mac": total_pim_mac_per_layer * nl,
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
                "avg_pim_latency_cycles": 0})

    _write_json(path, {
        "schema_version": 1,
        "figure_id": "fig21_llama2_7b_prefill_prompt_sweep",
        "description": "Llama2-7B full-depth prefill backend replay across prompt lengths",
        "model": "Llama2-7B", "phase": "prefill",
        "sweep": {"prompt_len_values": list(F5_PROMPT_LENGTHS), "modes": list(MODES)},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": rows,
        "caveats": ["Simulator-diagnostic cycles, not silicon-calibrated"]})
    print(f"wrote {path} ({len(rows)} rows)")


def render_f5(output_dir: Path) -> None:
    try:
        paper_figures = _load_paper_figure_module()
        paper_figures.PREFILL_PROMPT_SWEEP_CACHE = output_dir / F5_JSON
        paper_figures.gen_f5(output_dir / FIGURE_DIRNAME)
        return
    except (ImportError, ModuleNotFoundError):
        pass

    # Standalone renderer
    path = output_dir / F5_JSON
    if not path.exists():
        print(f"  WARNING: missing {path} — run --collect f5 first")
        return

    from lib.backend_replay import prefill_formula

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload.get("rows", []) if isinstance(r, dict)]
    model = str(payload.get("model", "Llama2-7B"))

    steady = sorted([r for r in rows if str(r.get("mode")) == "steady_state"] or rows,
                    key=lambda r: int(r["prompt_len"]))
    if not steady:
        print("  WARNING: empty sweep — skipping F5")
        return

    pls = [int(r["prompt_len"]) for r in steady]
    attn = []
    for r in steady:
        pm = float(r["pim_mac"])
        b = r.get("per_layer_pim_mac_buckets", {})
        nl = float(r.get("num_layers", 1))
        a = pm - float(b.get("qkvo_projection", 0)) * nl - float(b.get("ffn", 0)) * nl
        if a <= 0 and "attention" in b:
            a = float(b["attention"]) * nl
        attn.append(max(a, 0))

    ext_n, ext_a = [], []
    if model == "Llama2-7B":
        for pl_ext in [32, 64, 128, 256, 512]:
            if pl_ext <= max(pls):
                continue
            try:
                f = prefill_formula("llama2-7b", prompt_len=pl_ext)
                bk = f.get("per_layer_pim_mac_buckets", {})
                if isinstance(bk, dict) and "attention" in bk:
                    ext_n.append(pl_ext)
                    ext_a.append(float(bk["attention"]) * float(f["model_total_layers"]))
            except Exception:
                break

    an, aa = pls + ext_n, attn + ext_a
    fx = np.array(an, dtype=float)
    fy = np.array(aa, dtype=float)
    co = np.polyfit(fx, fy, 2) if len(fx) >= 3 else np.array([fy[-1] / fx[-1]**2, 0, 0])
    fg = np.linspace(min(fx), max(fx), 200)
    fv = np.polyval(co, fg)
    ny = [v / (n * n) for n, v in zip(an, aa)]

    fig = plt.figure(figsize=(5.0, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    axn = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(pls, attn, "o-", color=C_BLUE, lw=1.6, ms=4, label="Backend-derived attention")
    if ext_n:
        ax.plot(ext_n, ext_a, "o--", color=C_BLUE, lw=1.0, ms=3,
                markerfacecolor="white", label="Formula extension")
    ax.plot(fg, fv, ":", color=C_COPPER, lw=1.6, label="Quadratic fit")
    for n, v in zip(pls, attn):
        ax.text(n, v * 1.06, f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.1f}K",
                ha="center", va="bottom", fontsize=6, color=C_ANNOT)
    for n, v in zip(ext_n, ext_a):
        ax.text(n, v * 1.06, f"{v/1e6:.1f}M", ha="center", va="bottom",
                fontsize=6, color=C_ANNOT)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(min(an) * 0.85, max(an) * 1.15)
    ax.set_ylim(min(aa) * 0.5, max(aa) * 1.8)
    ax.set_ylabel("Attention PIM MACs")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    ax.set_title(f"Causal-attention quadratic scaling ({model} prefill)", pad=6)
    ax.legend(frameon=False, handlelength=1.2)

    axn.plot(an, ny, ".-", color=C_GREEN, lw=1.2, ms=3)
    hi = max(1, len(ny) // 2)
    ref = float(np.median(ny[hi:]))
    axn.axhline(ref, color=C_REF, ls=":", lw=0.8)
    axn.text(max(an) * 1.02, ref, f"{ref/1e3:.2f}K MACs/n²", fontsize=6, va="center", color=C_REF)
    axn.set_xscale("log", base=2)
    axn.set_xlabel("Prompt length n (tokens)")
    axn.set_ylabel("PIM MACs / n²")
    axn.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.1f}K"))
    axn.set_xticks(sorted(set(an)), [str(n) for n in sorted(set(an))])

    fig.tight_layout(pad=0.6)
    _save(fig, output_dir / FIGURE_DIRNAME, "f5_prefill_attention_scaling")


# ── F6 — parameter sensitivity ──────────────────────────────────────────

def _f6_linear_fit(xs: list[float], ys: list[float]) -> dict:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"slope": 0.0, "intercept": ys[0] if ys else 0.0, "r2": 1.0 if len(ys) == 1 else 0.0}
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom if denom else 0.0
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _f6_base_cfg(*, active_banks: int = F6_ACTIVE_BANKS, banks_per_mpu: int = F6_BASE_BANKS_PER_MPU) -> dict:
    cfg = make_rr_cfg(active_banks, banks_per_mpu)
    cfg["num_pim_requests"] = F6_NUM_PIM_REQUESTS
    return cfg


def _f6_row_from_stats(task: dict, stats: dict, time_unit_ns: float) -> dict:
    ctrl = _f6_nested(stats, "memory_system", "controller", default={}) or {}
    dram = _f6_nested(stats, "memory_system", "dram", default={}) or {}
    extracted = extract_pim_stats(stats, time_unit_ns)
    command_counts = extracted.pop("command_counts")
    cycles = int(extracted["cycles"])
    served = int(extracted["num_pim_reqs_served"])
    total_time_ns = float(extracted["total_time_ns"])
    pim_mac = int(command_counts.get("PIM_MAC", extracted.get("pim_mac_issued", 0)) or 0)
    pim_mac_ab = int(command_counts.get("PIM_MAC_AB", 0) or 0)
    energy_stats = dram if dram else ctrl
    memory_energy_pJ = float(energy_stats.get("total_energy", 0.0) or 0.0)
    pim_incremental_energy_pJ = float(energy_stats.get("total_incremental_cmd_energy", 0.0) or 0.0)
    total_energy_pJ = memory_energy_pJ + pim_incremental_energy_pJ
    ops_per_request = float(ctrl.get("pim_ops_per_request", ctrl.get("pim_ops_per_block_issue", 64.0)) or 64.0)
    lanes = float(ctrl.get("pim_lanes", 32.0) or 32.0)
    e_comp = float(task.get("e_comp_pJ_per_mac", ctrl.get("pim_compute_energy_pJ_per_mac", 0.0)) or 0.0)
    e_array_local = float(ctrl.get("pim_array_local_energy_pJ", 0.0) or 0.0)
    e_c2p = float(ctrl.get("pim_cell_to_pim_energy_pJ_per_256b", 0.0) or 0.0)
    e_vrf = float(ctrl.get("pim_vrf_access_energy_pJ", 0.0) or 0.0)
    e_srf = float(ctrl.get("pim_srf_access_energy_pJ", 0.0) or 0.0)
    pim_mac_events = pim_mac + pim_mac_ab
    pim_event_energy_pJ = pim_mac_events * (e_array_local + lanes * e_comp + e_c2p + e_vrf + e_srf)

    return {
        "figure": "f6",
        "sweep": str(task["sweep"]),
        "active_banks": int(task.get("active_banks", F6_ACTIVE_BANKS)),
        "pim_banks_per_mpu": int(task.get("banks_per_mpu", F6_BASE_BANKS_PER_MPU)),
        "nop": int(task.get("nop", F6_BASE_NOP)),
        "nPIM_MAC_II": int(ctrl.get("pim_mac_issue_interval_cycles", task.get("nPIM_MAC_II", 0)) or 0),
        "nPIM_MAC_LAT": int(ctrl.get("pim_mac_pipeline_latency_cycles", task.get("nPIM_MAC_LAT", 0)) or 0),
        "e_comp_pJ_per_mac": e_comp,
        "cycles": cycles,
        "total_time_ns": total_time_ns,
        "avg_pim_latency_ns": float(extracted["avg_pim_latency_ns"]),
        "num_pim_reqs_served": served,
        "request_throughput": float(extracted["request_throughput"]),
        "pim_mac_issued": int(extracted["pim_mac_issued"]),
        "count_PIM_MAC": pim_mac,
        "count_PIM_MAC_AB": pim_mac_ab,
        "pim_lanes": lanes,
        "pim_ops_per_request": ops_per_request,
        "pim_array_local_energy_pJ": e_array_local,
        "pim_cell_to_pim_energy_pJ_per_256b": e_c2p,
        "pim_vrf_access_energy_pJ": e_vrf,
        "pim_srf_access_energy_pJ": e_srf,
        "pim_mpu_group_count": int(ctrl.get("pim_mpu_group_count", 0) or 0),
        "effective_mpu_groups": int(ctrl.get("effective_mpu_groups", 0) or 0),
        "pim_inflight_peak": int(ctrl.get("pim_inflight_peak", extracted.get("pim_inflight_peak", 0)) or 0),
        "pim_capacity_stalls": int(extracted.get("pim_capacity_stalls", 0) or 0),
        "pim_dependency_stalls": int(extracted.get("pim_dependency_stalls", 0) or 0),
        "pim_mpu_group_stalls": int(extracted.get("pim_mpu_group_stalls", 0) or 0),
        "memory_energy_pJ": memory_energy_pJ,
        "pim_incremental_energy_pJ": pim_incremental_energy_pJ,
        "total_energy_pJ": total_energy_pJ,
        "memory_energy_per_request_pJ": memory_energy_pJ / served if served else 0.0,
        "pim_incremental_energy_per_request_pJ": pim_incremental_energy_pJ / served if served else 0.0,
        "total_energy_per_request_pJ": total_energy_pJ / served if served else 0.0,
        "pim_compute_event_energy_pJ": pim_mac_events * lanes * e_comp,
        "pim_parameterized_event_energy_pJ": pim_event_energy_pJ,
        "simulated_ops_per_ns": (served * ops_per_request / total_time_ns) if total_time_ns > 0 else 0.0}


def _f6_pim_ops_roofline(rows: list[dict], time_unit_ns: float) -> dict:
    mapping = sorted((r for r in rows if r["sweep"] == "bank_mapping"), key=lambda r: r["pim_banks_per_mpu"])
    pim_base = next((r for r in mapping if int(r["pim_banks_per_mpu"]) == F6_BASE_BANKS_PER_MPU),
                    mapping[0] if mapping else None)
    if pim_base:
        ops_ceiling = float(pim_base["pim_ops_per_request"]) / time_unit_ns
        observed_ops = float(pim_base["simulated_ops_per_ns"])
        ratio = observed_ops / ops_ceiling if ops_ceiling > 0 else 0.0
    else:
        ops_ceiling = 0.0; observed_ops = 0.0; ratio = 0.0
    return {
        "name": "pim_mac_global_issue_ops",
        "observed_ops_per_ns": observed_ops,
        "theoretical_ops_per_ns": ops_ceiling,
        "observed_GOPS": observed_ops, "theoretical_GOPS": ops_ceiling,
        "ratio": ratio, "pass": ratio >= 0.95}


def _collect_f6_pim_task(task: dict) -> dict:
    cfg = _f6_base_cfg(
        active_banks=int(task.get("active_banks", F6_ACTIVE_BANKS)),
        banks_per_mpu=int(task.get("banks_per_mpu", F6_BASE_BANKS_PER_MPU)))
    dram_kwargs = cfg.setdefault("dram_kwargs", {})
    if "nPIM_MAC_II" in task:
        dram_kwargs["nPIM_MAC_II"] = int(task["nPIM_MAC_II"])
    if "nPIM_MAC_LAT" in task:
        dram_kwargs["nPIM_MAC_LAT"] = int(task["nPIM_MAC_LAT"])
    if "e_comp_pJ_per_mac" in task:
        dram_kwargs["pim_compute_energy_pJ_per_mac"] = float(task["e_comp_pJ_per_mac"])
    stats = run_single(
        cfg_override=cfg,
        nop=int(task.get("nop", F6_BASE_NOP)),
        num_probes=F6_NUM_PIM_REQUESTS, warmup=10000)
    return _f6_row_from_stats(task, stats, float(task["time_unit_ns"]))


def _run_f6_lpddr5_bandwidth_roofline(time_unit_ns: float, *, num_reads: int = F6_ROOFLINE_READS) -> dict:
    import ramulator
    from lib.runner import _extract_dram_layout

    dram = ramulator.dram.LPDDR5(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400")
    org, timing = dram.resolve()
    tx_bytes = int(type(dram).internal_prefetch_size * int(org["channel_width"]) / 8)
    theoretical_bytes_per_ns = tx_bytes / (float(timing["nBL"]) * time_unit_ns)
    layout = _extract_dram_layout(dram)
    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=4, nop_counter=1, num_probe_requests=1,
        streaming_only=True, num_streaming_requests=int(num_reads),
        pim_mode=False, stream_cols=int(layout["num_cols"]),
        warmup_cycles=0, read_ratio=100, seed=12345, **layout)
    ctrl = ramulator.controller.LPDDR5(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper())
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper())
    sim = ramulator.Simulation(frontend, mem)
    sim.run()
    stats = sim.stats

    ctrl_stats = _nested(stats, "memory_system", "controller", default={}) or {}
    frontend_stats = stats.get("frontend", {}) if isinstance(stats.get("frontend", {}), dict) else {}
    cycles = int(ctrl_stats.get("cycles", 0) or 0)
    runtime_ns = cycles * time_unit_ns
    streamed = int(frontend_stats.get("streaming_requests_sent", num_reads) or 0)
    observed_bytes_per_ns = (streamed * tx_bytes / runtime_ns) if runtime_ns > 0 else 0.0
    ratio = observed_bytes_per_ns / theoretical_bytes_per_ns if theoretical_bytes_per_ns > 0 else 0.0
    return {
        "name": "lpddr5_row_hit_read_bandwidth",
        "num_reads": int(num_reads), "streaming_reads_sent": streamed,
        "cycles": cycles, "runtime_ns": runtime_ns,
        "tx_bytes": tx_bytes, "nBL": int(timing["nBL"]), "tCK_ns": time_unit_ns,
        "observed_bytes_per_ns": observed_bytes_per_ns,
        "theoretical_bytes_per_ns": theoretical_bytes_per_ns,
        "observed_GBps": observed_bytes_per_ns, "theoretical_GBps": theoretical_bytes_per_ns,
        "ratio": ratio, "pass": ratio >= 0.98}


def _f6_checks(rows: list[dict], roofline: dict, time_unit_ns: float) -> dict:
    timing = sorted((r for r in rows if r["sweep"] == "timing"), key=lambda r: r["nPIM_MAC_II"])
    energy = sorted((r for r in rows if r["sweep"] == "energy"), key=lambda r: r["e_comp_pJ_per_mac"])

    timing_base = next((r for r in timing if int(r["nPIM_MAC_II"]) == F6_TIMING_BASE_VALUE),
                       timing[0] if timing else None)
    timing_max_rel_error = 0.0
    if timing_base:
        bx = float(timing_base["nPIM_MAC_II"]); by = float(timing_base["cycles"])
        for r in timing:
            expected = float(r["nPIM_MAC_II"]) / bx
            observed = float(r["cycles"]) / by if by else 0.0
            timing_max_rel_error = max(timing_max_rel_error, abs(observed - expected))
    timing_fit = _f6_linear_fit(
        [float(r["nPIM_MAC_II"]) for r in timing],
        [float(r["cycles"]) for r in timing])

    layer1_vals = [float(r["memory_energy_per_request_pJ"]) for r in energy]
    layer2_vals = [float(r["pim_incremental_energy_per_request_pJ"]) for r in energy]
    layer1_mean = sum(layer1_vals) / len(layer1_vals) if layer1_vals else 0.0
    layer1_rel_range = ((max(layer1_vals) - min(layer1_vals)) / layer1_mean) if layer1_mean else 0.0
    layer2_fit = _f6_linear_fit(
        [float(r["e_comp_pJ_per_mac"]) for r in energy], layer2_vals)

    return {
        "timing_linear_fit": timing_fit,
        "timing_max_normalized_error_vs_ii": timing_max_rel_error,
        "energy_layer1_relative_range": layer1_rel_range,
        "energy_layer2_linear_fit": layer2_fit,
        "roofline_pass": bool(
            roofline.get("lpddr5_bandwidth", {}).get("pass", False)
            and roofline.get("pim_mac_ops", {}).get("pass", False))}


def collect_f6(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    path = output_dir / F6_JSON
    if not force and path.exists():
        print(f"using existing {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    time_unit_ns = _time_unit_ns()

    tasks: list[dict] = []
    tasks.extend({"sweep": "timing", "active_banks": F6_TIMING_ACTIVE_BANKS,
                  "banks_per_mpu": F6_BASE_BANKS_PER_MPU, "nop": F6_BASE_NOP,
                  "nPIM_MAC_II": v, "nPIM_MAC_LAT": F6_TIMING_ISOLATION_LAT,
                  "time_unit_ns": time_unit_ns}
                 for v in F6_TIMING_VALUES)
    tasks.extend({"sweep": "energy", "active_banks": F6_ACTIVE_BANKS,
                  "banks_per_mpu": F6_BASE_BANKS_PER_MPU, "nop": F6_BASE_NOP,
                  "e_comp_pJ_per_mac": v, "time_unit_ns": time_unit_ns}
                 for v in F6_E_COMP_VALUES)
    tasks.extend({"sweep": "bank_mapping", "active_banks": F6_ACTIVE_BANKS,
                  "banks_per_mpu": v, "nop": F6_BASE_NOP,
                  "time_unit_ns": time_unit_ns}
                 for v in F6_BANKS_PER_MPU)

    rows: list[dict] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_collect_f6_pim_task(task))
            print(f"[f6] {index}/{len(tasks)}: {task['sweep']} {task}", flush=True)
    else:
        failures: list[tuple[dict, Exception]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {pool.submit(_collect_f6_pim_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    rows.append(future.result())
                    print(f"[f6] {index}/{len(tasks)}: {task['sweep']} {task}", flush=True)
                except Exception as exc:
                    failures.append((task, exc))
                    print(f"[f6] FAILED {index}/{len(tasks)}: {task['sweep']} {task}: {exc}", flush=True)
        if failures:
            raise RuntimeError(f"F6 collection failed for {len(failures)} task(s)")

    rows.sort(key=lambda r: (str(r["sweep"]), int(r["pim_banks_per_mpu"]),
                              float(r["nPIM_MAC_II"]), float(r["e_comp_pJ_per_mac"])))

    roofline = {
        "lpddr5_bandwidth": _run_f6_lpddr5_bandwidth_roofline(time_unit_ns),
        "pim_mac_ops": _f6_pim_ops_roofline(rows, time_unit_ns)}
    checks = _f6_checks(rows, roofline, time_unit_ns)

    _write_json(path, {
        "schema_version": 1,
        "figure_id": "f6_parameter_sensitivity",
        "description": "LPDDR5-PIM parameter sensitivity and roofline checks",
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "baseline": {
            "active_banks": F6_ACTIVE_BANKS, "timing_active_banks": F6_TIMING_ACTIVE_BANKS,
            "timing_nPIM_MAC_LAT": F6_TIMING_ISOLATION_LAT,
            "banks_per_mpu": F6_BASE_BANKS_PER_MPU, "num_pim_requests": F6_NUM_PIM_REQUESTS,
            "nop": F6_BASE_NOP, "tCK_ns": time_unit_ns},
        "sweeps": {"nPIM_MAC_II": list(F6_TIMING_VALUES),
                   "e_comp_pJ_per_mac": list(F6_E_COMP_VALUES),
                   "banks_per_mpu": list(F6_BANKS_PER_MPU)},
        "roofline": roofline, "checks": checks, "rows": rows})
    print(f"wrote {path} ({len(rows)} rows)")


def _read_f6_payload(output_dir: Path) -> dict:
    path = output_dir / F6_JSON
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run --collect f6 first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} does not contain a valid F6 payload")
    return payload


def render_f6(output_dir: Path) -> None:
    payload = _read_f6_payload(output_dir)
    rows = [row for row in payload["rows"] if isinstance(row, dict)]
    timing = sorted((r for r in rows if r["sweep"] == "timing"), key=lambda r: r["nPIM_MAC_II"])
    energy = sorted((r for r in rows if r["sweep"] == "energy"), key=lambda r: r["e_comp_pJ_per_mac"])
    mapping = sorted((r for r in rows if r["sweep"] == "bank_mapping"), key=lambda r: r["pim_banks_per_mpu"])
    if not timing or not energy or not mapping:
        raise ValueError("F6 payload must contain timing, energy, and bank_mapping rows")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), sharey=False)

    # (a) Timing sensitivity
    ax = axes[0]
    base = next((r for r in timing if int(r["nPIM_MAC_II"]) == F6_TIMING_BASE_VALUE), timing[0])
    bx = float(base["nPIM_MAC_II"]); by = float(base["cycles"])
    xs = [float(r["nPIM_MAC_II"]) for r in timing]
    ys = [float(r["cycles"]) / by for r in timing]
    ideal = [x / bx for x in xs]
    ax.plot(xs, ys, marker="o", color=C_BLUE, lw=1.8, ms=4, label="Measured cycles")
    ax.plot(xs, ideal, linestyle=":", color=C_REF, lw=1.4, label="Ideal linear")
    ax.set_xlabel("nPIM_MAC_II (cycles)")
    ax.set_ylabel("Normalized cycles")
    ax.set_title("(a) Timing knob (1 bank)", pad=6)
    ax.set_xticks(xs, [str(int(x)) for x in xs])
    fit = payload.get("checks", {}).get("timing_linear_fit", {})
    ax.text(0.04, 0.94, f"R²={float(fit.get('r2', 0.0)):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=C_ANNOT)
    _grid(ax, "both")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)

    # (b) Energy sensitivity
    ax = axes[1]
    ex = [float(r["e_comp_pJ_per_mac"]) for r in energy]
    l1 = [float(r["memory_energy_per_request_pJ"]) for r in energy]
    l2 = [float(r["pim_incremental_energy_per_request_pJ"]) for r in energy]
    ax.plot(ex, l1, marker="s", color=C_GREEN, lw=1.6, ms=4, label="Layer-1 memory")
    ax.plot(ex, l2, marker="o", color=C_COPPER, lw=1.8, ms=4, label="Layer-2 PIM incr.")
    ax.set_xlabel("e_comp (pJ/MAC)")
    ax.set_ylabel("Energy/request (pJ)")
    ax.set_title("(b) Energy knob", pad=6)
    ax.set_xticks(ex, [f"{x:g}" for x in ex])
    ef = payload.get("checks", {}).get("energy_layer2_linear_fit", {})
    ax.text(0.04, 0.94, f"Layer-2 R²={float(ef.get('r2', 0.0)):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=C_ANNOT)
    _grid(ax, "both")
    ax.legend(frameon=False, loc="best", handlelength=1.4)

    # (c) Bank mapping sensitivity
    ax = axes[2]
    bpm = [int(r["pim_banks_per_mpu"]) for r in mapping]
    tp = [float(r["request_throughput"]) for r in mapping]
    epr = [float(r["total_energy_per_request_pJ"]) for r in mapping]
    ax.plot(bpm, tp, marker="o", color=C_BLUE, lw=1.8, ms=4, label="Throughput")
    ax.set_xlabel("Banks per PIM block (16 banks)")
    ax.set_ylabel("Throughput (req/ns)", color=C_BLUE)
    ax.tick_params(axis="y", labelcolor=C_BLUE)
    ax.set_title("(c) Bank mapping", pad=6)
    ax.set_xticks(bpm, [str(v) for v in bpm])
    _grid(ax, "y")
    ax2 = ax.twinx()
    ax2.plot(bpm, epr, marker="^", color=C_COPPER, lw=1.6, ms=4, linestyle="--", label="Energy/request")
    ax2.set_ylabel("Energy/request (pJ)", color=C_COPPER)
    ax2.tick_params(axis="y", labelcolor=C_COPPER)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best", handlelength=1.4)

    roof = payload.get("roofline", {})
    bw = roof.get("lpddr5_bandwidth", {})
    ops = roof.get("pim_mac_ops", {})
    bw_pct = 100.0 * float(bw.get("ratio", 0.0))
    ops_pct = 100.0 * float(ops.get("ratio", 0.0))
    fig.text(0.995, 0.01, f"Rooflines: LPDDR5 BW {bw_pct:.1f}% • PIM ops {ops_pct:.1f}%",
             ha="right", va="bottom", fontsize=7, color=C_ANNOT)
    fig.tight_layout(pad=0.8, rect=(0, 0.04, 1, 1))
    _save(fig, output_dir / FIGURE_DIRNAME, "f6_parameter_sensitivity")


# ── Transformer-trace k=1 vs k=2 PIM comparison table ──────────────────

FTABLE_WORKLOADS = (
    {"model_key": "llama2-7b",    "phase": "decode", "past_len": 1024},
    {"model_key": "llama2-13b",   "phase": "decode", "past_len": 1024},
    {"model_key": "llama2-70b",   "phase": "decode", "past_len": 1024},
    {"model_key": "opt-125m",     "phase": "decode", "past_len": 1024},
    {"model_key": "opt-350m",     "phase": "decode", "past_len": 1024},
    {"model_key": "opt-1.3b",     "phase": "decode", "past_len": 1024},
    {"model_key": "qwen25-7b",    "phase": "decode", "past_len": 1024},
    {"model_key": "qwen25-14b",   "phase": "decode", "past_len": 1024},
    {"model_key": "qwen25-32b",   "phase": "decode", "past_len": 1024},
    {"model_key": "qwen25-72b",   "phase": "decode", "past_len": 1024},
    {"model_key": "gemma-2b",     "phase": "decode", "past_len": 1024},
    {"model_key": "gemma-7b",     "phase": "decode", "past_len": 1024},
    {"model_key": "gemma2-9b",    "phase": "decode", "past_len": 1024},
    {"model_key": "gemma2-27b",   "phase": "decode", "past_len": 1024},
    {"model_key": "mixtral-8x7b", "phase": "decode", "past_len": 1024},
)

PIM_CONFIGS = {
    "k1": {"pim_banks_per_mpu": 1, "pim_mac_execution_model": "shared_mpu_serial"},
    "k2": {"pim_banks_per_mpu": 2, "pim_mac_execution_model": "shared_mpu_serial"},
}


def _run_ftable_task(task: dict) -> dict:
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len") or 1024,
        prompt_len=task.get("prompt_len") or 12,
        materialize_weights=task["materialize_weights"],
        pim_cfg_override=task["pim_cfg_override"],
        max_inflight_requests=task.get("max_inflight_requests", 16),
        mac_mode=task.get("mac_mode", "per_bank"))
    part_path = Path(task["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(part_path)
    return result


def collect_ftable(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    path = output_dir / FTABLE_JSON
    if not force and path.exists():
        print(f"using existing {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    parts_dir = output_dir / "ftable_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(model: str, pim_label: str) -> Path:
        safe = model.replace("-", "_").replace(".", "_")
        return parts_dir / f"{safe}__{pim_label}.json"

    tasks: list[dict] = []
    for wl in FTABLE_WORKLOADS:
        for label, cfg in PIM_CONFIGS.items():
            part = _part_path(wl["model_key"], label)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": wl["model_key"], "phase": wl["phase"],
                "past_len": wl["past_len"],
                "materialize_weights": False,
                "pim_cfg_override": cfg,
                "part_path": str(part), "pim_label": label,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})

    total = len(tasks)
    expected = len(FTABLE_WORKLOADS) * len(PIM_CONFIGS)
    cached = expected - total
    if cached > 0:
        print(f"[ftable] {cached} parts already cached, {total} remaining", flush=True)
    if total > 0:
        print(f"[ftable] collecting {total} simulation points with {workers} workers", flush=True)

    if total > 0:
        if workers <= 1:
            for idx, task in enumerate(tasks, 1):
                _run_ftable_task(task)
                print(f"[ftable] {idx}/{total}: {task['model_key']} {task['pim_label']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(_run_ftable_task, t): t for t in tasks}
                for idx, future in enumerate(as_completed(future_map), 1):
                    task = future_map[future]
                    try:
                        future.result()
                        print(f"[ftable] {idx}/{total}: {task['model_key']} {task['pim_label']}", flush=True)
                    except Exception as exc:
                        print(f"[ftable] FAILED {idx}/{total}: {task['model_key']} {task['pim_label']}: {exc}", flush=True)

    # Assemble rows
    from lib.backend_replay import _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    rows: list[dict] = []
    for wl in FTABLE_WORKLOADS:
        model_key = wl["model_key"]
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
            hidden_size = 4096; num_layers = 32
        else:
            model_name = model_key
            hidden_size = 0; num_layers = 0

        k1_part = _part_path(model_key, "k1")
        k2_part = _part_path(model_key, "k2")
        if not k1_part.exists():
            print(f"  WARNING: missing k1 part {k1_part}")
            continue
        if not k2_part.exists():
            print(f"  WARNING: missing k2 part {k2_part}")
            continue

        k1 = json.loads(k1_part.read_text(encoding="utf-8"))
        k2 = json.loads(k2_part.read_text(encoding="utf-8"))

        cycles_k1 = int(k1["cycles"]); cycles_k2 = int(k2["cycles"])
        runtime_ns_k1 = float(k1["runtime_ns"]); runtime_ns_k2 = float(k2["runtime_ns"])
        slowdown = (cycles_k2 / cycles_k1) if cycles_k1 > 0 else 0.0
        mpu_stalls_k2 = int(k2.get("pim_mpu_group_stalls", 0) or 0)
        shared_block_stall_pct = (mpu_stalls_k2 / cycles_k2 * 100.0) if cycles_k2 > 0 else 0.0

        workload_label = f"{model_name} {wl['phase']}"
        if wl["phase"] == "decode" and wl.get("past_len"):
            workload_label += f" (past={wl['past_len']})"

        rows.append({
            "workload": workload_label, "model_key": model_key,
            "model_family": _infer_model_family(model_name),
            "phase": wl["phase"],
            "hidden_size": hidden_size, "num_layers": num_layers,
            "cycles_k1": cycles_k1, "cycles_k2": cycles_k2,
            "runtime_ns_k1": runtime_ns_k1, "runtime_ns_k2": runtime_ns_k2,
            "slowdown": round(slowdown, 4),
            "pim_simultaneous_active_banks_peak_k1": int(k1.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_simultaneous_active_banks_peak_k2": int(k2.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_ab_mac_latency_cycles_k1": int(k1.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_ab_mac_latency_cycles_k2": int(k2.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_mpu_group_stalls_k2": mpu_stalls_k2,
            "pim_dependency_stalls_k2": int(k2.get("pim_dependency_stalls", 0) or 0),
            "pim_capacity_stalls_k2": int(k2.get("pim_capacity_stalls", 0) or 0),
            "num_bank_timing_blocked_k2": int(k2.get("num_bank_timing_blocked_cycles", 0) or 0),
            "shared_block_stall_pct": round(shared_block_stall_pct, 2),
            "pim_banks_per_mpu_k1": int(k1.get("pim_banks_per_mpu", 1) or 1),
            "pim_banks_per_mpu_k2": int(k2.get("pim_banks_per_mpu", 2) or 2),
            "replay_ok_k1": bool(k1.get("replay_ok")),
            "replay_ok_k2": bool(k2.get("replay_ok")),
        })

    _write_json(path, {
        "schema_version": 1,
        "description": "Transformer-trace PIM comparison: CD-PIM dedicated per-bank (k=1) vs shared-MPU 2-banks/MPU (k=2)",
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": rows})
    print(f"wrote {path} ({len(rows)} rows)")


# ── CLI ─────────────────────────────────────────────────────────────────

COLLECTORS = {"f2": collect_f2, "f3": collect_f3, "f4": collect_f4, "f5": collect_f5,
              "f6": collect_f6, "ftable": collect_ftable}
RENDERERS = {"f2": render_f2, "f3": render_f3, "f4": render_f4, "f5": render_f5, "f6": render_f6}


def _expand_target(target: str | None) -> list[str]:
    if target in (None, "all"):
        return ["f2", "f3", "f4", "f5", "f6", "ftable"]
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
    figure_choices = ["f2", "f3", "f4", "f5", "f6", "ftable", "all"]
    parser = argparse.ArgumentParser(description="Collect and render LPDDR5-PIM F2-F6 artifacts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--collect", nargs="?", const="all", choices=figure_choices)
    group.add_argument("--render", nargs="?", const="all", choices=figure_choices)
    group.add_argument("--all", nargs="?", const="all", choices=figure_choices)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
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
