#!/usr/bin/env python3
"""Reproduce the LPDDR-PIM paper artifacts that appear in main.tex.

Two targets, matching the two artifacts in the paper:

    cross-model   Fig. cross_model_cycles     (15 decode + 14 prefill configs,
                  cold-start vs steady-state).  Data: decode_prefill_cycles_parts/
    pim-sharing   Table tab:pim-sharing        (per-bank b=1 vs shared b=2).
                  Data: pim_sharing_parts/

Usage:
    python scripts/gen_figures.py --collect cross-model --workers 8
    python scripts/gen_figures.py --render  cross-model
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_DIRNAME = "figures"
CROSS_MODEL_FIGURE_NAME = "cross_model_cycles"
DECODE_JSON = "decode_cycles.json"
PREFILL_JSON = "prefill_cycles.json"
PIM_SHARING_JSON = "pim_sharing_comparison.json"
CROSS_MODEL_PARTS_DIRNAME = "decode_prefill_cycles_parts"
PIM_SHARING_PARTS_DIRNAME = "pim_sharing_parts"

MODES = ("steady_state", "cold_start")

C_EDGE = "#555555"
C_GRID = "0.78"
C_ANNOT = "0.35"


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
        print(f"saved figure: {path}")
    plt.close(fig)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_part(result: dict, part_path: str) -> None:
    p = Path(part_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def _load_paper_figure_module():
    module_path = PROJECT_ROOT / "paper" / "scripts" / "gen_paper_figures.py"
    spec = importlib.util.spec_from_file_location("paper_gen_figures", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── cross-model — decode + prefill cycles (cold-start vs steady-state) ──

CROSS_MODEL_DECODE_MODELS = (
    "llama2-7b", "llama2-13b", "llama2-70b",
    "opt-125m", "opt-350m", "opt-1.3b",
    "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
    "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b",
    "mixtral-8x7b")
CROSS_MODEL_PREFILL_MODELS = (
    "llama2-7b", "llama2-13b", "llama2-70b",
    "opt-125m", "opt-350m", "opt-1.3b",
    "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
    "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b")
CROSS_MODEL_PREFILL_PROMPT_LEN = 12
DECODE_PAST_LEN = 1024


def _cross_model_part_path(output_dir: Path, model: str, phase: str, mode: str) -> Path:
    safe = model.replace("-", "_").replace(".", "_")
    return output_dir / CROSS_MODEL_PARTS_DIRNAME / f"{safe}__{phase}__{mode}.json"


def _run_cross_model_task(task: dict) -> dict:
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len", 1024),
        prompt_len=task.get("prompt_len", 12),
        materialize_weights=task["materialize_weights"],
        pim_cfg_override=task.get("pim_cfg_override"),
        max_inflight_requests=task.get("max_inflight_requests", 1),
        mac_mode=task.get("mac_mode", "per_kind"))
    _write_part(result, task["part_path"])
    return result


def collect_cross_model(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    from lib.backend_replay import prefill_formula, pim_cfg_shared, _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    decode_path = output_dir / DECODE_JSON
    prefill_path = output_dir / PREFILL_JSON
    if not force and decode_path.exists() and prefill_path.exists():
        print(f"[cross-model] using existing data: {decode_path}, {prefill_path}")
        print(f"[cross-model] part cache: {output_dir / CROSS_MODEL_PARTS_DIRNAME}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CROSS_MODEL_PARTS_DIRNAME).mkdir(parents=True, exist_ok=True)

    pim_cfg = pim_cfg_shared()
    tasks: list[dict] = []
    for model in CROSS_MODEL_DECODE_MODELS:
        for mode in MODES:
            part = _cross_model_part_path(output_dir, model, "decode", mode)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": model, "phase": "decode", "mode": mode,
                "past_len": DECODE_PAST_LEN,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part), "pim_cfg_override": pim_cfg,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})
    for model in CROSS_MODEL_PREFILL_MODELS:
        for mode in MODES:
            part = _cross_model_part_path(output_dir, model, "prefill", mode)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": model, "phase": "prefill", "mode": mode,
                "prompt_len": CROSS_MODEL_PREFILL_PROMPT_LEN,
                "materialize_weights": mode == "cold_start",
                "part_path": str(part), "pim_cfg_override": pim_cfg,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})

    total = len(tasks)
    expected = (len(CROSS_MODEL_DECODE_MODELS) + len(CROSS_MODEL_PREFILL_MODELS)) * 2
    if expected - total > 0:
        print(
            f"[cross-model] {expected - total} parts already cached in "
            f"{output_dir / CROSS_MODEL_PARTS_DIRNAME}; {total} remaining",
            flush=True)
    if total > 0:
        print(
            f"[cross-model] collecting {total} simulation points with {workers} workers; "
            f"data -> {decode_path}, {prefill_path}",
            flush=True)
        _run_tasks(tasks, workers, _run_cross_model_task,
                   lambda t: f"{t['model_key']} {t['phase']} {t['mode']} -> {t['part_path']}",
                   "cross-model")

    _assemble_cross_model(output_dir, decode_path, prefill_path)


def _run_tasks(tasks: list[dict], workers: int, fn, label, tag: str) -> None:
    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            fn(task)
            print(f"[{tag}] {idx}/{len(tasks)}: {label(task)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fn, t): t for t in tasks}
        for idx, future in enumerate(as_completed(future_map), 1):
            task = future_map[future]
            try:
                future.result()
                print(f"[{tag}] {idx}/{len(tasks)}: {label(task)}", flush=True)
            except Exception as exc:
                print(f"[{tag}] FAILED {idx}/{len(tasks)}: {label(task)}: {exc}", flush=True)


def _assemble_cross_model(output_dir: Path, decode_path: Path, prefill_path: Path) -> None:
    from lib.backend_replay import prefill_formula, _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    decode_rows: list[dict] = []
    for model in CROSS_MODEL_DECODE_MODELS:
        spec = get_model_spec(model) if model != "mixtral-8x7b" else None
        for mode in MODES:
            part = _cross_model_part_path(output_dir, model, "decode", mode)
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
    _assemble_cross_model_prefill(output_dir, prefill_path)


def _assemble_cross_model_prefill(output_dir: Path, prefill_path: Path) -> None:
    from lib.backend_replay import prefill_formula, _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    P = CROSS_MODEL_PREFILL_PROMPT_LEN
    prefill_rows: list[dict] = []
    for model in CROSS_MODEL_PREFILL_MODELS:
        formula = prefill_formula(model, prompt_len=P)
        spec = get_model_spec(model)
        for mode in MODES:
            part = _cross_model_part_path(output_dir, model, "prefill", mode)
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
                "prompt_len": P,
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
                "trace_name": f"{model}_prefill_P{P}_{mode}",
                "command_counts": data.get("opcode_counts", {}),
                "pim_mac_density": 0.0})
    _write_json(prefill_path, {
        "schema_version": 1,
        "figure_id": "fig22_cross_model_prefill_cycles",
        "description": "Cross-model dense prefill backend replay cycles",
        "phase": "prefill",
        "metric_units": {"cycles": "cycles", "runtime_ns": "ns"},
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py",
                       "prompt_len": P},
        "rows": prefill_rows,
        "caveats": ["Simulator-diagnostic cycles, not silicon-calibrated"]})
    print(f"wrote {prefill_path} ({len(prefill_rows)} rows)")


def render_cross_model(output_dir: Path) -> None:
    try:
        paper_figures = _load_paper_figure_module()
        paper_figures.CROSS_MODEL_DECODE_CACHE = output_dir / DECODE_JSON
        paper_figures.CROSS_MODEL_PREFILL_CACHE = output_dir / PREFILL_JSON
        figure_dir = output_dir / FIGURE_DIRNAME
        paper_figures.gen_f4(figure_dir)
        for ext in ("pdf", "png"):
            legacy_path = figure_dir / f"f4_{CROSS_MODEL_FIGURE_NAME}.{ext}"
            path = figure_dir / f"{CROSS_MODEL_FIGURE_NAME}.{ext}"
            if legacy_path.exists():
                legacy_path.replace(path)
                print(f"saved figure: {path}")
        return
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        pass

    C_BAR_A = "#b8c8dc"
    C_BAR_B = "#d8c0b0"

    def _cycles_label(v: float) -> str:
        if v >= 1e9: return f"{v/1e9:.1f}B"
        if v >= 1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"

    def _panel(ax, rows: list[dict], *, title: str, ylabel: bool = True) -> None:
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

    d_rows = json.loads((output_dir / DECODE_JSON).read_text("utf-8"))["rows"]
    p_rows = json.loads((output_dir / PREFILL_JSON).read_text("utf-8"))["rows"]
    fig = plt.figure(figsize=(10.5, 3.2))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.32, top=0.88, wspace=0.15)
    _panel(fig.add_subplot(1, 2, 1), d_rows, title="(a) Decode backend cycles")
    _panel(fig.add_subplot(1, 2, 2), p_rows, title="(b) Prefill backend cycles", ylabel=False)
    _save(fig, output_dir / FIGURE_DIRNAME, CROSS_MODEL_FIGURE_NAME)


# ── pim-sharing — per-bank (b=1) vs two-bank shared (b=2) decode table ──

PIM_SHARING_WORKLOADS = tuple(
    {"model_key": m, "phase": "decode", "past_len": 1024}
    for m in (
        "llama2-7b", "llama2-13b", "llama2-70b",
        "opt-125m", "opt-350m", "opt-1.3b",
        "qwen25-7b", "qwen25-14b", "qwen25-32b", "qwen25-72b",
        "gemma-2b", "gemma-7b", "gemma2-9b", "gemma2-27b", "mixtral-8x7b"))

PIM_CONFIGS = {
    "k1": {"pim_banks_per_mpu": 1, "pim_mac_execution_model": "shared_mpu_serial"},
    "k2": {"pim_banks_per_mpu": 2, "pim_mac_execution_model": "shared_mpu_serial"},
}


def _pim_sharing_part_path(output_dir: Path, model: str, label: str) -> Path:
    safe = model.replace("-", "_").replace(".", "_")
    return output_dir / PIM_SHARING_PARTS_DIRNAME / f"{safe}__{label}.json"


def _run_pim_sharing_task(task: dict) -> dict:
    from lib.backend_replay import generate_and_replay

    result = generate_and_replay(
        task["phase"], task["model_key"],
        past_len=task.get("past_len") or 1024,
        prompt_len=task.get("prompt_len") or 12,
        materialize_weights=task["materialize_weights"],
        pim_cfg_override=task["pim_cfg_override"],
        max_inflight_requests=task.get("max_inflight_requests", 16),
        mac_mode=task.get("mac_mode", "per_kind"))
    _write_part(result, task["part_path"])
    return result


def collect_pim_sharing(output_dir: Path, *, force: bool = False, workers: int = 1) -> None:
    path = output_dir / PIM_SHARING_JSON
    if not force and path.exists():
        print(f"[pim-sharing] using existing data: {path}")
        print(f"[pim-sharing] part cache: {output_dir / PIM_SHARING_PARTS_DIRNAME}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PIM_SHARING_PARTS_DIRNAME).mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for wl in PIM_SHARING_WORKLOADS:
        for label, cfg in PIM_CONFIGS.items():
            part = _pim_sharing_part_path(output_dir, wl["model_key"], label)
            if not force and part.exists():
                continue
            tasks.append({
                "model_key": wl["model_key"], "phase": wl["phase"],
                "past_len": wl["past_len"], "materialize_weights": False,
                "pim_cfg_override": cfg, "part_path": str(part), "pim_label": label,
                "max_inflight_requests": 16, "mac_mode": "per_kind"})

    total = len(tasks)
    expected = len(PIM_SHARING_WORKLOADS) * len(PIM_CONFIGS)
    if expected - total > 0:
        print(
            f"[pim-sharing] {expected - total} parts already cached in "
            f"{output_dir / PIM_SHARING_PARTS_DIRNAME}; {total} remaining",
            flush=True)
    if total > 0:
        print(
            f"[pim-sharing] collecting {total} simulation points with {workers} workers; "
            f"data -> {path}",
            flush=True)
        _run_tasks(tasks, workers, _run_pim_sharing_task,
                   lambda t: f"{t['model_key']} {t['pim_label']} -> {t['part_path']}",
                   "pim-sharing")

    _assemble_pim_sharing(output_dir, path)


def _assemble_pim_sharing(output_dir: Path, path: Path) -> None:
    from lib.backend_replay import _infer_model_family
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec

    rows: list[dict] = []
    for wl in PIM_SHARING_WORKLOADS:
        model_key = wl["model_key"]
        try:
            spec = get_model_spec(model_key)
        except (KeyError, ValueError):
            spec = None
        if spec is not None:
            model_name, hidden_size, num_layers = spec.name, int(spec.hidden_size), int(spec.num_layers)
        elif model_key == "mixtral-8x7b":
            model_name, hidden_size, num_layers = "Mixtral-8x7B", 4096, 32
        else:
            model_name, hidden_size, num_layers = model_key, 0, 0

        k1_part = _pim_sharing_part_path(output_dir, model_key, "k1")
        k2_part = _pim_sharing_part_path(output_dir, model_key, "k2")
        if not k1_part.exists():
            print(f"  WARNING: missing k1 part {k1_part}"); continue
        if not k2_part.exists():
            print(f"  WARNING: missing k2 part {k2_part}"); continue
        k1 = json.loads(k1_part.read_text(encoding="utf-8"))
        k2 = json.loads(k2_part.read_text(encoding="utf-8"))

        cycles_k1, cycles_k2 = int(k1["cycles"]), int(k2["cycles"])
        slowdown = (cycles_k2 / cycles_k1) if cycles_k1 > 0 else 0.0
        mpu_stalls_k2 = int(k2.get("pim_mpu_group_stalls", 0) or 0)
        stall_pct = (mpu_stalls_k2 / cycles_k2 * 100.0) if cycles_k2 > 0 else 0.0
        label = f"{model_name} {wl['phase']}"
        if wl["phase"] == "decode" and wl.get("past_len"):
            label += f" (past={wl['past_len']})"

        rows.append({
            "workload": label, "model_key": model_key,
            "model_family": _infer_model_family(model_name),
            "phase": wl["phase"], "hidden_size": hidden_size, "num_layers": num_layers,
            "cycles_k1": cycles_k1, "cycles_k2": cycles_k2,
            "runtime_ns_k1": float(k1["runtime_ns"]), "runtime_ns_k2": float(k2["runtime_ns"]),
            "slowdown": round(slowdown, 4),
            "pim_simultaneous_active_banks_peak_k1": int(k1.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_simultaneous_active_banks_peak_k2": int(k2.get("pim_simultaneous_active_banks_peak", 0) or 0),
            "pim_ab_mac_latency_cycles_k1": int(k1.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_ab_mac_latency_cycles_k2": int(k2.get("pim_ab_mac_latency_cycles", 0) or 0),
            "pim_mpu_group_stalls_k2": mpu_stalls_k2,
            "pim_dependency_stalls_k2": int(k2.get("pim_dependency_stalls", 0) or 0),
            "pim_capacity_stalls_k2": int(k2.get("pim_capacity_stalls", 0) or 0),
            "num_bank_timing_blocked_k2": int(k2.get("num_bank_timing_blocked_cycles", 0) or 0),
            "shared_block_stall_pct": round(stall_pct, 2),
            "pim_banks_per_mpu_k1": int(k1.get("pim_banks_per_mpu", 1) or 1),
            "pim_banks_per_mpu_k2": int(k2.get("pim_banks_per_mpu", 2) or 2),
            "replay_ok_k1": bool(k1.get("replay_ok")), "replay_ok_k2": bool(k2.get("replay_ok"))})

    _write_json(path, {
        "schema_version": 1,
        "description": "Transformer-trace PIM comparison: CD-PIM dedicated per-bank (k=1) vs shared-MPU 2-banks/MPU (k=2)",
        "provenance": {"date": date.today().isoformat(), "generator": "scripts/gen_figures.py"},
        "rows": rows})
    print(f"wrote {path} ({len(rows)} rows)")


COLLECTORS = {"cross-model": collect_cross_model, "pim-sharing": collect_pim_sharing}
RENDERERS = {"cross-model": render_cross_model}
TARGETS = ("cross-model", "pim-sharing", "all")


def _expand_target(target: str | None) -> list[str]:
    if target in (None, "all"):
        return ["cross-model", "pim-sharing"]
    if target not in COLLECTORS:
        raise ValueError(f"unknown target: {target}")
    return [target]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the LPDDR-PIM paper figure (cross-model) and table (pim-sharing)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--collect", nargs="?", const="all", choices=TARGETS)
    group.add_argument("--render", nargs="?", const="all", choices=TARGETS)
    group.add_argument("--all", nargs="?", const="all", choices=TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    _apply_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.collect is not None:
        for target in _expand_target(args.collect):
            COLLECTORS[target](args.output_dir, force=args.force, workers=args.workers)
        return 0
    if args.render is not None:
        for target in _expand_target(args.render):
            if target in RENDERERS:
                RENDERERS[target](args.output_dir)
        return 0

    for target in _expand_target(args.all):
        COLLECTORS[target](args.output_dir, force=args.force, workers=args.workers)
    for target in _expand_target(args.all):
        if target in RENDERERS:
            RENDERERS[target](args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
