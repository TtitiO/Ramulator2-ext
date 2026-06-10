"""Generate P4 paper figures: attention/FFN/MoE decomposition and replay validation.

Usage:
    PYTHONPATH="ramulator2/python" .venv/bin/python \
        ramulator2/tests/analysis/figures/gen_p4_figures.py \
        --output-dir paper/figures/

Produces:
    fig4_attention_decomposition.png/pdf
    fig5_ffn_moe_decomposition.png/pdf
    p4_replay_validation.tex
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# Ensure the ramulator package is importable
_here = Path(__file__).resolve().parent
# Add ramulator2/python for the ramulator package
sys.path.insert(0, str(_here.parent.parent.parent.parent / "python"))
# Add ramulator2/ for tests and other local modules
sys.path.insert(0, str(_here.parent.parent.parent.parent))



def _fig_path(output_dir: Path, name: str, ext: str = "png") -> Path:
    return output_dir / f"{name}.{ext}"


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = _fig_path(output_dir, name, ext)
        fig.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paper_table_dir(output_dir: Path) -> Path:
    """Route paper runs to paper/tables while keeping tests local to tmp dirs."""
    return output_dir.parent / "tables" if output_dir.name == "figures" else output_dir


CMD_ORDER = ["SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"]
OP_ORDER = ["READ", "WRITE", "RESIDENCY", "PIM_MAC"]
CMD_COLORS = {
    "SB": "#0072B2",
    "HAB": "#D55E00",
    "HAB_PIM": "#E69F00",
    "PIM_BCAST": "#56B4E9",
    "PIM_MAC": "#009E73",
    "PIM_MAC_AB": "#CC79A7",
    "READ": "#0072B2",
    "WRITE": "#D55E00",
    "RESIDENCY": "#7F7F7F",
}
LLAMA2_SCALING_FIGURE_ID = "fig9_llama2_7b_13b_models_latency_breakdown"
DEFAULT_LLAMA2_SCALING_CACHE = Path("paper/data") / f"{LLAMA2_SCALING_FIGURE_ID}.json"
DECODE_CONTEXT_SWEEP_FIGURE_ID = "fig11_llama2_7b_decode_context_length_sweep"
DEFAULT_DECODE_CONTEXT_SWEEP_CACHE = Path("paper/data") / f"{DECODE_CONTEXT_SWEEP_FIGURE_ID}.json"
GENERATED_TOKEN_SWEEP_FIGURE_ID = "fig12_llama2_7b_generated_token_sweep"
DEFAULT_GENERATED_TOKEN_SWEEP_CACHE = Path("paper/data") / f"{GENERATED_TOKEN_SWEEP_FIGURE_ID}.json"
PREFILL_PROMPT_SWEEP_FIGURE_ID = "fig21_llama2_7b_prefill_prompt_sweep"
DEFAULT_PREFILL_PROMPT_SWEEP_CACHE = Path("paper/data") / f"{PREFILL_PROMPT_SWEEP_FIGURE_ID}.json"
MIXTRAL_BREAKDOWN_FIGURE_ID = "fig13_mixtral_8x7b_32_layer_breakdown"
DEFAULT_MIXTRAL_BREAKDOWN_CACHE = Path("paper/data") / f"{MIXTRAL_BREAKDOWN_FIGURE_ID}.json"
MIXTRAL_VS_LLAMA2_FIGURE_ID = "fig14_mixtral_vs_llama2_decode_comparison"
MOE_SENSITIVITY_FIGURE_ID = "fig15_moe_expert_scaling_sensitivity"
DEFAULT_MOE_SENSITIVITY_CACHE = Path("paper/data") / f"{MOE_SENSITIVITY_FIGURE_ID}.json"
MOE_OPERATOR_DIAG_FIGURE_ID = "fig16_mixtral_operator_diagnostics"
OPT_LATENCY_FIGURE_ID = "fig19_opt_dense_model_latency_breakdown"
DEFAULT_OPT_LATENCY_CACHE = Path("paper/data") / f"{OPT_LATENCY_FIGURE_ID}.json"
QWEN_GEMMA_LATENCY_FIGURE_ID = "fig20_qwen_gemma_dense_model_latency_breakdown"
DEFAULT_QWEN_GEMMA_LATENCY_CACHE = Path("paper/data") / f"{QWEN_GEMMA_LATENCY_FIGURE_ID}.json"
CROSS_MODEL_LATENCY_FIGURE_ID = "fig18_cross_model_decode_cycles"
DEFAULT_CROSS_MODEL_LATENCY_CACHE = Path("paper/data") / f"{CROSS_MODEL_LATENCY_FIGURE_ID}.json"
CROSS_MODEL_PREFILL_FIGURE_ID = "fig22_cross_model_prefill_cycles"
CROSS_MODEL_DECODE_PREFILL_FIGURE_ID = "fig22_cross_model_decode_prefill_cycles"
CROSS_MODEL_PREFILL_ATTENTION_SCALING_FIGURE_ID = "fig22_prefill_attention_scaling"
DEFAULT_CROSS_MODEL_PREFILL_CACHE = Path("paper/data") / f"{CROSS_MODEL_PREFILL_FIGURE_ID}.json"
DEFAULT_CROSS_MODEL_SOURCE_CACHES = (
    DEFAULT_LLAMA2_SCALING_CACHE,
    DEFAULT_MIXTRAL_BREAKDOWN_CACHE,
    DEFAULT_OPT_LATENCY_CACHE,
    DEFAULT_QWEN_GEMMA_LATENCY_CACHE,
)
DEFAULT_CROSS_MODEL_PREFILL_MODEL_KEYS = (
    "llama2-7b",
    "llama2-13b",
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "qwen25-7b",
    "qwen25-14b",
    "qwen25-32b",
    "qwen25-72b",
    "gemma-2b",
    "gemma-7b",
    "gemma2-9b",
    "gemma2-27b",
    "llama2-70b",
)


def _plot_repeat_expanded_counts(ax: plt.Axes, counts: dict[str, int]) -> None:
    order = OP_ORDER + [c for c in CMD_ORDER if c not in OP_ORDER]
    present_cmds = [c for c in order if c in counts]
    present_cmds.extend(c for c in counts if c not in present_cmds)
    cmd_values = [counts.get(c, 0) for c in present_cmds]
    bars = ax.bar(
        present_cmds,
        cmd_values,
        color=[CMD_COLORS[c] for c in present_cmds],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(cmd_values + [1]) * 2.5)
    for bar, val in zip(bars, cmd_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 1) * 1.12,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=30)


def _plot_pim_bcast_modes(ax: plt.Axes, mode_counts: dict[str, int], *, title: str) -> None:
    labels = ["steady-state", "cold-start"]
    keys = ["steady_state", "cold_start"]
    values = [int(mode_counts.get(key, 0)) for key in keys]
    bars = ax.bar(labels, values, color=["#999999", CMD_COLORS["PIM_BCAST"]], edgecolor="white", linewidth=0.5)
    ymax = max(values + [1])
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("PIM_BCAST repeats", fontweight="bold")
    ax.set_title(title, fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.04,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ───────────────────────────────────────────────────────────────────────
# Attention Operator Decomposition
# ───────────────────────────────────────────────────────────────────────


def gen_figure_4(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_attention_decomposition,
        collect_attention_sweep,
        collect_attention_paper,
        collect_attention_sweep_paper,
    )

    if use_tiny:
        data = collect_attention_decomposition()
        sweep = collect_attention_sweep(
            num_heads_list=[1, 2, 4, 8],
            past_len_list=[32, 64, 128, 256],
        )
    else:
        data = collect_attention_paper()
        sweep = collect_attention_sweep_paper()

    fig = plt.figure(figsize=(15, 4.8))

    # --- Panel A: Command breakdown bar chart ---
    ax1 = fig.add_subplot(1, 3, 1)
    counts = data.get("operator_counts", data["concrete_counts"])
    semantic_counts = data["semantic_counts"]

    _plot_repeat_expanded_counts(ax1, counts)
    ax1.set_ylabel("Repeat-expanded op/request or semantic-residency count", fontweight="bold")
    ax1.set_title(
        f"Panel A: Attention traffic, residency, and PIM MACs\n"
        f"({data['num_heads']} head{'s' if data['num_heads'] > 1 else ''}, head_dim={data['head_dim']}, "
        f"past_len={data['past_len']}, {data['datatype']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel B: materialization-mode PIM_BCAST comparison ---
    ax_bcast = fig.add_subplot(1, 3, 2)
    _plot_pim_bcast_modes(
        ax_bcast,
        data.get("pim_bcast_by_mode", {}),
        title="Panel B: true PIM_BCAST setup only\n(residency is semantic-only)",
    )

    # --- Panel B: Parameter sensitivity ---
    ax2 = fig.add_subplot(1, 3, 3)
    head_vals = sorted(set(r["_num_heads"] for r in sweep))
    markers = ["o", "s", "^", "D"]
    colors_line = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for head_idx, num_heads in enumerate(head_vals):
        subset = [r for r in sweep if r["_num_heads"] == num_heads]
        subset.sort(key=lambda r: r["_past_len"])
        x = [r["_past_len"] for r in subset]
        y = [r["concrete_counts"].get("PIM_MAC", 0) for r in subset]
        ax2.plot(x, y, marker=markers[head_idx], color=colors_line[head_idx],
                 label=f"{num_heads} heads", linewidth=1.5, markersize=6)

    ax2.set_xlabel("past_len", fontweight="bold")
    ax2.set_ylabel("PIM_MAC repeats", fontweight="bold")
    ax2.set_title("Panel C: Attention compute scales with context", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig4_attention_decomposition")


# ───────────────────────────────────────────────────────────────────────
# FFN/SwiGLU and MoE Decomposition
# ───────────────────────────────────────────────────────────────────────


def gen_figure_5(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_ffn_decomposition,
        collect_moe_decomposition,
        collect_ffn_paper,
        collect_moe_paper,
    )

    if use_tiny:
        ffn_data = collect_ffn_decomposition()
        moe_data = collect_moe_decomposition()
    else:
        ffn_data = collect_ffn_paper()
        moe_data = collect_moe_paper()

    fig = plt.figure(figsize=(15, 7.4))

    # --- Panel A: FFN ---
    ax1 = fig.add_subplot(2, 2, 1)
    ffn_counts = ffn_data.get("operator_counts", ffn_data["concrete_counts"])

    _plot_repeat_expanded_counts(ax1, ffn_counts)
    ax1.set_ylabel("Repeat-expanded op/request or semantic-residency count", fontweight="bold")
    ax1.set_title(
        f"Panel A: FFN/SwiGLU traffic, residency, and PIM MACs\n"
        f"(hidden={ffn_data['hidden_size']}, ffn_hidden={ffn_data['ffn_hidden_size']}, "
        f"{ffn_data['datatype']}, act={ffn_data['activation']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1_bcast = fig.add_subplot(2, 2, 2)
    _plot_pim_bcast_modes(
        ax1_bcast,
        ffn_data.get("pim_bcast_by_mode", {}),
        title="Panel B: FFN true PIM_BCAST setup only\n(residency is semantic-only)",
    )

    # --- Panel B: MoE ---
    ax2 = fig.add_subplot(2, 2, 3)
    moe_counts = moe_data.get("operator_counts", moe_data["concrete_counts"])

    _plot_repeat_expanded_counts(ax2, moe_counts)
    ax2.set_ylabel("Repeat-expanded op/request or semantic-residency count", fontweight="bold")
    ax2.set_title(
        f"Panel C: MoE selected-expert traffic, residency, and PIM MACs\n"
        f"({moe_data['num_experts']} experts, top-{moe_data['top_k']}, "
        f"selected {moe_data['selected_experts']}, {moe_data['datatype']})",
        fontsize=10,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2_bcast = fig.add_subplot(2, 2, 4)
    _plot_pim_bcast_modes(
        ax2_bcast,
        moe_data.get("pim_bcast_by_mode", {}),
        title="Panel D: MoE true PIM_BCAST setup only\n(residency is semantic-only)",
    )

    fig.tight_layout()
    _save(fig, output_dir, "fig5_ffn_moe_decomposition")


# ───────────────────────────────────────────────────────────────────────
# End-to-End Replay Validation
# ───────────────────────────────────────────────────────────────────────


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_replay_validation_table(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write replay validation as a LaTeX table/prose artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("fig6_replay_validation.png", "fig6_replay_validation.pdf"):
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end replay validation for generated P4 traces. The tiny surrogate rows use reduced parameters for fast replay; Llama2-7B rows use full-depth surrogate dimensions (32 layers, \texttt{past\_len}=1024, \texttt{head\_dim}=128, INT8). PIM\_MAC and PIM\_BCAST columns are repeat-expanded command counts. One INT8 PIM\_MAC repeat represents 32 scalar MACs (64 primitive multiply/add ops). All values are simulator-internal diagnostics (non-silicon-calibrated).}",
        r"\label{tab:p4-replay-validation}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Trace & Semantic records & Concrete opcodes & Status & PIM\_MAC issued & PIM\_BCAST issued & Runtime (ns) \\",
        r"\midrule",
    ]
    for row in rows:
        runtime = row.get("runtime_ns", 0)
        try:
            runtime_text = f"{float(str(runtime)):.1f}"
        except (TypeError, ValueError):
            runtime_text = str(runtime)
        command_counts = row.get("command_counts", {})
        pim_bcast = row.get("pim_bcast_issued", "")
        if isinstance(command_counts, dict):
            pim_bcast = command_counts.get("PIM_BCAST", pim_bcast)
        lines.append(
            " & ".join(
                [
                    str(row.get("trace_name", "")),
                    _latex_escape(row.get("semantic_records", "")),
                    _latex_escape(row.get("concrete_records", "")),
                    _latex_escape(row.get("replay_status", "PASS")),
                    _latex_escape(row.get("pim_mac_issued", "")),
                    _latex_escape(pim_bcast),
                    _latex_escape(runtime_text),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize Status PASS indicates the replay completed successfully (all opcode requests completed). Runtime uses a controller-internal clock (tCK~$=$~0.625\,ns for LPDDR5-6400). The Llama2 dense-decoder rows are structured workload surrogates, not full end-to-end model execution.",
            r"\end{table}",
            "",
        ]
    )
    path = output_dir / "p4_replay_validation.tex"
    path.write_text("\n".join(lines))
    return path


def gen_figure_6(output_dir: Path) -> None:
    from tests.analysis.figures import p4_figure_data

    rows = p4_figure_data.collect_replay_stats()
    rows.extend(p4_figure_data.collect_llama2_7b_replay_stats())
    rows.extend(p4_figure_data.collect_llama2_13b_replay_stats())
    write_replay_validation_table(rows, _paper_table_dir(output_dir))


def _first_latency(replay_rows: list[dict[str, object]]) -> tuple[float, float]:
    if not replay_rows:
        return 0.0, 0.0
    row = replay_rows[0]
    return float(row.get("runtime_ns", 0) or 0), float(row.get("cycles", 0) or 0)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _normalize_llama2_scaling_model(data: dict[str, object]) -> dict[str, object]:
    replay_rows = data.get("backend_replay_stats", [])
    if isinstance(replay_rows, list):
        replay_stats = {}
        for row in replay_rows:
            if not isinstance(row, dict):
                continue
            trace_name = str(row.get("trace_name", ""))
            mode = "cold_start" if "cold" in trace_name else "steady_state"
            replay_stats[mode] = {
                "runtime_ns": row.get("runtime_ns", 0),
                "cycles": row.get("cycles", 0),
                "command_counts": row.get("command_counts", {}),
            }
    else:
        replay_stats = dict(data.get("backend_replay_stats", {}))

    return {
        "model_name": data.get("model_name", data.get("manifest_name", "Llama2")),
        "dimensions": {
            "num_layers": data.get("num_layers", 0),
            "hidden_size": data.get("hidden_size", 0),
            "num_heads": data.get("num_heads", 0),
            "head_dim": data.get("head_dim", 0),
            "ffn_hidden_size": data.get("ffn_hidden_size", 0),
            "past_len": data.get("past_len", 0),
        },
        "per_layer_pim_mac_buckets": {
            "qkvo_projection": data.get("qkvo_projection_pim_mac_per_layer", 0),
            "attention": data.get("attention_pim_mac_per_layer", 0),
            "ffn": data.get("ffn_pim_mac_per_layer", 0),
        },
        "replay_stats": replay_stats,
        "command_counts": data.get("concrete_counts", {}),
    }


def write_llama2_scaling_cache(cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    datasets = [
        p4_figure_data.collect_llama2_7b_dense_decoder_data(),
        p4_figure_data.collect_llama2_13b_dense_decoder_data(),
    ]
    payload = {
        "schema_version": 1,
        "figure_id": LLAMA2_SCALING_FIGURE_ID,
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "commit": _git_commit(),
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "models": [_normalize_llama2_scaling_model(data) for data in datasets],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def _llama2_spec_summary(data: dict[str, object]) -> tuple[int, str]:
    """Return approximate FP16 weight size and compact dimension summary for display."""
    model_name = str(data.get("model_name", ""))
    if "13B" in model_name:
        weight_gb = 26
    else:
        weight_gb = 14
    dims = (
        f"{int(data.get('num_layers', 0))}L, "
        f"H={int(data.get('hidden_size', 0))}, "
        f"heads={int(data.get('num_heads', 0))}, "
        f"d={int(data.get('head_dim', 0))}"
    )
    return weight_gb, dims


def _plot_llama2_scaling_payload(payload: dict[str, object]) -> plt.Figure:
    models = [model for model in payload.get("models", []) if isinstance(model, dict)]
    labels = [str(model.get("model_name", "Llama2")) for model in models]
    x = range(len(labels))

    fig = plt.figure(figsize=(12.5, 4.3))
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.35
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]
    for idx, (mode_key, mode_label, color) in enumerate(modes):
        raw_values = [model.get("replay_stats", {}).get(mode_key, {}).get("runtime_ns") for model in models]
        values = [float(value) if value is not None else 0.0 for value in raw_values]
        offsets = [pos + (idx - 0.5) * width for pos in x]
        bars = ax1.bar(offsets, values, width=width, label=mode_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value, raw_value in zip(bars, values, raw_values):
            if raw_value is None:
                bar.set_alpha(0.18)
                bar.set_hatch("//")
                label = "not collected"
            else:
                label = f"{value:,.0f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.03, label, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(1, 2, 2)
    components = [("qkvo_projection", "Q/K/V/O", "#009E73"), ("attention", "Attention", "#0072B2"), ("ffn", "FFN/SwiGLU", "#D55E00")]
    width = 0.24
    for idx, (bucket_key, bucket_label, color) in enumerate(components):
        values = [float(model.get("per_layer_pim_mac_buckets", {}).get(bucket_key, 0)) for model in models]
        offsets = [pos + (idx - 1) * width for pos in x]
        ax2.bar(offsets, values, width=width, label=bucket_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_xticks(list(x), labels)
    ax2.set_yscale("log")
    ax2.set_ylabel("Per-layer PIM_MAC repeats", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_llama2_scaling_figure_from_cache(cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, output_dir: Path = Path("paper/figures")) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_llama2_scaling_payload(payload)
    _save(fig, Path(output_dir), LLAMA2_SCALING_FIGURE_ID)


def _model_dimension_field(model: dict[str, object], key: str, default: object = None) -> object:
    dimensions = model.get("dimensions", {})
    if isinstance(dimensions, dict) and key in dimensions:
        return dimensions[key]
    return model.get(key, default)


def _infer_model_family(model_name: str, model: dict[str, object]) -> str:
    family = model.get("model_family")
    if family:
        return str(family)
    prefix = model_name.split("-", 1)[0]
    return "Llama2" if prefix.startswith("Llama") else prefix


def _cross_model_rows_from_cache(cache_path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{cache_path} must contain a models list")

    rows: list[dict[str, object]] = []
    source_cache = Path(cache_path).as_posix()
    for model in models:
        if not isinstance(model, dict):
            continue
        model_name = str(model.get("model_name", "unknown"))
        replay_stats = model.get("replay_stats", {})
        if not isinstance(replay_stats, dict):
            raise ValueError(f"{cache_path}: model {model_name} missing replay_stats")
        for mode in ("steady_state", "cold_start"):
            stats = replay_stats.get(mode)
            if not isinstance(stats, dict):
                continue
            command_counts = stats.get("command_counts", {})
            if not isinstance(command_counts, dict):
                command_counts = {}
            cycles = stats.get("cycles")
            runtime_ns = stats.get("runtime_ns")
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": _infer_model_family(model_name, model),
                    "mode": mode,
                    "cycles": int(cycles) if cycles is not None else None,
                    "runtime_ns": float(runtime_ns) if runtime_ns is not None else None,
                    "runtime_s": (float(runtime_ns) / 1e9) if runtime_ns is not None else None,
                    "pim_mac_issued": int(stats.get("pim_mac_issued", command_counts.get("PIM_MAC", 0)) or 0),
                    "num_layers": int(_model_dimension_field(model, "num_layers", model.get("model_total_layers", 0)) or 0),
                    "hidden_size": int(_model_dimension_field(model, "hidden_size", 0) or 0),
                    "ffn_hidden_size": int(
                        _model_dimension_field(model, "ffn_hidden_size", _model_dimension_field(model, "expert_hidden_size", 0)) or 0
                    ),
                    "dimension_scope": str(model.get("dimension_scope", "real")),
                    "replay_status": stats.get("replay_status"),
                    "data_source": stats.get("data_source"),
                    "source_cache": source_cache,
                }
            )
    return rows


def write_cross_model_latency_cache(
    cache_path: Path = DEFAULT_CROSS_MODEL_LATENCY_CACHE,
    *,
    source_caches: tuple[Path, ...] = DEFAULT_CROSS_MODEL_SOURCE_CACHES,
) -> Path:
    rows: list[dict[str, object]] = []
    for source in source_caches:
        if not Path(source).exists():
            raise FileNotFoundError(f"missing cross-model source cache: {source}")
        rows.extend(_cross_model_rows_from_cache(Path(source)))
    if not rows:
        raise ValueError("cross-model latency cache would be empty")

    payload = {
        "description": "Cross-model decode backend replay cycles derived from Llama2, Mixtral, OPT, Qwen, and Gemma caches",
        "figure_id": CROSS_MODEL_LATENCY_FIGURE_ID,
        "metric_units": {
            "cycles": "cycles",
            "runtime_ns": "ns",
            "runtime_s": "s",
            "tCK_ns": 0.625,
        },
        "phase": "decode",
        "provenance": {
            "commit": _git_commit(),
            "date": _dt.date.today().isoformat(),
            "derived_from_caches": [Path(source).as_posix() for source in source_caches],
            "replay_mode": "backend_cache_derived",
        },
        "rows": rows,
    }
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def _validate_cross_model_latency_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("figure_id") != CROSS_MODEL_LATENCY_FIGURE_ID:
        raise ValueError(f"cross-model cache figure_id must be {CROSS_MODEL_LATENCY_FIGURE_ID}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cross-model cache must contain non-empty rows")
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"cross-model row {index} must be an object")
        for field in ("model_name", "mode", "cycles", "runtime_ns", "pim_mac_issued", "source_cache"):
            if field not in row:
                raise ValueError(f"cross-model row {index} missing {field}")
        normalized.append(row)
    return normalized


def _plot_cross_model_latency_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_cross_model_latency_payload(payload)
    model_order = []
    for row in rows:
        name = str(row["model_name"])
        if name not in model_order:
            model_order.append(name)

    by_model_mode = {(str(row["model_name"]), str(row["mode"])): row for row in rows}
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]
    x = list(range(len(model_order)))
    width = 0.36

    fig = plt.figure(figsize=(max(13.5, 0.75 * len(model_order)), 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for idx, (mode, label, color) in enumerate(modes):
        offsets = [pos + (idx - 0.5) * width for pos in x]
        cycle_values = []
        runtime_values = []
        for name in model_order:
            row = by_model_mode.get((name, mode), {})
            cycle_values.append(float(row.get("cycles") or 0))
            runtime_values.append(float(row.get("runtime_ns") or 0) / 1e9)
        bars = ax1.bar(offsets, cycle_values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax2.bar(offsets, runtime_values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value in zip(bars, cycle_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, value * 1.05, f"{value/1e9:.2f}B", ha="center", va="bottom", fontsize=7, rotation=90)

    for ax in (ax1, ax2):
        ax.set_xticks(x, model_order, rotation=45, ha="right")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.set_yscale("log")
    ax1.set_ylabel("Backend cycles (log scale)", fontweight="bold")
    ax1.set_title("Panel A: Decode backend cycles", fontsize=10)
    ax2.set_ylabel("Backend runtime (s)", fontweight="bold")
    ax2.set_title("Panel B: Controller-clock runtime", fontsize=10)
    ax1.text(
        0.02,
        0.98,
        "Cache-backed full-model decode surrogate\nnot serving throughput; inherits source-cache caveats",
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    fig.tight_layout()
    return fig


def render_cross_model_latency_figure_from_cache(
    cache_path: Path = DEFAULT_CROSS_MODEL_LATENCY_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_cross_model_latency_payload(payload)
    _save(fig, Path(output_dir), CROSS_MODEL_LATENCY_FIGURE_ID)


def _model_order_from_rows(rows: list[dict[str, object]]) -> list[str]:
    order: list[str] = []
    for row in rows:
        name = str(row["model_name"])
        if name not in order:
            order.append(name)
    return order


def _plot_phase_cycle_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    title: str,
    include_ylabel: bool,
) -> None:
    model_order = _model_order_from_rows(rows)
    by_model_mode = {(str(row["model_name"]), str(row["mode"])): row for row in rows}
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]
    x = list(range(len(model_order)))
    width = 0.36
    all_values: list[float] = []
    for idx, (mode, label, color) in enumerate(modes):
        offsets = [pos + (idx - 0.5) * width for pos in x]
        cycle_values = []
        for name in model_order:
            row = by_model_mode.get((name, mode), {})
            cycle_values.append(float(row.get("cycles") or 0))
        all_values.extend(value for value in cycle_values if value > 0)
        bars = ax.bar(offsets, cycle_values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value in zip(bars, cycle_values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.05,
                    f"{value / 1e9:.2f}B",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )
    ax.set_xticks(x, model_order, rotation=45, ha="right")
    ax.set_yscale("log")
    if all_values:
        ax.set_ylim(min(all_values) * 0.45, max(all_values) * 2.8)
    if include_ylabel:
        ax.set_ylabel("Backend cycles (log scale)", fontweight="bold")
    ax.set_title(title, fontsize=10, pad=18)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_cross_model_decode_prefill_figure_from_cache(
    decode_cache_path: Path = DEFAULT_CROSS_MODEL_LATENCY_CACHE,
    prefill_cache_path: Path = DEFAULT_CROSS_MODEL_PREFILL_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    decode_payload = json.loads(Path(decode_cache_path).read_text(encoding="utf-8"))
    prefill_payload = json.loads(Path(prefill_cache_path).read_text(encoding="utf-8"))
    decode_rows = _validate_cross_model_latency_payload(decode_payload)
    prefill_rows = _validate_cross_model_prefill_payload(prefill_payload)
    fig = plt.figure(figsize=(18.0, 5.2))
    ax_decode = fig.add_subplot(1, 2, 1)
    ax_prefill = fig.add_subplot(1, 2, 2)
    _plot_phase_cycle_panel(ax_decode, decode_rows, title="Decode backend cycles", include_ylabel=True)
    _plot_phase_cycle_panel(ax_prefill, prefill_rows, title="Prefill backend cycles", include_ylabel=False)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.34, top=0.82, wspace=0.16)
    _save(fig, Path(output_dir), CROSS_MODEL_DECODE_PREFILL_FIGURE_ID)


def _cross_model_prefill_formula(model_key: str, *, prompt_len: int) -> dict[str, object]:
    from ramulator.dram.lpddr5_pim import PIM_DATATYPE_RESOURCES
    from ramulator.workload_surrogate.generate_full_transformer import (
        FFN_VARIANT_PROJECTION_COUNTS,
        get_dense_prefill_manifests,
        get_model_spec,
    )
    from tests.analysis.figures import p4_figure_data

    spec = get_model_spec(model_key)
    attention_manifest, _ = get_dense_prefill_manifests(spec, prompt_len=prompt_len)
    resources = PIM_DATATYPE_RESOURCES[spec.datatype]
    lanes = int(resources["pim_lanes"])
    primitive_ops_per_mac = int(resources["pim_ops_per_mac"])
    num_kv_heads = int(spec.num_kv_heads or spec.num_heads)
    q_proj_dim = int(spec.num_heads) * int(spec.head_dim)
    kv_proj_dim = num_kv_heads * int(spec.head_dim)
    q_projection = p4_figure_data._ceil_div(prompt_len * spec.hidden_size * q_proj_dim, lanes)
    k_projection = p4_figure_data._ceil_div(prompt_len * spec.hidden_size * kv_proj_dim, lanes)
    v_projection = p4_figure_data._ceil_div(prompt_len * spec.hidden_size * kv_proj_dim, lanes)
    o_projection = p4_figure_data._ceil_div(prompt_len * q_proj_dim * spec.hidden_size, lanes)
    qkvo_per_layer = q_projection + k_projection + v_projection + o_projection
    score_tile_tokens = int(attention_manifest["score_tile_tokens"])
    context_tile_tokens = int(attention_manifest["context_tile_tokens"])
    causal_pairs = prompt_len * (prompt_len + 1) // 2
    valid_attention_pairs_per_layer = causal_pairs * int(spec.num_heads)
    attention_per_layer = p4_figure_data._prefill_attention_pim_mac_per_layer(
        prompt_len=prompt_len,
        num_heads=int(spec.num_heads),
        head_dim=int(spec.head_dim),
        lanes=lanes,
        score_tile_tokens=score_tile_tokens,
        context_tile_tokens=context_tile_tokens,
    )
    num_projections = FFN_VARIANT_PROJECTION_COUNTS.get(spec.ffn_variant, 3)
    ffn_per_layer = num_projections * p4_figure_data._ceil_div(
        prompt_len * spec.hidden_size * spec.ffn_hidden_size,
        lanes,
    )
    return {
        "model_name": spec.name,
        "model_family": _infer_model_family(spec.name, {}),
        "model_key": model_key,
        "model_total_layers": int(spec.num_layers),
        "hidden_size": int(spec.hidden_size),
        "ffn_hidden_size": int(spec.ffn_hidden_size),
        "ffn_variant": spec.ffn_variant,
        "activation": spec.activation,
        "num_heads": int(spec.num_heads),
        "num_kv_heads": num_kv_heads,
        "head_dim": int(spec.head_dim),
        "datatype": spec.datatype,
        "citation": spec.citation,
        "prompt_len": int(prompt_len),
        "seq_len": int(prompt_len),
        "prefill_causal_pairs": int(causal_pairs),
        "valid_attention_pairs_per_layer": int(valid_attention_pairs_per_layer),
        "attention_issued_work_elements_per_layer": int(2 * valid_attention_pairs_per_layer * spec.head_dim),
        "score_tile_tokens": score_tile_tokens,
        "context_tile_tokens": context_tile_tokens,
        "pim_mac_lanes": lanes,
        "primitive_ops_per_mac": primitive_ops_per_mac,
        "per_layer_pim_mac_buckets": {
            "qkvo_projection": int(qkvo_per_layer),
            "attention": int(attention_per_layer),
            "ffn": int(ffn_per_layer),
        },
        "kv_residency_policy": attention_manifest.get("residency_policy", {}),
    }


def _cross_model_prefill_part_path(parts_dir: Path, model_key: str, mode: str) -> Path:
    safe_model = model_key.replace("/", "_").replace(".", "_").replace("-", "_")
    safe_mode = mode.replace("/", "_").replace(".", "_").replace("-", "_")
    return parts_dir / f"{safe_model}__{safe_mode}.json"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _cross_model_prefill_payload(
    rows: list[dict[str, object]],
    *,
    prompt_len: int,
    num_layers: int | None,
    modes: tuple[str, ...],
    model_keys: tuple[str, ...],
    generator_version: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "figure_id": CROSS_MODEL_PREFILL_FIGURE_ID,
        "description": "Cross-model dense prefill backend replay cycles using real model dimensions at full depth",
        "phase": "prefill",
        "metric_units": {
            "cycles": "cycles",
            "runtime_ns": "ns",
            "runtime_s": "s",
            "tCK_ns": 0.625,
        },
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "commit": _git_commit(),
            "generator_version": generator_version,
            "prompt_len": int(prompt_len),
            "num_layers_override": int(num_layers) if num_layers is not None else None,
            "modes": list(modes),
            "replay_mode": "backend_full_depth_prefill_replays",
            "model_keys": list(model_keys),
            "excluded_decode_fig18_model_families": {
                "Mixtral": "No MoE prefill trace path is implemented; dense-model comparison only."
            },
            "incremental_cache": True,
        },
        "rows": rows,
        "caveats": [
            "prefill_only_no_decode_mixing",
            "real_model_dimensions_full_depth",
            "uniform_prompt_length_for_cross_model_comparison",
            "READ_zero_expected_kv_from_layer_local_projection_residency",
            "not_flashattention_or_chunked_prefill_runtime",
            "mixtral_excluded_no_moe_prefill_trace_path",
            "bounded_surrogate_not_serving",
            "non_silicon_calibrated",
        ],
    }


def _cross_model_prefill_cache_matches(
    payload: dict[str, object],
    *,
    prompt_len: int,
    num_layers: int | None,
    modes: tuple[str, ...],
) -> bool:
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        return False
    try:
        cached_prompt_len = int(provenance.get("prompt_len", -1))
    except (TypeError, ValueError):
        return False
    return (
        payload.get("figure_id") == CROSS_MODEL_PREFILL_FIGURE_ID
        and payload.get("phase") == "prefill"
        and cached_prompt_len == int(prompt_len)
        and provenance.get("num_layers_override") == (int(num_layers) if num_layers is not None else None)
        and list(provenance.get("modes", [])) == list(modes)
    )


def _ordered_cross_model_prefill_rows(
    rows: list[dict[str, object]],
    *,
    model_keys: tuple[str, ...],
    modes: tuple[str, ...],
) -> list[dict[str, object]]:
    by_key_mode = {(str(row.get("model_key")), str(row.get("mode"))): row for row in rows}
    ordered: list[dict[str, object]] = []
    for model_key in model_keys:
        for mode in modes:
            row = by_key_mode.get((model_key, mode))
            if row is not None:
                ordered.append(row)
    return ordered


def _collect_cross_model_prefill_one_task(task: dict[str, object]) -> dict[str, object]:
    from ramulator.workload_surrogate.generate_full_transformer import get_model_spec
    from tests.analysis.figures.p4_backend_data import collect_all_backend_stats_dense_model_prefill

    model_key = str(task["model_key"])
    mode = str(task["mode"])
    prompt_len = int(task["prompt_len"])
    raw_num_layers = task.get("num_layers")
    num_layers = int(raw_num_layers) if raw_num_layers is not None else None
    part_path = Path(str(task["part_path"]))

    formula = _cross_model_prefill_formula(model_key, prompt_len=prompt_len)
    spec = get_model_spec(model_key)
    actual_layers = spec.num_layers if num_layers is None else int(num_layers)
    model_slug = str(formula["model_name"]).lower().replace("-", "_").replace(".", "_")
    trace_name = f"{model_slug}_{actual_layers}_layer_prefill_P{prompt_len}_{mode}"
    backend = collect_all_backend_stats_dense_model_prefill(
        model_key,
        prompt_len=prompt_len,
        num_layers_override=num_layers,
        modes=(mode,),
    )
    stats = backend[trace_name]
    command_counts = dict(stats.get("command_counts", {}))
    cycles = int(stats.get("cycles", 0) or 0)
    row = {
        **formula,
        "phase": "prefill",
        "mode": mode,
        "materialize_weights": mode == "cold_start",
        "replay_layers": actual_layers,
        "dimension_scope": "real_dimensions_full_depth",
        "trace_name": trace_name,
        "cycles": cycles,
        "runtime_ns": stats.get("runtime_ns"),
        "runtime_s": (float(stats["runtime_ns"]) / 1e9) if stats.get("runtime_ns") is not None else None,
        "pim_mac_issued": int(stats.get("pim_mac_issued", command_counts.get("PIM_MAC", 0)) or 0),
        "pim_bcast_issued": int(stats.get("pim_bcast_issued", command_counts.get("PIM_BCAST", 0)) or 0),
        "pim_mac_density": (int(command_counts.get("PIM_MAC", 0)) / cycles) if cycles > 0 else None,
        "command_counts": command_counts,
        "replay_status": "PASS" if stats.get("replay_ok", False) else "FAIL",
        "data_source": "real_backend_simulation",
    }
    _atomic_write_json(
        part_path,
        {
            "schema_version": 1,
            "figure_id": CROSS_MODEL_PREFILL_FIGURE_ID,
            "phase": "prefill",
            "prompt_len": prompt_len,
            "num_layers_override": num_layers,
            "model_key": model_key,
            "mode": mode,
            "row": row,
        },
    )
    return row


def _load_cross_model_prefill_part(
    part_path: Path,
    *,
    prompt_len: int,
    num_layers: int | None,
    model_key: str,
    mode: str,
) -> dict[str, object] | None:
    if not part_path.exists():
        return None
    try:
        payload = json.loads(part_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        payload.get("figure_id") != CROSS_MODEL_PREFILL_FIGURE_ID
        or payload.get("phase") != "prefill"
        or payload.get("prompt_len") != int(prompt_len)
        or payload.get("num_layers_override") != (int(num_layers) if num_layers is not None else None)
        or payload.get("model_key") != model_key
        or payload.get("mode") != mode
    ):
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def _write_cross_model_prefill_cache_parallel(
    cache_path: Path,
    *,
    prompt_len: int,
    num_layers: int | None,
    modes: tuple[str, ...],
    model_keys: tuple[str, ...],
    workers: int,
    generator_version: str,
) -> Path:
    parts_dir = cache_path.parent / f"{cache_path.stem}_parts"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Seed per-task part files from any valid existing monolithic cache so a
    # sequential run can be resumed by the parallel collector without rework.
    if cache_path.exists():
        try:
            existing_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_payload = {}
        if isinstance(existing_payload, dict) and _cross_model_prefill_cache_matches(
            existing_payload,
            prompt_len=prompt_len,
            num_layers=num_layers,
            modes=modes,
        ):
            for row in existing_payload.get("rows", []):
                if not isinstance(row, dict):
                    continue
                model_key = str(row.get("model_key"))
                mode = str(row.get("mode"))
                if model_key in model_keys and mode in modes:
                    part_path = _cross_model_prefill_part_path(parts_dir, model_key, mode)
                    if not part_path.exists():
                        _atomic_write_json(
                            part_path,
                            {
                                "schema_version": 1,
                                "figure_id": CROSS_MODEL_PREFILL_FIGURE_ID,
                                "phase": "prefill",
                                "prompt_len": int(prompt_len),
                                "num_layers_override": int(num_layers) if num_layers is not None else None,
                                "model_key": model_key,
                                "mode": mode,
                                "row": row,
                            },
                        )

    def _load_all_part_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model_key in model_keys:
            for mode in modes:
                row = _load_cross_model_prefill_part(
                    _cross_model_prefill_part_path(parts_dir, model_key, mode),
                    prompt_len=prompt_len,
                    num_layers=num_layers,
                    model_key=model_key,
                    mode=mode,
                )
                if row is not None:
                    rows.append(row)
        return _ordered_cross_model_prefill_rows(rows, model_keys=model_keys, modes=modes)

    def _write_merged_cache() -> list[dict[str, object]]:
        rows = _load_all_part_rows()
        _atomic_write_json(
            cache_path,
            _cross_model_prefill_payload(
                rows,
                prompt_len=prompt_len,
                num_layers=num_layers,
                modes=modes,
                model_keys=model_keys,
                generator_version=generator_version,
            ),
        )
        return rows

    completed_rows = _write_merged_cache()
    completed = {(str(row.get("model_key")), str(row.get("mode"))) for row in completed_rows}
    total_tasks = len(model_keys) * len(modes)
    tasks: list[dict[str, object]] = []
    for model_key in model_keys:
        for mode in modes:
            if (model_key, mode) in completed:
                print(f"[fig22-prefill] SKIP {model_key} {mode}: part cached", flush=True)
                continue
            tasks.append(
                {
                    "model_key": model_key,
                    "mode": mode,
                    "prompt_len": int(prompt_len),
                    "num_layers": int(num_layers) if num_layers is not None else None,
                    "part_path": str(_cross_model_prefill_part_path(parts_dir, model_key, mode)),
                }
            )

    print(
        f"[fig22-prefill] Parallel collection start: cache={cache_path}, "
        f"parts_dir={parts_dir}, completed={len(completed)}/{total_tasks}, "
        f"pending={len(tasks)}, workers={workers}",
        flush=True,
    )
    if not tasks:
        print(f"[fig22-prefill] COMPLETE: rows={len(completed_rows)}, cache={cache_path}", flush=True)
        return cache_path

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task in tasks:
            print(
                f"[fig22-prefill] SUBMIT model={task['model_key']}, mode={task['mode']}, "
                f"part={task['part_path']}",
                flush=True,
            )
            futures[pool.submit(_collect_cross_model_prefill_one_task, task)] = task
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            completed.add((str(row.get("model_key")), str(row.get("mode"))))
            merged_rows = _write_merged_cache()
            print(
                f"[fig22-prefill] DONE {len(completed)}/{total_tasks}: "
                f"model={task['model_key']}, mode={task['mode']}, cycles={row.get('cycles')}, "
                f"cache_rows={len(merged_rows)}, cache={cache_path}",
                flush=True,
            )

    merged_rows = _write_merged_cache()
    print(f"[fig22-prefill] COMPLETE: rows={len(merged_rows)}, cache={cache_path}", flush=True)
    return cache_path


def write_cross_model_prefill_cache(
    cache_path: Path = DEFAULT_CROSS_MODEL_PREFILL_CACHE,
    *,
    prompt_len: int = 12,
    num_layers: int | None = None,
    modes: tuple[str, ...] = ("steady_state", "cold_start"),
    model_keys: tuple[str, ...] = DEFAULT_CROSS_MODEL_PREFILL_MODEL_KEYS,
    workers: int = 1,
) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import (
        FULL_TRANSFORMER_GENERATOR_VERSION,
        get_model_spec,
    )
    from tests.analysis.figures.p4_backend_data import collect_all_backend_stats_dense_model_prefill

    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if num_layers is not None and num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    cache_path = Path(cache_path)

    if workers > 1:
        return _write_cross_model_prefill_cache_parallel(
            cache_path,
            prompt_len=prompt_len,
            num_layers=num_layers,
            modes=modes,
            model_keys=model_keys,
            workers=workers,
            generator_version=FULL_TRANSFORMER_GENERATOR_VERSION,
        )

    def _payload_for(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "figure_id": CROSS_MODEL_PREFILL_FIGURE_ID,
            "description": "Cross-model dense prefill backend replay cycles using real model dimensions at full depth",
            "phase": "prefill",
            "metric_units": {
                "cycles": "cycles",
                "runtime_ns": "ns",
                "runtime_s": "s",
                "tCK_ns": 0.625,
            },
            "provenance": {
                "date": _dt.date.today().isoformat(),
                "commit": _git_commit(),
                "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
                "prompt_len": int(prompt_len),
                "num_layers_override": int(num_layers) if num_layers is not None else None,
                "modes": list(modes),
                "replay_mode": "backend_full_depth_prefill_replays",
                "model_keys": list(model_keys),
                "excluded_decode_fig18_model_families": {
                    "Mixtral": "No MoE prefill trace path is implemented; dense-model comparison only."
                },
                "incremental_cache": True,
            },
            "rows": rows,
            "caveats": [
                "prefill_only_no_decode_mixing",
                "real_model_dimensions_full_depth",
                "uniform_prompt_length_for_cross_model_comparison",
                "READ_zero_expected_kv_from_layer_local_projection_residency",
                "not_flashattention_or_chunked_prefill_runtime",
                "mixtral_excluded_no_moe_prefill_trace_path",
                "bounded_surrogate_not_serving",
                "non_silicon_calibrated",
            ],
        }

    def _write_incremental(rows: list[dict[str, object]]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(_payload_for(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(cache_path)

    rows: list[dict[str, object]] = []
    if cache_path.exists():
        try:
            existing_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            existing_prov = existing_payload.get("provenance", {})
            if (
                existing_payload.get("figure_id") == CROSS_MODEL_PREFILL_FIGURE_ID
                and existing_payload.get("phase") == "prefill"
                and isinstance(existing_prov, dict)
                and int(existing_prov.get("prompt_len", -1)) == int(prompt_len)
                and existing_prov.get("num_layers_override") == (int(num_layers) if num_layers is not None else None)
                and list(existing_prov.get("modes", [])) == list(modes)
            ):
                existing_rows = existing_payload.get("rows", [])
                if isinstance(existing_rows, list):
                    rows = [row for row in existing_rows if isinstance(row, dict)]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            rows = []

    completed = {(str(row.get("model_key")), str(row.get("mode"))) for row in rows}
    total_tasks = len(model_keys) * len(modes)
    print(
        f"[fig22-prefill] Resumable collection start: cache={cache_path}, "
        f"completed={len(completed)}/{total_tasks}, prompt_len={prompt_len}, "
        f"num_layers={'full' if num_layers is None else num_layers}, modes={list(modes)}",
        flush=True,
    )
    _write_incremental(rows)
    for model_key in model_keys:
        formula = _cross_model_prefill_formula(model_key, prompt_len=prompt_len)
        spec = get_model_spec(model_key)
        actual_layers = spec.num_layers if num_layers is None else int(num_layers)
        pending_modes = tuple(mode for mode in modes if (model_key, mode) not in completed)
        if not pending_modes:
            print(f"[fig22-prefill] SKIP {model_key}: all modes cached", flush=True)
            continue
        model_slug = str(formula["model_name"]).lower().replace("-", "_").replace(".", "_")
        for mode in pending_modes:
            ordinal = len(completed) + 1
            print(
                f"[fig22-prefill] START {ordinal}/{total_tasks}: "
                f"model={model_key}, mode={mode}, layers={actual_layers}, prompt_len={prompt_len}",
                flush=True,
            )
            backend = collect_all_backend_stats_dense_model_prefill(
                model_key,
                prompt_len=prompt_len,
                num_layers_override=num_layers,
                modes=(mode,),
            )
            trace_name = f"{model_slug}_{actual_layers}_layer_prefill_P{prompt_len}_{mode}"
            stats = backend[trace_name]
            command_counts = dict(stats.get("command_counts", {}))
            cycles = int(stats.get("cycles", 0) or 0)
            rows.append(
                {
                    **formula,
                    "phase": "prefill",
                    "mode": mode,
                    "materialize_weights": mode == "cold_start",
                    "replay_layers": actual_layers,
                    "dimension_scope": "real_dimensions_full_depth",
                    "trace_name": trace_name,
                    "cycles": cycles,
                    "runtime_ns": stats.get("runtime_ns"),
                    "runtime_s": (float(stats["runtime_ns"]) / 1e9) if stats.get("runtime_ns") is not None else None,
                    "pim_mac_issued": int(stats.get("pim_mac_issued", command_counts.get("PIM_MAC", 0)) or 0),
                    "pim_bcast_issued": int(stats.get("pim_bcast_issued", command_counts.get("PIM_BCAST", 0)) or 0),
                    "pim_mac_density": (int(command_counts.get("PIM_MAC", 0)) / cycles) if cycles > 0 else None,
                    "command_counts": command_counts,
                    "replay_status": "PASS" if stats.get("replay_ok", False) else "FAIL",
                    "data_source": "real_backend_simulation",
                }
            )
            completed.add((model_key, mode))
            _write_incremental(rows)
            print(
                f"[fig22-prefill] DONE {len(completed)}/{total_tasks}: "
                f"model={model_key}, mode={mode}, cycles={cycles}, "
                f"cache_rows={len(rows)}, cache={cache_path}",
                flush=True,
            )

    _write_incremental(rows)
    print(f"[fig22-prefill] COMPLETE: rows={len(rows)}, cache={cache_path}", flush=True)
    return cache_path


def _validate_cross_model_prefill_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("figure_id") != CROSS_MODEL_PREFILL_FIGURE_ID:
        raise ValueError(f"cross-model prefill cache figure_id must be {CROSS_MODEL_PREFILL_FIGURE_ID}")
    if payload.get("phase") != "prefill":
        raise ValueError("cross-model prefill cache must declare phase='prefill'")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cross-model prefill cache must contain non-empty rows")
    required = {"model_name", "mode", "cycles", "runtime_ns", "pim_mac_issued", "prompt_len", "replay_layers"}
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"cross-model prefill row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"cross-model prefill row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_cross_model_prefill_payload(
    payload: dict[str, object],
) -> plt.Figure:
    """Plot the cross-model prefill-cycle comparison as a standalone figure."""
    rows = _validate_cross_model_prefill_payload(payload)
    model_order: list[str] = []
    for row in rows:
        name = str(row["model_name"])
        if name not in model_order:
            model_order.append(name)
    by_model_mode = {(str(row["model_name"]), str(row["mode"])): row for row in rows}
    modes = [
        ("steady_state", "Steady-state", "#0072B2"),
        ("cold_start", "Cold-start", "#D55E00"),
    ]
    x = list(range(len(model_order)))
    width = 0.36

    fig = plt.figure(figsize=(max(10.5, 0.62 * len(model_order)), 4.8))
    ax1 = fig.add_subplot(1, 1, 1)

    all_cycle_values: list[float] = []
    for idx, (mode, label, color) in enumerate(modes):
        offsets = [pos + (idx - 0.5) * width for pos in x]
        cycle_values = []
        for name in model_order:
            row = by_model_mode.get((name, mode), {})
            cycle_values.append(float(row.get("cycles") or 0))
        all_cycle_values.extend(value for value in cycle_values if value > 0)
        bars = ax1.bar(offsets, cycle_values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value in zip(bars, cycle_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, value * 1.05, f"{value/1e9:.2f}B", ha="center", va="bottom", fontsize=7, rotation=90)

    ax1.set_xticks(x, model_order, rotation=45, ha="right")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_yscale("log")
    if all_cycle_values:
        ax1.set_ylim(min(all_cycle_values) * 0.45, max(all_cycle_values) * 2.5)
    ax1.set_ylabel("Backend cycles (log scale)", fontweight="bold")
    ax1.set_title("Cross-model prefill backend cycles", fontsize=10, pad=18)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.36, top=0.82)
    return fig


def _plot_cross_model_prefill_attention_scaling_payload(
    sweep_payload: dict[str, object],
) -> plt.Figure:
    """Plot standalone causal-attention command scaling.

    The attention series is derived from backend-replayed concrete PIM_MAC input
    by subtracting the linear Q/K/V/O and FFN buckets from the replayed total.
    The plot intentionally excludes backend cycles so the message is only the
    causal-attention O(n^2) command-growth relation.
    """
    sweep_rows: list[dict[str, object]] = []
    sweep_model = "Llama2-7B"
    try:
        sweep_rows = _validate_prefill_prompt_sweep_payload(sweep_payload)
    except ValueError:
        sweep_rows = []
    if isinstance(sweep_payload.get("model"), str):
        sweep_model = str(sweep_payload["model"])

    steady_sweep = [row for row in sweep_rows if str(row.get("mode")) == "steady_state"] or sweep_rows
    steady_sweep.sort(key=lambda row: int(row["prompt_len"]))
    if not steady_sweep:
        raise ValueError("prefill attention scaling figure requires non-empty sweep rows")
    prompt_lengths = [int(row["prompt_len"]) for row in steady_sweep]
    attention_pim_mac = []
    for row in steady_sweep:
        if "cycles" not in row or "pim_mac" not in row:
            raise ValueError("prefill sweep rows must include backend replay cycles and pim_mac")
        pim_mac = float(row["pim_mac"])
        buckets = row.get("per_layer_pim_mac_buckets", {})
        if not isinstance(buckets, dict):
            raise ValueError("prefill sweep rows must include per_layer_pim_mac_buckets")
        num_layers = float(row.get("num_layers", 1))
        qkvo = float(buckets.get("qkvo_projection", 0)) * num_layers
        ffn = float(buckets.get("ffn", 0)) * num_layers
        attention = pim_mac - qkvo - ffn
        if attention <= 0 and "attention" in buckets:
            attention = float(buckets["attention"]) * num_layers
        if pim_mac <= 0:
            raise ValueError("prefill sweep backend replay pim_mac must be positive")
        if attention <= 0:
            raise ValueError("prefill sweep backend replay data must imply positive attention PIM_MAC")
        attention_pim_mac.append(attention)

    extended_prompt_lengths: list[int] = []
    extended_attention: list[float] = []
    if sweep_model == "Llama2-7B":
        for prompt_len in [32, 64, 128, 256, 512]:
            if prompt_len <= max(prompt_lengths):
                continue
            try:
                formula = _cross_model_prefill_formula("llama2-7b", prompt_len=prompt_len)
                buckets = formula.get("per_layer_pim_mac_buckets", {})
                if isinstance(buckets, dict):
                    extended_prompt_lengths.append(prompt_len)
                    extended_attention.append(float(buckets["attention"]) * float(formula["model_total_layers"]))
            except Exception:
                extended_prompt_lengths = []
                extended_attention = []
                break

    fit_x = np.array(prompt_lengths + extended_prompt_lengths, dtype=float)
    fit_y = np.array(attention_pim_mac + extended_attention, dtype=float)
    if len(fit_x) >= 3:
        coefficients = np.polyfit(fit_x, fit_y, deg=2)
    else:
        coefficients = np.array([0.0, fit_y[-1] / (fit_x[-1] ** 2), 0.0])
    fit_grid = np.linspace(min(fit_x), max(fit_x), 200)
    fit_values = np.polyval(coefficients, fit_grid)
    normalized_x = prompt_lengths + extended_prompt_lengths
    normalized_y = [value / (n * n) for n, value in zip(normalized_x, attention_pim_mac + extended_attention)]

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax_norm = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(
        prompt_lengths,
        attention_pim_mac,
        marker="o",
        color="#0072B2",
        label="backend-replay-derived attention",
        linewidth=1.8,
        markersize=5,
    )
    if extended_prompt_lengths:
        ax.plot(
            extended_prompt_lengths,
            extended_attention,
            marker="o",
            color="#56B4E9",
            label="trace-derived larger-n extension",
            linewidth=1.3,
            markersize=4,
            linestyle="--",
            markerfacecolor="white",
        )
    ax.plot(fit_grid, fit_values, color="#D55E00", linestyle=":", linewidth=2.0, label="quadratic fit")
    for n, value in zip(prompt_lengths, attention_pim_mac):
        label = f"{value/1e6:.1f}M" if value >= 1e6 else f"{value/1e3:.1f}K"
        ax.text(n, value * 1.035, label, ha="center", va="bottom", fontsize=7, color="#333333")
    for n, value in zip(extended_prompt_lengths, extended_attention):
        ax.text(n, value * 1.035, f"{value/1e6:.1f}M", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(min(normalized_x) * 0.86, max(normalized_x) * 1.14)
    ax.set_ylim(min(attention_pim_mac + extended_attention) * 0.55, max(attention_pim_mac + extended_attention) * 1.7)
    ax.set_ylabel("Attention PIM_MAC commands", fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1e6:.1f}M"))
    ax.set_title("Quadratic scaling of causal-attention PIM commands during prefill (Llama2-7B)", fontsize=10, pad=12)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax_norm.plot(normalized_x, normalized_y, marker=".", color="#009E73", linewidth=1.4)
    ax_norm.axhline(float(np.mean(normalized_y)), color="#666666", linestyle=":", linewidth=1.0)
    ax_norm.set_xscale("log", base=2)
    ax_norm.set_xlabel("Prefill prompt length n (tokens)", fontweight="bold")
    ax_norm.set_ylabel("Commands / n²", fontweight="bold")
    ax_norm.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value/1e3:.1f}K"))
    ax_norm.set_title("Normalized check: near-constant commands/n² indicates quadratic scaling", fontsize=9, pad=4)
    tick_values = sorted(set(prompt_lengths + extended_prompt_lengths))
    ax_norm.set_xticks(tick_values, [str(n) for n in tick_values])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_norm.spines["top"].set_visible(False)
    ax_norm.spines["right"].set_visible(False)
    return fig


def render_cross_model_prefill_figure_from_cache(
    cache_path: Path = DEFAULT_CROSS_MODEL_PREFILL_CACHE,
    output_dir: Path = Path("paper/figures"),
    sweep_cache_path: Path = DEFAULT_PREFILL_PROMPT_SWEEP_CACHE,
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_cross_model_prefill_payload(payload)
    _save(fig, Path(output_dir), CROSS_MODEL_PREFILL_FIGURE_ID)
    if DEFAULT_CROSS_MODEL_LATENCY_CACHE.exists():
        render_cross_model_decode_prefill_figure_from_cache(
            DEFAULT_CROSS_MODEL_LATENCY_CACHE,
            Path(cache_path),
            Path(output_dir),
        )


def _validate_decode_context_sweep_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != 1:
        raise ValueError("decode context sweep cache schema_version must be 1")
    if payload.get("figure_id") != DECODE_CONTEXT_SWEEP_FIGURE_ID:
        raise ValueError(f"decode context sweep cache figure_id must be {DECODE_CONTEXT_SWEEP_FIGURE_ID}")
    if payload.get("phase") != "decode" or payload.get("seq_len") != 1:
        raise ValueError("decode context sweep cache must declare phase='decode' and seq_len=1")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("decode context sweep cache must contain non-empty rows")
    required = {"past_len", "mode", "status", "runtime_ns", "cycles", "pim_mac", "pim_bcast", "pim_mac_density"}
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"decode context sweep row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"decode context sweep row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_decode_context_sweep_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_decode_context_sweep_payload(payload)
    modes = sorted({str(row["mode"]) for row in rows})
    colors = {"steady_state": "#0072B2", "cold_start": "#D55E00"}
    labels = {"steady_state": "Steady-state", "cold_start": "Cold-start"}

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for mode in modes:
        subset = [row for row in rows if str(row["mode"]) == mode]
        subset.sort(key=lambda row: int(row["past_len"]))
        x = [int(row["past_len"]) for row in subset]
        runtime = [float(row["runtime_ns"]) for row in subset]
        pim_mac = [float(row["pim_mac"]) for row in subset]
        density = [float(row["pim_mac_density"]) for row in subset]
        color = colors.get(mode, "#009E73")
        label = labels.get(mode, mode.replace("_", " ").title())
        ax1.plot(x, runtime, marker="o", color=color, label=label, linewidth=1.8)
        ax2.plot(x, pim_mac, marker="s", color=color, label=label, linewidth=1.8)
        ax3.plot(x, density, marker="^", color=color, label=label, linewidth=1.8)

    xlabel = "Decode context length (past_len), seq_len=1"
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    ax1.set_title("Panel A: Runtime diagnostic", fontsize=10)
    ax2.set_ylabel("Replayed PIM_MAC repeats", fontweight="bold")
    ax2.set_title("Panel B: Concrete opcode count", fontsize=10)
    ax3.set_ylabel("PIM_MAC repeats / cycle", fontweight="bold")
    ax3.set_title("Panel C: Command density", fontsize=10)

    caveat = (
        "Llama2-7B bounded full-depth decode surrogate\n"
        "decode-only, seq_len=1; x-axis varies past_len\n"
        "simulator-internal backend diagnostics"
    )
    ax1.text(0.02, 0.98, caveat, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))
    fig.tight_layout()
    return fig


def write_decode_context_sweep_cache(
    cache_path: Path = DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
    *,
    past_len_values: list[int] | None = None,
    modes: tuple[str, ...] | None = None,
) -> Path:
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    payload = p4_figure_data.collect_llama2_7b_decode_context_length_sweep(
        past_len_values=past_len_values,
        modes=modes,
    )
    payload.setdefault("provenance", {})
    if isinstance(payload["provenance"], dict):
        payload["provenance"].setdefault("date", _dt.date.today().isoformat())
        payload["provenance"].setdefault("commit", _git_commit())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_decode_context_sweep_figure_from_cache(
    cache_path: Path = DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_decode_context_sweep_payload(payload)
    _save(fig, Path(output_dir), DECODE_CONTEXT_SWEEP_FIGURE_ID)


def _validate_generated_token_sweep_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != 1:
        raise ValueError("generated token sweep cache schema_version must be 1")
    if payload.get("figure_id") != GENERATED_TOKEN_SWEEP_FIGURE_ID:
        raise ValueError(f"generated token sweep cache figure_id must be {GENERATED_TOKEN_SWEEP_FIGURE_ID}")
    if payload.get("phase") != "decode" or payload.get("seq_len_per_step") != 1:
        raise ValueError("generated token sweep cache must declare phase='decode' and seq_len_per_step=1")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generated token sweep cache must contain non-empty rows")
    required = {
        "generated_token_index",
        "generated_tokens_total",
        "past_len",
        "mode",
        "status",
        "runtime_ns",
        "cumulative_runtime_ns",
        "cycles",
        "pim_mac",
        "pim_bcast",
        "pim_mac_density",
    }
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"generated token sweep row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"generated token sweep row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_generated_token_sweep_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_generated_token_sweep_payload(payload)
    modes = sorted({str(row["mode"]) for row in rows})
    colors = {"steady_state": "#0072B2", "cold_start": "#D55E00"}
    labels = {"steady_state": "Steady-state", "cold_start": "Cold-start"}

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for mode in modes:
        subset = [row for row in rows if str(row["mode"]) == mode]
        subset.sort(key=lambda row: int(row["generated_tokens_total"]))
        x = [int(row["generated_tokens_total"]) for row in subset]
        cumulative_runtime = [float(row["cumulative_runtime_ns"]) for row in subset]
        per_token_runtime = [float(row["runtime_ns"]) for row in subset]
        pim_mac = [float(row["pim_mac"]) for row in subset]
        color = colors.get(mode, "#009E73")
        label = labels.get(mode, mode.replace("_", " ").title())
        ax1.plot(x, cumulative_runtime, marker="o", color=color, label=label, linewidth=1.8)
        ax2.plot(x, per_token_runtime, marker="s", color=color, label=label, linewidth=1.8)
        ax3.plot(x, pim_mac, marker="^", color=color, label=label, linewidth=1.8)

    xlabel = "Generated tokens total (independent single-token replays, seq_len=1)"
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylabel("Cumulative backend runtime (ns)", fontweight="bold")
    ax1.set_title("Panel A: Cumulative runtime", fontsize=10)
    ax2.set_ylabel("Per-token backend runtime (ns)", fontweight="bold")
    ax2.set_title("Panel B: Per-token runtime", fontsize=10)
    ax3.set_ylabel("Replayed PIM_MAC repeats / token", fontweight="bold")
    ax3.set_title("Panel C: Per-token command count", fontsize=10)

    caveat = (
        "Llama2-7B bounded decode surrogate\n"
        "independent single-token backend replays\n"
        "past_len grows by token; seq_len=1 per replay"
    )
    ax1.text(0.02, 0.98, caveat, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))
    fig.tight_layout()
    return fig


def write_generated_token_sweep_cache(
    cache_path: Path = DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
    *,
    initial_past_len: int = 1024,
    num_generated_tokens: int = 4,
    modes: tuple[str, ...] | None = None,
) -> Path:
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    payload = p4_figure_data.collect_llama2_7b_generated_token_sweep(
        initial_past_len=initial_past_len,
        num_generated_tokens=num_generated_tokens,
        modes=modes,
    )
    payload.setdefault("provenance", {})
    if isinstance(payload["provenance"], dict):
        payload["provenance"].setdefault("date", _dt.date.today().isoformat())
        payload["provenance"].setdefault("commit", _git_commit())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_generated_token_sweep_figure_from_cache(
    cache_path: Path = DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_generated_token_sweep_payload(payload)
    _save(fig, Path(output_dir), GENERATED_TOKEN_SWEEP_FIGURE_ID)


def _validate_prefill_prompt_sweep_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != 1:
        raise ValueError("prefill prompt sweep cache schema_version must be 1")
    if payload.get("figure_id") != PREFILL_PROMPT_SWEEP_FIGURE_ID:
        raise ValueError(f"prefill prompt sweep cache figure_id must be {PREFILL_PROMPT_SWEEP_FIGURE_ID}")
    if payload.get("phase") != "prefill":
        raise ValueError("prefill prompt sweep cache must declare phase='prefill'")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prefill prompt sweep cache must contain non-empty rows")
    required = {
        "prompt_len",
        "mode",
        "status",
        "runtime_ns",
        "cycles",
        "pim_mac",
        "pim_bcast",
        "pim_mac_density",
        "prefill_causal_pairs",
        "valid_attention_pairs_per_layer",
        "attention_issued_work_elements_per_layer",
    }
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prefill prompt sweep row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"prefill prompt sweep row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_prefill_prompt_sweep_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_prefill_prompt_sweep_payload(payload)
    modes = sorted({str(row["mode"]) for row in rows})
    colors = {"steady_state": "#0072B2", "cold_start": "#D55E00"}
    labels = {"steady_state": "Steady-state", "cold_start": "Cold-start"}

    fig = plt.figure(figsize=(13.8, 8.2))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for mode in modes:
        subset = [row for row in rows if str(row["mode"]) == mode]
        subset.sort(key=lambda row: int(row["prompt_len"]))
        x = [int(row["prompt_len"]) for row in subset]
        runtime = [float(row["runtime_ns"]) for row in subset]
        pim_mac = [float(row["pim_mac"]) for row in subset]
        causal_pairs = [float(row["prefill_causal_pairs"]) for row in subset]
        color = colors.get(mode, "#009E73")
        label = labels.get(mode, mode.replace("_", " ").title())
        ax1.plot(x, runtime, marker="o", color=color, label=label, linewidth=1.8)
        ax3.plot(x, pim_mac, marker="s", color=color, label=label, linewidth=1.8)
        ax4.plot(x, causal_pairs, marker="^", color=color, label=f"valid causal pairs ({label})", linewidth=1.8)

    steady_rows = [row for row in rows if str(row["mode"]) == "steady_state"] or rows
    steady_rows.sort(key=lambda row: int(row["prompt_len"]))
    x = list(range(len(steady_rows)))
    xlabels = [str(int(row["prompt_len"])) for row in steady_rows]
    component_specs = [
        ("qkvo_projection", "Q/K/V/O", "#009E73"),
        ("attention", "Attention", "#0072B2"),
        ("ffn", "FFN", "#D55E00"),
    ]
    bottom = [0.0] * len(steady_rows)
    for bucket_key, bucket_label, color in component_specs:
        values = [float(dict(row.get("per_layer_pim_mac_buckets", {})).get(bucket_key, 0)) for row in steady_rows]
        ax2.bar(x, values, bottom=bottom, label=bucket_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.88)
        bottom = [prev + value for prev, value in zip(bottom, values)]

    xlabel = "Prefill prompt length (tokens)"
    for ax in (ax1, ax3, ax4):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
    ax2.set_xlabel(xlabel, fontweight="bold")
    ax2.set_xticks(x, xlabels)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    ax1.set_title("Panel A: Runtime diagnostic", fontsize=10)
    ax2.set_ylabel("Per-layer PIM_MAC repeats", fontweight="bold")
    ax2.set_title("Panel B: Component decomposition (Q/K/V/O + Attention + FFN)", fontsize=10)
    ax2.set_yscale("log")
    ax3.set_xlabel(xlabel, fontweight="bold")
    ax3.set_ylabel("Total replayed PIM_MAC repeats", fontweight="bold")
    ax3.set_title("Panel C: Concrete opcode count", fontsize=10)
    ax3.set_yscale("log")
    ax4.set_ylabel("Valid causal QK pairs per layer", fontweight="bold")
    ax4.set_title("Panel D: Valid-pair semantics", fontsize=10)

    caveat = (
        "Llama2-7B full-depth prefill-only backend replay\n"
        "real model dimensions; small prompts for runtime\n"
        "K/V residency is layer-local: READ=0 for prefill attention\n"
        "not FlashAttention/chunked prefill or serving goodput"
    )
    ax1.text(0.02, 0.98, caveat, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))
    fig.tight_layout()
    return fig


def write_prefill_prompt_sweep_cache(
    cache_path: Path = DEFAULT_PREFILL_PROMPT_SWEEP_CACHE,
    *,
    prompt_len_values: list[int] | None = None,
    modes: tuple[str, ...] | None = None,
    workers: int = 1,
) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if prompt_len_values is None:
        prompt_len_values = [2, 4]
    if modes is None:
        modes = ("steady_state",)
    if workers > 1:
        return _write_prefill_prompt_sweep_cache_parallel(
            cache_path,
            prompt_len_values=[int(value) for value in prompt_len_values],
            modes=modes,
            workers=workers,
            generator_version=FULL_TRANSFORMER_GENERATOR_VERSION,
        )
    payload = p4_figure_data.collect_llama2_7b_prefill_prompt_sweep(
        prompt_len_values=prompt_len_values,
        modes=modes,
    )
    payload.setdefault("provenance", {})
    if isinstance(payload["provenance"], dict):
        payload["provenance"].setdefault("date", _dt.date.today().isoformat())
        payload["provenance"].setdefault("commit", _git_commit())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_prefill_prompt_sweep_figure_from_cache(
    cache_path: Path = DEFAULT_PREFILL_PROMPT_SWEEP_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_prefill_prompt_sweep_payload(payload)
    _save(fig, Path(output_dir), PREFILL_PROMPT_SWEEP_FIGURE_ID)
    attention_scaling_fig = _plot_cross_model_prefill_attention_scaling_payload(payload)
    _save(attention_scaling_fig, Path(output_dir), CROSS_MODEL_PREFILL_ATTENTION_SCALING_FIGURE_ID)


def _prefill_prompt_sweep_part_path(parts_dir: Path, prompt_len: int) -> Path:
    return parts_dir / f"P{int(prompt_len)}.json"


def _collect_prefill_prompt_sweep_one_task(task: dict[str, object]) -> list[dict[str, object]]:
    from tests.analysis.figures import p4_figure_data

    prompt_len = int(task["prompt_len"])
    modes = tuple(str(mode) for mode in task.get("modes", ("steady_state",)))
    part_path = Path(str(task["part_path"]))
    payload = p4_figure_data.collect_llama2_7b_prefill_prompt_sweep(
        prompt_len_values=[prompt_len],
        modes=modes,
    )
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    _atomic_write_json(
        part_path,
        {
            "schema_version": 1,
            "figure_id": PREFILL_PROMPT_SWEEP_FIGURE_ID,
            "phase": "prefill",
            "prompt_len": prompt_len,
            "modes": list(modes),
            "rows": rows,
        },
    )
    return rows


def _load_prefill_prompt_sweep_part(part_path: Path, *, prompt_len: int, modes: tuple[str, ...]) -> list[dict[str, object]]:
    if not part_path.exists():
        return []
    try:
        payload = json.loads(part_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if (
        payload.get("figure_id") != PREFILL_PROMPT_SWEEP_FIGURE_ID
        or payload.get("phase") != "prefill"
        or int(payload.get("prompt_len", -1)) != int(prompt_len)
        or list(payload.get("modes", [])) != list(modes)
    ):
        return []
    rows = payload.get("rows", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _prefill_prompt_sweep_payload_from_rows(
    rows: list[dict[str, object]],
    *,
    prompt_len_values: list[int],
    modes: tuple[str, ...],
    generator_version: str,
) -> dict[str, object]:
    max_prompt = max(prompt_len_values)
    return {
        "schema_version": 1,
        "figure_id": PREFILL_PROMPT_SWEEP_FIGURE_ID,
        "description": "Prefill-only prompt-length backend sweep using full-depth real-dimension Llama2-7B traces",
        "phase": "prefill",
        "model": "Llama2-7B",
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "commit": _git_commit(),
            "generator_version": generator_version,
            "replay_mode": "backend_full_depth_prefill_replays",
            "modes": list(modes),
            "incremental_cache": True,
        },
        "sweep": {
            "prompt_len_values": [int(value) for value in prompt_len_values],
            "score_tile_tokens_policy": "min(256, prompt_len)",
            "context_tile_tokens_policy": "min(256, prompt_len)",
            "default_score_tile_tokens_last": min(256, int(max_prompt)),
            "default_context_tile_tokens_last": min(256, int(max_prompt)),
        },
        "rows": rows,
        "caveats": [
            "prefill_only_no_decode_mixing",
            "full_depth_real_llama2_7b_dimensions",
            "longer_prefill_backend_replays_use_increased_collection_timeout",
            "READ_zero_expected_kv_from_layer_local_projection_residency",
            "not_flashattention_or_chunked_prefill_runtime",
            "bounded_surrogate_not_serving",
            "non_silicon_calibrated",
        ],
    }


def _write_prefill_prompt_sweep_cache_parallel(
    cache_path: Path,
    *,
    prompt_len_values: list[int],
    modes: tuple[str, ...],
    workers: int,
    generator_version: str,
) -> Path:
    cache_path = Path(cache_path)
    parts_dir = cache_path.parent / f"{cache_path.stem}_parts"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            existing_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_payload = {}
        if isinstance(existing_payload, dict) and existing_payload.get("figure_id") == PREFILL_PROMPT_SWEEP_FIGURE_ID:
            rows_by_prompt: dict[int, list[dict[str, object]]] = {}
            for row in existing_payload.get("rows", []):
                if isinstance(row, dict) and int(row.get("prompt_len", -1)) in prompt_len_values:
                    rows_by_prompt.setdefault(int(row["prompt_len"]), []).append(row)
            for prompt_len, rows in rows_by_prompt.items():
                part_path = _prefill_prompt_sweep_part_path(parts_dir, prompt_len)
                if not part_path.exists():
                    _atomic_write_json(
                        part_path,
                        {
                            "schema_version": 1,
                            "figure_id": PREFILL_PROMPT_SWEEP_FIGURE_ID,
                            "phase": "prefill",
                            "prompt_len": int(prompt_len),
                            "modes": list(modes),
                            "rows": rows,
                        },
                    )

    def _load_all_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for prompt_len in prompt_len_values:
            rows.extend(_load_prefill_prompt_sweep_part(_prefill_prompt_sweep_part_path(parts_dir, prompt_len), prompt_len=prompt_len, modes=modes))
        mode_order = {mode: index for index, mode in enumerate(modes)}
        rows.sort(key=lambda row: (int(row.get("prompt_len", 0)), mode_order.get(str(row.get("mode")), 999)))
        return rows

    def _write_merged_cache() -> list[dict[str, object]]:
        rows = _load_all_rows()
        _atomic_write_json(
            cache_path,
            _prefill_prompt_sweep_payload_from_rows(
                rows,
                prompt_len_values=prompt_len_values,
                modes=modes,
                generator_version=generator_version,
            ),
        )
        return rows

    completed_rows = _write_merged_cache()
    completed_prompts = {int(row["prompt_len"]) for row in completed_rows if str(row.get("mode")) in modes}
    tasks = [
        {
            "prompt_len": int(prompt_len),
            "modes": list(modes),
            "part_path": str(_prefill_prompt_sweep_part_path(parts_dir, int(prompt_len))),
        }
        for prompt_len in prompt_len_values
        if int(prompt_len) not in completed_prompts
    ]
    print(
        f"[prefill-sweep] Parallel collection start: cache={cache_path}, parts_dir={parts_dir}, "
        f"completed={len(completed_prompts)}/{len(prompt_len_values)}, pending={len(tasks)}, workers={workers}",
        flush=True,
    )
    if not tasks:
        print(f"[prefill-sweep] COMPLETE: rows={len(completed_rows)}, cache={cache_path}", flush=True)
        return cache_path
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_collect_prefill_prompt_sweep_one_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            rows = future.result()
            merged_rows = _write_merged_cache()
            print(
                f"[prefill-sweep] DONE prompt_len={task['prompt_len']}: rows={len(rows)}, cache_rows={len(merged_rows)}",
                flush=True,
            )
    merged_rows = _write_merged_cache()
    print(f"[prefill-sweep] COMPLETE: rows={len(merged_rows)}, cache={cache_path}", flush=True)
    return cache_path


def gen_multi_model_llama2_latency_breakdown(output_dir: Path) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_llama2_7b_dense_decoder_data,
        collect_llama2_13b_dense_decoder_data,
    )

    datasets = [collect_llama2_7b_dense_decoder_data(), collect_llama2_13b_dense_decoder_data()]
    labels = [data.get("model_name", data.get("manifest_name", "Llama2")) for data in datasets]
    runtimes = [_first_latency(data.get("backend_replay_stats", []))[0] for data in datasets]
    cycles = [_first_latency(data.get("backend_replay_stats", []))[1] for data in datasets]

    weight_sizes = [_llama2_spec_summary(data)[0] for data in datasets]
    spec_summaries = [_llama2_spec_summary(data)[1] for data in datasets]

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    x = range(len(labels))
    bars = ax1.bar(list(x), runtimes, color="#0072B2", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    for bar, runtime, cyc in zip(bars, runtimes, cycles):
        ax1.text(bar.get_x() + bar.get_width() / 2, max(runtime, 1) * 1.05,
                 f"{runtime:,.1f} ns\n{int(cyc):,} cyc", ha="center", va="bottom", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(1, 3, 2)
    components = ["Q/K/V/O", "Attention", "FFN/SwiGLU"]
    width = 0.35
    for idx, data in enumerate(datasets):
        values = [
            data["qkvo_projection_pim_mac_per_layer"],
            data["attention_pim_mac_per_layer"],
            data["ffn_pim_mac_per_layer"],
        ]
        offset = (idx - (len(datasets) - 1) / 2) * width
        ax2.bar([pos + offset for pos in range(len(components))], values, width=width,
                label=str(labels[idx]), edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_xticks(list(range(len(components))), components)
    ax2.set_yscale("log")
    ax2.set_ylabel("Per-layer PIM_MAC repeats", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(1, 3, 3)
    weight_bars = ax3.bar(list(x), weight_sizes, color="#009E73", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.set_xticks(list(x), labels)
    ax3.set_ylabel("Approx. FP16 weight size (GB)", fontweight="bold")
    for bar, weight_gb, summary in zip(weight_bars, weight_sizes, spec_summaries):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            max(weight_gb, 1) * 1.03,
            f"~{weight_gb} GB\n{summary}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig9_llama2_7b_13b_models_latency_breakdown")


def _cached_or_collect_llama2_scaling(output_dir: Path) -> None:
    if DEFAULT_LLAMA2_SCALING_CACHE.exists():
        render_llama2_scaling_figure_from_cache(DEFAULT_LLAMA2_SCALING_CACHE, output_dir)
    else:
        print(f"  Skipped Llama2 scaling figure; cache missing at {DEFAULT_LLAMA2_SCALING_CACHE}")
        print("  Run with --collect-llama2-scaling-cache first, then --render-llama2-scaling-cache.")


def _cached_or_collect_decode_context_sweep(output_dir: Path) -> None:
    if DEFAULT_DECODE_CONTEXT_SWEEP_CACHE.exists():
        render_decode_context_sweep_figure_from_cache(DEFAULT_DECODE_CONTEXT_SWEEP_CACHE, output_dir)
    else:
        print(f"  Skipped decode context-length sweep figure; cache missing at {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE}")
        print("  Run with --collect-decode-context-sweep-cache first, then --render-decode-context-sweep-cache.")


def _cached_or_collect_generated_token_sweep(output_dir: Path) -> None:
    if DEFAULT_GENERATED_TOKEN_SWEEP_CACHE.exists():
        render_generated_token_sweep_figure_from_cache(DEFAULT_GENERATED_TOKEN_SWEEP_CACHE, output_dir)
    else:
        print(f"  Skipped generated-token sweep figure; cache missing at {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE}")
        print("  Run with --collect-generated-token-sweep-cache first, then --render-generated-token-sweep-cache.")


def _cached_or_collect_prefill_prompt_sweep(output_dir: Path) -> None:
    if DEFAULT_PREFILL_PROMPT_SWEEP_CACHE.exists():
        render_prefill_prompt_sweep_figure_from_cache(DEFAULT_PREFILL_PROMPT_SWEEP_CACHE, output_dir)
    else:
        print(f"  Skipped prefill prompt-length sweep figure; cache missing at {DEFAULT_PREFILL_PROMPT_SWEEP_CACHE}")
        print("  Run with --collect-prefill-prompt-sweep-cache first, then --render-prefill-prompt-sweep-cache.")


def gen_figure_9_llama2_7b_32_layer(output_dir: Path) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_llama2_7b_dense_decoder_data,
    )

    data = collect_llama2_7b_dense_decoder_data()
    replay_rows = data["backend_replay_stats"]

    fig = plt.figure(figsize=(12, 5))

    # --- Panel A: backend replay command comparison ---
    ax1 = fig.add_subplot(1, 2, 1)
    labels = ["Steady-state", "Cold-start"]
    command_names = ["PIM_MAC", "PIM_BCAST"]
    x = range(len(labels))
    width = 0.35
    for idx, command in enumerate(command_names):
        values = [row["command_counts"].get(command, 0) for row in replay_rows]
        offset = (idx - 0.5) * width
        bars = ax1.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=command,
            color=CMD_COLORS[command],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, max(val, 1) * 1.05,
                     f"{int(val):,}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(list(x), labels)
    ax1.set_yscale("log")
    ax1.set_ylim(0.8, max([row["command_counts"].get(cmd, 0) for row in replay_rows for cmd in command_names] + [1]) * 2.5)
    ax1.set_ylabel("Backend command count", fontweight="bold")
    ax1.set_title("Panel A: Backend replay commands", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    caption = (
        "Backend replay\n"
        "Steady-state vs Cold-start\n"
        "Llama2-7B 32-layer dense-decoder surrogate (hidden=4096, ffn_hidden=11008, "
        "32 heads, head_dim=128, past_len=1024, INT8).\n"
        "Decode-block v2 scope: Q/K/V/O projections plus QK^T + PV attention core; "
        "semantic KV HostRead/HostWrite lower to native READ/WRITE concrete requests; Barrier/Drain remain ordering annotations. "
        "Panel A shows replayed READ/WRITE and PIM opcodes. Excludes RoPE, norm, residual, and softmax hardware cost."
    )
    ax1.text(0.02, 0.98, caption, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    # --- Panel B: per-layer attention vs FFN PIM_MAC ---
    ax2 = fig.add_subplot(1, 2, 2)
    lanes = int(data["pim_mac_lanes"])
    qkvo_per_layer = data["qkvo_projection_pim_mac_per_layer"]
    attention_per_layer = data["attention_pim_mac_per_layer"]
    ffn_per_layer = data["ffn_pim_mac_per_layer"]
    labels = ["Q/K/V/O\nprojections", "Attention\n(QK^T+PV)", "FFN/SwiGLU\n(up+gate+down)"]
    values = [qkvo_per_layer, attention_per_layer, ffn_per_layer]
    bars = ax2.bar(labels, values, color=["#009E73", "#0072B2", "#D55E00"], edgecolor="white",
                   linewidth=0.5, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_ylabel(f"Per-layer PIM_MAC repeats\n({lanes} scalar INT8 MACs/repeat)", fontweight="bold")
    ax2.set_title("Panel B: Modeled per-layer compute slice", fontsize=10)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.12,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig9_llama2_7b_32_layer_breakdown")


# ─── Mixtral-8x7B MoE breakdown figure ──────────────────────────────────

def _plot_mixtral_breakdown_payload(payload: dict[str, object]) -> plt.Figure:
    models = [model for model in payload.get("models", []) if isinstance(model, dict)]
    if not models:
        raise ValueError("Mixtral breakdown cache must contain at least one model")
    model = models[0]
    dims = (
        f"{int(model.get('num_layers', 0))}L, "
        f"H={int(model.get('hidden_size', 0))}, "
        f"E={int(model.get('num_experts', 0))}, "
        f"top_k={int(model.get('top_k', 0))}"
    )

    fig = plt.figure(figsize=(12.5, 4.3))
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]

    # Panel A: backend cycles and PIM_MAC issued
    replay = model.get("replay_stats", {})
    ax1 = fig.add_subplot(1, 2, 1)
    metrics = [("cycles", "Cycles"), ("pim_mac_issued", "PIM_MAC issued")]
    x = range(len(metrics))
    width = 0.35
    for idx, (mode_key, mode_label, color) in enumerate(modes):
        mode_stats = replay.get(mode_key, {}) if isinstance(replay, dict) else {}
        values = []
        for metric_key, _ in metrics:
            val = mode_stats.get(metric_key, 0) if isinstance(mode_stats, dict) else 0
            if metric_key == "cycles":
                val = int(val) if val else 0
            else:
                val = int(val) if val else 0
            values.append(val)
        offsets = [pos + (idx - 0.5) * width for pos in x]
        bars = ax1.bar(offsets, values, width=width, label=mode_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value in zip(bars, values):
            label = f"{value:,.0f}" if value > 1000 else str(value)
            ax1.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.03, label, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(list(x), [m[1] for m in metrics])
    ax1.set_title(f"Panel A: Backend replay — {dims}", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: per-layer PIM_MAC buckets
    ax2 = fig.add_subplot(1, 2, 2)
    buckets = model.get("per_layer_pim_mac_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}
    bucket_order = ["qkvo_projection", "attention", "router", "expert_ffn_fused"]
    bucket_labels = ["Q/K/V/O", "Attention", "Router", "Expert FFN\n(fused)"]
    bucket_colors = ["#009E73", "#0072B2", "#E69F00", "#D55E00"]
    present = [(k, l, c) for k, l, c in zip(bucket_order, bucket_labels, bucket_colors) if k in buckets]
    x_buckets = range(len(present))
    values_buckets = [int(buckets.get(k, 0)) for k, _, _ in present]
    bars2 = ax2.bar(x_buckets, values_buckets, color=[c for _, _, c in present], edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars2, values_buckets):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(x_buckets), [l for _, l, _ in present], fontsize=8)
    ax2.set_yscale("log")
    ax2.set_title("Panel B: Per-layer PIM_MAC repeats", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def write_mixtral_breakdown_cache(cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    data = p4_figure_data.collect_mixtral_8x7b_moe_decoder_data(
        replay_stats_fn=p4_figure_data.collect_mixtral_8x7b_replay_stats if collect_backend else None,
    )
    backend_replay = data.pop("backend_replay_stats", [])
    replay_stats: dict[str, dict[str, object]] = {}
    if isinstance(backend_replay, list):
        for entry in backend_replay:
            if not isinstance(entry, dict):
                continue
            trace_name = str(entry.get("trace_name", ""))
            is_cold = "cold" in trace_name.lower()
            mode = "cold_start" if is_cold else "steady_state"
            replay_stats[mode] = {
                "cycles": int(entry.get("cycles", 0) or 0),
                "runtime_ns": entry.get("runtime_ns"),
                "pim_mac_issued": int(entry.get("pim_mac_issued", 0) or 0),
                "avg_pim_latency_cycles": entry.get("avg_pim_latency_cycles"),
                "command_counts": entry.get("command_counts", {}),
            }

    payload = {
        "schema_version": 1,
        "figure_id": MIXTRAL_BREAKDOWN_FIGURE_ID,
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "commit": _git_commit(),
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "models": [
            {
                "model_name": data.get("model_name", "Mixtral-8x7B"),
                "num_layers": int(data.get("num_layers", 32)),
                "hidden_size": int(data.get("hidden_size", 512)),
                "expert_hidden_size": int(data.get("expert_hidden_size", 2048)),
                "real_hidden_size": int(data.get("real_hidden_size", 4096)),
                "real_expert_hidden_size": int(data.get("real_expert_hidden_size", 14336)),
                "num_experts": int(data.get("num_experts", 8)),
                "top_k": int(data.get("top_k", 2)),
                "num_heads": int(data.get("num_heads", 4)),
                "num_kv_heads": int(data.get("num_kv_heads", 1)),
                "head_dim": int(data.get("head_dim", 128)),
                "past_len": int(data.get("past_len", 1024)),
                "datatype": str(data.get("datatype", "int8")),
                "per_layer_pim_mac_buckets": {
                    "qkvo_projection": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
                    "attention": int(data.get("attention_pim_mac_per_layer", 0)),
                    "router": int(data.get("router_pim_mac_per_layer", 0)),
                    "expert_ffn_fused": int(data.get("expert_ffn_pim_mac_fused_per_layer", 0)),
                    "expert_ffn_real": int(data.get("expert_ffn_pim_mac_real_per_layer", 0)),
                    "expert_ffn_full_dim_real_analytic": int(data.get("expert_ffn_pim_mac_full_dim_real_per_layer", 0)),
                },
                "replay_stats": replay_stats,
                "concrete_counts": data.get("concrete_counts", {}),
                "caveats": [
                    "scaled_dimensions_hidden_512_expert_ffn_2048",
                    "gqa_no_reuse_4to1_ratio",
                    "fused_single_gemv_expert_mac_3x_factor_for_real_swiglu",
                    "deterministic_routing_selected_experts_0_1",
                    "decode_only_seq_len_1",
                    "bounded_surrogate_not_serving",
                    "non_silicon_calibrated",
                ],
            }
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def write_opt_latency_cache(cache_path: Path = DEFAULT_OPT_LATENCY_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    replay = None if collect_backend else (lambda: [])
    models_data = [
        p4_figure_data.collect_opt_125m_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_opt_350m_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_opt_1_3b_dense_decoder_data(replay_stats_fn=replay),
    ]

    def _build_model_entry(data: dict[str, object]) -> dict[str, object]:
        replay_rows = data.get("backend_replay_stats", [])
        replay_stats: dict[str, dict[str, object]] = {}
        for row in replay_rows if isinstance(replay_rows, list) else []:
            if not isinstance(row, dict):
                continue
            trace_name = str(row.get("trace_name", ""))
            mode = "cold_start" if "cold" in trace_name else "steady_state"
            replay_stats[mode] = {
                "cycles": int(row.get("cycles", 0) or 0),
                "runtime_ns": row.get("runtime_ns"),
                "command_counts": row.get("command_counts", {}),
                "pim_mac_issued": int(row.get("pim_mac_issued", 0) or 0),
                "data_source": "real_backend_simulation",
                "replay_status": row.get("replay_status", "PASS"),
            }
        return {
            "model_name": data.get("model_name", "OPT"),
            "model_family": "OPT",
            "dimension_scope": "real",
            "dimensions": {
                "num_layers": int(data.get("num_layers", 0)),
                "hidden_size": int(data.get("hidden_size", 0)),
                "num_heads": int(data.get("num_heads", 0)),
                "head_dim": int(data.get("head_dim", 0)),
                "ffn_hidden_size": int(data.get("ffn_hidden_size", 0)),
                "ffn_variant": "relu_2proj",
                "activation": "relu",
                "past_len": int(data.get("past_len", 512)),
            },
            "per_layer_pim_mac_buckets": {
                "qkvo_projection": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
                "attention": int(data.get("attention_pim_mac_per_layer", 0)),
                "ffn": int(data.get("ffn_pim_mac_per_layer", 0)),
            },
            "command_counts": data.get("concrete_counts", {}),
            "replay_stats": replay_stats,
            "caveats": [
                "full_published_opt_dimensions",
                "backend_replay_required_for_latency_claims",
                "opt_ffn_fc1_relu_fc2",
                "decode_only_not_serving_throughput",
            ],
        }

    payload = {
        "figure_id": OPT_LATENCY_FIGURE_ID,
        "models": [_build_model_entry(d) for d in models_data],
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "commit": _git_commit(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "schema_version": 1,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def write_qwen_gemma_latency_cache(cache_path: Path = DEFAULT_QWEN_GEMMA_LATENCY_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import (
        FULL_TRANSFORMER_GENERATOR_VERSION,
        LLAMA2_70B_NUM_LAYERS,
    )
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    replay = None if collect_backend else (lambda: [])
    models_data = [
        p4_figure_data.collect_qwen25_7b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_qwen25_14b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_qwen25_32b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_qwen25_72b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_gemma_2b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_gemma_7b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_gemma2_9b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data.collect_gemma2_27b_dense_decoder_data(replay_stats_fn=replay),
        p4_figure_data._collect_dense_decoder_data_generic(
            "llama2-70b",
            f"llama2_70b_{LLAMA2_70B_NUM_LAYERS}_layer_dense_decoder",
            None,
            past_len=1024,
        ),
    ]

    def _replay_stats(rows_obj: object) -> dict[str, dict[str, object]]:
        replay_stats: dict[str, dict[str, object]] = {}
        for row in rows_obj if isinstance(rows_obj, list) else []:
            if not isinstance(row, dict):
                continue
            trace_name = str(row.get("trace_name", ""))
            mode = "cold_start" if "cold" in trace_name else "steady_state"
            replay_stats[mode] = {
                "cycles": int(row.get("cycles", 0) or 0),
                "runtime_ns": row.get("runtime_ns"),
                "command_counts": row.get("command_counts", {}),
                "opcodes_sent": int(row.get("opcodes_sent", 0) or 0),
                "opcodes_completed": int(row.get("opcodes_completed", 0) or 0),
                "pim_bcast_issued": int(row.get("pim_bcast_issued", 0) or 0),
                "pim_mac_issued": int(row.get("pim_mac_issued", 0) or 0),
                "data_source": "real_backend_simulation",
                "replay_status": row.get("replay_status", "PASS"),
            }
        return replay_stats

    def _build_qg_model_entry(data: dict[str, object]) -> dict[str, object]:
        model_name = str(data.get("model_name", ""))
        family = "Qwen" if model_name.startswith("Qwen") else "Gemma" if model_name.startswith("Gemma") else "Llama2"
        head_dim = int(data.get("head_dim", 0))
        num_kv_heads = int(data.get("num_kv_heads", data.get("num_heads", 0)) or 0)
        return {
            "model_name": model_name,
            "model_family": family,
            "model_total_layers": int(data.get("num_layers", 0)),
            "num_layers": int(data.get("num_layers", 0)),
            "citation": data.get("citation"),
            "datatype": str(data.get("datatype", "int8")),
            "dimension_scope": "real",
            "hidden_size": int(data.get("hidden_size", 0)),
            "ffn_hidden_size": int(data.get("ffn_hidden_size", 0)),
            "ffn_variant": str(data.get("ffn_variant", "swiglu_3proj")),
            "activation": str(data.get("activation", "silu")),
            "num_heads": int(data.get("num_heads", 0)),
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "kv_proj_dim": num_kv_heads * head_dim,
            "q_proj_dim": int(data.get("hidden_size", 0)),
            "past_len": int(data.get("past_len", 1024)),
            "per_layer_pim_mac_buckets": {
                "qkvo_projection": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
                "attention": int(data.get("attention_pim_mac_per_layer", 0)),
                "ffn": int(data.get("ffn_pim_mac_per_layer", 0)),
            },
            "command_counts": data.get("concrete_counts", {}),
            "replay_stats": _replay_stats(data.get("backend_replay_stats", [])),
            "caveats": [
                "decode_only_seq_len_1",
                "bounded_surrogate_not_serving",
                "non_silicon_calibrated",
                "norm_rope_residual_softcap_accounting_not_lowered",
                "qkv_bias_accounting_not_lowered",
            ],
        }

    payload = {
        "figure_id": QWEN_GEMMA_LATENCY_FIGURE_ID,
        "models": [_build_qg_model_entry(d) for d in models_data],
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "commit": _git_commit(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "includes_llama2_70b_scaling_reference": True,
            "num_layers_override": None,
            "past_len": 1024,
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "schema_version": 1,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_mixtral_breakdown_figure_from_cache(cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE, output_dir: Path = Path("paper/figures")) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_mixtral_breakdown_payload(payload)
    _save(fig, Path(output_dir), MIXTRAL_BREAKDOWN_FIGURE_ID)


# ─── P1: Mixtral vs Llama2 comparison ───────────────────────────────────

def gen_figure_mixtral_vs_llama2(output_dir: Path, *, llama2_cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, mixtral_cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE) -> None:
    """Render Mixtral vs Llama2 comparison from both caches.

    Backend cycles are carried in the caches for provenance, but this figure
    intentionally plots PIM_MAC repeats only: raw cycles would be misleading
    because the Mixtral trace is dimension-scaled while Llama2-7B is full-size.
    """
    llama2_payload = json.loads(llama2_cache_path.read_text(encoding="utf-8")) if llama2_cache_path.exists() else None
    mixtral_payload = json.loads(mixtral_cache_path.read_text(encoding="utf-8")) if mixtral_cache_path.exists() else None

    fig = plt.figure(figsize=(12.5, 4.3))
    models_data: list[dict[str, object]] = []

    if llama2_payload:
        for m in llama2_payload.get("models", []):
            if isinstance(m, dict) and "Llama2-7B" in str(m.get("model_name", "")):
                dims = m.get("dimensions", {}) if isinstance(m.get("dimensions"), dict) else {}
                replay = m.get("replay_stats", {})
                steady = replay.get("steady_state", {}) if isinstance(replay, dict) else {}
                buckets = m.get("per_layer_pim_mac_buckets", {})
                models_data.append({
                    "label": "Llama2-7B",
                    "cycles": int(steady.get("cycles", 0) or 0),
                    "pim_mac": sum(int(buckets.get(k, 0)) for k in ["qkvo_projection", "attention", "ffn"]),
                    "buckets": {
                        "Q/K/V/O": int(buckets.get("qkvo_projection", 0)),
                        "Attention": int(buckets.get("attention", 0)),
                        "FFN/SwiGLU": int(buckets.get("ffn", 0)),
                    },
                })

    if mixtral_payload:
        for m in mixtral_payload.get("models", []):
            if isinstance(m, dict):
                replay = m.get("replay_stats", {})
                steady = replay.get("steady_state", {}) if isinstance(replay, dict) else {}
                buckets = m.get("per_layer_pim_mac_buckets", {})
                models_data.append({
                    "label": "Mixtral-8x7B\n(scaled)",
                    "cycles": int(steady.get("cycles", 0) or 0),
                    "pim_mac": sum(int(buckets.get(k, 0)) for k in ["qkvo_projection", "attention", "router", "expert_ffn_fused"]),
                    "buckets": {
                        "Q/K/V/O": int(buckets.get("qkvo_projection", 0)),
                        "Attention": int(buckets.get("attention", 0)),
                        "Router+Experts": int(buckets.get("router", 0)) + int(buckets.get("expert_ffn_fused", 0)),
                    },
                })

    if not models_data:
        print("  WARNING: No cached data for Mixtral vs Llama2 comparison — skipping fig14")
        return

    # Panel A: normalized per-layer PIM_MAC composition.  The raw totals differ
    # by construction (scaled Mixtral vs full-dim Llama2), so percentages make
    # the intra-model operator mix visible without implying latency fairness.
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.35
    x = range(len(models_data))
    bucket_order = ["Q/K/V/O", "Attention", "FFN/SwiGLU", "Router+Experts"]
    all_bucket_keys = [k for k in bucket_order if any(k in m["buckets"] for m in models_data)]
    colors = {"Q/K/V/O": "#009E73", "Attention": "#0072B2", "FFN/SwiGLU": "#D55E00", "Router+Experts": "#E69F00"}
    bottom = [0.0] * len(models_data)
    for bk in all_bucket_keys:
        vals = []
        for m in models_data:
            total = max(1, int(m.get("pim_mac", 0)))
            vals.append(float(m["buckets"].get(bk, 0)) / total * 100.0)
        bars = ax1.bar(x, vals, width, bottom=bottom, label=bk, color=colors.get(bk, "#999999"), edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_xticks(list(x), [m["label"] for m in models_data], fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Share of per-layer PIM_MAC repeats (%)", fontweight="bold")
    ax1.set_title("Panel A: Per-layer PIM_MAC composition", fontsize=10)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel B: PIM_MAC total
    ax2 = fig.add_subplot(1, 2, 2)
    vals2 = [m["pim_mac"] for m in models_data]
    bars2 = ax2.bar(x, vals2, width * 1.2, color=["#0072B2", "#D55E00"], edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(list(x), [m["label"] for m in models_data], fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("PIM_MAC repeats per layer", fontweight="bold")
    ax2.set_title("Panel B: Total PIM_MAC (1 layer, log scale)", fontsize=10)
    ax2.text(
        0.02,
        0.03,
        "Caveat: Mixtral is scaled (H=512, expert=2048);\n"
        "Llama2-7B is full-dim (H=4096, FFN=11008).\n"
        "Raw backend cycles exist in caches but are not fairness-normalized.",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MIXTRAL_VS_LLAMA2_FIGURE_ID)


# ─── P2: Expert scaling sensitivity ──────────────────────────────────────

def gen_figure_moe_expert_sensitivity(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures import p4_figure_data

    sweep = p4_figure_data.collect_moe_expert_sensitivity_sweep()
    # Aggregate by (num_experts, top_k)
    aggregated: dict[tuple[int, int], dict[str, int]] = {}
    for r in sweep:
        key = (r["_num_experts"], r["_top_k"])
        if key not in aggregated:
            aggregated[key] = {"router_mac": 0, "expert_mac": 0, "total_pim_mac": 0}
        concrete = r.get("concrete_counts", {})
        aggregated[key]["total_pim_mac"] = int(concrete.get("PIM_MAC", 0))
        # Router PIM_MAC is hidden_size × num_experts, ceil-divided by PIM lanes.
        hidden = r.get("hidden_size", 32)
        num_experts = r.get("_num_experts", 4)
        lanes = 32
        aggregated[key]["router_mac"] = max(1, (1 * num_experts * hidden + lanes - 1) // lanes)

    num_experts_keys = sorted(set(k[0] for k in aggregated))
    top_k_keys = sorted(set(k[1] for k in aggregated))

    fig = plt.figure(figsize=(12.5, 4.3))
    # Panel A: total PIM_MAC heatmap/grouped bars
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.8 / len(top_k_keys)
    colors_tk = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    for ti, tk in enumerate(top_k_keys):
        vals = []
        for ne in num_experts_keys:
            vals.append(aggregated.get((ne, tk), {}).get("total_pim_mac", 0))
        offsets = [pos + (ti - (len(top_k_keys) - 1) / 2) * width for pos in range(len(num_experts_keys))]
        ax1.bar(offsets, vals, width=width * 0.9, label=f"top_k={tk}", color=colors_tk[ti % len(colors_tk)], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_xticks(range(len(num_experts_keys)), [str(ne) for ne in num_experts_keys])
    ax1.set_xlabel("num_experts", fontweight="bold")
    ax1.set_ylabel("PIM_MAC repeats", fontweight="bold")
    ax1.set_title("Panel A: PIM_MAC vs num_experts × top_k", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel B: router overhead fraction
    ax2 = fig.add_subplot(1, 2, 2)
    markers = ["o", "s", "^", "D", "P", "X"]
    for ti, tk in enumerate(top_k_keys):
        fracs = []
        for ne in num_experts_keys:
            a = aggregated.get((ne, tk), {})
            total = a.get("total_pim_mac", 1)
            router = a.get("router_mac", 0)
            fracs.append(router / max(1, total) * 100 if total else 0)
        ax2.plot(
            range(len(num_experts_keys)),
            fracs,
            linestyle="None",
            marker=markers[ti % len(markers)],
            label=f"top_k={tk}",
            color=colors_tk[ti % len(colors_tk)],
            markersize=6,
        )
    ax2.set_xticks(range(len(num_experts_keys)), [str(ne) for ne in num_experts_keys])
    ax2.set_xlabel("num_experts", fontweight="bold")
    ax2.set_ylabel("Router overhead (%)", fontweight="bold")
    ax2.set_ylim(0, 22)
    ax2.set_title("Panel B: Router MAC fraction", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.text(
        0.02,
        0.03,
        "Generator-level sweep only\nTiny manifest: H=32, expert=64\nNo backend timing/cycles",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MOE_SENSITIVITY_FIGURE_ID)


# ─── P3: Per-operator diagnostics ───────────────────────────────────────

def gen_figure_moe_operator_diagnostics(output_dir: Path) -> None:
    from tests.analysis.figures import p4_figure_data

    data = p4_figure_data.collect_mixtral_8x7b_moe_decoder_data(replay_stats_fn=None)
    op_buckets = {
        "Q/K/V/O": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
        "Attention\n(QK^T+PV)": int(data.get("attention_pim_mac_per_layer", 0)),
        "Router": int(data.get("router_pim_mac_per_layer", 0)),
        "Expert FFN\n(fused, ×1)": int(data.get("expert_ffn_pim_mac_fused_per_layer", 0)),
        "Expert FFN\n(real, ×3; scaled)": int(data.get("expert_ffn_pim_mac_real_per_layer", 0)),
        "Expert FFN\n(full dim, ×3)\n(analytic)": int(data.get("expert_ffn_pim_mac_full_dim_real_per_layer", 0)),
    }

    fig = plt.figure(figsize=(10.5, 4.3))
    ax1 = fig.add_subplot(1, 1, 1)
    labels = list(op_buckets.keys())
    values = list(op_buckets.values())
    colors = ["#009E73", "#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#999999"]
    bars = ax1.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    bars[-1].set_hatch("//")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(labels)), labels, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("PIM_MAC repeats per layer\n(1 repeat = 32 INT8 MACs)", fontweight="bold")
    ax1.set_title("Mixtral-8x7B: per-operator PIM_MAC (scaled, 1 layer)", fontsize=10)
    ax1.text(
        0.02,
        0.97,
        "Traceable bars use scaled dims (H=512, expert=2048).\n"
        "Full-dim ×3 bar is analytical only, not backend-replayed.",
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MOE_OPERATOR_DIAG_FIGURE_ID)


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate P4 paper figures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures (default: paper/figures)",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip replay validation to save time",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny/test manifests instead of paper-scale model configurations",
    )
    parser.add_argument(
        "--collect-llama2-scaling-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_LLAMA2_SCALING_CACHE,
        default=None,
        help=f"Run expensive Llama2 7B/13B backend collection and write JSON cache (default: {DEFAULT_LLAMA2_SCALING_CACHE})",
    )
    parser.add_argument(
        "--render-llama2-scaling-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_LLAMA2_SCALING_CACHE,
        default=None,
        help=f"Render Llama2 7B/13B scaling figure from JSON cache without backend simulation (default: {DEFAULT_LLAMA2_SCALING_CACHE})",
    )
    parser.add_argument(
        "--collect-decode-context-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
        default=None,
        help=f"Run Llama2-7B backend decode context-length sweep and write JSON cache (default: {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--render-decode-context-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
        default=None,
        help=f"Render decode context-length sweep figure from JSON cache without backend simulation (default: {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--collect-generated-token-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
        default=None,
        help=f"Run Llama2-7B generated-token backend sweep and write JSON cache (default: {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--render-generated-token-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
        default=None,
        help=f"Render generated-token sweep figure from JSON cache without backend simulation (default: {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--collect-prefill-prompt-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_PREFILL_PROMPT_SWEEP_CACHE,
        default=None,
        help=f"Run Llama2-7B backend prefill prompt-length sweep and write JSON cache (default: {DEFAULT_PREFILL_PROMPT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--render-prefill-prompt-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_PREFILL_PROMPT_SWEEP_CACHE,
        default=None,
        help=f"Render prefill prompt-length sweep figure from JSON cache without backend simulation (default: {DEFAULT_PREFILL_PROMPT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--prefill-sweep-prompt-lens",
        type=int,
        nargs="+",
        default=None,
        help="Prompt lengths for the prefill prompt sweep (default: [2, 4])",
    )
    parser.add_argument(
        "--collect-mixtral-breakdown-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_MIXTRAL_BREAKDOWN_CACHE,
        default=None,
        help=f"Run Mixtral-8x7B 32-layer backend collection and write JSON cache (default: {DEFAULT_MIXTRAL_BREAKDOWN_CACHE})",
    )
    parser.add_argument(
        "--render-mixtral-breakdown-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_MIXTRAL_BREAKDOWN_CACHE,
        default=None,
        help=f"Render Mixtral-8x7B breakdown figure from JSON cache without backend simulation (default: {DEFAULT_MIXTRAL_BREAKDOWN_CACHE})",
    )
    parser.add_argument(
        "--collect-opt-latency-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_OPT_LATENCY_CACHE,
        default=None,
        help=f"Run OPT backend collection and write JSON cache (default: {DEFAULT_OPT_LATENCY_CACHE})",
    )
    parser.add_argument(
        "--collect-qwen-gemma-latency-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_QWEN_GEMMA_LATENCY_CACHE,
        default=None,
        help=f"Run Qwen/Gemma backend collection and write JSON cache (default: {DEFAULT_QWEN_GEMMA_LATENCY_CACHE})",
    )
    parser.add_argument(
        "--collect-cross-model-latency-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_CROSS_MODEL_LATENCY_CACHE,
        default=None,
        help=(
            "Derive cross-model decode-cycle JSON cache from existing backend-collected source caches "
            f"(default: {DEFAULT_CROSS_MODEL_LATENCY_CACHE})"
        ),
    )
    parser.add_argument(
        "--render-cross-model-latency-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_CROSS_MODEL_LATENCY_CACHE,
        default=None,
        help=f"Render cross-model decode-cycle figure from JSON cache without backend simulation (default: {DEFAULT_CROSS_MODEL_LATENCY_CACHE})",
    )
    parser.add_argument(
        "--collect-cross-model-prefill-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_CROSS_MODEL_PREFILL_CACHE,
        default=None,
        help=(
            "Run cross-model dense prefill backend collection and write JSON cache "
            f"(default: {DEFAULT_CROSS_MODEL_PREFILL_CACHE})"
        ),
    )
    parser.add_argument(
        "--render-cross-model-prefill-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_CROSS_MODEL_PREFILL_CACHE,
        default=None,
        help=f"Render cross-model prefill-cycle figure from JSON cache without backend simulation (default: {DEFAULT_CROSS_MODEL_PREFILL_CACHE})",
    )
    parser.add_argument(
        "--cross-model-prefill-prompt-len",
        type=int,
        default=12,
        help="Prompt length for --collect-cross-model-prefill-cache (default: 12)",
    )
    parser.add_argument(
        "--cross-model-prefill-layers",
        type=int,
        default=None,
        help="Replay layer count for --collect-cross-model-prefill-cache (default: full model depth)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for supported collection commands (default: 1)",
    )
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    if opts.collect_llama2_scaling_cache is not None:
        path = write_llama2_scaling_cache(opts.collect_llama2_scaling_cache, collect_backend=True)
        print(f"Wrote Llama2 scaling cache to {path.resolve()}")
        return 0

    if opts.render_llama2_scaling_cache is not None:
        render_llama2_scaling_figure_from_cache(opts.render_llama2_scaling_cache, opts.output_dir)
        print(f"Rendered Llama2 scaling figure from {opts.render_llama2_scaling_cache.resolve()}")
        return 0

    if opts.collect_decode_context_sweep_cache is not None:
        path = write_decode_context_sweep_cache(opts.collect_decode_context_sweep_cache)
        print(f"Wrote decode context-length sweep cache to {path.resolve()}")
        return 0

    if opts.render_decode_context_sweep_cache is not None:
        render_decode_context_sweep_figure_from_cache(opts.render_decode_context_sweep_cache, opts.output_dir)
        print(f"Rendered decode context-length sweep figure from {opts.render_decode_context_sweep_cache.resolve()}")
        return 0

    if opts.collect_generated_token_sweep_cache is not None:
        path = write_generated_token_sweep_cache(opts.collect_generated_token_sweep_cache)
        print(f"Wrote generated-token sweep cache to {path.resolve()}")
        return 0

    if opts.render_generated_token_sweep_cache is not None:
        render_generated_token_sweep_figure_from_cache(opts.render_generated_token_sweep_cache, opts.output_dir)
        print(f"Rendered generated-token sweep figure from {opts.render_generated_token_sweep_cache.resolve()}")
        return 0

    if opts.collect_prefill_prompt_sweep_cache is not None:
        path = write_prefill_prompt_sweep_cache(
            opts.collect_prefill_prompt_sweep_cache,
            prompt_len_values=opts.prefill_sweep_prompt_lens,
            workers=opts.workers,
        )
        print(f"Wrote prefill prompt-length sweep cache to {path.resolve()}")
        return 0

    if opts.render_prefill_prompt_sweep_cache is not None:
        render_prefill_prompt_sweep_figure_from_cache(opts.render_prefill_prompt_sweep_cache, opts.output_dir)
        print(f"Rendered prefill prompt-length sweep figure from {opts.render_prefill_prompt_sweep_cache.resolve()}")
        return 0

    if opts.collect_mixtral_breakdown_cache is not None:
        path = write_mixtral_breakdown_cache(opts.collect_mixtral_breakdown_cache, collect_backend=True)
        print(f"Wrote Mixtral breakdown cache to {path.resolve()}")
        return 0

    if opts.render_mixtral_breakdown_cache is not None:
        render_mixtral_breakdown_figure_from_cache(opts.render_mixtral_breakdown_cache, opts.output_dir)
        print(f"Rendered Mixtral breakdown figure from {opts.render_mixtral_breakdown_cache.resolve()}")
        return 0

    if opts.collect_opt_latency_cache is not None:
        path = write_opt_latency_cache(opts.collect_opt_latency_cache, collect_backend=True)
        print(f"Wrote OPT latency cache to {path.resolve()}")
        return 0

    if opts.collect_qwen_gemma_latency_cache is not None:
        path = write_qwen_gemma_latency_cache(opts.collect_qwen_gemma_latency_cache, collect_backend=True)
        print(f"Wrote Qwen/Gemma latency cache to {path.resolve()}")
        return 0

    if opts.collect_cross_model_latency_cache is not None:
        path = write_cross_model_latency_cache(opts.collect_cross_model_latency_cache)
        print(f"Wrote cross-model latency cache to {path.resolve()}")
        return 0

    if opts.render_cross_model_latency_cache is not None:
        render_cross_model_latency_figure_from_cache(opts.render_cross_model_latency_cache, opts.output_dir)
        print(f"Rendered cross-model latency figure from {opts.render_cross_model_latency_cache.resolve()}")
        return 0

    if opts.collect_cross_model_prefill_cache is not None:
        path = write_cross_model_prefill_cache(
            opts.collect_cross_model_prefill_cache,
            prompt_len=opts.cross_model_prefill_prompt_len,
            num_layers=opts.cross_model_prefill_layers,
            modes=("steady_state", "cold_start"),
            workers=opts.workers,
        )
        print(f"Wrote cross-model prefill cache to {path.resolve()}")
        return 0

    if opts.render_cross_model_prefill_cache is not None:
        render_cross_model_prefill_figure_from_cache(opts.render_cross_model_prefill_cache, opts.output_dir)
        print(f"Rendered cross-model prefill figure from {opts.render_cross_model_prefill_cache.resolve()}")
        return 0

    manifest_mode = "TINY (unit-test scale)" if opts.tiny else "PAPER (representative model scale)"
    print(f"Manifest mode: {manifest_mode}")

    # Clean stale fig6 raster artifacts — fig6 is now exported as LaTeX table
    for stale in ("fig6_replay_validation.png", "fig6_replay_validation.pdf"):
        stale_path = opts.output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    print("Generating attention decomposition...")
    gen_figure_4(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating FFN/MoE decomposition...")
    gen_figure_5(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    if not opts.skip_replay:
        print("Generating replay validation table (this runs the simulator)...")
        gen_figure_6(opts.output_dir)
        print("  OK")
    else:
        print("  Skipped (--skip-replay)")

    print("Generating Llama2-7B 32-layer dense decoder breakdown...")
    gen_figure_9_llama2_7b_32_layer(opts.output_dir)
    print("  OK")

    print("Generating Llama2 7B/13B latency and compute comparison from cache...")
    _cached_or_collect_llama2_scaling(opts.output_dir)

    print("Generating Llama2-7B decode context-length sweep from cache...")
    _cached_or_collect_decode_context_sweep(opts.output_dir)

    print("Generating Llama2-7B generated-token sweep from cache...")
    _cached_or_collect_generated_token_sweep(opts.output_dir)

    print("Generating Llama2-7B prefill prompt-length sweep from cache...")
    _cached_or_collect_prefill_prompt_sweep(opts.output_dir)

    print("Generating MoE expert scaling sensitivity...")
    gen_figure_moe_expert_sensitivity(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating MoE per-operator diagnostics...")
    gen_figure_moe_operator_diagnostics(opts.output_dir)
    print("  OK")

    print("Generating Mixtral vs Llama2 comparison from caches...")
    gen_figure_mixtral_vs_llama2(opts.output_dir)
    print("  OK")

    print("Generating cross-model decode-cycle comparison from cache...")
    if DEFAULT_CROSS_MODEL_LATENCY_CACHE.exists():
        render_cross_model_latency_figure_from_cache(DEFAULT_CROSS_MODEL_LATENCY_CACHE, opts.output_dir)
        print("  OK")
    else:
        print(f"  Skipped; cache missing at {DEFAULT_CROSS_MODEL_LATENCY_CACHE}")
        print("  Run with --collect-cross-model-latency-cache first, then --render-cross-model-latency-cache.")

    print("Generating cross-model prefill-cycle comparison from cache...")
    if DEFAULT_CROSS_MODEL_PREFILL_CACHE.exists():
        render_cross_model_prefill_figure_from_cache(DEFAULT_CROSS_MODEL_PREFILL_CACHE, opts.output_dir)
        print("  OK")
    else:
        print(f"  Skipped; cache missing at {DEFAULT_CROSS_MODEL_PREFILL_CACHE}")
        print("  Run with --collect-cross-model-prefill-cache first, then --render-cross-model-prefill-cache.")

    print(f"\nAll figures written to {opts.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
