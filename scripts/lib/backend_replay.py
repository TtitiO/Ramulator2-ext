"""LPDDR5-PIM concrete-trace backend replay.

Used by ``scripts/gen_figures.py`` for F4, F5, and ftable data collection.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

# Raise the expanded-record ceiling for large models and long prefill sweeps.
_ONE_TRILLION = "1000000000000"
os.environ.setdefault("RAMULATOR_MAX_EXPANDED_RECORDS", _ONE_TRILLION)

from .runner import _extract_dram_layout


LPDDR5_PIM_CONFIG = {
    "dram_class": "LPDDR5PIM",
    "org_preset": "LPDDR5_8Gb_x16",
    "timing_preset": "LPDDR5_6400",
    "dram_kwargs": {
        "pim_enabled": True,
        "pim_mode": "bank",
        "pim_datatype": "int8",
        "pim_banks_per_mpu": 2,
        "pim_mac_execution_model": "shared_mpu_serial",
    },
    "frontend_clock_ratio": 4,
}


def pim_cfg_per_bank() -> dict:
    """k=1: dedicated per-bank PIM — one bank per MPU."""
    return {"pim_banks_per_mpu": 1, "pim_mac_execution_model": "shared_mpu_serial"}


def pim_cfg_shared() -> dict:
    """k=2: shared-MPU PIM — two banks per MPU."""
    return {"pim_banks_per_mpu": 2, "pim_mac_execution_model": "shared_mpu_serial"}


def create_dram(cfg: dict | None = None, *, dram_kwargs_overrides: dict | None = None):
    import ramulator

    cfg = cfg or LPDDR5_PIM_CONFIG
    dram_kwargs = dict(cfg.get("dram_kwargs", {}))
    if dram_kwargs_overrides:
        dram_kwargs.update(dram_kwargs_overrides)
    return ramulator.dram.LPDDR5PIM(
        org_preset=cfg["org_preset"],
        timing_preset=cfg["timing_preset"],
        **dram_kwargs,
    )


def _make_frontend(trace_path: Path, dram, *, clock_ratio: int = 4,
                   max_trace_bytes: int | None = None,
                   max_expanded_records: int | None = None,
                   max_inflight_requests: int = 1):
    import ramulator

    request_type_ids = {
        name: idx for idx, name in enumerate(type(dram).supported_requests.keys())}
    command_ids = {name: idx for idx, name in enumerate(type(dram).commands)}
    layout = _extract_dram_layout(dram)
    kwargs = dict(
        clock_ratio=clock_ratio,
        path=str(trace_path),
        pim_compute_request_type_id=request_type_ids["PIMCompute"],
        pim_load_all_request_type_id=request_type_ids["PIMLoadAll"],
        pim_compute_all_request_type_id=request_type_ids["PIMComputeAll"],
        sb_command_id=command_ids["SB"],
        hab_command_id=command_ids["HAB"],
        hab_pim_command_id=command_ids["HAB_PIM"],
        addr_vec_size=layout["addr_vec_size"],
        max_repeat=100_000_000,
        max_records=10_000_000,
        max_inflight_requests=max_inflight_requests,
    )
    if max_trace_bytes is not None:
        kwargs["max_trace_bytes"] = max_trace_bytes
    if max_expanded_records is not None:
        kwargs["max_expanded_records"] = max_expanded_records
    return ramulator.frontend.LPDDR5PIMConcreteTrace(**kwargs)


def _make_mem(dram):
    import ramulator

    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )


def count_concrete_opcodes(concrete: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in concrete:
        counts[str(record["opcode"])] += int(record.get("repeat", 1))
    return dict(sorted(counts.items()))


def time_unit_ns(cfg: dict | None = None) -> float:
    dram = create_dram(cfg)
    _, timing = dram.resolve()
    return float(timing["tCK_ps"]) / 1000.0


def replay_concrete_trace(
    concrete_records: list[dict],
    *,
    materialize_weights: bool = False,
    max_trace_bytes: int = 1024 * 1024 * 1024,
    max_expanded_records: int = 100_000_000_000,
    pim_cfg_override: dict | None = None,
    max_inflight_requests: int = 1,
) -> dict:
    """Replay already-lowered concrete opcodes through the Ramulator backend.

    pim_cfg_override is merged into the DRAM kwargs (e.g. pim_banks_per_mpu).
    max_inflight_requests controls the frontend's inflight window.
    """
    import ramulator
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import write_jsonl

    dram = create_dram(dram_kwargs_overrides=pim_cfg_override)
    tck_ns = time_unit_ns()

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl"
        write_jsonl(concrete_records, trace_path)
        frontend = _make_frontend(
            trace_path, dram,
            clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
            max_trace_bytes=max_trace_bytes,
            max_expanded_records=max_expanded_records,
            max_inflight_requests=max_inflight_requests,
        )
        mem = _make_mem(dram)
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        stats = sim.stats

    ctrl = stats.get("memory_system", {}).get("controller", {})
    fe = stats.get("frontend", {})
    cycles = int(ctrl.get("cycles", 0) or 0)
    pim_mac = int(ctrl.get("num_issued_pim_mac", 0) or 0)
    pim_mac_ab = int(ctrl.get("num_issued_pim_mac_ab", 0) or 0)
    replay_ok = cycles > 0 and (pim_mac > 0 or pim_mac_ab > 0)

    _s = lambda k: int(ctrl.get(k, 0) or 0)
    return {
        "cycles": cycles,
        "runtime_ns": cycles * tck_ns,
        "command_counts": count_concrete_opcodes(concrete_records),
        "pim_mac_issued": pim_mac,
        "pim_mac_ab_issued": pim_mac_ab,
        "pim_bcast_issued": int(ctrl.get("num_issued_pim_bcast", 0) or 0),
        "replay_ok": replay_ok,
        "frontend_stats": {
            k: fe[k] for k in ("requests_issued", "pim_requests_completed",
                                "completed", "total_records_replayed") if k in fe},
        "pim_mpu_group_stalls": _s("pim_mpu_group_stalls"),
        "pim_dependency_stalls": _s("pim_dependency_stalls"),
        "pim_capacity_stalls": _s("pim_capacity_stalls"),
        "pim_inflight_peak": _s("pim_inflight_peak"),
        "pim_simultaneous_active_banks_peak": _s("pim_simultaneous_active_banks_peak"),
        "pim_banks_per_mpu": _s("pim_banks_per_mpu"),
        "effective_mpu_groups": _s("effective_mpu_groups"),
        "pim_ab_mac_latency_cycles": _s("pim_ab_mac_latency_cycles"),
        "num_bank_timing_blocked_cycles": _s("num_bank_timing_blocked_cycles"),
        "num_mpu_group_busy_blocked_cycles": _s("num_mpu_group_busy_blocked_cycles"),
    }


def generate_and_replay(
    phase: str,
    model_key: str,
    *,
    past_len: int = 1024,
    prompt_len: int = 12,
    materialize_weights: bool = False,
    pim_cfg_override: dict | None = None,
    max_inflight_requests: int = 1,
    interleave_depth: int = 4,
    mac_mode: str = "per_kind",
) -> dict:
    """Generate semantic records → lower to concrete → replay through backend.

    mac_mode: "per_kind" (weight-stationary FFN=all-bank, attention=per-bank),
              "per_bank" (all per-bank PIM_MAC), or "all_bank" (all PIM_MAC_AB).
    """
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_dense_decoder_records_for_model,
        generate_dense_prefill_records_for_model,
        generate_mixtral_8x7b_decoder_records,
    )
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
        lower_semantic_records_to_concrete,
    )

    if model_key == "mixtral-8x7b":
        if phase != "decode":
            raise ValueError("Mixtral-8x7B only supports decode phase")
        semantic = generate_mixtral_8x7b_decoder_records()
    elif phase == "decode":
        semantic = generate_dense_decoder_records_for_model(model_key, past_len=past_len)
    elif phase == "prefill":
        semantic = generate_dense_prefill_records_for_model(model_key, prompt_len=prompt_len)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Bank interleaving is only applied when concurrent inflight is enabled.
    # We must pass the device's real multi-level bank decomposition so flat
    # bank indices map correctly into the addr_vec.
    interleave_banks = max_inflight_requests > 1
    lower_kwargs: dict = {
        "materialize_weights": materialize_weights,
        "interleave_banks": interleave_banks,
        "mac_mode": mac_mode,
    }
    if interleave_banks:
        layout = _extract_dram_layout(create_dram(dram_kwargs_overrides=pim_cfg_override))
        lower_kwargs["addr_vec_size"] = layout["addr_vec_size"]
        lower_kwargs["bank_positions"] = layout["bank_positions"]
        lower_kwargs["bank_counts"] = layout["bank_counts"]
        lower_kwargs["row_level"] = layout["row_pos"]
        lower_kwargs["col_level"] = layout["col_pos"]
        lower_kwargs["interleave_depth"] = interleave_depth
    concrete = lower_semantic_records_to_concrete(semantic, **lower_kwargs)
    opcode_counts = count_concrete_opcodes(concrete)

    # Inflight window: all-bank ops are serialized by the controller so stay
    # at width 1. per_kind/per_bank auto-scale to span all banks.
    effective_inflight = max_inflight_requests
    if interleave_banks and mac_mode in ("per_kind", "per_bank"):
        max_span = max(
            (len(r["bank_sequence"]) for r in semantic if r.get("bank_sequence")), default=1)
        effective_inflight = max(max_inflight_requests, max_span * interleave_depth)
    elif mac_mode == "all_bank":
        effective_inflight = 1

    result = replay_concrete_trace(
        concrete, pim_cfg_override=pim_cfg_override,
        max_inflight_requests=effective_inflight)
    mode = "cold_start" if materialize_weights else "steady_state"

    return {
        "model_key": model_key,
        "phase": phase,
        "mode": mode,
        "past_len": past_len if phase == "decode" else None,
        "prompt_len": prompt_len if phase == "prefill" else None,
        "opcode_counts": opcode_counts,
        **result,
    }


# ── Analytical prefill formula (no simulation) ─────────────────────────


def _ceil_div(n: int, d: int) -> int:
    return (int(n) + int(d) - 1) // int(d)


def _prefill_tile_ranges(total: int, tile: int) -> list[tuple[int, int]]:
    return [(s, min(tile, total - s)) for s in range(0, total, tile)]


def _prefill_causal_pair_count(qs: int, qt: int, ks: int, kt: int) -> int:
    ke = ks + kt
    return sum(max(0, min(ke, qi + 1) - ks) for qi in range(qs, qs + qt))


def _prefill_attention_pim_mac_per_layer(
    *, prompt_len: int, num_heads: int, head_dim: int,
    lanes: int, score_tile_tokens: int, context_tile_tokens: int,
) -> int:
    q_ranges = _prefill_tile_ranges(prompt_len, score_tile_tokens)
    kv_ranges = _prefill_tile_ranges(prompt_len, context_tile_tokens)
    per_head = 0
    for qs, qt in q_ranges:
        for ks, kt in kv_ranges:
            pairs = _prefill_causal_pair_count(qs, qt, ks, kt)
            if pairs > 0:
                per_head += 2 * _ceil_div(pairs * head_dim, lanes)
    return num_heads * per_head


def _infer_model_family(name: str) -> str:
    nl = name.lower()
    if "llama" in nl:
        return "Llama"
    if "mixtral" in nl:
        return "Mixtral"
    if "opt" in nl:
        return "OPT"
    if "qwen" in nl:
        return "Qwen"
    if "gemma" in nl:
        return "Gemma"
    return "Unknown"


def prefill_formula(model_key: str, *, prompt_len: int) -> dict:
    """Analytical prefill PIM_MAC counts per layer (no simulation)."""
    from ramulator.dram.lpddr5_pim import PIM_DATATYPE_RESOURCES
    from ramulator.workload_surrogate.generate_full_transformer import (
        FFN_VARIANT_PROJECTION_COUNTS, get_dense_prefill_manifests, get_model_spec)

    spec = get_model_spec(model_key)
    attn_m, _ = get_dense_prefill_manifests(spec, prompt_len=prompt_len)
    res = PIM_DATATYPE_RESOURCES[spec.datatype]
    lanes = int(res["pim_lanes"])
    prim_ops = int(res["pim_ops_per_mac"])
    nkv = int(spec.num_kv_heads or spec.num_heads)
    q_proj = int(spec.num_heads) * int(spec.head_dim)
    kv_proj = nkv * int(spec.head_dim)

    qkvo = (
        _ceil_div(prompt_len * spec.hidden_size * q_proj, lanes)
        + _ceil_div(prompt_len * spec.hidden_size * kv_proj, lanes)
        + _ceil_div(prompt_len * spec.hidden_size * kv_proj, lanes)
        + _ceil_div(prompt_len * q_proj * spec.hidden_size, lanes))
    stile = int(attn_m["score_tile_tokens"])
    ctile = int(attn_m["context_tile_tokens"])
    causal_pairs = prompt_len * (prompt_len + 1) // 2
    valid_attn_pairs = causal_pairs * int(spec.num_heads)
    attn_per_layer = _prefill_attention_pim_mac_per_layer(
        prompt_len=prompt_len, num_heads=int(spec.num_heads),
        head_dim=int(spec.head_dim), lanes=lanes,
        score_tile_tokens=stile, context_tile_tokens=ctile)
    n_proj = FFN_VARIANT_PROJECTION_COUNTS.get(spec.ffn_variant, 3)
    ffn_per_layer = n_proj * _ceil_div(
        prompt_len * spec.hidden_size * spec.ffn_hidden_size, lanes)

    return {
        "model_name": spec.name,
        "model_family": _infer_model_family(spec.name),
        "model_key": model_key,
        "model_total_layers": int(spec.num_layers),
        "hidden_size": int(spec.hidden_size),
        "ffn_hidden_size": int(spec.ffn_hidden_size),
        "ffn_variant": spec.ffn_variant,
        "activation": spec.activation,
        "num_heads": int(spec.num_heads),
        "num_kv_heads": nkv,
        "head_dim": int(spec.head_dim),
        "datatype": spec.datatype,
        "citation": spec.citation,
        "prompt_len": int(prompt_len),
        "seq_len": int(prompt_len),
        "prefill_causal_pairs": int(causal_pairs),
        "valid_attention_pairs_per_layer": int(valid_attn_pairs),
        "attention_issued_work_elements_per_layer": int(2 * valid_attn_pairs * spec.head_dim),
        "score_tile_tokens": stile,
        "context_tile_tokens": ctile,
        "pim_mac_lanes": lanes,
        "primitive_ops_per_mac": prim_ops,
        "per_layer_pim_mac_buckets": {
            "qkvo_projection": int(qkvo),
            "attention": int(attn_per_layer),
            "ffn": int(ffn_per_layer),
        },
        "kv_residency_policy": attn_m.get("residency_policy", {}),
    }
