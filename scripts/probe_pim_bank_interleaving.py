#!/usr/bin/env python3
"""Probe and verify k=1 (per-bank PIM) vs k=2 (Samsung-style shared-MPU PIM).

This script records the experiment used to diagnose and validate the ftable
(per-bank vs shared-PIM-block) comparison.  It has two independent modes:

  --mode synthetic   Fast, self-contained probe.  Emits ONE compact PIM_MAC
                     record that round-robins over `--banks` banks at
                     `--interleave-depth` ops/bank, replays it under k1 and k2,
                     and prints cycles / active-banks / MPU-stalls / slowdown.
                     No model generation — runs in seconds.  Use this to sweep
                     interleave_depth and see exactly how bank concurrency and
                     the k1/k2 slowdown respond.

  --mode pipeline    Full end-to-end check on a real transformer trace:
                     generate semantic records -> lower to compact interleaved
                     concrete opcodes (carrying the device's real multi-level
                     bank decomposition) -> replay under k1 and k2.  Confirms
                     the in-memory interleaving works on actual model traces.

Background (why this exists)
----------------------------
The LPDDR5-PIM controller serializes each bank to one in-flight PIM op, so
`pim_inflight_peak` is always 1 and is the WRONG health metric.  The real metric
is `pim_simultaneous_active_banks_peak`.  To run N banks in parallel the
frontend's inflight window must span N distinct banks at once.  Coarse bank
ordering (all of bank0, then bank1, ...) never lets the window span >1 bank, so
k1 == k2 and the comparison shows no effect.

The fix interleaves the PIM_MAC issue stream finely *in memory* in the C++
concrete frontend (one compact record carries bank_sequence / interleave_depth /
dependency_count and the frontend rotates banks per issue) so it costs zero
trace-file growth.  `interleave_depth` = number of ops issued to one bank before
switching banks.  The inflight window must be >= len(bank_sequence) *
interleave_depth to expose every bank; `generate_and_replay` auto-scales it.

Validated results (8Gb x16 preset, 16 banks, interleave_depth=4):
  k1 (banks_per_mpu=1): all banks active in parallel
  k2 (banks_per_mpu=2): shared-MPU model throttles concurrency ~2x
  -> ~1.7x slowdown on llama2-7b / qwen2.5-14b / gemma2-9b

Usage
-----
    export PYTHONPATH="ramulator2/python:ramulator2"
    export RAMULATOR_MAX_EXPANDED_RECORDS=1000000000000

    # Synthetic sweep of interleave_depth on 16 banks
    .venv/bin/python scripts/probe_pim_bank_interleaving.py --mode synthetic \
        --banks 16 --depth-sweep 1,4,16,64

    # Single synthetic point
    .venv/bin/python scripts/probe_pim_bank_interleaving.py --mode synthetic \
        --banks 4 --interleave-depth 4 --groups 500

    # End-to-end on a real model
    .venv/bin/python scripts/probe_pim_bank_interleaving.py --mode pipeline \
        --model llama2-7b --phase decode --past-len 8

See also: scripts/gen_figures.py --collect ftable   (the production table).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make `lib` importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ramulator2" / "python"))


def _ensure_expanded_budget() -> None:
    """The interleaved records expand to millions of issues; lift the cap."""
    os.environ.setdefault("RAMULATOR_MAX_EXPANDED_RECORDS", "1000000000000")


# ── Synthetic probe ──────────────────────────────────────────────────────────

def _build_synthetic_records(n_groups: int, banks: list[int], depth: int,
                             dep_count: int = 8,
                             bank_positions: list[int] | None = None,
                             bank_counts: list[int] | None = None) -> list[dict]:
    """One SB + one compact PIM_MAC record round-robining over `banks`.

    total ops = n_groups * depth * len(banks).  The C++ frontend expands this
    single record into that many interleaved per-bank issues at replay time.

    `bank_positions`/`bank_counts` give the device's real multi-level bank
    decomposition (e.g. [1,3,2]/[1,4,4] for the 8Gb x16 preset = 16 banks).
    They are REQUIRED whenever any bank index exceeds the capacity of a single
    addr_vec bank slot, otherwise high banks alias / crash the backend.
    """
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import concrete_provenance

    total = n_groups * depth * len(banks)
    mac = {
        "schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "m0", "opcode": "PIM_MAC",
        "repeat": total, "addr_vec": [0, 0, 0, 0, 0, 0],
        "provenance": concrete_provenance(), "notes": "mac",
        "bank_sequence": list(banks), "dependency_count": dep_count,
        "row_count": 1, "row_start": 0, "column_start": 0,
        "resolved_row_offset": 0, "resolved_col_offset": 0,
        "interleave_depth": depth, "interleave_start_idx": 0,
        "row_level": 4, "col_level": 5,
    }
    if bank_positions is not None and bank_counts is not None:
        mac["bank_positions"] = list(bank_positions)
        mac["bank_counts"] = list(bank_counts)
    else:
        mac["bank_level"] = 3
    return [
        {
            "schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "sb", "opcode": "SB",
            "repeat": 1, "addr_vec": [0, 0, 0, 0, 0, 0],
            "provenance": concrete_provenance(), "notes": "sb",
        },
        mac,
    ]


def _build_synthetic_ab_records(n_ab: int, *, dep_count: int = 8) -> list[dict]:
    """HAB → PIM_BCAST → HAB_PIM → PIM_MAC_AB×n_ab → SB sequence.

    One PIM_MAC_AB performs one MAC on every bank simultaneously.
    n_ab = ceil(total_macs / bank_count) conserves arithmetic vs per-bank scheme.
    """
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import concrete_provenance

    base_av = [0, 0, 0, 0, 0, 0]
    prov = concrete_provenance()
    return [
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "hab", "opcode": "HAB",
         "repeat": 1, "addr_vec": list(base_av), "provenance": prov, "notes": "enter HAB"},
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "bcast", "opcode": "PIM_BCAST",
         "repeat": 1, "addr_vec": list(base_av), "provenance": prov, "notes": "all-bank load"},
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "hab_pim", "opcode": "HAB_PIM",
         "repeat": 1, "addr_vec": list(base_av), "provenance": prov, "notes": "enter PIM AB"},
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "mac_ab", "opcode": "PIM_MAC_AB",
         "repeat": n_ab, "addr_vec": list(base_av), "provenance": prov, "notes": "all-bank MAC"},
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "sb", "opcode": "SB",
         "repeat": 1, "addr_vec": list(base_av), "provenance": prov, "notes": "return to SB"},
    ]


def _run_records(records: list[dict], banks_per_mpu: int, max_inflight: int) -> dict:
    import ramulator
    from lib.backend_replay import create_dram, _make_frontend, _make_mem
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import write_jsonl

    dram = create_dram(dram_kwargs_overrides={
        "pim_banks_per_mpu": banks_per_mpu,
        "pim_mac_execution_model": "shared_mpu_serial",
    })
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "trace.jsonl"
        write_jsonl(records, tp)
        fe = _make_frontend(tp, dram, clock_ratio=4, max_inflight_requests=max_inflight)
        mem = _make_mem(dram)
        sim = ramulator.Simulation(fe, mem)
        sim.run()
        ctrl = sim.stats.get("memory_system", {}).get("controller", {})
    g = lambda k: int(ctrl.get(k, 0) or 0)
    return {
        "cycles": g("cycles"),
        "active_banks": g("pim_simultaneous_active_banks_peak"),
        "mpu_stalls": g("pim_mpu_group_stalls"),
        "issued_mac": g("num_issued_pim_mac"),
        "issued_mac_ab": g("num_issued_pim_mac_ab"),
        "ab_mac_latency_cycles": g("pim_ab_mac_latency_cycles"),
    }


def run_synthetic(banks: int, depth: int, groups: int, dep_count: int) -> None:
    from lib.backend_replay import create_dram
    from lib.runner import _extract_dram_layout

    # Use the device's real multi-level bank decomposition so banks beyond a
    # single addr_vec slot's capacity map correctly (16-bank preset = [1,3,2]/[1,4,4]).
    layout = _extract_dram_layout(create_dram())
    if banks > layout["total_bank_units"]:
        raise SystemExit(
            f"--banks {banks} exceeds device bank units {layout['total_bank_units']}")

    bank_list = list(range(banks))
    records = _build_synthetic_records(
        groups, bank_list, depth, dep_count,
        bank_positions=layout["bank_positions"], bank_counts=layout["bank_counts"])
    # Window must span every bank at this depth to expose full concurrency.
    window = banks * depth
    print(f"[synthetic] banks={banks} interleave_depth={depth} groups={groups} "
          f"dep_count={dep_count} inflight_window={window}")
    k1 = _run_records(records, banks_per_mpu=1, max_inflight=window)
    k2 = _run_records(records, banks_per_mpu=2, max_inflight=window)
    slowdown = k2["cycles"] / k1["cycles"] if k1["cycles"] else 0.0
    print(f"  k1 (per-bank):  cycles={k1['cycles']:>12d}  active_banks={k1['active_banks']:>3d}  "
          f"mpu_stalls={k1['mpu_stalls']:>12d}  issued_mac={k1['issued_mac']}")
    print(f"  k2 (shared):    cycles={k2['cycles']:>12d}  active_banks={k2['active_banks']:>3d}  "
          f"mpu_stalls={k2['mpu_stalls']:>12d}  issued_mac={k2['issued_mac']}")
    print(f"  SLOWDOWN (k2/k1): {slowdown:.4f}x")


def run_synthetic_ab(banks: int, groups: int, depth: int) -> None:
    """Synthetic all-bank MAC probe: isolates the AB latency scaling.

    Emits n_ab = groups * depth PIM_MAC_AB ops (same total work as the per-bank
    probe would spread across `banks` banks).  Under k1 the AB latency = 1×
    completion_latency; under k2 it = 2× completion_latency.
    """
    # Total per-bank MACs in the equivalent per-bank probe:
    #   total = groups * depth * banks, spread across banks -> per-bank = groups * depth
    # One PIM_MAC_AB does one MAC on every bank, so n_ab = groups * depth.
    n_ab = groups * depth
    records = _build_synthetic_ab_records(n_ab)
    print(f"[synthetic-ab] banks={banks} n_ab={n_ab} (groups={groups} depth={depth})")
    k1 = _run_records(records, banks_per_mpu=1, max_inflight=1)
    k2 = _run_records(records, banks_per_mpu=2, max_inflight=1)
    slowdown = k2["cycles"] / k1["cycles"] if k1["cycles"] else 0.0
    print(f"  k1 (per-bank):  cycles={k1['cycles']:>12d}  active_banks={k1['active_banks']:>3d}  "
          f"ab_latency={k1['ab_mac_latency_cycles']:>4d}  issued_mac_ab={k1['issued_mac_ab']}")
    print(f"  k2 (shared):    cycles={k2['cycles']:>12d}  active_banks={k2['active_banks']:>3d}  "
          f"ab_latency={k2['ab_mac_latency_cycles']:>4d}  issued_mac_ab={k2['issued_mac_ab']}")
    print(f"  SLOWDOWN (k2/k1): {slowdown:.4f}x")
    print(f"  AB latency ratio (k2/k1): "
          f"{k2['ab_mac_latency_cycles'] / k1['ab_mac_latency_cycles']:.4f}x"
          if k1['ab_mac_latency_cycles'] else "  AB latency ratio: N/A")


# ── End-to-end pipeline check ────────────────────────────────────────────────

def run_pipeline(model: str, phase: str, past_len: int, prompt_len: int,
                 interleave_depth: int, mac_mode: str = "per_bank") -> None:
    from lib.backend_replay import generate_and_replay, pim_cfg_per_bank, pim_cfg_shared

    print(f"[pipeline] model={model} phase={phase} past_len={past_len} "
          f"prompt_len={prompt_len} interleave_depth={interleave_depth} mac_mode={mac_mode}")
    common = dict(past_len=past_len, prompt_len=prompt_len,
                  max_inflight_requests=16, interleave_depth=interleave_depth,
                  mac_mode=mac_mode)
    k1 = generate_and_replay(phase, model, pim_cfg_override=pim_cfg_per_bank(), **common)
    print(f"  k1 (per-bank):  cycles={k1['cycles']:>14d}  "
          f"active_banks={k1['pim_simultaneous_active_banks_peak']:>3d}  "
          f"bpm={k1['pim_banks_per_mpu']}")
    k2 = generate_and_replay(phase, model, pim_cfg_override=pim_cfg_shared(), **common)
    print(f"  k2 (shared):    cycles={k2['cycles']:>14d}  "
          f"active_banks={k2['pim_simultaneous_active_banks_peak']:>3d}  "
          f"bpm={k2['pim_banks_per_mpu']}  mpu_stalls={k2.get('pim_mpu_group_stalls')}")
    slowdown = k2["cycles"] / k1["cycles"] if k1["cycles"] else 0.0
    print(f"  SLOWDOWN (k2/k1): {slowdown:.4f}x")


def _parse_depth_sweep(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Probe k=1 vs k=2 PIM bank interleaving (synthetic or end-to-end).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["synthetic", "pipeline"], default="synthetic")
    # synthetic
    p.add_argument("--banks", type=int, default=16, help="bank count for synthetic mode")
    p.add_argument("--interleave-depth", type=int, default=4, help="ops per bank before switching")
    p.add_argument("--depth-sweep", type=str, default=None,
                   help="comma list of depths to sweep, e.g. 1,4,16,64 (synthetic mode)")
    p.add_argument("--groups", type=int, default=500, help="number of full bank-cycle groups")
    p.add_argument("--dep-count", type=int, default=8, help="dependency column rotation modulus")
    p.add_argument("--mac-mode", choices=["per_bank", "all_bank"], default="per_bank",
                   help="MAC emission mode: per_bank (SB PIM_MAC) or all_bank (PIM_MAC_AB)")
    # pipeline
    p.add_argument("--model", type=str, default="llama2-7b")
    p.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=12)
    args = p.parse_args()

    _ensure_expanded_budget()

    if args.mode == "synthetic":
        if args.mac_mode == "all_bank":
            run_synthetic_ab(args.banks, args.groups, args.interleave_depth)
        else:
            depths = _parse_depth_sweep(args.depth_sweep) if args.depth_sweep else [args.interleave_depth]
            for depth in depths:
                run_synthetic(args.banks, depth, args.groups, args.dep_count)
    else:
        run_pipeline(args.model, args.phase, args.past_len, args.prompt_len,
                     args.interleave_depth, mac_mode=args.mac_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
