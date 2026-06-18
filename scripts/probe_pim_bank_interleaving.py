#!/usr/bin/env python3
"""Probe k=1 (per-bank) vs k=2 (shared-MPU) PIM bank interleaving.

Two modes:
  --mode synthetic   Fast probe: emits compact PIM_MAC records round-robining
                     over N banks, replays under k1 and k2. Sweep interleave_depth.
  --mode pipeline    End-to-end check on a real transformer trace.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ramulator2" / "python"))


def _ensure_expanded_budget() -> None:
    os.environ.setdefault("RAMULATOR_MAX_EXPANDED_RECORDS", "1000000000000")


# ── Synthetic probe ─────────────────────────────────────────────────────

def _build_synthetic_records(n_groups: int, banks: list[int], depth: int,
                             dep_count: int = 8,
                             bank_positions: list[int] | None = None,
                             bank_counts: list[int] | None = None) -> list[dict]:
    """One SB + one compact PIM_MAC record with bank interleaving metadata.

    bank_positions/bank_counts give the device's multi-level bank decomposition
    (e.g. [1,3,2]/[1,4,4] for 8Gb x16 = 16 banks). Required when any bank index
    exceeds a single addr_vec slot's capacity.
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
        {"schema_version": "lpddr5-pim-opcode-v0.1", "record_id": "sb", "opcode": "SB",
         "repeat": 1, "addr_vec": [0, 0, 0, 0, 0, 0],
         "provenance": concrete_provenance(), "notes": "sb"},
        mac,
    ]


def _build_synthetic_ab_records(n_ab: int, *, dep_count: int = 8) -> list[dict]:
    """HAB → PIM_BCAST → HAB_PIM → PIM_MAC_AB×n_ab → SB."""
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

    layout = _extract_dram_layout(create_dram())
    if banks > layout["total_bank_units"]:
        raise SystemExit(
            f"--banks {banks} exceeds device bank units {layout['total_bank_units']}")

    bank_list = list(range(banks))
    records = _build_synthetic_records(
        groups, bank_list, depth, dep_count,
        bank_positions=layout["bank_positions"], bank_counts=layout["bank_counts"])
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
    """All-bank MAC probe: isolates the AB latency scaling under k1 vs k2."""
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
    if k1['ab_mac_latency_cycles']:
        print(f"  AB latency ratio (k2/k1): "
              f"{k2['ab_mac_latency_cycles'] / k1['ab_mac_latency_cycles']:.4f}x")


# ── End-to-end pipeline check ───────────────────────────────────────────

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
        description="Probe k=1 vs k=2 PIM bank interleaving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--mode", choices=["synthetic", "pipeline"], default="synthetic")
    p.add_argument("--banks", type=int, default=16)
    p.add_argument("--interleave-depth", type=int, default=4)
    p.add_argument("--depth-sweep", type=str, default=None,
                   help="comma-separated depths, e.g. 1,4,16,64")
    p.add_argument("--groups", type=int, default=500)
    p.add_argument("--dep-count", type=int, default=8)
    p.add_argument("--mac-mode", choices=["per_bank", "all_bank"], default="per_bank")
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
