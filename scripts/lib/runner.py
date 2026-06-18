"""Direct Ramulator2 runner — instantiate components without the test registry."""

from __future__ import annotations

import copy
import csv
import tempfile
from collections import Counter
from pathlib import Path


DEFAULT_CFG = {
    "org_preset": "LPDDR5_8Gb_x16",
    "timing_preset": "LPDDR5_6400",
    "dram_kwargs": {
        "pim_enabled": True,
        "pim_mode": "bank",
        "pim_datatype": "int8",
    },
    "frontend_clock_ratio": 4,
    "stream_cols": 8,
    "pim_mode": True,
    "num_pim_requests": 4096,
    "pim_distribution_mode": "same_bank",
    "pim_same_bank": True,
    "pim_dependency_count": 1,
    "pim_bank_group_size": 0,
    "pim_bank_sequence": [],
    "pim_bank_sequence_order": "frontend",
    "pim_burst_length": 1,
    "pim_row_start": 0,
    "pim_row_count": 1,
    "pim_split_all_bank": False,
}

COMMANDS_TO_COUNT = [
    "ACT1", "ACT2", "CAS_RD", "CAS_WR", "RD", "WR", "RDA", "WRA",
    "SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB",
    "PREpb", "PREab", "REFab",
]


def _merge_cfg(base: dict, override: dict | None) -> dict:
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_cfg(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _serialize_int_list(values) -> str:
    return ",".join(str(v) for v in values)


def _extract_dram_layout(dram) -> dict:
    """Extract the device's multi-level bank decomposition from a DRAM object.

    Returns bank_positions, bank_counts, and total_bank_units so callers can
    map flat bank indices 0..N-1 into the addr_vec correctly.
    """
    cls = type(dram)
    level_names = list(cls.levels.keys())
    org_dict, _ = dram.resolve()
    org_counts = [org_dict.get(name.lower(), 1) for name in level_names]
    row_idx = level_names.index("Row")
    col_idx = level_names.index("Column")
    bank_positions = list(range(1, row_idx))
    bank_counts = [org_counts[i] for i in bank_positions]

    if "BankGroup" in level_names:
        bg_idx = level_names.index("BankGroup") - 1
        if bg_idx < len(bank_positions) - 1:
            bank_positions.append(bank_positions.pop(bg_idx))
            bank_counts.append(bank_counts.pop(bg_idx))

    if "PseudoChannel" in level_names:
        pc_pos = level_names.index("PseudoChannel")
        pc_idx = [i for i, pos in enumerate(bank_positions) if pos == pc_pos][0]
        bank_positions.append(bank_positions.pop(pc_idx))
        bank_counts.append(bank_counts.pop(pc_idx))

    total_bank_units = 1
    for count in bank_counts:
        total_bank_units *= count
    return {
        "addr_vec_size": len(level_names),
        "bank_positions": bank_positions,
        "bank_counts": bank_counts,
        "total_bank_units": total_bank_units,
        "row_pos": row_idx,
        "col_pos": col_idx,
        "num_rows": org_counts[row_idx],
        "num_cols": org_counts[col_idx],
    }


def _make_dram(ramulator, cfg: dict):
    return ramulator.dram.LPDDR5PIM(
        org_preset=cfg["org_preset"],
        timing_preset=cfg["timing_preset"],
        **cfg.get("dram_kwargs", {}),
    )


def _pim_request_ids(dram, *, split_all_bank: bool = False) -> tuple[int, int, int]:
    names = list(type(dram).supported_requests.keys())
    pim_compute = names.index("PIMCompute") if "PIMCompute" in names else -1
    if not split_all_bank:
        return pim_compute, -1, -1
    return (
        pim_compute,
        names.index("PIMLoadAll") if "PIMLoadAll" in names else -1,
        names.index("PIMComputeAll") if "PIMComputeAll" in names else -1,
    )


def _read_command_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        command, count = [part.strip() for part in line.split(",", maxsplit=1)]
        counts[command] = int(count)
    return dict(sorted(counts.items()))


def _read_command_traces(prefix: Path) -> list[dict]:
    traces = []
    for trace_path in sorted(prefix.parent.glob(f"{prefix.name}.ch*")):
        with trace_path.open(newline="", encoding="utf-8") as handle:
            commands = [row["command"] for row in csv.DictReader(handle)]
        traces.append({
            "path": str(trace_path),
            "channel": trace_path.suffix.replace(".ch", ""),
            "command_count": len(commands),
            "command_counts": dict(sorted(Counter(commands).items())),
            "commands_preview": commands[:12],
        })
    return traces


def _attach_plugins(ramulator, tmpdir: Path):
    counts_path = tmpdir / "command_counts.csv"
    trace_prefix = tmpdir / "command_trace.csv"
    return [
        ramulator.controller_plugin.CommandCounter(
            commands_to_count=COMMANDS_TO_COUNT, path=str(counts_path)),
        ramulator.controller_plugin.CmdTraceRecorder(path=str(trace_prefix)),
    ]


def _collect_observability(stats: dict, tmpdir: Path, cfg: dict) -> dict:
    ctrl = stats.get("memory_system", {}).get("controller", {})
    selected = {}
    for key in (
        "cycles", "num_pim_reqs_served", "num_issued_pim_mac",
        "avg_pim_latency", "avg_pim_service_latency",
        "avg_pim_launch_wait", "avg_pim_response_latency",
        "pim_capacity_stalls", "pim_mpu_group_stalls", "pim_dependency_stalls",
        "pim_inflight_peak", "pim_banks_per_mpu", "pim_mpu_group_count",
        "total_banks", "effective_mpu_groups",
    ):
        if key in ctrl:
            selected[key] = ctrl[key]
    return {
        "modeled": {
            "command_counts": _read_command_counts(tmpdir / "command_counts.csv"),
            "command_traces": _read_command_traces(tmpdir / "command_trace.csv"),
            "controller_stats": selected,
            "pim_datatype": cfg.get("dram_kwargs", {}).get("pim_datatype", "unknown"),
        }
    }


def _make_controller_and_mem(ramulator, dram, plugins):
    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        controller_plugins=plugins,
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )


def run_single(
    dram=None,
    cfg_override: dict | None = None,
    nop: int = 1,
    num_probes: int = 10000,
    warmup: int = 10000,
    read_ratio: int = 100,
    observability_dir: Path | None = None,
) -> dict:
    """Run one LatencyThroughputTrace LPDDR5-PIM simulation point."""
    import ramulator

    cfg = _merge_cfg(DEFAULT_CFG, cfg_override)
    dram = dram if dram is not None else _make_dram(ramulator, cfg)
    layout = _extract_dram_layout(dram)
    pim_request, pim_load, pim_compute_all = _pim_request_ids(
        dram, split_all_bank=bool(cfg.get("pim_split_all_bank", False)))

    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=int(cfg["frontend_clock_ratio"]),
        nop_counter=int(nop),
        num_probe_requests=int(num_probes),
        pim_mode=bool(cfg.get("pim_mode", True)),
        num_pim_requests=int(cfg.get("num_pim_requests", 4096)),
        pim_distribution_mode=str(cfg.get("pim_distribution_mode", "same_bank")),
        pim_same_bank=bool(cfg.get("pim_same_bank", True)),
        pim_dependency_count=int(cfg.get("pim_dependency_count", 1)),
        pim_bank_group_size=int(cfg.get("pim_bank_group_size", 0)),
        pim_bank_sequence=_serialize_int_list(cfg.get("pim_bank_sequence", [])),
        pim_bank_sequence_order=str(cfg.get("pim_bank_sequence_order", "frontend")),
        pim_burst_length=int(cfg.get("pim_burst_length", 1)),
        pim_row_start=int(cfg.get("pim_row_start", 0)),
        pim_row_count=int(cfg.get("pim_row_count", 1)),
        pim_request_type_id=pim_request,
        pim_load_request_type_id=pim_load,
        pim_compute_all_request_type_id=pim_compute_all,
        pim_split_all_bank=bool(cfg.get("pim_split_all_bank", False)),
        warmup_cycles=int(warmup),
        seed=12345,
        read_ratio=int(read_ratio),
        stream_cols=int(cfg.get("stream_cols", 8)),
        **layout,
    )
    with tempfile.TemporaryDirectory(dir=observability_dir) as tmp:
        tmpdir = Path(tmp)
        mem = _make_controller_and_mem(ramulator, dram, _attach_plugins(ramulator, tmpdir))
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        stats = sim.stats
        stats.setdefault("evidence", {})["pim_energy_observability"] = \
            _collect_observability(stats, tmpdir, cfg)
        return stats
