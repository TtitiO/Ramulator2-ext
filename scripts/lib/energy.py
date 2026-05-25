"""Small stats extractor for the paper-facing LPDDR5-PIM figures."""

from __future__ import annotations


def _nested(stats: dict, *keys: str, default=None):
    cur = stats
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_pim_stats(stats: dict, time_unit_ns: float) -> dict:
    """Extract only the Ramulator stats used by F2-F5.

    Expected keys are under ``memory_system.controller``: ``cycles``,
    ``avg_pim_latency``, ``num_pim_reqs_served``/``num_issued_pim_mac`` and
    optional PIM stall/resource counters.  Command counts are read from
    ``evidence.pim_energy_observability.modeled.command_counts`` when the runner
    attaches observability plugins.
    """
    ctrl = _nested(stats, "memory_system", "controller", default={}) or {}
    frontend = stats.get("frontend", {}) if isinstance(stats.get("frontend", {}), dict) else {}
    cycles = int(ctrl.get("cycles", 0) or 0)
    total_time_ns = cycles * float(time_unit_ns)
    served = int(
        ctrl.get(
            "num_pim_reqs_served",
            ctrl.get("num_issued_pim_mac", frontend.get("pim_requests_completed", 0)),
        )
        or 0
    )
    command_counts = _nested(
        stats,
        "evidence",
        "pim_energy_observability",
        "modeled",
        "command_counts",
        default={},
    ) or {}
    throughput = (served / total_time_ns) if total_time_ns > 0 else 0.0

    return {
        "avg_pim_latency_ns": float(ctrl.get("avg_pim_latency", 0.0) or 0.0) * float(time_unit_ns),
        "request_throughput": throughput,
        "total_time_ns": total_time_ns,
        "cycles": cycles,
        "num_pim_reqs_served": served,
        "command_counts": dict(sorted(command_counts.items())),
        "pim_mac_issued": int(ctrl.get("num_issued_pim_mac", command_counts.get("PIM_MAC", 0)) or 0),
        "pim_bcast_issued": int(command_counts.get("PIM_BCAST", 0) or 0),
        "pim_inflight_peak": int(ctrl.get("pim_inflight_peak", 0) or 0),
        "pim_capacity_stalls": int(ctrl.get("pim_capacity_stalls", 0) or 0),
        "pim_dependency_stalls": int(ctrl.get("pim_dependency_stalls", 0) or 0),
        "pim_mpu_group_stalls": int(ctrl.get("pim_mpu_group_stalls", 0) or 0),
    }
