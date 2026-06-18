"""LPDDR5-PIM round-robin sweep configuration."""

from __future__ import annotations

from copy import deepcopy


LPDDR5_ANALYSIS_POWER = {
    "enabled": True,
    "VDD1": 1.80, "VDD2H": 1.05, "VDD2L": 0.90, "VDDQ": 0.50,
    "IDD01": 2.80, "IDD02H": 32.00, "IDD02L": 0.25, "IDD0Q": 0.75,
    "IDD2N1": 1.20, "IDD2N2H": 16.00, "IDD2N2L": 0.25, "IDD2NQ": 0.75,
    "IDD3N1": 1.20, "IDD3N2H": 16.00, "IDD3N2L": 0.25, "IDD3NQ": 0.75,
    "IDD4R1": 2.00, "IDD4R2H": 18.00, "IDD4R2L": 0.30, "IDD4RQ": 0.85,
    "IDD4W1": 2.10, "IDD4W2H": 19.00, "IDD4W2L": 0.35, "IDD4WQ": 0.90,
    "IDD5AB1": 2.20, "IDD5AB2H": 35.00, "IDD5AB2L": 0.25, "IDD5ABQ": 0.75,
}


def make_rr_cfg(active_banks: int, banks_per_mpu: int) -> dict:
    """Build a round-robin PIM sweep config for one bank/BPM combination."""
    if active_banks <= 0 or banks_per_mpu <= 0:
        raise ValueError("active_banks and banks_per_mpu must be positive")

    rank = 2 if active_banks > 16 else 1
    return {
        "dram_kwargs": {
            "pim_enabled": True,
            "pim_mode": "bank",
            "pim_banks_per_mpu": int(banks_per_mpu),
            "pim_mac_execution_model": "shared_mpu_serial",
            "pim_datatype": "int8",
            "rank": rank,
            "power": deepcopy(LPDDR5_ANALYSIS_POWER),
        },
        "pim_mode": True,
        "num_pim_requests": 4096,
        "pim_distribution_mode": "bank_sequence",
        "pim_same_bank": False,
        "pim_bank_sequence": list(range(int(active_banks))),
        "pim_bank_sequence_order": "controller",
        "pim_bank_group_size": int(active_banks),
        "pim_dependency_count": min(int(active_banks), 4),
        "pim_burst_length": 1,
        "pim_row_start": 0,
        "pim_row_count": 1,
    }
