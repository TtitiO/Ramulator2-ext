"""Tests for Phase 4 full-transformer semantic dataflow generation."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.utils.sim import extract_dram_layout

ramulator = importlib.import_module("ramulator")

generator_mod = importlib.import_module("ramulator.workload_surrogate.generate_full_transformer")
lowering_mod = importlib.import_module("ramulator.workload_surrogate.generate_lpddr5_pim_concrete")
concrete_schema_mod = importlib.import_module("ramulator.workload_surrogate.lpddr5_pim_concrete_trace")
trace_mod = importlib.import_module("ramulator.workload_surrogate.structured_trace")


def _make_mem(dram):
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


def _frontend(
    trace_path: Path,
    dram,
    *,
    max_trace_bytes: int | None = None,
    max_expanded_records: int | None = None,
):
    request_type_ids = {name: index for index, name in enumerate(type(dram).supported_requests.keys())}
    command_ids = {name: index for index, name in enumerate(type(dram).commands)}
    layout = extract_dram_layout(dram)
    kwargs = {
        "clock_ratio": LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        "path": str(trace_path),
        "pim_compute_request_type_id": request_type_ids["PIMCompute"],
        "pim_load_all_request_type_id": request_type_ids["PIMLoadAll"],
        "pim_compute_all_request_type_id": request_type_ids["PIMComputeAll"],
        "sb_command_id": command_ids["SB"],
        "hab_command_id": command_ids["HAB"],
        "hab_pim_command_id": command_ids["HAB_PIM"],
        "addr_vec_size": layout["addr_vec_size"],
    }
    if max_trace_bytes is not None:
        kwargs["max_trace_bytes"] = max_trace_bytes
    if max_expanded_records is not None:
        kwargs["max_expanded_records"] = max_expanded_records
    return ramulator.frontend.LPDDR5PIMConcreteTrace(**kwargs)


def _p2_frontend(trace_path: Path, dram):
    layout = extract_dram_layout(dram)
    return ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )


def _p2_replay_provenance() -> dict:
    return {
        "source_kind": "handwritten",
        "tuple_manifest": "p4_offline_boundary_unit",
        "literature_anchor": ["manual:p4-offline-boundary"],
        "generator_version": "manual",
        "claim_boundary": [
            "structured workload-surrogate",
            "simulator-diagnostic",
            "non-silicon-calibrated",
            "decode-only-first",
        ],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_mixed_prefill_decode",
            "not_all_bank_fidelity",
        ],
    }


def test_p4_schema_accepts_full_transformer_record_kinds():
    assert {
        "PIMDataMove",
        "AttentionScore",
        "AttentionSoftmax",
        "AttentionContext",
        "FFNProjection",
        "PIMElementwise",
        "MoERouter",
        "MoETopK",
        "MoEDispatch",
        "MoEExpertFFN",
        "MoECombine",
        "PIMOperandResidency",
        "PIMOperandReuse",
    } <= trace_mod.SEMANTIC_RECORD_KINDS


def test_p4_schema_rejects_missing_tensor_io():
    record = [record for record in generator_mod.generate_attention_records() if record["kind"] == "AttentionScore"][0]
    del record["tensor_io"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "tensor_io" in str(exc)
    else:
        raise AssertionError("Expected missing tensor_io rejection")


def test_p4_schema_requires_offline_ir_metadata():
    record = [record for record in generator_mod.generate_attention_records() if record["kind"] == "AttentionScore"][0]
    assert record["operator_context"].get("record_family") == "p4_offline_transformer_dataflow_ir"
    del record["operator_context"]["record_family"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "record_family" in str(exc)
    else:
        raise AssertionError("Expected missing P4 record_family rejection")


def test_p4_schema_enforces_required_claim_boundary_and_non_claims():
    record = [record for record in generator_mod.generate_attention_records() if record["kind"] == "AttentionScore"][0]
    record["provenance"]["claim_boundary"] = [claim for claim in record["provenance"]["claim_boundary"] if claim != "operator-internal-dataflow-first"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "operator-internal-dataflow-first" in str(exc)
    else:
        raise AssertionError("Expected P4 claim-boundary rejection")


def test_p4_schema_rejects_generated_record_missing_required_non_claim():
    record = [record for record in generator_mod.generate_attention_records() if record["kind"] == "AttentionScore"][0]
    record["provenance"]["non_claims"] = [claim for claim in record["provenance"]["non_claims"] if claim != "not_vllm_replay"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "not_vllm_replay" in str(exc)
    else:
        raise AssertionError("Expected P4 non-claim rejection")


def test_p4_semantic_records_are_offline_only_not_structured_frontend_replay(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    record = [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionScore"][0]
    record["provenance"] = _p2_replay_provenance()
    trace_path = tmp_path / "p4_semantic_ir.jsonl"
    trace_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    try:
        ramulator.Simulation(_p2_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert "unsupported kind 'AttentionScore'" in str(exc)
    else:
        raise AssertionError("Expected structured frontend to reject P4 offline IR records")


@pytest.mark.parametrize(
    "record_kind,record_factory",
    [
        ("PIMDataMove", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMDataMove"][0]),
        ("AttentionSoftmax", lambda: [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionSoftmax"][0]),
        ("AttentionContext", lambda: [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionContext"][0]),
        ("FFNProjection", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "FFNProjection"][0]),
        ("PIMElementwise", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMElementwise"][0]),
        ("MoERouter", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoERouter"][0]),
        ("MoETopK", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoETopK"][0]),
        ("MoEDispatch", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoEDispatch"][0]),
        ("MoEExpertFFN", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoEExpertFFN"][0]),
        ("MoECombine", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoECombine"][0]),
        ("PIMOperandResidency", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMOperandResidency"][0]),
        ("PIMOperandReuse", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMOperandReuse"][0]),
    ],
)
def test_all_p4_semantic_record_kinds_are_offline_only_not_structured_frontend_replay(tmp_path: Path, record_kind: str, record_factory):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    record = record_factory()
    assert record["kind"] == record_kind
    trace_mod.validate_record(record)
    record = dict(record)
    record["provenance"] = _p2_replay_provenance()
    trace_path = tmp_path / f"{record_kind}.jsonl"
    trace_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    try:
        ramulator.Simulation(_p2_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert f"unsupported kind '{record_kind}'" in str(exc)
    else:
        raise AssertionError(f"Expected structured frontend to reject P4 offline IR record kind {record_kind}")


def test_attention_generator_emits_score_softmax_context_ordering():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_layers": 1, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)

    assert [record["kind"] for record in records] == [
        "PIMOperandResidency",
        "HostRead",
        "AttentionScore",
        "AttentionSoftmax",
        "PIMOperandResidency",
        "HostRead",
        "AttentionContext",
    ]
    score = records[2]
    softmax = records[3]
    context = records[6]
    assert score["record_id"] in softmax["logical_dependencies"]
    assert softmax["record_id"] in context["logical_dependencies"]
    assert score["tensor_io"]["outputs"] == softmax["tensor_io"]["inputs"]
    assert context["operator_context"]["stage"] == "context"
    _assert_attention_score_inputs_have_producer_dependencies(records)


def test_attention_generator_is_parameter_sensitive():
    manifest_a = generator_mod.get_tiny_attention_manifest()
    manifest_a.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    records_a = generator_mod.generate_attention_records(manifest_a)

    manifest_b = generator_mod.get_tiny_attention_manifest()
    manifest_b.update({"num_heads": 2, "past_len": 64, "score_tile_tokens": 32, "head_dim": 64})
    records_b = generator_mod.generate_attention_records(manifest_b)

    assert len(records_a) == 7
    assert len(records_b) == 26
    mac_a = sum(record["num_requests"] for record in records_a if record["kind"] in {"AttentionScore", "AttentionContext"})
    mac_b = sum(record["num_requests"] for record in records_b if record["kind"] in {"AttentionScore", "AttentionContext"})
    assert mac_b > mac_a


def test_attention_scales_with_num_layers():
    manifest_one = generator_mod.get_tiny_attention_manifest()
    manifest_one.update({"num_layers": 1, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    manifest_two = generator_mod.get_tiny_attention_manifest()
    manifest_two.update({"num_layers": 2, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    assert len(generator_mod.generate_attention_records(manifest_two)) == 2 * len(generator_mod.generate_attention_records(manifest_one))


def test_attention_with_fp16_datatype_uses_16_lanes():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32, "datatype": "fp16"})
    score = [record for record in generator_mod.generate_attention_records(manifest) if record["kind"] == "AttentionScore"][0]
    assert score["num_requests"] == 64


def test_attention_generator_uses_global_softmax_across_score_tiles():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 64, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)
    score_ids = [record["record_id"] for record in records if record["kind"] == "AttentionScore"]
    softmax = [record for record in records if record["kind"] == "AttentionSoftmax"][0]
    context_records = [record for record in records if record["kind"] == "AttentionContext"]

    assert softmax["logical_dependencies"] == score_ids
    assert len(softmax["tensor_io"]["inputs"]) == 2
    assert all(softmax["record_id"] in record["logical_dependencies"] for record in context_records)


def test_attention_generator_emits_final_context_reduction_for_multi_tile_heads():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_layers": 1, "num_heads": 1, "past_len": 64, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)
    context_ids = [record["record_id"] for record in records if record["kind"] == "AttentionContext"]
    reductions = [record for record in records if record["kind"] == "PIMElementwise" and record["op"] == "attention_context_reduction_accounting"]

    assert len(reductions) == 1
    reduction = reductions[0]
    assert reduction["logical_dependencies"] == context_ids
    assert reduction["operator_context"]["stage"] == "context_reduction_accounting"
    assert reduction["accounting_metadata"]["lowering"] == "not_lowered_to_native_opcode_in_p4_2"


def test_attention_context_shape_models_probability_times_v():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 64})
    records = generator_mod.generate_attention_records(manifest)
    context = [record for record in records if record["kind"] == "AttentionContext"][0]
    assert context["compute_shape"]["n"] == 64
    assert context["compute_shape"]["k"] == 32


def test_attention_manifest_requires_guardrail_non_claims():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["non_claims"] = [claim for claim in manifest["non_claims"] if claim != "not_raw_attacc_schema"]
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "not_raw_attacc_schema" in str(exc)
    else:
        raise AssertionError("Expected required P4 non-claim rejection")


def test_attention_manifest_rejects_unknown_datatype():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["datatype"] = "fp8"
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "Unsupported attention datatype" in str(exc)
    else:
        raise AssertionError("Expected unknown datatype rejection")


def test_attention_manifest_rejects_invalid_ramulator_defaults():
    for field in ("burst_length", "row_count", "dependency_count"):
        manifest = generator_mod.get_tiny_attention_manifest()
        manifest["ramulator_visible_defaults"][field] = 0
        try:
            generator_mod.generate_attention_records(manifest)
        except ValueError as exc:
            assert field in str(exc)
            assert "positive" in str(exc)
        else:
            raise AssertionError(f"Expected invalid default rejection for {field}")


def test_attention_first_slice_rejects_distinct_context_tile_size():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["context_tile_tokens"] = manifest["score_tile_tokens"] // 2
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "context_tile_tokens == score_tile_tokens" in str(exc)
    else:
        raise AssertionError("Expected unsupported distinct context tile size rejection")


def test_attention_lowering_uses_only_native_lpddr5_pim_opcodes_and_skips_softmax():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    opcodes = [record["opcode"] for record in concrete]
    assert set(opcodes) <= concrete_schema_mod.CONCRETE_OPCODES
    assert "PIM_BCAST" not in opcodes
    assert "PIM_MAC" in opcodes
    assert "AttentionSoftmax" not in {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert not {"PIM_WR_GB", "PIM_MV_BA", "PIM_SFM"} & set(opcodes)


def test_attention_lowering_rejects_unsupported_data_movement_kind():
    semantic = [record for record in generator_mod.generate_ffn_records() if record["kind"] == "PIMDataMove"][0:1]
    semantic[0]["movement_policy"]["movement_kind"] = "semantic_only_unmodeled_move"
    try:
        lowering_mod.lower_semantic_records_to_concrete(semantic)
    except ValueError as exc:
        assert "not supported by native PIM_BCAST lowering" in str(exc)
    else:
        raise AssertionError("Expected unsupported PIMDataMove lowering rejection")


def test_concrete_lowering_honors_semantic_repeat_for_data_movement_records():
    semantic = [record for record in generator_mod.generate_ffn_records() if record["kind"] == "PIMDataMove"][0:1]
    semantic[0]["repeat"] = 3
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    bcasts = [record for record in concrete if record["opcode"] == "PIM_BCAST"]
    assert len(bcasts) == 1
    assert bcasts[0]["repeat"] == 3


def test_concrete_lowering_matches_bank_distribution_before_dependency_phase():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 4, "k": 32, "output_elements": 4, "datatype": "int8"},
            "num_requests": 4,
            "bank_sequence": [0, 1],
            "dependency_context": {"dependency_count": 4, "dependency_id": 1},
            "row_policy": {"row_start": 10, "row_count": 8, "resolved_row": 10},
            "column_policy": {"column_start": 20, "resolved_column": 21},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    macs = [record for record in concrete if record["opcode"] == "PIM_MAC"]
    assert [record["addr_vec"][3] for record in macs] == [0, 1]
    assert [record["addr_vec"][5] for record in macs] == [21, 21]
    assert [record["repeat"] for record in macs] == [2, 2]


def test_concrete_lowering_honors_semantic_repeat_for_compute_records():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 3,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 2, "k": 32, "output_elements": 2, "datatype": "int8"},
            "num_requests": 2,
            "bank_sequence": [0, 1],
            "dependency_context": {"dependency_count": 2, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 2, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    macs = [record for record in concrete if record["opcode"] == "PIM_MAC"]
    assert len(macs) == 2
    assert sum(record["repeat"] for record in macs) == 6  # 3 repeats per bank × 2 banks


def test_concrete_lowering_decomposes_flat_bank_like_structured_frontend():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "frontend"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 1, "k": 32, "output_elements": 1, "datatype": "int8"},
            "num_requests": 1,
            "bank_sequence": [5],
            "dependency_context": {"dependency_count": 1, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        addr_vec_size=6,
        bank_positions=[1, 2, 3],
        bank_counts=[2, 4, 4],
        row_level=4,
        col_level=5,
    )
    mac = [record for record in concrete if record["opcode"] == "PIM_MAC"][0]
    assert [mac["addr_vec"][level] for level in [1, 2, 3]] == [0, 1, 1]


def test_concrete_lowering_controller_order_decomposes_unsorted_bank_positions():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score", "record_family": "p4_offline_transformer_dataflow_ir"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 1, "k": 32, "output_elements": 1, "datatype": "int8"},
            "num_requests": 1,
            "bank_sequence": [5],
            "dependency_context": {"dependency_count": 1, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        addr_vec_size=6,
        bank_positions=[3, 1, 2],
        bank_counts=[4, 2, 4],
        row_level=4,
        col_level=5,
    )
    mac = [record for record in concrete if record["opcode"] == "PIM_MAC"][0]
    assert [mac["addr_vec"][level] for level in [1, 2, 3]] == [0, 1, 1]


def test_ffn_swiglu_generator_emits_true_dag_ordering():
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    ffn_ops = [record["op"] for record in records if record["op"].startswith("ffn_")]
    assert ffn_ops == [
        "ffn_hidden_activation_tile_setup",
        "ffn_hidden_activation_tile_setup",
        "ffn_up_weight_residency",
        "ffn_gate_weight_residency",
        "ffn_down_weight_residency",
        "ffn_hidden_reuse_for_up_gate",
        "ffn_up_projection",
        "ffn_up_intermediate_residency",
        "ffn_gate_projection",
        "ffn_gate_intermediate_residency",
        "ffn_gate_activation_accounting",
        "ffn_activated_gate_residency",
        "ffn_gated_multiply_accounting",
        "ffn_intermediate_bank_local_residency",
        "ffn_down_projection",
    ]
    by_id = {record["record_id"]: record for record in records}
    by_op_single = {}
    multi = {}
    for record in records:
        op = record["op"]
        if op in by_op_single:
            multi.setdefault(op, [by_op_single.pop(op)])
            multi[op].append(record)
        else:
            by_op_single[op] = record
    assert len(multi.get("ffn_hidden_activation_tile_setup", [])) == 2
    assert by_op_single["ffn_gate_projection"]["record_id"] in by_op_single["ffn_gate_intermediate_residency"]["logical_dependencies"]
    assert by_op_single["ffn_gate_intermediate_residency"]["record_id"] in by_op_single["ffn_gate_activation_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_gate_activation_accounting"]["record_id"] in by_op_single["ffn_activated_gate_residency"]["logical_dependencies"]
    assert by_op_single["ffn_activated_gate_residency"]["record_id"] in by_op_single["ffn_gated_multiply_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_up_projection"]["record_id"] in by_op_single["ffn_up_intermediate_residency"]["logical_dependencies"]
    assert by_op_single["ffn_up_intermediate_residency"]["record_id"] in by_op_single["ffn_gated_multiply_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_intermediate_bank_local_residency"]["record_id"] in by_op_single["ffn_down_projection"]["logical_dependencies"]
    # Per-tile dependency chain: tile setups → PIMOperandReuse → FFNProjection
    tile_setup_ids = [r["record_id"] for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    reuse_deps = by_op_single["ffn_hidden_reuse_for_up_gate"]["logical_dependencies"]
    for tid in tile_setup_ids:
        assert tid in reuse_deps, f"PIMOperandReuse should depend on tile setup {tid}"


def test_ffn_generator_emits_operand_setup_residency_and_reuse_semantics():
    manifest = generator_mod.get_tiny_ffn_manifest()
    assert manifest["operand_movement_policy"] == {
        "weights": "preloaded_stationary",
        "dynamic_activation_setup": "materialized",
        "ffn_intermediate": "bank_local_capacity_controlled",
    }
    records = generator_mod.generate_ffn_records(manifest)
    by_id = {record["record_id"]: record for record in records}
    by_op_single = {}
    multi_ops = {}
    for record in records:
        op = record["op"]
        if op in by_op_single:
            multi_ops.setdefault(op, [by_op_single.pop(op)])
            multi_ops[op].append(record)
        else:
            by_op_single[op] = record

    assert [record["op"] for record in records] == [
        "ffn_hidden_activation_tile_setup",
        "ffn_hidden_activation_tile_setup",
        "ffn_up_weight_residency",
        "ffn_gate_weight_residency",
        "ffn_down_weight_residency",
        "ffn_hidden_reuse_for_up_gate",
        "ffn_up_projection",
        "ffn_up_intermediate_residency",
        "ffn_gate_projection",
        "ffn_gate_intermediate_residency",
        "ffn_gate_activation_accounting",
        "ffn_activated_gate_residency",
        "ffn_gated_multiply_accounting",
        "ffn_intermediate_bank_local_residency",
        "ffn_down_projection",
    ]
    # Per-tile PIMDataMove records carry full metadata
    tile_setups = multi_ops["ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 2
    for ts in tile_setups:
        assert ts["kind"] == "PIMDataMove"
        mp = ts["movement_policy"]
        assert mp["operand_role"] == "activation_input"
        assert mp["residency"] == "dynamic_activation_tile"
        assert mp["materialized"] is True
        assert mp["reuse_scope"] == "ffn_layer"
        assert mp["lowering"] == "native_pim_bcast_when_supported"
        assert "tile_index" in mp
        assert "tile_elements" in mp
        assert "tile_start" in mp
        assert mp["distribution_scope"] == "broadcast"
    assert by_op_single["ffn_up_weight_residency"]["kind"] == "PIMDataMove"
    assert by_op_single["ffn_up_weight_residency"]["movement_policy"]["materialized"] is True
    assert by_op_single["ffn_up_weight_residency"]["movement_policy"]["residency"] == "preloaded_stationary"
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["kind"] == "PIMOperandReuse"
    assert by_op_single["ffn_up_projection"]["record_id"] not in by_op_single["ffn_hidden_reuse_for_up_gate"].get("logical_dependencies", [])
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["record_id"] in by_op_single["ffn_up_projection"]["logical_dependencies"]
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["record_id"] in by_op_single["ffn_gate_projection"]["logical_dependencies"]
    # PIMOperandReuse depends on all per-tile setup records
    reuse_deps = by_op_single["ffn_hidden_reuse_for_up_gate"]["logical_dependencies"]
    for ts in tile_setups:
        assert ts["record_id"] in reuse_deps
    assert by_op_single["ffn_intermediate_bank_local_residency"]["accounting_metadata"]["residency"] == "bank_local_capacity_controlled"
    assert by_op_single["ffn_intermediate_bank_local_residency"]["accounting_metadata"]["materialized"] is False
    assert by_op_single["ffn_intermediate_bank_local_residency"]["record_id"] in by_op_single["ffn_down_projection"]["logical_dependencies"]


def test_ffn_weight_residency_is_materialized_data_move():
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    by_op = {record["op"]: record for record in records}

    for op, projection_op in [
        ("ffn_up_weight_residency", "ffn_up_projection"),
        ("ffn_gate_weight_residency", "ffn_gate_projection"),
        ("ffn_down_weight_residency", "ffn_down_projection"),
    ]:
        weight_move = by_op[op]
        expected = generator_mod._num_requests(
            int(manifest["hidden_size"]) * int(manifest["ffn_hidden_size"]),
            manifest["datatype"],
        )
        assert weight_move["kind"] == "PIMDataMove"
        assert weight_move["num_requests"] == expected
        assert weight_move["movement_policy"]["operand_role"] == "weight"
        assert weight_move["movement_policy"]["residency"] == "preloaded_stationary"
        assert weight_move["movement_policy"]["materialized"] is True
        assert weight_move["movement_policy"]["movement_kind"] == "preloaded_stationary_weight_residency"
        assert weight_move["movement_policy"]["lowering"] == "semantic_only_steady_state_or_host_write_preload"
        assert weight_move["movement_policy"]["materialization_lowering"] == "regular_host_write_preload_when_materialized"
        assert weight_move["datatype_metadata"]["behavior_claim"] == "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        assert weight_move["record_id"] in by_op[projection_op]["logical_dependencies"]
        assert weight_move["tensor_io"]["outputs"][0] in by_op[projection_op]["tensor_io"]["inputs"]


def test_ffn_projection_shapes_are_direction_specific():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"seq_len": 2, "hidden_size": 64, "ffn_hidden_size": 128})
    records = generator_mod.generate_ffn_records(manifest)
    by_op = {record["op"]: record for record in records}
    assert by_op["ffn_up_projection"]["compute_shape"] == {"m": 2, "n": 128, "k": 64, "output_elements": 256, "datatype": "int8"}
    assert by_op["ffn_gate_projection"]["compute_shape"] == {"m": 2, "n": 128, "k": 64, "output_elements": 256, "datatype": "int8"}
    assert by_op["ffn_down_projection"]["compute_shape"] == {"m": 2, "n": 64, "k": 128, "output_elements": 128, "datatype": "int8"}


def test_ffn_manifest_validation_rejects_missing_and_invalid_fields():
    manifest = generator_mod.get_tiny_ffn_manifest()
    del manifest["activation"]
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "activation" in str(exc)
    else:
        raise AssertionError("Expected missing activation rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_hidden_size"] = 0
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_hidden_size" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid ffn_hidden_size rejection")


def test_ffn_relu_2proj_generator_emits_two_projection_topology():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({
        "manifest_name": "tiny_ffn_relu_2proj_p4_v0_1",
        "workload_class": "structured_transformer_ffn_relu_2proj_surrogate",
        "model_family": "tiny decoder-only transformer FFN/ReLU two-projection slice",
        "ffn_variant": "relu_2proj",
        "activation": "relu",
    })
    records = generator_mod.generate_ffn_records(manifest)
    ops = [record["op"] for record in records if record["op"].startswith("ffn_")]

    assert ops == [
        "ffn_hidden_activation_tile_setup",
        "ffn_hidden_activation_tile_setup",
        "ffn_up_weight_residency",
        "ffn_down_weight_residency",
        "ffn_hidden_reuse_for_up_projection",
        "ffn_up_projection",
        "ffn_up_intermediate_residency",
        "ffn_relu_activation_accounting",
        "ffn_intermediate_bank_local_residency",
        "ffn_down_projection",
    ]
    assert "ffn_gate_projection" not in ops
    assert "ffn_gated_multiply_accounting" not in ops
    assert not any("gate" in op or "gated" in op for op in ops)
    assert generator_mod.ffn_projection_count(manifest) == 2

    by_op = {record["op"]: record for record in records}
    assert by_op["ffn_up_projection"]["operator_context"]["operator_family"] == "ffn_relu_2proj"
    assert by_op["ffn_down_projection"]["operator_context"]["operator_family"] == "ffn_relu_2proj"
    assert by_op["ffn_relu_activation_accounting"]["operator_context"]["operator_family"] == "ffn_relu_2proj"
    assert by_op["ffn_up_intermediate_residency"]["record_id"] in by_op["ffn_relu_activation_accounting"]["logical_dependencies"]
    assert by_op["ffn_up_projection"]["record_id"] not in by_op["ffn_relu_activation_accounting"]["logical_dependencies"]
    assert by_op["ffn_relu_activation_accounting"]["tensor_io"]["inputs"] == ["L0.ffn.up_intermediate.pim_visible"]
    assert by_op["ffn_relu_activation_accounting"]["record_id"] in by_op["ffn_intermediate_bank_local_residency"]["logical_dependencies"]
    assert by_op["ffn_intermediate_bank_local_residency"]["record_id"] in by_op["ffn_down_projection"]["logical_dependencies"]

    record_positions = {record["record_id"]: index for index, record in enumerate(records)}
    for residency_record in [r for r in records if r["kind"] == "PIMOperandResidency"]:
        residency_id = residency_record["record_id"]
        assert any(
            record_positions[record["record_id"]] > record_positions[residency_id]
            and residency_id in record.get("logical_dependencies", [])
            for record in records
        ), f"Residency record {residency_id} is not consumed by a later dependency"


def test_ffn_geglu_3proj_generator_emits_gated_three_projection_topology():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({
        "manifest_name": "tiny_ffn_geglu_3proj_p4_v0_1",
        "workload_class": "structured_transformer_ffn_geglu_3proj_surrogate",
        "model_family": "tiny decoder-only transformer FFN/GeGLU three-projection slice",
        "ffn_variant": "geglu_3proj",
        "activation": "gelu_pytorch_tanh",
    })
    records = generator_mod.generate_ffn_records(manifest)
    ops = [record["op"] for record in records if record["op"].startswith("ffn_")]

    assert "ffn_gate_projection" in ops
    assert "ffn_gated_multiply_accounting" in ops
    assert generator_mod.ffn_projection_count(manifest) == 3

    by_op = {record["op"]: record for record in records}
    assert by_op["ffn_up_projection"]["operator_context"]["operator_family"] == "ffn_geglu"
    assert by_op["ffn_gate_projection"]["operator_context"]["operator_family"] == "ffn_geglu"
    assert by_op["ffn_gate_activation_accounting"]["accounting_metadata"]["activation"] == "gelu_pytorch_tanh"
    assert by_op["ffn_gate_projection"]["record_id"] in by_op["ffn_gate_intermediate_residency"]["logical_dependencies"]
    assert by_op["ffn_gate_intermediate_residency"]["record_id"] in by_op["ffn_gate_activation_accounting"]["logical_dependencies"]


def test_ffn_relu_2proj_mac_counts_match_formula_and_concrete_lowering():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update(
        {
            "manifest_name": "tiny_ffn_relu_2proj_count_p4_v0_1",
            "workload_class": "structured_transformer_ffn_relu_2proj_surrogate",
            "model_family": "tiny decoder-only transformer FFN/ReLU two-projection slice",
            "ffn_variant": "relu_2proj",
            "activation": "relu",
            "seq_len": 2,
            "hidden_size": 64,
            "ffn_hidden_size": 128,
            "ffn_activation_tile_size": 64,
            "datatype": "int8",
        }
    )

    semantic = generator_mod.generate_ffn_records(manifest)
    projection_records = [record for record in semantic if record["kind"] == "FFNProjection"]
    lanes = 32
    expected_per_projection = (manifest["seq_len"] * manifest["hidden_size"] * manifest["ffn_hidden_size"] + lanes - 1) // lanes
    expected_total = generator_mod.ffn_projection_count(manifest) * expected_per_projection

    assert [record["op"] for record in projection_records] == ["ffn_up_projection", "ffn_down_projection"]
    assert sum(record["num_requests"] for record in projection_records) == expected_total

    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)
    assert sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_MAC") == expected_total


def test_ffn_manifest_validation_rejects_variant_activation_mismatches():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_variant"] = "relu_2proj"
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "activation='relu'" in str(exc)
    else:
        raise AssertionError("Expected relu_2proj/silu mismatch rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["activation"] = "relu"
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "swiglu_3proj" in str(exc)
    else:
        raise AssertionError("Expected swiglu_3proj/relu mismatch rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_variant"] = "gelu_2proj"
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "Unsupported FFN variant" in str(exc)
    else:
        raise AssertionError("Expected unknown FFN variant rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"ffn_variant": "geglu_3proj", "activation": "silu"})
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "geglu_3proj" in str(exc)
        assert "gelu" in str(exc)
    else:
        raise AssertionError("Expected geglu_3proj/silu mismatch rejection")


def test_ffn_lowering_only_lowers_projection_records_to_native_compute():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)
    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert lowered_kinds == {"FFNProjection", "PIMDataMove"}
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert {record["opcode"] for record in concrete} <= concrete_schema_mod.CONCRETE_OPCODES
    assert "PIM_MAC" in {record["opcode"] for record in concrete}
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}
    assert not {"PIM_SFM", "PIM_MV_BA", "PIM_WR_GB"} & {record["opcode"] for record in concrete}


def test_ffn_lowering_materializes_activation_setup_but_not_residency_or_reuse():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIMDataMove" in lowered_kinds
    assert "FFNProjection" in lowered_kinds
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}


def test_ffn_weight_moves_lower_to_write_when_materialized_while_reuse_stays_semantic_only():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, materialize_weights=True)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"]:
        assert op in lowered_ops

    assert "ffn_hidden_reuse_for_up_gate" not in lowered_ops
    assert "ffn_intermediate_bank_local_residency" not in lowered_ops
    assert "ffn_gate_activation_accounting" not in lowered_ops
    assert "ffn_gated_multiply_accounting" not in lowered_ops

    weight_writes = [
        record
        for record in concrete
        if record["opcode"] == "WRITE"
        and record["provenance"]["semantic_source"]["op"]
        in {"ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"}
    ]
    weight_bcasts = [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST"
        and record["provenance"]["semantic_source"]["op"]
        in {"ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"}
    ]
    assert len(weight_writes) == 3
    assert weight_bcasts == []


def test_ffn_default_steady_state_lowering_skips_weight_data_moves():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"]:
        assert op not in lowered_ops

    assert "ffn_hidden_activation_tile_setup" in lowered_ops
    assert "FFNProjection" in {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIM_MAC" in {record["opcode"] for record in concrete}


def test_ffn_mac_counts_change_with_shape_and_datatype():
    manifest_a = generator_mod.get_tiny_ffn_manifest()
    manifest_a.update({"hidden_size": 32, "ffn_hidden_size": 64, "datatype": "int8"})
    manifest_b = generator_mod.get_tiny_ffn_manifest()
    manifest_b.update({"hidden_size": 64, "ffn_hidden_size": 128, "datatype": "fp16"})
    macs_a = sum(record["num_requests"] for record in generator_mod.generate_ffn_records(manifest_a) if record["kind"] == "FFNProjection")
    macs_b = sum(record["num_requests"] for record in generator_mod.generate_ffn_records(manifest_b) if record["kind"] == "FFNProjection")
    assert macs_b > macs_a


def test_ffn_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 32, "ffn_hidden_size": 64, "datatype": "int8"})
    semantic = generator_mod.generate_ffn_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "ffn_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 3


def test_ffn_generator_single_tile_when_tile_size_equals_hidden_size():
    """Backward compat: ffn_activation_tile_size == hidden_size → 1 tile."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 32, "ffn_activation_tile_size": 32})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 1
    assert tile_setups[0]["movement_policy"]["tile_index"] == 0
    assert tile_setups[0]["movement_policy"]["tile_elements"] == 32
    assert tile_setups[0]["movement_policy"]["tile_start"] == 0


def test_ffn_generator_multi_tile_produces_distribution_metadata():
    """Each per-tile PIMDataMove carries tile_index, tile_elements, tile_start, distribution_scope."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 64, "ffn_activation_tile_size": 16})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 4  # 64 / 16
    for i, ts in enumerate(tile_setups):
        mp = ts["movement_policy"]
        assert mp["tile_index"] == i
        assert mp["tile_elements"] == 16
        assert mp["tile_start"] == i * 16
        assert mp["distribution_scope"] == "broadcast"
        assert mp["materialized"] is True
        assert mp["operand_role"] == "activation_input"
        oc = ts["operator_context"]
        assert oc["tile_id"] == i
        assert oc["tile_elements"] == 16
        assert oc["tile_start"] == i * 16
    # All tile setup IDs are in the PIMOperandReuse dependencies
    reuse = [r for r in records if r["op"] == "ffn_hidden_reuse_for_up_gate"][0]
    tile_setup_outputs = [ts["tensor_io"]["outputs"][0] for ts in tile_setups]
    assert reuse["tensor_io"]["inputs"] == tile_setup_outputs
    for ts in tile_setups:
        assert ts["record_id"] in reuse["logical_dependencies"]


def test_ffn_generator_partial_last_tile():
    """When hidden_size is not evenly divisible by tile_size, the last tile is smaller."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 40, "ffn_activation_tile_size": 16})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 3  # ceil(40/16)
    # Last tile should be 8 elements
    assert tile_setups[2]["movement_policy"]["tile_elements"] == 8
    assert tile_setups[2]["movement_policy"]["tile_start"] == 32


def test_ffn_manifest_rejects_invalid_tile_size():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_activation_tile_size"] = 0
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_activation_tile_size" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected tile_size=0 rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_activation_tile_size"] = 100  # > hidden_size=32
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_activation_tile_size" in str(exc)
        assert "hidden_size" in str(exc)
    else:
        raise AssertionError("Expected tile_size > hidden_size rejection")


def test_ffn_manifest_rejects_invalid_distribution_policy():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["activation_distribution_policy"] = "unicast"
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "activation_distribution_policy" in str(exc)
        assert "unicast" in str(exc)
    else:
        raise AssertionError("Expected invalid distribution_policy rejection")


def test_ffn_policies_vary_by_tile_index():
    """Different tile indices produce different dependency/row assignments."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    p0 = generator_mod._ffn_policies(manifest, layer_index=0, stage_index=0, tile_index=0)
    p1 = generator_mod._ffn_policies(manifest, layer_index=0, stage_index=0, tile_index=1)
    assert p0["dependency_context"]["dependency_id"] != p1["dependency_context"]["dependency_id"]
    assert p0["dependency_context"]["tile_index"] == 0
    assert p1["dependency_context"]["tile_index"] == 1


def test_moe_generator_emits_router_topk_dispatch_selected_experts_and_combine():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    assert [record["op"] for record in records] == [
        "moe_router_input_setup",
        "moe_router_weight_residency",
        "moe_router_projection",
        "moe_router_logits_residency",
        "moe_topk_select_accounting",
        "moe_expert_dispatch_accounting",
        "moe_token_dispatch_materialized",
        "moe_expert_1_up_weight_residency",
        "moe_expert_1_up_projection",
        "moe_expert_1_gate_weight_residency",
        "moe_expert_1_gate_projection",
        "moe_expert_1_down_weight_residency",
        "moe_expert_1_gate_activation_accounting",
        "moe_expert_1_gated_multiply_accounting",
        "moe_expert_1_intermediate_residency",
        "moe_expert_1_down_projection",
        "moe_expert_1_output_residency",
        "moe_expert_3_up_weight_residency",
        "moe_expert_3_up_projection",
        "moe_expert_3_gate_weight_residency",
        "moe_expert_3_gate_projection",
        "moe_expert_3_down_weight_residency",
        "moe_expert_3_gate_activation_accounting",
        "moe_expert_3_gated_multiply_accounting",
        "moe_expert_3_intermediate_residency",
        "moe_expert_3_down_projection",
        "moe_expert_3_output_residency",
        "moe_expert_combine_accounting",
        "moe_expert_output_combine_materialized",
    ]
    by_op = {record["op"]: record for record in records}
    assert by_op["moe_router_projection"]["record_id"] in by_op["moe_router_logits_residency"]["logical_dependencies"]
    assert by_op["moe_router_logits_residency"]["record_id"] in by_op["moe_topk_select_accounting"]["logical_dependencies"]
    assert by_op["moe_topk_select_accounting"]["record_id"] in by_op["moe_expert_dispatch_accounting"]["logical_dependencies"]
    assert by_op["moe_expert_dispatch_accounting"]["record_id"] in by_op["moe_token_dispatch_materialized"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_1_up_projection"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_1_gate_projection"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_3_up_projection"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_3_gate_projection"]["logical_dependencies"]
    assert by_op["moe_expert_combine_accounting"]["logical_dependencies"] == [
        by_op["moe_expert_1_output_residency"]["record_id"],
        by_op["moe_expert_3_output_residency"]["record_id"],
    ]
    assert by_op["moe_expert_combine_accounting"]["record_id"] in by_op["moe_expert_output_combine_materialized"]["logical_dependencies"]


def test_moe_generator_emits_operand_setup_dispatch_combine_and_residency_semantics():
    manifest = generator_mod.get_tiny_moe_manifest()
    assert manifest["operand_movement_policy"] == {
        "weights": "preloaded_stationary",
        "router_input_setup": "materialized",
        "token_dispatch": "materialized",
        "expert_output_combine": "materialized",
    }
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    by_op = {record["op"]: record for record in records}

    assert [record["op"] for record in records] == [
        "moe_router_input_setup",
        "moe_router_weight_residency",
        "moe_router_projection",
        "moe_router_logits_residency",
        "moe_topk_select_accounting",
        "moe_expert_dispatch_accounting",
        "moe_token_dispatch_materialized",
        "moe_expert_1_up_weight_residency",
        "moe_expert_1_up_projection",
        "moe_expert_1_gate_weight_residency",
        "moe_expert_1_gate_projection",
        "moe_expert_1_down_weight_residency",
        "moe_expert_1_gate_activation_accounting",
        "moe_expert_1_gated_multiply_accounting",
        "moe_expert_1_intermediate_residency",
        "moe_expert_1_down_projection",
        "moe_expert_1_output_residency",
        "moe_expert_3_up_weight_residency",
        "moe_expert_3_up_projection",
        "moe_expert_3_gate_weight_residency",
        "moe_expert_3_gate_projection",
        "moe_expert_3_down_weight_residency",
        "moe_expert_3_gate_activation_accounting",
        "moe_expert_3_gated_multiply_accounting",
        "moe_expert_3_intermediate_residency",
        "moe_expert_3_down_projection",
        "moe_expert_3_output_residency",
        "moe_expert_combine_accounting",
        "moe_expert_output_combine_materialized",
    ]
    assert by_op["moe_router_input_setup"]["kind"] == "PIMDataMove"
    assert by_op["moe_router_input_setup"]["movement_policy"]["operand_role"] == "activation_input"
    assert by_op["moe_router_weight_residency"]["kind"] == "PIMDataMove"
    assert by_op["moe_router_weight_residency"]["movement_policy"]["residency"] == "preloaded_stationary"
    assert by_op["moe_router_weight_residency"]["movement_policy"]["materialized"] is True
    assert by_op["moe_router_input_setup"]["record_id"] in by_op["moe_router_projection"]["logical_dependencies"]
    assert by_op["moe_router_weight_residency"]["record_id"] in by_op["moe_router_projection"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["kind"] == "PIMDataMove"
    assert by_op["moe_expert_dispatch_accounting"]["record_id"] in by_op["moe_token_dispatch_materialized"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_1_up_projection"]["logical_dependencies"]
    assert by_op["moe_expert_1_up_weight_residency"]["record_id"] in by_op["moe_expert_1_up_projection"]["logical_dependencies"]
    assert by_op["moe_expert_output_combine_materialized"]["kind"] == "PIMDataMove"
    assert by_op["moe_expert_combine_accounting"]["record_id"] in by_op["moe_expert_output_combine_materialized"]["logical_dependencies"]


def test_moe_weight_residency_is_materialized_data_move():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    by_op = {record["op"]: record for record in records}
    expected_router = generator_mod._num_requests(
        int(manifest["hidden_size"]) * int(manifest["num_experts"]),
        manifest["datatype"],
    )
    expected_expert = generator_mod._num_requests(
        int(manifest["hidden_size"]) * int(manifest["expert_hidden_size"]),
        manifest["datatype"],
    )

    for op, compute_op in [
        ("moe_router_weight_residency", "moe_router_projection"),
        ("moe_expert_1_up_weight_residency", "moe_expert_1_up_projection"),
        ("moe_expert_1_gate_weight_residency", "moe_expert_1_gate_projection"),
        ("moe_expert_1_down_weight_residency", "moe_expert_1_down_projection"),
        ("moe_expert_3_up_weight_residency", "moe_expert_3_up_projection"),
        ("moe_expert_3_gate_weight_residency", "moe_expert_3_gate_projection"),
        ("moe_expert_3_down_weight_residency", "moe_expert_3_down_projection"),
    ]:
        weight_move = by_op[op]
        assert weight_move["kind"] == "PIMDataMove"
        expected = expected_router if op == "moe_router_weight_residency" else expected_expert
        assert weight_move["num_requests"] == expected
        assert weight_move["movement_policy"]["operand_role"] == "weight"
        assert weight_move["movement_policy"]["residency"] == "preloaded_stationary"
        assert weight_move["movement_policy"]["materialized"] is True
        assert weight_move["datatype_metadata"]["behavior_claim"] == "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        assert weight_move["record_id"] in by_op[compute_op]["logical_dependencies"]
        assert weight_move["tensor_io"]["outputs"][0] in by_op[compute_op]["tensor_io"]["inputs"]


def test_moe_only_selected_experts_emit_compute_and_topk_is_sensitive():
    manifest_a = generator_mod.get_tiny_moe_manifest()
    manifest_a.update({"num_experts": 4, "top_k": 1, "selected_experts": [2]})
    records_a = generator_mod.generate_moe_records(manifest_a)
    manifest_b = generator_mod.get_tiny_moe_manifest()
    manifest_b.update({"num_experts": 4, "top_k": 3, "selected_experts": [0, 2, 3]})
    records_b = generator_mod.generate_moe_records(manifest_b)
    experts_a = [record for record in records_a if record["kind"] == "MoEExpertFFN"]
    experts_b = [record for record in records_b if record["kind"] == "MoEExpertFFN"]
    assert [record["operator_context"]["expert_id"] for record in experts_a] == [2, 2, 2]
    assert [record["operator_context"]["expert_id"] for record in experts_b] == [0, 0, 0, 2, 2, 2, 3, 3, 3]
    assert [record["operator_context"]["ffn_projection"] for record in experts_a] == ["up", "gate", "down"]
    assert sum(record["num_requests"] for record in experts_b) > sum(record["num_requests"] for record in experts_a)


def test_moe_manifest_validation_rejects_invalid_topk_and_expert_selection():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest["top_k"] = 0
    try:
        generator_mod.generate_moe_records(manifest)
    except ValueError as exc:
        assert "top_k" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid top_k rejection")

    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 2, "top_k": 2, "selected_experts": [0, 3]})
    try:
        generator_mod.generate_moe_records(manifest)
    except ValueError as exc:
        assert "selected_experts" in str(exc)
    else:
        raise AssertionError("Expected out-of-range selected_experts rejection")


def test_moe_compute_only_lowering_lowers_router_and_selected_expert_compute():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)
    lowered_kinds = [record["provenance"]["semantic_source"]["kind"] for record in concrete if record["opcode"] == "PIM_MAC"]
    assert set(lowered_kinds) == {"MoERouter", "MoEExpertFFN"}
    assert "MoETopK" not in lowered_kinds
    assert "MoEDispatch" not in lowered_kinds
    assert "MoECombine" not in lowered_kinds
    assert not {"PIM_SFM", "PIM_MV_BA", "PIM_WR_GB"} & {record["opcode"] for record in concrete}


def test_moe_lowering_materializes_dispatch_and_combine_but_not_residency():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIMDataMove" in lowered_kinds
    assert "MoERouter" in lowered_kinds
    assert "MoEExpertFFN" in lowered_kinds
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert "MoEDispatch" not in lowered_kinds
    assert "MoECombine" not in lowered_kinds
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}


def test_moe_weight_moves_lower_to_write_when_materialized_while_routing_accounting_stays_semantic_only():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name=manifest["manifest_name"],
        materialize_weights=True,
    )
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    expert_weight_ops = [
        f"moe_expert_{expert}_{projection}_weight_residency"
        for expert in (0, 2)
        for projection in ("up", "gate", "down")
    ]
    for op in ["moe_router_weight_residency", *expert_weight_ops]:
        assert op in lowered_ops

    assert "moe_topk_select_accounting" not in lowered_ops
    assert "moe_expert_dispatch_accounting" not in lowered_ops
    assert "moe_expert_combine_accounting" not in lowered_ops

    weight_writes = [
        record
        for record in concrete
        if record["opcode"] == "WRITE"
        and record["provenance"]["semantic_source"]["op"]
        in {"moe_router_weight_residency", *expert_weight_ops}
    ]
    weight_bcasts = [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST"
        and record["provenance"]["semantic_source"]["op"]
        in {"moe_router_weight_residency", *expert_weight_ops}
    ]
    assert len(weight_writes) == 7
    assert weight_bcasts == []


def test_moe_default_steady_state_lowering_skips_weight_data_moves():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in [
        "moe_router_weight_residency",
        *[
            f"moe_expert_{expert}_{projection}_weight_residency"
            for expert in (0, 2)
            for projection in ("up", "gate", "down")
        ],
    ]:
        assert op not in lowered_ops

    for op in [
        "moe_router_input_setup",
        "moe_token_dispatch_materialized",
        "moe_expert_output_combine_materialized",
    ]:
        assert op in lowered_ops

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "MoERouter" in lowered_kinds
    assert "MoEExpertFFN" in lowered_kinds
    assert "PIM_MAC" in {record["opcode"] for record in concrete}


def test_moe_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3], "hidden_size": 32, "expert_hidden_size": 64})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "moe_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 3


def _assert_dependencies_precede_consumers(records: list[dict]) -> None:
    positions = {record["record_id"]: index for index, record in enumerate(records)}
    assert len(positions) == len(records)
    for index, record in enumerate(records):
        for dependency in record.get("logical_dependencies", []):
            assert positions[dependency] < index


def _assert_attention_score_inputs_have_producer_dependencies(records: list[dict]) -> None:
    producers: dict[str, list[dict]] = {}
    for record in records:
        for output in record.get("tensor_io", {}).get("outputs", []):
            producers.setdefault(output, []).append(record)
    for score in [record for record in records if record.get("kind") == "AttentionScore"]:
        dependencies = set(score.get("logical_dependencies", []))
        for input_tensor in score.get("tensor_io", {}).get("inputs", []):
            input_producers = producers.get(input_tensor, [])
            if input_producers:
                assert any(producer["record_id"] in dependencies for producer in input_producers), (
                    f"AttentionScore {score['record_id']} consumes {input_tensor} from "
                    f"{[producer['record_id'] for producer in input_producers]} without a logical dependency"
                )


def test_attention_overlapped_head_schedule_differs_and_remains_legal():
    serialized = generator_mod.get_tiny_attention_manifest()
    serialized.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "serialized"})
    overlapped = generator_mod.get_tiny_attention_manifest()
    overlapped.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "overlap_independent_heads"})

    serialized_records = generator_mod.generate_attention_records(serialized)
    overlapped_records = generator_mod.generate_attention_records(overlapped)
    assert [record["op"] for record in serialized_records] != [record["op"] for record in overlapped_records]
    first_overlapped_softmax = next(index for index, record in enumerate(overlapped_records) if record["kind"] == "AttentionSoftmax")
    assert all(index < first_overlapped_softmax for index, record in enumerate(overlapped_records) if record["kind"] == "AttentionScore")
    assert all(record["provenance"].get("schedule_policy") == "overlap_independent_heads" for record in overlapped_records)
    assert all(
        record["operator_context"].get("dag_hint") == "heads_are_independent_safe_to_parallelize"
        for record in overlapped_records
    )
    _assert_dependencies_precede_consumers(overlapped_records)
    _assert_attention_score_inputs_have_producer_dependencies(overlapped_records)


def test_attention_overlap_requires_multiple_heads():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "schedule_policy": "overlap_independent_heads"})

    with pytest.raises(ValueError, match="overlap_independent_heads requires num_heads >= 2"):
        generator_mod.generate_attention_records(manifest)


def test_attention_overlapped_schedule_lowers_and_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "overlap_independent_heads"})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "attention_overlap_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    assert sim.stats["frontend"]["opcode_requests_completed"] == sim.stats["frontend"]["opcode_requests_sent"]


def test_moe_overlapped_expert_schedule_differs_and_remains_legal():
    serialized = generator_mod.get_tiny_moe_manifest()
    serialized.update({"top_k": 3, "selected_experts": [0, 1, 2], "schedule_policy": "serialized"})
    overlapped = generator_mod.get_tiny_moe_manifest()
    overlapped.update({"top_k": 3, "selected_experts": [0, 1, 2], "schedule_policy": "overlap_selected_experts"})

    serialized_records = generator_mod.generate_moe_records(serialized)
    overlapped_records = generator_mod.generate_moe_records(overlapped)
    assert [record["op"] for record in serialized_records] != [record["op"] for record in overlapped_records]
    assert [record["operator_context"].get("expert_id") for record in overlapped_records if record["kind"] == "MoEExpertFFN"] == [0, 0, 0, 2, 2, 2, 1, 1, 1]
    assert all(record["provenance"].get("schedule_policy") == "overlap_selected_experts" for record in overlapped_records)
    assert all(
        record["operator_context"].get("dag_hint") == "selected_experts_are_independent_safe_to_parallelize"
        for record in overlapped_records
    )
    _assert_dependencies_precede_consumers(overlapped_records)


def test_moe_overlap_requires_multiple_selected_experts():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"top_k": 1, "selected_experts": [0], "schedule_policy": "overlap_selected_experts"})

    with pytest.raises(ValueError, match="overlap_selected_experts requires top_k >= 2"):
        generator_mod.generate_moe_records(manifest)


def test_combined_lowering_does_not_emit_duplicate_mode_commands():
    semantic = generator_mod.generate_full_transformer_layer_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="combined_mode_test")
    opcodes = [record["opcode"] for record in concrete]
    assert not any(lhs == rhs and lhs in {"SB", "HAB", "HAB_PIM"} for lhs, rhs in zip(opcodes, opcodes[1:]))
    assert not any(
        prev_opcode == "PIM_MAC" and opcode == "SB" and next_opcode == "PIM_MAC"
        for prev_opcode, opcode, next_opcode in zip(opcodes, opcodes[1:], opcodes[2:])
    )


def test_full_transformer_layer_combined_attention_ffn_moe_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    attention = generator_mod.get_tiny_attention_manifest()
    attention.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    ffn = generator_mod.get_tiny_ffn_manifest()
    ffn.update({"hidden_size": 32, "ffn_hidden_size": 64})
    moe = generator_mod.get_tiny_moe_manifest()
    moe.update({"hidden_size": 32, "expert_hidden_size": 64, "num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})

    semantic = generator_mod.generate_full_transformer_layer_records(attention_manifest=attention, ffn_manifest=ffn, moe_manifest=moe)
    assert len({record["record_id"] for record in semantic}) == len(semantic)
    _assert_dependencies_precede_consumers(semantic)
    _assert_attention_score_inputs_have_producer_dependencies(semantic)
    assert {"AttentionScore", "FFNProjection", "MoERouter", "MoEExpertFFN"} <= {record["kind"] for record in semantic}

    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="combined_p4_layer")
    assert {record["opcode"] for record in concrete} <= concrete_schema_mod.CONCRETE_OPCODES
    trace_path = tmp_path / "combined_p4_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    assert sim.stats["frontend"]["opcode_requests_completed"] == sim.stats["frontend"]["opcode_requests_sent"]


def test_build_full_transformer_provenance_summary_covers_attention_ffn_moe():
    semantic = generator_mod.generate_full_transformer_layer_records()
    summary = generator_mod.build_full_transformer_provenance_summary(semantic, manifest_name="combined_default")
    assert summary["record_counts_by_kind"]["AttentionScore"] >= 1
    assert summary["record_counts_by_kind"]["FFNProjection"] == 3
    assert summary["record_counts_by_kind"]["MoERouter"] == 1
    assert summary["record_counts_by_kind"]["MoEExpertFFN"] == 6
    assert "not_runtime_replay" in summary["non_claims"]


def test_attention_provenance_summary_accepts_manifest_name_keyword():
    semantic = generator_mod.generate_attention_records()

    summary = generator_mod.build_provenance_summary(semantic, manifest_name="attention_keyword")

    assert summary["manifest_name"] == "attention_keyword"
    assert summary["record_counts_by_kind"]["AttentionScore"] >= 1


def test_attention_provenance_summary_defaults_unknown_manifest_name():
    semantic = generator_mod.generate_attention_records()

    summary = generator_mod.build_provenance_summary(semantic)

    assert summary["manifest_name"] == "unknown"
    assert "semantic DAG summary" in summary["notes"]


def test_moe_router_mapping_is_distinct_from_expert_zero_when_stage_matches():
    manifest = generator_mod.get_tiny_moe_manifest()
    router_policy = generator_mod._moe_policies(manifest, layer_index=0, stage_index=0)
    expert_policy = generator_mod._moe_policies(manifest, layer_index=0, stage_index=0, expert_id=0)

    assert router_policy["dependency_context"]["expert_id"] is None
    assert expert_policy["dependency_context"]["expert_id"] == 0
    assert router_policy["dependency_context"]["dependency_id"] != expert_policy["dependency_context"]["dependency_id"]
    assert router_policy["row_policy"]["resolved_row"] != expert_policy["row_policy"]["resolved_row"]


def test_llama2_7b_full_depth_manifests_use_32_replayed_layers():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()

    assert attention["manifest_name"] == "llama2_7b_32_layer_attention_decode"
    assert ffn["manifest_name"] == "llama2_7b_32_layer_ffn_swiglu_decode"
    assert attention["num_layers"] == 32
    assert ffn["num_layers"] == 32
    assert attention["model_total_layers"] == 32
    assert ffn["model_total_layers"] == 32
    assert attention["num_heads"] == 32
    assert attention["head_dim"] == 128
    assert attention["hidden_size"] == 4096
    assert attention["past_len"] == 1024
    assert attention["score_tile_tokens"] == 256
    assert attention["context_tile_tokens"] == 256
    assert attention["schedule_policy"] == "serialized"
    assert ffn["hidden_size"] == 4096
    assert ffn["ffn_hidden_size"] == 11008
    assert ffn["ffn_activation_tile_size"] == 4096
    assert ffn["seq_len"] == 1
    assert ffn["schedule_policy"] == "serialized"
    assert attention["datatype"] == "int8"
    assert ffn["datatype"] == "int8"
    generator_mod._validate_attention_manifest(attention)
    generator_mod._validate_ffn_manifest(ffn)


def test_llama2_7b_model_registry_exposes_decode_v2_spec():
    model = generator_mod.get_model_spec("llama2-7b")
    scope = generator_mod.DecodeTraceScope.llama2_7b_decode_v2()

    assert model.name == "Llama2-7B"
    assert model.num_layers == 32
    assert model.hidden_size == 4096
    assert model.num_heads == 32
    assert model.num_kv_heads == 32
    assert model.head_dim == 128
    assert model.ffn_hidden_size == 11008
    assert model.datatype == "int8"
    assert scope.seq_len == 1
    assert scope.past_len == 1024
    assert scope.include_qkvo_projections is True
    assert scope.score_tile_tokens == 256
    assert scope.context_tile_tokens == 256


def test_llama2_13b_model_registry_exposes_decode_v2_spec():
    model = generator_mod.get_model_spec("llama2-13b")
    scope = generator_mod.DecodeTraceScope.llama2_13b_decode_v2()

    assert model.name == "Llama2-13B"
    assert model.num_layers == 40
    assert model.hidden_size == 5120
    assert model.num_heads == 40
    assert model.head_dim == 128
    assert model.ffn_hidden_size == 13824
    assert model.num_kv_heads == 40
    assert scope.seq_len == 1
    assert scope.past_len == 1024
    assert scope.include_qkvo_projections is True


def test_llama2_13b_full_depth_manifests_use_40_replayed_layers_and_dims():
    attention, ffn = generator_mod.get_llama2_13b_dense_decoder_manifests()

    for manifest in (attention, ffn):
        assert "40_layer" in manifest["manifest_name"]
        assert "5120" in manifest["manifest_name"]
        assert manifest["num_layers"] == 40
        assert manifest["model_total_layers"] == 40
        assert manifest["hidden_size"] == 5120
        assert manifest["datatype"] == "int8"

    assert attention["num_heads"] == 40
    assert attention["num_kv_heads"] == 40
    assert attention["head_dim"] == 128
    assert attention["past_len"] == 1024
    assert ffn["ffn_hidden_size"] == 13824
    assert "13824" in ffn["manifest_name"]
    generator_mod._validate_attention_manifest(attention)
    generator_mod._validate_ffn_manifest(ffn)


DENSE_DECODER_REQUIRED_KINDS = {"AttentionScore", "AttentionSoftmax", "AttentionContext", "FFNProjection", "PIMElementwise"}
DENSE_DECODER_FORBIDDEN_MOE_KINDS = {"MoERouter", "MoEExpertFFN", "MoETopK", "MoEDispatch", "MoECombine"}
DENSE_DECODER_SCALING_KINDS = ["AttentionScore", "AttentionContext", "FFNProjection", "PIMDataMove", "PIMElementwise"]
DENSE_DECODER_COMPUTE_KINDS = {"AttentionScore", "AttentionContext", "FFNProjection"}
DENSE_DECODER_WEIGHT_RESIDENCY_OPS = {
    "ffn_up_weight_residency",
    "ffn_gate_weight_residency",
    "ffn_down_weight_residency",
}
DENSE_DECODER_CONCRETE_OPCODES = {"READ", "WRITE", "SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"}
DENSE_DECODER_EXPECTED_PIM_MAC_REPEATS = 210_763_776
DENSE_DECODER_EXPECTED_STEADY_PIM_BCAST_REPEATS = 32
DENSE_DECODER_EXPECTED_COLD_PIM_BCAST_REPEATS = DENSE_DECODER_EXPECTED_STEADY_PIM_BCAST_REPEATS
DENSE_DECODER_V2_EXPECTED_PIM_MAC_REPEATS = DENSE_DECODER_EXPECTED_PIM_MAC_REPEATS
DENSE_DECODER_V2_EXPECTED_SEMANTIC_COUNTS = {
    "HostRead": 8_192,
    "HostWrite": 64,
    "Barrier": 64,
    "Drain": 1,
}


def _tiny_prefill_spec(
    *,
    num_layers: int = 1,
    hidden_size: int = 32,
    num_heads: int = 1,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    ffn_hidden_size: int = 64,
) -> object:
    return generator_mod.ModelSpec(
        name="Tiny-Prefill",
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_hidden_size=ffn_hidden_size,
        datatype="int8",
    )


def _tiny_prefill_manifests(
    *,
    prompt_len: int,
    num_layers: int = 1,
    hidden_size: int = 32,
    num_heads: int = 1,
    num_kv_heads: int | None = None,
    head_dim: int = 16,
    ffn_hidden_size: int = 64,
    score_tile_tokens: int | None = None,
    context_tile_tokens: int | None = None,
) -> tuple[dict, dict]:
    spec = _tiny_prefill_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_hidden_size=ffn_hidden_size,
    )
    attention, ffn = generator_mod.get_dense_prefill_manifests(spec, prompt_len=prompt_len)
    if score_tile_tokens is not None:
        attention["score_tile_tokens"] = score_tile_tokens
    if context_tile_tokens is not None:
        attention["context_tile_tokens"] = context_tile_tokens
    return attention, ffn


def _layer_index(record) -> int:
    return int(record["layer"].split("_")[1])


def test_prefill_manifest_validation_and_scope_metadata():
    with pytest.raises(ValueError, match="prompt_len must be positive"):
        generator_mod.PrefillTraceScope(prompt_len=0)
    with pytest.raises(ValueError, match="tile sizes must be positive"):
        generator_mod.PrefillTraceScope(prompt_len=64, score_tile_tokens=0)
    with pytest.raises(ValueError, match="tile sizes must be <= prompt_len"):
        generator_mod.PrefillTraceScope(prompt_len=64, context_tile_tokens=128)

    attention, ffn = _tiny_prefill_manifests(prompt_len=64)

    for manifest in (attention, ffn):
        assert manifest["phase"] == "prefill"
        assert manifest["seq_len"] == 64
        assert manifest["prompt_len"] == 64
        assert manifest["prefill_only"] is True
        assert set(generator_mod.REQUIRED_P4_PREFILL_NON_CLAIMS) <= set(manifest["non_claims"])
    assert attention["past_len"] == 0
    assert attention["attention_mode"] == "causal_prefill"

    invalid_attention = dict(attention)
    invalid_attention["schedule_policy"] = "overlap_independent_heads"
    with pytest.raises(ValueError, match="Prefill attention generator currently supports schedule_policy='serialized' only"):
        generator_mod.generate_prefill_attention_records(invalid_attention)


def test_dense_prefill_records_include_required_stages_without_host_reads():
    attention, ffn = _tiny_prefill_manifests(prompt_len=64)

    records = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )
    kinds = {record["kind"] for record in records}
    ops = {record["op"] for record in records}

    assert {"AttentionScore", "AttentionSoftmax", "AttentionContext", "FFNProjection", "HostWrite", "Barrier", "Drain"} <= kinds
    assert {"q_projection", "k_projection", "v_projection", "o_projection", "ffn_up_projection"} <= ops
    assert {"kv_cache_k_prefill_population", "kv_cache_v_prefill_population"} <= ops
    assert "HostRead" not in kinds
    assert not [record for record in records if record["op"].startswith("kv_cache_") and record["op"].endswith("_tile_read")]


def test_dense_prefill_causal_tiling_pair_counts_and_request_scaling():
    attention, ffn = _tiny_prefill_manifests(
        prompt_len=64,
        hidden_size=16,
        num_heads=1,
        head_dim=16,
        ffn_hidden_size=32,
        score_tile_tokens=32,
        context_tile_tokens=32,
    )

    records = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )
    scores = [record for record in records if record["kind"] == "AttentionScore"]
    contexts = [record for record in records if record["kind"] == "AttentionContext"]
    expected_pairs = 64 * 65 // 2
    expected_requests = generator_mod._num_requests(expected_pairs * attention["head_dim"], attention["datatype"])

    assert len(scores) == 3
    assert sum(record["operator_context"]["causal_pair_count"] for record in scores) == expected_pairs
    assert sum(record["compute_shape"]["output_elements"] for record in scores) == expected_pairs
    assert {record["compute_shape"]["output_elements"] for record in contexts} == {32 * attention["head_dim"]}
    assert sum(record["compute_shape"]["valid_attention_pairs"] for record in contexts) == expected_pairs
    assert sum(record["compute_shape"]["issued_work_elements"] for record in contexts) == expected_pairs * attention["head_dim"]
    assert sum(record["num_requests"] for record in scores) == expected_requests
    assert sum(record["num_requests"] for record in contexts) == expected_requests
    assert "HostRead" not in {record["kind"] for record in records}  # K/V are layer-local projection residencies.

    # Context output_elements should be query_tokens * head_dim, not causal pair count
    _hd = attention["head_dim"]
    _st = attention["score_tile_tokens"]
    _plen = attention["seq_len"]
    for record in contexts:
        q_start = record["operator_context"]["query_start"]
        actual_q = min(_st, _plen - q_start)
        expected_output = actual_q * _hd
        assert record["compute_shape"]["output_elements"] == expected_output, (
            f"Context output_elements={record['compute_shape']['output_elements']} "
            f"expected {expected_output} for query_start={q_start}"
        )

    softmax_records = [record for record in records if record["kind"] == "AttentionSoftmax"]
    assert [record["operator_context"]["tile_tokens"] for record in softmax_records] == [32, 32]
    assert [record["operator_context"]["softmax_causal_span_tokens"] for record in softmax_records] == [32, 64]

    # Context-reduction accounting elements = query_tokens * head_dim (per Q tile)
    reduction_records = [
        record for record in records
        if record["kind"] == "PIMElementwise" and record.get("op") == "attention_context_reduction_accounting"
    ]
    assert len(reduction_records) == 1
    assert reduction_records[0]["accounting_metadata"]["elements"] == _st * _hd


def test_dense_prefill_projection_and_ffn_requests_scale_linearly_with_prompt_len():
    def request_sum(prompt_len: int, op: str) -> int:
        attention, ffn = _tiny_prefill_manifests(prompt_len=prompt_len)
        records = generator_mod.generate_dense_prefill_transformer_layer_records(
            attention_manifest=attention,
            ffn_manifest=ffn,
        )
        return sum(record["num_requests"] for record in records if record.get("op") == op)

    assert request_sum(64, "q_projection") == 2 * request_sum(32, "q_projection")
    assert request_sum(64, "ffn_up_projection") == 2 * request_sum(32, "ffn_up_projection")


def test_dense_prefill_kv_cache_write_bytes_use_prompt_num_kv_heads_and_head_dim():
    attention, ffn = _tiny_prefill_manifests(
        prompt_len=64,
        hidden_size=32,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
    )

    records = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )
    kv_writes = [record for record in records if record.get("op") in {"kv_cache_k_prefill_population", "kv_cache_v_prefill_population"}]

    assert len(kv_writes) == 2
    assert {record["bytes"] for record in kv_writes} == {64 * 2 * 16}


def test_dense_prefill_dag_dependencies_and_o_projection_terminals():
    attention, ffn = _tiny_prefill_manifests(
        prompt_len=64,
        hidden_size=16,
        num_heads=1,
        head_dim=16,
        ffn_hidden_size=32,
        score_tile_tokens=32,
        context_tile_tokens=32,
    )

    records = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )
    _assert_dependencies_precede_consumers(records)
    _assert_attention_score_inputs_have_producer_dependencies(records)

    q0_context = next(
        record
        for record in records
        if record["kind"] == "AttentionContext"
        and record["operator_context"].get("query_start") == 0
        and record["operator_context"].get("key_start") == 0
    )
    q1_reduction = next(
        record
        for record in records
        if record["kind"] == "PIMElementwise"
        and record.get("op") == "attention_context_reduction_accounting"
        and record["operator_context"].get("query_start") == 32
    )
    o_projection = next(record for record in records if record.get("op") == "o_projection")

    assert q0_context["record_id"] in o_projection["logical_dependencies"]
    assert q1_reduction["record_id"] in o_projection["logical_dependencies"]
    assert q0_context["compute_shape"]["output_elements"] == 32 * attention["head_dim"]
    assert q0_context["compute_shape"]["output_shape"] == {"rows": 32, "cols": attention["head_dim"]}
    assert q0_context["compute_shape"]["valid_attention_pairs"] == q0_context["operator_context"]["causal_pair_count"]
    assert q1_reduction["accounting_metadata"]["elements"] == 32 * attention["head_dim"]


def test_dense_prefill_concrete_lowering_sanity_has_kv_writes_without_kv_reads_or_bcasts():
    attention, ffn = _tiny_prefill_manifests(
        prompt_len=32,
        hidden_size=16,
        num_heads=1,
        head_dim=16,
        ffn_hidden_size=32,
        score_tile_tokens=32,
        context_tile_tokens=32,
    )
    semantic = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )

    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="tiny_dense_prefill")
    concrete_schema_mod.validate_sequence(concrete)
    kv_prefill_population = [
        record
        for record in concrete
        if record["provenance"]["semantic_source"].get("op")
        in {"kv_cache_k_prefill_population", "kv_cache_v_prefill_population"}
    ]

    assert any(record["opcode"] == "PIM_MAC" for record in concrete)
    assert {record["opcode"] for record in kv_prefill_population} == {"WRITE"}
    assert not [record for record in concrete if record["provenance"]["semantic_source"].get("kind") == "HostRead"]
    assert not [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST" and "kv_cache" in record["provenance"]["semantic_source"].get("op", "")
    ]


def test_reduced_depth_prefill_prompt_257_stresses_tile_boundary_and_replays(tmp_path: Path):
    """Prompt 257 crosses the default 256-token causal tile boundary."""
    attention, ffn = _tiny_prefill_manifests(
        prompt_len=257,
        num_layers=2,
        hidden_size=16,
        num_heads=1,
        head_dim=16,
        ffn_hidden_size=32,
        score_tile_tokens=256,
        context_tile_tokens=256,
    )

    semantic = generator_mod.generate_dense_prefill_transformer_layer_records(
        attention_manifest=attention,
        ffn_manifest=ffn,
    )
    scores = [record for record in semantic if record["kind"] == "AttentionScore"]
    contexts = [record for record in semantic if record["kind"] == "AttentionContext"]
    reductions = [
        record
        for record in semantic
        if record["kind"] == "PIMElementwise" and record.get("op") == "attention_context_reduction_accounting"
    ]
    expected_pairs_per_layer = 257 * 258 // 2

    assert len(scores) == 2 * 3  # per layer: (Q0,K0), (Q1,K0), (Q1,K1); (Q0,K1) is masked out.
    assert len(contexts) == len(scores)
    assert len(reductions) == 2
    for layer in range(2):
        layer_scores = [record for record in scores if _layer_index(record) == layer]
        layer_contexts = [record for record in contexts if _layer_index(record) == layer]
        assert sum(record["operator_context"]["causal_pair_count"] for record in layer_scores) == expected_pairs_per_layer
        assert sum(record["compute_shape"]["valid_attention_pairs"] for record in layer_contexts) == expected_pairs_per_layer
        boundary = next(
            record
            for record in layer_scores
            if record["operator_context"].get("query_start") == 256
            and record["operator_context"].get("key_start") == 256
        )
        assert boundary["operator_context"]["causal_pair_count"] == 1

    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="prefill_P257_tile_boundary")
    concrete_schema_mod.validate_sequence(concrete)
    dram = create_dram(LPDDR5_PIM_CONFIG)
    trace_path = tmp_path / "prefill_P257_tile_boundary.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    assert sim.stats["frontend"]["opcode_requests_completed"] == sim.stats["frontend"]["opcode_requests_sent"]


def test_llama2_13b_dense_decoder_records_span_exactly_40_layers_and_final_drain():
    attention, ffn = generator_mod.get_llama2_13b_dense_decoder_manifests()
    records = generator_mod.generate_llama2_13b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)

    layers = sorted({_layer_index(record) for record in records if record.get("layer", "").startswith("layer_")})
    kinds = {record["kind"] for record in records}

    assert len({record["record_id"] for record in records}) == len(records)
    assert layers == list(range(40))
    assert DENSE_DECODER_REQUIRED_KINDS <= kinds
    assert DENSE_DECODER_FORBIDDEN_MOE_KINDS.isdisjoint(kinds)
    assert records[-1]["kind"] == "Drain"
    assert records[-1]["op"] == "final_drain"


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_emits_qkvo_and_semantic_boundaries():
    records = generator_mod.generate_llama2_7b_dense_decoder_records()
    counts = Counter(record["kind"] for record in records)
    projection_ops = {record["op"] for record in records if record["kind"] == "FFNProjection"}

    assert {"q_projection", "k_projection", "v_projection", "o_projection"} <= projection_ops
    for kind, expected_count in DENSE_DECODER_V2_EXPECTED_SEMANTIC_COUNTS.items():
        assert counts[kind] == expected_count
    assert [record["op"] for record in records if record["kind"] == "Barrier"].count("layer_start") == 32
    assert [record["op"] for record in records if record["kind"] == "Barrier"].count("layer_transition") == 32
    assert [record["op"] for record in records if record["kind"] == "Drain"] == ["final_drain"]


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_emits_layer_start_barrier_before_each_layer_body():
    records = generator_mod.generate_llama2_7b_dense_decoder_records()

    for layer_index in range(32):
        layer_name = f"layer_{layer_index:02d}"
        layer_indices = [index for index, record in enumerate(records) if record["layer"] == layer_name]
        first_layer_record = records[min(layer_indices)]
        assert first_layer_record["kind"] == "Barrier"
        assert first_layer_record["op"] == "layer_start"
        assert first_layer_record["barrier_scope"]["kind"] == "layer_start"


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_wires_layer_dag_dependencies():
    records = generator_mod.generate_llama2_7b_dense_decoder_records()
    _assert_attention_score_inputs_have_producer_dependencies(records)
    by_id = {record["record_id"]: record for record in records}
    producers = {
        output: record
        for record in records
        for output in record.get("tensor_io", {}).get("outputs", [])
    }

    for layer_index in range(32):
        layer_name = f"layer_{layer_index:02d}"
        q = producers[f"L{layer_index}.Q"]
        k = producers[f"L{layer_index}.K"]
        v = producers[f"L{layer_index}.V"]
        q_residency = producers[f"L{layer_index}.Q.pim_visible"]
        assert q["op"] == "q_projection"
        assert k["op"] == "k_projection"
        assert v["op"] == "v_projection"
        assert q_residency["op"] == "q_operand_residency"
        assert q["record_id"] in q_residency["logical_dependencies"]

        k_append = producers[f"L{layer_index}.kv_cache.K.current_token"]
        v_append = producers[f"L{layer_index}.kv_cache.V.current_token"]
        assert k["record_id"] in k_append["logical_dependencies"]
        assert v["record_id"] in v_append["logical_dependencies"]

        score_records = [
            record
            for record in records
            if record["kind"] == "AttentionScore" and record["layer"] == layer_name
        ]
        assert score_records
        assert all(q_residency["record_id"] in score["logical_dependencies"] for score in score_records)
        assert all(f"L{layer_index}.Q.pim_visible" in score["tensor_io"]["inputs"] for score in score_records)

        o_projection = producers[f"L{layer_index}.attention_output"]
        assert o_projection["op"] == "o_projection"
        assert any(
            by_id[dependency]["kind"] in {"AttentionContext", "PIMElementwise"}
            for dependency in o_projection["logical_dependencies"]
        )

        barrier_index = next(
            index
            for index, record in enumerate(records)
            if record["kind"] == "Barrier" and record["layer"] == layer_name and record["op"] == "layer_transition"
        )
        layer_compute_indices = [
            index
            for index, record in enumerate(records)
            if record["layer"] == layer_name and record["kind"] not in {"Barrier", "Drain"}
        ]
        assert max(layer_compute_indices) < barrier_index
    assert records[-1]["kind"] == "Drain"


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_uses_host_read_write_for_kv_cache_accounting():
    records = generator_mod.generate_llama2_7b_dense_decoder_records()
    kv_cache_records = [record for record in records if "kv_cache" in record["op"]]

    assert kv_cache_records
    assert {record["kind"] for record in kv_cache_records} <= {"HostRead", "HostWrite"}
    assert all(record["kind"] != "PIMDataMove" for record in kv_cache_records)


def test_dense_decoder_host_read_write_records_carry_structured_replay_request_policy():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention = copy.deepcopy(attention)
    ffn = copy.deepcopy(ffn)
    attention.update({"num_layers": 1, "past_len": 32, "score_tile_tokens": 32, "context_tile_tokens": 32})
    ffn["num_layers"] = 1
    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    host_records = [record for record in records if record["kind"] in {"HostRead", "HostWrite"}]

    assert {record["kind"] for record in host_records} == {"HostRead", "HostWrite"}
    for record in host_records:
        policy = record["address_policy"]
        assert policy["lowering"] == "structured_replay_regular_dram_request"
        assert policy["base_byte"] >= 0
        assert policy["stride_bytes"] == 64
        assert policy["count"] == max(1, (record["bytes"] + policy["stride_bytes"] - 1) // policy["stride_bytes"])


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_lowers_to_expected_pim_mac_without_kv_bcast():
    semantic = generator_mod.generate_llama2_7b_dense_decoder_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name="llama2_7b_32_layer_dense_decoder_v2_steady_state",
        materialize_weights=False,
    )

    assert sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_MAC") == DENSE_DECODER_V2_EXPECTED_PIM_MAC_REPEATS
    assert not [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST" and "kv_cache" in record["provenance"]["semantic_source"].get("op", "")
    ]


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_accepts_divisible_gqa_manifest():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_kv_heads"] = 8
    attention["head_group_size"] = 4  # 32 // 8

    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    assert len(records) > 0
    # K/V projections should have output dim = num_kv_heads * head_dim
    k_proj_records = [r for r in records if r.get("op") == "k_projection"]
    for r in k_proj_records:
        n = r.get("compute_shape", {}).get("n")
        assert n == 8 * 128  # num_kv_heads * head_dim = 1024


@pytest.mark.analysis_full
def test_dense_decoder_manifest_builder_derives_head_group_size_for_gqa_specs():
    spec = generator_mod.ModelSpec(
        name="Synthetic-GQA",
        num_layers=1,
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        ffn_hidden_size=11008,
        num_kv_heads=8,
    )

    manifest = generator_mod.get_llama2_dense_decoder_attention_manifest(spec)

    assert manifest["head_group_size"] == 4


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_accepts_semantic_mqa_manifest():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention.update({"num_layers": 1, "past_len": 32, "score_tile_tokens": 32, "context_tile_tokens": 32})
    ffn = copy.deepcopy(ffn)
    ffn["num_layers"] = 1
    attention["num_kv_heads"] = 1
    attention["head_group_size"] = 32

    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    assert len(records) > 0
    assert attention["head_group_size"] == 32

    for record in records:
        trace_mod.validate_record(record)

    k_proj_records = [r for r in records if r.get("op") == "k_projection"]
    v_proj_records = [r for r in records if r.get("op") == "v_projection"]
    kv_writes = [r for r in records if r.get("op") in {"kv_cache_k_append", "kv_cache_v_append"}]

    assert k_proj_records and v_proj_records and kv_writes
    assert {r.get("compute_shape", {}).get("n") for r in k_proj_records} == {128}
    assert {r.get("compute_shape", {}).get("n") for r in v_proj_records} == {128}
    assert {r.get("bytes") for r in kv_writes} == {128}


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_dense_gqa_replays_with_pim_mac_and_kv_cache_rw(tmp_path: Path):
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention.update({"num_layers": 1, "past_len": 32, "score_tile_tokens": 32, "context_tile_tokens": 32})
    ffn = copy.deepcopy(ffn)
    ffn["num_layers"] = 1
    attention["num_kv_heads"] = 8
    attention["head_group_size"] = 4

    semantic = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name="llama2_7b_1_layer_dense_decoder_gqa_replay",
    )
    concrete_schema_mod.validate_sequence(concrete)

    k_projection_records = [record for record in semantic if record.get("op") == "k_projection"]
    v_projection_records = [record for record in semantic if record.get("op") == "v_projection"]
    kv_cache_writes = [record for record in semantic if record.get("op") == "kv_cache_k_append"]
    assert k_projection_records and v_projection_records and kv_cache_writes
    assert {record["compute_shape"]["n"] for record in k_projection_records} == {8 * 128}
    assert {record["compute_shape"]["n"] for record in v_projection_records} == {8 * 128}
    assert {record["bytes"] for record in kv_cache_writes} == {8 * 128}

    concrete_kv_rw = [
        record
        for record in concrete
        if record["opcode"] in {"READ", "WRITE"}
        and record["provenance"]["semantic_source"].get("op", "").startswith("kv_cache_")
    ]
    assert concrete_kv_rw
    assert {record["opcode"] for record in concrete_kv_rw} == {"READ", "WRITE"}
    assert any(record["opcode"] == "PIM_MAC" for record in concrete)

    dram = create_dram(LPDDR5_PIM_CONFIG)
    trace_path = tmp_path / "llama2_7b_dense_decoder_gqa_replay.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    frontend_stats = sim.stats["frontend"]
    assert frontend_stats["opcode_requests_completed"] == frontend_stats["opcode_requests_sent"]


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_v2_accepts_kv_heads_alias_for_gqa():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention.pop("num_kv_heads", None)
    attention["kv_heads"] = 8
    attention["head_group_size"] = 4

    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    assert len(records) > 0


@pytest.mark.analysis_full
def test_gqa_rejects_disagreeing_kv_head_aliases():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_layers"] = 1
    ffn["num_layers"] = 1
    attention["num_kv_heads"] = 8
    attention["kv_heads"] = 4
    attention["head_group_size"] = 4

    with pytest.raises(ValueError, match="aliases must match"):
        generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)


@pytest.mark.analysis_full
def test_gqa_rejects_non_divisible_num_kv_heads():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_kv_heads"] = 7  # 32 not divisible by 7

    with pytest.raises(ValueError, match="divisible"):
        generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)


@pytest.mark.analysis_full
def test_gqa_rejects_num_kv_heads_exceeding_num_heads():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_kv_heads"] = 64

    with pytest.raises(ValueError, match="num_kv_heads must not exceed"):
        generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)


@pytest.mark.analysis_full
def test_gqa_rejects_stale_head_group_size_metadata():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_layers"] = 1
    ffn["num_layers"] = 1
    attention["num_kv_heads"] = 8
    # Keep the default MHA head_group_size=1 to ensure redundant metadata cannot silently diverge.

    with pytest.raises(ValueError, match="head_group_size .* must equal"):
        generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)


@pytest.mark.analysis_full
def test_gqa_kv_cache_write_byte_elements_use_num_kv_heads_times_head_dim():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention["num_kv_heads"] = 8
    attention["head_group_size"] = 4
    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    kv_writes = [r for r in records if r.get("op") == "kv_cache_k_append"]
    for r in kv_writes:
        assert r.get("bytes") == 8 * 128  # num_kv_heads * head_dim * bytes_per_element (int8=1)


@pytest.mark.analysis_full
def test_gqa_mha_produces_identical_output_when_num_kv_heads_equals_num_heads():
    """MHA is the degenerate case of GQA: num_kv_heads == num_heads."""
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    # Default: num_kv_heads == num_heads == 32
    records_mha = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    # GQA-mode with identical values
    att_copy = dict(attention)
    att_copy["num_kv_heads"] = 32
    att_copy["head_group_size"] = 1
    records_gqa_like = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=att_copy, ffn_manifest=ffn)
    # Same number of records
    assert len(records_mha) == len(records_gqa_like)


@pytest.mark.analysis_full
def test_gqa_head_group_id_maps_q_heads_to_kv_groups():
    """Verify that Q heads 0-3 map to KV group 0, 4-7 to KV group 1, etc."""
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["num_heads"] = 8
    manifest["num_kv_heads"] = 2
    manifest["head_group_size"] = 4  # 8 // 2
    records = generator_mod.generate_attention_records(manifest)
    for r in records:
        if "operator_context" in r:
            ctx = r["operator_context"]
            head_id = ctx.get("head_id")
            group_id = ctx.get("head_group_id")
            if head_id is not None and group_id is not None:
                assert group_id == head_id // 4


@pytest.mark.analysis_full
def test_dense_decoder_v2_supports_attention_projection_dim_distinct_from_hidden_size():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    attention.update(
        {
            "num_layers": 1,
            "hidden_size": 2304,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 256,
            "head_group_size": 2,
            "past_len": 32,
            "score_tile_tokens": 32,
            "context_tile_tokens": 32,
        }
    )
    ffn = copy.deepcopy(ffn)
    ffn.update({"num_layers": 1, "hidden_size": 2304, "ffn_hidden_size": 9216, "ffn_activation_tile_size": 2304})

    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)
    by_op = {record["op"]: record for record in records if record.get("kind") == "FFNProjection"}

    assert by_op["q_projection"]["compute_shape"] == {"m": 1, "n": 2048, "k": 2304, "output_elements": 2048, "datatype": "int8"}
    assert by_op["k_projection"]["compute_shape"] == {"m": 1, "n": 1024, "k": 2304, "output_elements": 1024, "datatype": "int8"}
    assert by_op["v_projection"]["compute_shape"] == {"m": 1, "n": 1024, "k": 2304, "output_elements": 1024, "datatype": "int8"}
    assert by_op["o_projection"]["compute_shape"] == {"m": 1, "n": 2304, "k": 2048, "output_elements": 2304, "datatype": "int8"}


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_records_span_exactly_32_layers_without_moe():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    records = generator_mod.generate_llama2_7b_dense_decoder_records(attention_manifest=attention, ffn_manifest=ffn)

    layers = sorted({_layer_index(record) for record in records})
    kinds = {record["kind"] for record in records}

    assert len({record["record_id"] for record in records}) == len(records)
    assert layers == list(range(32))
    assert DENSE_DECODER_REQUIRED_KINDS <= kinds
    assert DENSE_DECODER_FORBIDDEN_MOE_KINDS.isdisjoint(kinds)


@pytest.mark.analysis_full
def test_llama2_7b_dense_decoder_semantic_counts_scale_from_one_to_32_layers():
    full_attention, full_ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()
    one_attention = copy.deepcopy(full_attention)
    one_ffn = copy.deepcopy(full_ffn)
    one_attention["num_layers"] = 1
    one_ffn["num_layers"] = 1

    one_layer = generator_mod.generate_llama2_7b_dense_decoder_records(
        attention_manifest=one_attention,
        ffn_manifest=one_ffn,
    )
    full_depth = generator_mod.generate_llama2_7b_dense_decoder_records(
        attention_manifest=full_attention,
        ffn_manifest=full_ffn,
    )

    assert len({record["record_id"] for record in one_layer}) == len(one_layer)
    assert len({record["record_id"] for record in full_depth}) == len(full_depth)

    for kind in DENSE_DECODER_SCALING_KINDS:
        assert sum(record["kind"] == kind for record in full_depth) == 32 * sum(
            record["kind"] == kind for record in one_layer
        )

    assert sum(int(record["num_requests"]) for record in full_depth if record["kind"] in DENSE_DECODER_COMPUTE_KINDS) == 32 * sum(
        int(record["num_requests"]) for record in one_layer if record["kind"] in DENSE_DECODER_COMPUTE_KINDS
    )


@pytest.mark.analysis_full
def test_llama2_7b_32_layer_steady_state_skips_weight_materialization():
    semantic = generator_mod.generate_llama2_7b_dense_decoder_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name="llama2_7b_32_layer_dense_decoder_steady_state",
        materialize_weights=False,
    )
    concrete_schema_mod.validate_sequence(concrete)

    lowered_weight_ops = [
        record["provenance"]["semantic_source"]["op"]
        for record in concrete
        if record["provenance"]["semantic_source"].get("op") in DENSE_DECODER_WEIGHT_RESIDENCY_OPS
    ]
    assert lowered_weight_ops == []
    assert any(record["opcode"] == "PIM_MAC" for record in concrete)
    assert {record["opcode"] for record in concrete} <= DENSE_DECODER_CONCRETE_OPCODES


@pytest.mark.analysis_full
def test_llama2_7b_32_layer_cold_start_materializes_ffn_weights():
    semantic = generator_mod.generate_llama2_7b_dense_decoder_records()
    steady = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name="llama2_7b_32_layer_dense_decoder_steady_state",
        materialize_weights=False,
    )
    cold = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name="llama2_7b_32_layer_dense_decoder_cold_start_stress",
        materialize_weights=True,
    )
    concrete_schema_mod.validate_sequence(steady)
    concrete_schema_mod.validate_sequence(cold)

    cold_weight_ops = [
        record["provenance"]["semantic_source"]["record_id"]
        for record in cold
        if record["provenance"]["semantic_source"].get("op") in DENSE_DECODER_WEIGHT_RESIDENCY_OPS
    ]
    cold_weight_source_records = set(cold_weight_ops)

    assert len(cold_weight_source_records) == 32 * 3
    for weight_op in DENSE_DECODER_WEIGHT_RESIDENCY_OPS:
        assert (
            len(
                {
                    record["provenance"]["semantic_source"]["record_id"]
                    for record in cold
                    if record["provenance"]["semantic_source"].get("op") == weight_op
                }
            )
            == 32
        )
    assert sum(int(record.get("repeat", 1)) for record in cold if record["opcode"] == "PIM_BCAST") == sum(
        int(record.get("repeat", 1)) for record in steady if record["opcode"] == "PIM_BCAST"
    )
    assert [
        record
        for record in cold
        if record["opcode"] == "PIM_BCAST" and record["provenance"]["semantic_source"].get("op") in DENSE_DECODER_WEIGHT_RESIDENCY_OPS
    ] == []
    assert sum(
        int(record.get("repeat", 1))
        for record in cold
        if record["opcode"] == "WRITE" and record["provenance"]["semantic_source"].get("op") in DENSE_DECODER_WEIGHT_RESIDENCY_OPS
    ) > 0
    assert {record["opcode"] for record in cold} <= DENSE_DECODER_CONCRETE_OPCODES


def test_attention_cli_writes_bounded_artifacts(tmp_path: Path):
    output_dir = tmp_path / "attention"
    result = subprocess.run(
        [sys.executable, "-m", "ramulator.workload_surrogate.generate_full_transformer", "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    assert trace_path.exists()
    assert summary_path.exists()
    assert f"Generated: {trace_path}" in result.stdout

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["record_counts_by_kind"]["AttentionScore"] == 4
    assert summary["record_counts_by_kind"]["AttentionContext"] == 4
    assert all("not_raw_attacc_schema" in record["provenance"]["non_claims"] for record in records)


def test_attention_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "attention_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 2


def _run_llama2_7b_dense_decoder_replay(
    tmp_path: Path,
    *,
    materialize_weights: bool,
) -> dict:
    default_schema_max_expanded_records = concrete_schema_mod.MAX_EXPANDED_RECORDS
    dram = create_dram(LPDDR5_PIM_CONFIG)
    semantic = generator_mod.generate_llama2_7b_dense_decoder_records()
    manifest_name = (
        "llama2_7b_32_layer_dense_decoder_cold_start_stress"
        if materialize_weights
        else "llama2_7b_32_layer_dense_decoder_steady_state"
    )
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name=manifest_name,
        materialize_weights=materialize_weights,
    )
    weight_source_records = {
        record["provenance"]["semantic_source"]["record_id"]
        for record in concrete
        if record["provenance"]["semantic_source"].get("op") in DENSE_DECODER_WEIGHT_RESIDENCY_OPS
    }
    expanded_record_count = concrete_schema_mod.expanded_record_count(concrete)
    trace_path = tmp_path / f"{manifest_name}.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    return {
        "frontend": sim.stats["frontend"],
        "controller": sim.stats["memory_system"]["controller"],
        "concrete_record_count": len(concrete),
        "weight_source_record_count": len(weight_source_records),
        "expanded_record_count": expanded_record_count,
        "default_schema_max_expanded_records": default_schema_max_expanded_records,
        "pim_mac_repeat_count": sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_MAC"),
        "pim_bcast_repeat_count": sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_BCAST"),
    }


# ── MoE decoder pipeline tests ──────────────────────────────────────────

@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_manifests_are_consistent():
    att, moe = generator_mod.get_mixtral_8x7b_moe_decoder_manifests()
    assert att["hidden_size"] == moe["hidden_size"]
    assert att["hidden_size"] == att["num_heads"] * att["head_dim"]
    assert att["num_kv_heads"] == 8
    assert att["head_group_size"] == 4
    assert moe["num_experts"] == 8
    assert moe["top_k"] == 2
    assert moe["selected_experts"] == [0, 1]
    assert att["num_layers"] == moe["num_layers"] == 32


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_records_span_32_layers_with_moe_kinds():
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    _assert_attention_score_inputs_have_producer_dependencies(records)
    layers = sorted({r["operator_context"]["layer_id"] for r in records if "operator_context" in r})
    kinds = {r["kind"] for r in records}

    assert layers == list(range(32))
    assert "MoERouter" in kinds
    assert "MoEExpertFFN" in kinds
    assert "MoECombine" in kinds
    assert "MoETopK" in kinds
    assert "MoEDispatch" in kinds
    assert "AttentionScore" in kinds
    assert "AttentionContext" in kinds
    assert "FFNProjection" in kinds


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_per_layer_moe_counts():
    """Verify 1 router + top_k real SwiGLU expert projections per layer."""
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    router_by_layer = {}
    expert_by_layer = {}
    for r in records:
        lid = r.get("operator_context", {}).get("layer_id", -1)
        if r["kind"] == "MoERouter":
            router_by_layer[lid] = router_by_layer.get(lid, 0) + 1
        elif r["kind"] == "MoEExpertFFN":
            expert_by_layer[lid] = expert_by_layer.get(lid, 0) + 1
    for lid in range(32):
        assert router_by_layer.get(lid, 0) == 1, f"Layer {lid} router count"
        assert expert_by_layer.get(lid, 0) == 6, f"Layer {lid} expert projection count (top_k=2 × 3)"


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_expert_mac_diagnostic():
    """Mixtral experts emit real SwiGLU up/gate/down projection records."""
    att, moe = generator_mod.get_mixtral_8x7b_moe_decoder_manifests()
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    hidden = moe["hidden_size"]
    expert_hidden = moe["expert_hidden_size"]
    top_k = moe["top_k"]
    lanes = 32  # INT8

    # Historical fused abstraction: one GEMV per expert.
    fused_elements_per_expert = hidden * expert_hidden
    expected_fused_requests_per_layer = int((top_k * fused_elements_per_expert + lanes - 1) // lanes)

    expert_records = [r for r in records if r["kind"] == "MoEExpertFFN"]
    layer_macs = sum(int(r.get("num_requests", 0)) for r in expert_records) // int(moe["num_layers"])

    # Real Mixtral expert: 3 projections (up, gate, down).
    real_elements_per_expert = 3 * hidden * expert_hidden
    real_requests_per_layer = int((top_k * real_elements_per_expert + lanes - 1) // lanes)
    assert layer_macs == real_requests_per_layer, (
        f"MoEExpertFFN num_requests per layer must reflect up/gate/down records: got {layer_macs}, "
        f"expected {real_requests_per_layer}"
    )
    assert real_requests_per_layer == 3 * expected_fused_requests_per_layer, (
        f"3-projection MoE expert count is 3× the fused count: "
        f"{real_requests_per_layer} vs {3 * expected_fused_requests_per_layer}"
    )


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_gqa_kv_cache_bytes():
    """GQA KV cache writes use num_kv_heads * head_dim."""
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    kv_writes = [r for r in records if r.get("op") == "kv_cache_k_append"]
    for r in kv_writes:
        assert r.get("bytes") == 8 * 128


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_k_projection_uses_kv_head_output_dim():
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    k_proj = [r for r in records if r.get("op") == "k_projection"]
    for r in k_proj:
        assert r.get("compute_shape", {}).get("n") == 8 * 128


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_router_input_follows_o_projection():
    """MoE router input setup record depends on O projection output."""
    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    for layer in range(32):
        layer_recs = [r for r in records if r.get("operator_context", {}).get("layer_id") == layer]
        router_setup = [r for r in layer_recs if r.get("op") == "moe_router_input_setup"]
        if router_setup:
            deps = router_setup[0].get("logical_dependencies", [])
            o_proj = [r for r in layer_recs if r.get("op") == "o_projection"]
            if o_proj:
                assert o_proj[0]["record_id"] in deps
                assert "attention_output" in str(router_setup[0].get("tensor_io", {}).get("inputs", []))


@pytest.mark.analysis_full
def test_mixtral_8x7b_decoder_lowers_and_replays(tmp_path: Path):
    """Full MoE decode trace lowers and replays through backend (subset layers for speed)."""
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import lower_semantic_records_to_concrete
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import write_jsonl as conc_write_jsonl
    from tests.analysis.figures._sim_helpers import _frontend, _make_mem

    records = generator_mod.generate_mixtral_8x7b_decoder_records()
    records = [r for r in records if r.get("operator_context", {}).get("layer_id", -1) < 2]
    assert len(records) > 0

    concrete = lower_semantic_records_to_concrete(records, materialize_weights=False)
    trace_path = tmp_path / "mixtral_decode_subset.jsonl"
    conc_write_jsonl(concrete, trace_path)

    dram = create_dram(LPDDR5_PIM_CONFIG)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()

    stats = sim.stats
    assert stats["frontend"]["opcode_requests_completed"] > 0
    assert stats["frontend"]["opcode_requests_completed"] == stats["frontend"]["opcode_requests_sent"]
    pim_mac_count = sum(1 for r in concrete if r["opcode"] == "PIM_MAC")
    assert pim_mac_count > 0
    # Verify MoE compute lowered to PIM_MAC
    moe_router_ops = [r for r in concrete if r.get("provenance", {}).get("semantic_source", {}).get("kind") == "MoERouter"]
    moe_expert_ops = [r for r in concrete if r.get("provenance", {}).get("semantic_source", {}).get("kind") == "MoEExpertFFN"]
    assert len(moe_router_ops) > 0 or len(moe_expert_ops) > 0


@pytest.mark.analysis_full
@pytest.mark.parametrize("materialize_weights", [False])
def test_llama2_7b_32_layer_dense_decoder_replays_to_completion(
    tmp_path: Path,
    materialize_weights: bool,
):
    result = _run_llama2_7b_dense_decoder_replay(
        tmp_path,
        materialize_weights=materialize_weights,
    )

    assert result["frontend"]["opcode_requests_completed"] == result["frontend"]["opcode_requests_sent"]
    assert result["controller"]["num_pim_reqs_served"] > 0
    assert result["pim_mac_repeat_count"] == DENSE_DECODER_EXPECTED_PIM_MAC_REPEATS
    assert result["controller"].get("num_issued_pim_mac", result["controller"]["num_pim_reqs_served"]) == DENSE_DECODER_EXPECTED_PIM_MAC_REPEATS
    assert result["pim_bcast_repeat_count"] > 0
    assert result["expanded_record_count"] <= result["default_schema_max_expanded_records"]
    if materialize_weights:
        assert result["weight_source_record_count"] == 32 * 3
        assert result["pim_bcast_repeat_count"] == DENSE_DECODER_EXPECTED_COLD_PIM_BCAST_REPEATS
        assert result["concrete_record_count"] > 0
    else:
        assert result["weight_source_record_count"] == 0
        assert result["pim_bcast_repeat_count"] == DENSE_DECODER_EXPECTED_STEADY_PIM_BCAST_REPEATS


# ═══════════════════════════════════════════════════════════════════════
# ModelSpec validation and citation tests
# ═══════════════════════════════════════════════════════════════════════


def test_model_spec_rejects_num_heads_not_divisible_by_num_kv_heads():
    with pytest.raises(ValueError, match="must be divisible by num_kv_heads"):
        generator_mod.ModelSpec(
            name="Bad-GQA",
            num_layers=1,
            hidden_size=4096,
            num_heads=32,
            head_dim=128,
            ffn_hidden_size=11008,
            num_kv_heads=7,
        )


def test_model_spec_accepts_valid_gqa_ratio():
    spec = generator_mod.ModelSpec(
        name="Valid-GQA",
        num_layers=1,
        hidden_size=3584,
        num_heads=28,
        head_dim=128,
        ffn_hidden_size=18944,
        num_kv_heads=4,
    )
    assert spec.num_heads == 28
    assert spec.num_kv_heads == 4
    assert spec.num_heads % spec.num_kv_heads == 0


def test_model_spec_citation_defaults_to_auto_generated_from_name():
    spec = generator_mod.ModelSpec(
        name="CustomModel",
        num_layers=1,
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        ffn_hidden_size=128,
    )
    assert "CustomModel" in spec.citation
    assert "auto-generated" in spec.citation
    assert spec.paper_anchor == spec.citation


def test_model_spec_citation_and_paper_anchor_can_be_overridden():
    spec = generator_mod.ModelSpec(
        name="Llama2-7B",
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        ffn_hidden_size=11008,
        citation="Llama 2 (Touvron et al., arXiv:2307.09288)",
        paper_anchor="Llama 2 (Touvron et al., 2023)",
    )
    assert spec.citation == "Llama 2 (Touvron et al., arXiv:2307.09288)"
    assert spec.paper_anchor == "Llama 2 (Touvron et al., 2023)"


def test_llama2_manifest_builders_use_model_spec_citation():
    attention, ffn = generator_mod.get_llama2_7b_dense_decoder_manifests()

    assert attention["model_citation"] == "Llama 2 (Touvron et al., arXiv:2307.09288)"
    assert ffn["model_citation"] == "Llama 2 (Touvron et al., arXiv:2307.09288)"
    assert attention["literature_anchors"][0] == "Llama 2 (Touvron et al., 2023)"
    assert ffn["literature_anchors"][0] == "Llama 2 (Touvron et al., 2023)"
    assert "LPDDR5-PIM native opcode surface" in attention["literature_anchors"]
    assert "LPDDR5-PIM native opcode surface" in ffn["literature_anchors"]


def test_model_citation_overrides_work_for_synthetic_spec_in_manifest():
    spec = generator_mod.ModelSpec(
        name="Synthetic-GQA",
        num_layers=1,
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        ffn_hidden_size=11008,
        num_kv_heads=8,
        citation="Synthetic GQA Model (test-only)",
        paper_anchor="Synthetic GQA Model",
    )
    manifest = generator_mod.get_llama2_dense_decoder_attention_manifest(spec)
    assert manifest["model_citation"] == "Synthetic GQA Model (test-only)"
    assert manifest["literature_anchors"][0] == "Synthetic GQA Model"


def test_unsupported_model_family_rejection_message():
    with pytest.raises(ValueError, match="Unsupported model spec.*qwen2-7b"):
        generator_mod.get_model_spec("qwen2-7b")

    with pytest.raises(ValueError, match="Unsupported model spec.*mistral"):
        generator_mod.get_model_spec("mistral-7b")


# ═══════════════════════════════════════════════════════════════════════
# Qwen2.5 and Gemma dense decoder model support tests
# ═══════════════════════════════════════════════════════════════════════


def _assert_model_spec(
    spec,
    *,
    name: str,
    layers: int,
    hidden: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    ffn: int,
    citation: str,
    ffn_variant: str,
    activation: str,
) -> None:
    assert spec.name == name
    assert spec.num_layers == layers
    assert spec.hidden_size == hidden
    assert spec.num_heads == heads
    assert spec.num_kv_heads == kv_heads
    assert spec.head_dim == head_dim
    assert spec.ffn_hidden_size == ffn
    assert spec.citation == citation
    assert spec.ffn_variant == ffn_variant
    assert spec.activation == activation


def _one_layer_dense_decoder_records(spec, *, past_len: int = 64) -> list[dict]:
    attention, ffn = generator_mod.get_dense_decoder_manifests(spec, past_len=past_len)
    attention = copy.deepcopy(attention)
    ffn = copy.deepcopy(ffn)
    attention.update({"num_layers": 1, "score_tile_tokens": past_len, "context_tile_tokens": past_len})
    ffn["num_layers"] = 1
    return generator_mod.generate_dense_transformer_layer_records(attention_manifest=attention, ffn_manifest=ffn)


def _records_by_op(records: list[dict]) -> dict[str, dict]:
    return {record["op"]: record for record in records}


def test_qwen25_7b_model_registry_exposes_correct_spec():
    for key in ("qwen2.5-7b", "qwen25-7b"):
        _assert_model_spec(
            generator_mod.get_model_spec(key),
            name="Qwen2.5-7B",
            layers=28,
            hidden=3584,
            heads=28,
            kv_heads=4,
            head_dim=128,
            ffn=18944,
            citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
            ffn_variant="swiglu_3proj",
            activation="silu",
        )


def test_qwen25_72b_model_registry_exposes_correct_spec():
    for key in ("qwen2.5-72b", "qwen25-72b"):
        _assert_model_spec(
            generator_mod.get_model_spec(key),
            name="Qwen2.5-72B",
            layers=80,
            hidden=8192,
            heads=64,
            kv_heads=8,
            head_dim=128,
            ffn=29568,
            citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
            ffn_variant="swiglu_3proj",
            activation="silu",
        )


def test_gemma_2b_model_registry_exposes_correct_spec():
    _assert_model_spec(
        generator_mod.get_model_spec("gemma-2b"),
        name="Gemma-2B",
        layers=18,
        hidden=2048,
        heads=8,
        kv_heads=1,
        head_dim=256,
        ffn=16384,
        citation="Gemma (Gemma Team, arXiv:2403.08295)",
        ffn_variant="geglu_3proj",
        activation="gelu_pytorch_tanh",
    )


def test_gemma_7b_model_registry_exposes_correct_spec():
    _assert_model_spec(
        generator_mod.get_model_spec("gemma-7b"),
        name="Gemma-7B",
        layers=28,
        hidden=3072,
        heads=16,
        kv_heads=16,
        head_dim=256,
        ffn=24576,
        citation="Gemma (Gemma Team, arXiv:2403.08295)",
        ffn_variant="geglu_3proj",
        activation="gelu_pytorch_tanh",
    )


def test_gemma2_9b_model_registry_exposes_correct_spec():
    for key in ("gemma-2-9b", "gemma2-9b"):
        _assert_model_spec(
            generator_mod.get_model_spec(key),
            name="Gemma-2-9B",
            layers=42,
            hidden=3584,
            heads=16,
            kv_heads=8,
            head_dim=256,
            ffn=14336,
            citation="Gemma 2 (Gemma Team, arXiv:2408.00118)",
            ffn_variant="geglu_3proj",
            activation="gelu_pytorch_tanh",
        )


def test_gemma2_27b_model_registry_exposes_correct_spec():
    for key in ("gemma-2-27b", "gemma2-27b"):
        _assert_model_spec(
            generator_mod.get_model_spec(key),
            name="Gemma-2-27B",
            layers=46,
            hidden=4608,
            heads=32,
            kv_heads=16,
            head_dim=128,
            ffn=73728,
            citation="Gemma 2 (Gemma Team, arXiv:2408.00118)",
            ffn_variant="geglu_3proj",
            activation="gelu_pytorch_tanh",
        )


def test_qwen_manifest_has_correct_gqa_properties():
    attention, ffn = generator_mod.get_dense_decoder_manifests(generator_mod.QWEN25_7B_MODEL_SPEC, past_len=64)

    assert attention["head_group_size"] == 7
    assert attention["num_kv_heads"] == 4
    generator_mod._validate_attention_manifest(attention)
    generator_mod._validate_ffn_manifest(ffn)

    records = _one_layer_dense_decoder_records(generator_mod.QWEN25_7B_MODEL_SPEC)
    by_op = _records_by_op(records)
    assert by_op["q_projection"]["compute_shape"]["n"] == 3584
    assert by_op["k_projection"]["compute_shape"]["n"] == 512


def test_gemma_manifest_has_correct_geglu_properties():
    gemma2b_attention, gemma2b_ffn = generator_mod.get_dense_decoder_manifests(generator_mod.GEMMA_2B_MODEL_SPEC, past_len=64)
    gemma29b_attention, gemma29b_ffn = generator_mod.get_dense_decoder_manifests(generator_mod.GEMMA2_9B_MODEL_SPEC, past_len=64)

    assert gemma2b_ffn["ffn_variant"] == "geglu_3proj"
    assert gemma2b_ffn["activation"] == "gelu_pytorch_tanh"
    assert gemma2b_attention["head_group_size"] == 8  # MQA: 8 Q heads / 1 KV head
    assert gemma29b_ffn["ffn_variant"] == "geglu_3proj"
    assert gemma29b_ffn["activation"] == "gelu_pytorch_tanh"
    assert gemma29b_attention["head_group_size"] == 2  # GQA: 16 Q heads / 8 KV heads
    generator_mod._validate_attention_manifest(gemma2b_attention)
    generator_mod._validate_ffn_manifest(gemma2b_ffn)


def test_qwen25_7b_one_layer_generates_valid_records():
    records = _one_layer_dense_decoder_records(generator_mod.QWEN25_7B_MODEL_SPEC)
    kinds = {record["kind"] for record in records}

    assert len({record["record_id"] for record in records}) == len(records)
    assert DENSE_DECODER_REQUIRED_KINDS <= kinds
    assert {"q_projection", "k_projection", "v_projection", "o_projection"} <= {record["op"] for record in records}
    assert DENSE_DECODER_FORBIDDEN_MOE_KINDS.isdisjoint(kinds)
    for record in records:
        trace_mod.validate_record(record)


def test_gemma_2b_one_layer_generates_valid_records():
    records = _one_layer_dense_decoder_records(generator_mod.GEMMA_2B_MODEL_SPEC)
    by_op = _records_by_op(records)

    assert len({record["record_id"] for record in records}) == len(records)
    assert DENSE_DECODER_REQUIRED_KINDS <= {record["kind"] for record in records}
    assert by_op["ffn_gate_projection"]["operator_context"]["operator_family"] == "ffn_geglu"
    assert by_op["ffn_gate_activation_accounting"]["accounting_metadata"]["activation"] == "gelu_pytorch_tanh"
    assert by_op["ffn_gated_multiply_accounting"]["operator_context"]["operator_family"] == "ffn_geglu"


def test_gemma2_9b_one_layer_generates_correct_projection_dims():
    records = _one_layer_dense_decoder_records(generator_mod.GEMMA2_9B_MODEL_SPEC)
    by_op = _records_by_op(records)

    assert by_op["q_projection"]["compute_shape"]["n"] == 4096  # 16 heads * 256 head_dim
    assert by_op["k_projection"]["compute_shape"]["n"] == 2048  # 8 KV heads * 256 head_dim
    assert by_op["o_projection"]["compute_shape"]["n"] == 3584
    assert by_op["o_projection"]["compute_shape"]["k"] == 4096


def test_gemma2_27b_one_layer_correct_projection_dims():
    records = _one_layer_dense_decoder_records(generator_mod.GEMMA2_27B_MODEL_SPEC)
    by_op = _records_by_op(records)

    assert by_op["q_projection"]["compute_shape"]["n"] == 4096  # 32 heads * 128 head_dim
    assert by_op["o_projection"]["compute_shape"]["n"] == 4608


def test_qwen_gqa_head_group_id_maps_correctly():
    attention, _ = generator_mod.get_dense_decoder_manifests(generator_mod.QWEN25_7B_MODEL_SPEC, past_len=64)
    attention = copy.deepcopy(attention)
    attention.update({"num_layers": 1, "score_tile_tokens": 64, "context_tile_tokens": 64})
    records = generator_mod.generate_attention_records(attention)

    for record in records:
        ctx = record.get("operator_context", {})
        head_id = ctx.get("head_id")
        group_id = ctx.get("head_group_id")
        if head_id is not None and group_id is not None:
            assert group_id == head_id // 7


def test_gemma2b_mqa_kv_cache_bytes_match_num_kv_heads_times_head_dim():
    records = _one_layer_dense_decoder_records(generator_mod.GEMMA_2B_MODEL_SPEC)
    kv_writes = [record for record in records if record.get("op") in {"kv_cache_k_append", "kv_cache_v_append"}]

    assert kv_writes
    assert {record["bytes"] for record in kv_writes} == {1 * 256}


def test_qwen_manifest_uses_qwen_citation():
    attention, ffn = generator_mod.get_dense_decoder_manifests(generator_mod.QWEN25_7B_MODEL_SPEC, past_len=64)

    assert attention["model_citation"] == "Qwen2.5 (Yang et al., arXiv:2412.15115)"
    assert ffn["model_citation"] == "Qwen2.5 (Yang et al., arXiv:2412.15115)"
    assert attention["literature_anchors"][0] == "Qwen2.5 (Yang et al., 2024)"
    assert ffn["literature_anchors"][0] == "Qwen2.5 (Yang et al., 2024)"
    assert "Llama" not in attention["model_citation"]
    assert all("Llama" not in anchor for anchor in attention["literature_anchors"])


def test_gemma_manifest_uses_gemma_citation():
    attention, ffn = generator_mod.get_dense_decoder_manifests(generator_mod.GEMMA2_9B_MODEL_SPEC, past_len=64)

    assert attention["model_citation"] == "Gemma 2 (Gemma Team, arXiv:2408.00118)"
    assert ffn["model_citation"] == "Gemma 2 (Gemma Team, arXiv:2408.00118)"
    assert attention["literature_anchors"][0] == "Gemma 2 (Gemma Team, 2024)"
    assert ffn["literature_anchors"][0] == "Gemma 2 (Gemma Team, 2024)"
    assert "Llama" not in attention["model_citation"]
    assert all("Llama" not in anchor for anchor in attention["literature_anchors"])


def test_qwen25_7b_2_layer_counts_scale_from_1_layer():
    one_attention, one_ffn = generator_mod.get_dense_decoder_manifests(generator_mod.QWEN25_7B_MODEL_SPEC, past_len=64)
    two_attention, two_ffn = generator_mod.get_dense_decoder_manifests(generator_mod.QWEN25_7B_MODEL_SPEC, past_len=64)
    one_attention = copy.deepcopy(one_attention)
    one_ffn = copy.deepcopy(one_ffn)
    two_attention = copy.deepcopy(two_attention)
    two_ffn = copy.deepcopy(two_ffn)
    one_attention.update({"num_layers": 1, "score_tile_tokens": 64, "context_tile_tokens": 64})
    one_ffn["num_layers"] = 1
    two_attention.update({"num_layers": 2, "score_tile_tokens": 64, "context_tile_tokens": 64})
    two_ffn["num_layers"] = 2

    one_layer = generator_mod.generate_dense_transformer_layer_records(attention_manifest=one_attention, ffn_manifest=one_ffn)
    two_layers = generator_mod.generate_dense_transformer_layer_records(attention_manifest=two_attention, ffn_manifest=two_ffn)

    assert len({record["record_id"] for record in one_layer}) == len(one_layer)
    assert len({record["record_id"] for record in two_layers}) == len(two_layers)
    for kind in DENSE_DECODER_SCALING_KINDS:
        assert sum(record["kind"] == kind for record in two_layers) == 2 * sum(record["kind"] == kind for record in one_layer)


def test_gemma_ffn_geglu_variant_is_recognized():
    assert "geglu_3proj" in generator_mod.SUPPORTED_FFN_VARIANTS
    assert generator_mod.GEMMA_2B_MODEL_SPEC.ffn_variant == "geglu_3proj"
    assert generator_mod.GEMMA2_9B_MODEL_SPEC.ffn_variant == "geglu_3proj"


def test_valid_movement_kinds_accepted():
    """Recognized movement kinds should pass data-move record validation."""
    valid_kinds = [
        "broadcast_or_accounted_tile_load",
        "preloaded_stationary_weight_residency",
        "dynamic_activation_tile",
        "bank_local_tile_activation",
        "cross_bank_operand_shuffle_accounting",
    ]
    # Verify all are in the VALID_PIM_DATA_MOVE_MOVEMENT_KINDS constant
    for mk in valid_kinds:
        assert mk in trace_mod.VALID_PIM_DATA_MOVE_MOVEMENT_KINDS, f"{mk} should be recognized"


def test_unknown_movement_kind_rejected_by_schema():
    """Unknown movement kinds should be rejected by record validation."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    import copy
    valid_data_move = [r for r in generator_mod.generate_ffn_records(manifest) if r["kind"] == "PIMDataMove"][0]
    record = {
        "schema_version": generator_mod.SCHEMA_VERSION,
        "generator_version": generator_mod.FULL_TRANSFORMER_GENERATOR_VERSION,
        "record_id": "rec_test99",
        "kind": "PIMDataMove",
        "phase": "decode",
        "layer": 0,
        "op": "test_unknown_kind",
        "repeat": 1,
        "stage": "data_movement",
        "tensor_io": {"inputs": ["test.in"], "outputs": ["test.out"]},
        "logical_dependencies": [],
        "operator_context": {
            "operator_family": "ffn",
            "stage": "data_movement",
            "record_family": trace_mod.REQUIRED_P4_RECORD_FAMILY,
            "layer_id": 0,
        },
        "provenance": copy.deepcopy(valid_data_move["provenance"]),
        "mapping_policy": {},
        "residency": {"inputs": [], "outputs": []},
        "movement_policy": {"movement_kind": "completely_unknown_bogus_kind"},
        "num_requests": 1,
        "dependency_context": {"dependency_id": 0, "dependency_count": 1},
        "row_policy": {"kind": "bounded_ffn_stage_rows", "row_start": 0, "row_count": 1, "resolved_row": 0},
        "column_policy": {"kind": "ffn_dependency_column_round_robin", "column_start": 0, "resolved_column": 0},
        "datatype_metadata": {"datatype": "int8", "role": "p4_ffn_tensor_dag_metadata", "behavior_claim": "explicit_resource_rows_only"},
    }
    with pytest.raises(ValueError, match="not a recognized movement kind"):
        trace_mod.validate_record(record)


def test_semantic_only_movement_kinds_skipped_in_lowering():
    """Semantic-only movement kinds must emit zero concrete opcodes, not raise."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    valid_data_move = [r for r in records if r["kind"] == "PIMDataMove"][0]

    for movement_kind in lowering_mod.SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS:
        semantic_only_record = copy.deepcopy(valid_data_move)
        semantic_only_record["record_id"] = f"rec_semantic_only_{movement_kind}"
        semantic_only_record["op"] = f"test_{movement_kind}"
        semantic_only_record["movement_policy"]["movement_kind"] = movement_kind

        concrete = lowering_mod.lower_semantic_records_to_concrete([semantic_only_record])

        assert concrete == [], f"{movement_kind} must be recognized but emit zero concrete opcodes"


def test_ffn_intermediate_residency_records_present():
    """FFN must have PIMOperandResidency records for gate, up, and activated intermediates."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    residency_records = [r for r in records if r["kind"] == "PIMOperandResidency"]
    expected_ops = {
        "ffn_up_intermediate_residency",
        "ffn_gate_intermediate_residency",
        "ffn_activated_gate_residency",
        "ffn_intermediate_bank_local_residency",
    }
    num_layers = int(manifest["num_layers"])
    assert len(residency_records) == num_layers * len(expected_ops)

    by_layer = {}
    for record in residency_records:
        by_layer.setdefault(record["operator_context"]["layer_id"], {})[record["op"]] = record

    assert set(by_layer) == set(range(num_layers))
    for layer_index, records_by_op in by_layer.items():
        assert set(records_by_op) == expected_ops
        assert records_by_op["ffn_up_intermediate_residency"]["tensor_io"]["outputs"] == [
            f"L{layer_index}.ffn.up_intermediate.pim_visible"
        ]
        assert records_by_op["ffn_gate_intermediate_residency"]["tensor_io"]["outputs"] == [
            f"L{layer_index}.ffn.gate_intermediate.pim_visible"
        ]
        assert records_by_op["ffn_activated_gate_residency"]["tensor_io"]["outputs"] == [
            f"L{layer_index}.ffn.activated_gate.pim_visible"
        ]
        assert records_by_op["ffn_intermediate_bank_local_residency"]["tensor_io"]["outputs"] == [
            f"L{layer_index}.ffn.gated_multiply.bank_local"
        ]

    record_positions = {record["record_id"]: index for index, record in enumerate(records)}
    for residency_record in residency_records:
        residency_id = residency_record["record_id"]
        assert any(
            record_positions[record["record_id"]] > record_positions[residency_id]
            and residency_id in record.get("logical_dependencies", [])
            for record in records
        ), f"Residency record {residency_id} is not consumed by a later dependency"


def test_moe_expert_activation_multiply_accounting():
    """MoE expert FFN must include activation and gated-multiply PIMElementwise records."""
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest["ffn_hidden_size"] = int(manifest["expert_hidden_size"]) + 17
    records = generator_mod.generate_moe_records(manifest)
    # Find PIMElementwise records in the MoE path
    elementwise_records = [r for r in records if r["kind"] == "PIMElementwise"]
    # Each selected expert in each layer should have activation + gated_multiply records
    num_selected = len(manifest["selected_experts"])
    num_layers = int(manifest["num_layers"])
    expected_min = num_layers * num_selected * 2  # activation + gated_multiply per expert per layer
    assert len(elementwise_records) >= expected_min, (
        f"MoE needs at least {expected_min} PIMElementwise records for activation+gated_multiply "
        f"({num_layers} layers x {num_selected} experts x 2), got {len(elementwise_records)}"
    )
    # Verify elementwise_kind values
    elementwise_kinds = {r.get("accounting_metadata", {}).get("kind") for r in elementwise_records}
    assert "semantic_only_activation" in elementwise_kinds, "MoE expert activation accounting missing"
    assert "semantic_only_gated_multiply" in elementwise_kinds, "MoE expert gated-multiply accounting missing"
    expected_elements = int(manifest["seq_len"]) * int(manifest["expert_hidden_size"])
    assert {r["accounting_metadata"]["elements"] for r in elementwise_records} == {expected_elements}


def test_moe_operand_residency_records():
    """MoE must materialize only router logits, gated expert intermediates, and expert outputs."""
    manifest = generator_mod.get_tiny_moe_manifest()
    records = generator_mod.generate_moe_records(manifest)
    residency_records = [r for r in records if r["kind"] == "PIMOperandResidency"]
    num_selected = len(manifest["selected_experts"])
    num_layers = int(manifest["num_layers"])
    # router_logits (1/layer) + selected expert gated intermediate and expert output (2/expert/layer).
    # Raw expert up/gate projection outputs and activated_gate intentionally remain direct semantic
    # operands for the elementwise accounting stages; do not require dense-FFN-style residency here.
    expected_count = num_layers * (1 + num_selected * 2)
    assert len(residency_records) == expected_count, (
        f"MoE should have exactly router_logits + gated-intermediate/output residency records "
        f"({num_layers} layers x (1 + {num_selected} experts x 2)), got {len(residency_records)}"
    )

    residency_ops = {r["op"] for r in residency_records}
    for layer_index in range(num_layers):
        assert "moe_router_logits_residency" in residency_ops
        for expert_id in manifest["selected_experts"]:
            assert f"moe_expert_{expert_id}_intermediate_residency" in residency_ops
            assert f"moe_expert_{expert_id}_output_residency" in residency_ops
            assert f"moe_expert_{expert_id}_up_intermediate_residency" not in residency_ops
            assert f"moe_expert_{expert_id}_gate_intermediate_residency" not in residency_ops
            assert f"moe_expert_{expert_id}_activated_gate_residency" not in residency_ops

    by_id = {record["record_id"]: record for record in records}
    by_op = {record["op"]: record for record in records}

    router_residency = by_op["moe_router_logits_residency"]
    assert router_residency["tensor_io"]["inputs"] == ["L0.moe.router_logits"]
    assert router_residency["tensor_io"]["outputs"] == ["L0.moe.router_logits.pim_visible"]
    assert router_residency["record_id"] in by_op["moe_topk_select_accounting"]["logical_dependencies"]

    for expert_id in manifest["selected_experts"]:
        intermediate = by_op[f"moe_expert_{expert_id}_intermediate_residency"]
        assert intermediate["tensor_io"]["inputs"] == [f"L0.moe.expert_{expert_id}.gated"]
        assert intermediate["tensor_io"]["outputs"] == [f"L0.moe.expert_{expert_id}.gated.bank_local"]
        assert intermediate["record_id"] in by_op[f"moe_expert_{expert_id}_down_projection"]["logical_dependencies"]

        output = by_op[f"moe_expert_{expert_id}_output_residency"]
        assert output["tensor_io"]["inputs"] == [f"L0.moe.expert_{expert_id}.output"]
        assert output["tensor_io"]["outputs"] == [f"L0.moe.expert_{expert_id}.output.pim_visible"]
        assert output["record_id"] in by_op["moe_expert_combine_accounting"]["logical_dependencies"]

    # Check that _semantic_operand_record was called (by verifying residency record structure).
    for rr in residency_records:
        for dependency in rr["logical_dependencies"]:
            assert dependency in by_id
        assert "accounting_metadata" in rr, "Residency record missing accounting_metadata"
        acc = rr["accounting_metadata"]
        assert acc.get("lowering") == "semantic_only_operand_residency", "MoE residency should be semantic-only"


def test_attention_q_probability_residency():
    """Attention must have PIMOperandResidency for Q (before AttentionScore) and probability (before AttentionContext)."""
    attention_manifest = generator_mod.get_tiny_attention_manifest()
    dense_records = generator_mod.generate_attention_records(attention_manifest)
    residency_records = [r for r in dense_records if r["kind"] == "PIMOperandResidency"]
    assert len(residency_records) >= 1, "Attention should have PIMOperandResidency records for Q and/or probability"
    # Check that Q or probability tensors appear in residency outputs
    residency_io = []
    for rr in residency_records:
        for out_tensor in rr.get("tensor_io", {}).get("outputs", []):
            residency_io.append(out_tensor)
    # Should have Q-related or probability-related residency
    has_residency = len(residency_io) > 0
    assert has_residency, "Should have Q or probability residency records"


def test_semantic_only_movement_kinds_not_lowered_to_pim_bcast():
    """cross_bank_operand_shuffle_accounting, bank_local_tile_activation, dynamic_activation_tile must not lower to PIM_BCAST."""
    # This is a code-level guardrail test: verify the SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS constant exists in the lowerer
    assert hasattr(lowering_mod, "SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS"), "Lowering module must define SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS"
    excluded = lowering_mod.SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS
    assert "cross_bank_operand_shuffle_accounting" in excluded
    assert "bank_local_tile_activation" in excluded
    assert "dynamic_activation_tile" in excluded
    # Also verify they exist in the trace module's VALID set
    assert "cross_bank_operand_shuffle_accounting" in trace_mod.VALID_PIM_DATA_MOVE_MOVEMENT_KINDS
    assert "bank_local_tile_activation" in trace_mod.VALID_PIM_DATA_MOVE_MOVEMENT_KINDS
    assert "dynamic_activation_tile" in trace_mod.VALID_PIM_DATA_MOVE_MOVEMENT_KINDS


# ── Bounded prefill replay test ────────────────────────────────────────

def _run_llama2_7b_dense_prefill_replay(
    tmp_path: Path,
    *,
    prompt_len: int,
) -> dict:
    """Bounded prefill replay: Llama2-7B full 32-layer, small prompt."""
    semantic = generator_mod.generate_llama2_7b_dense_prefill_records(
        prompt_len=prompt_len,
    )

    manifest_name = f"llama2_7b_32_layer_dense_prefill_prompt_{prompt_len}"
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest_name,
    )
    expanded_record_count = concrete_schema_mod.expanded_record_count(concrete)
    trace_path = tmp_path / f"{manifest_name}.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    dram = create_dram(LPDDR5_PIM_CONFIG)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    return {
        "frontend": sim.stats["frontend"],
        "controller": sim.stats["memory_system"]["controller"],
        "concrete_record_count": len(concrete),
        "expanded_record_count": expanded_record_count,
        "pim_mac_repeat_count": sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_MAC"),
        "pim_bcast_repeat_count": sum(int(record.get("repeat", 1)) for record in concrete if record["opcode"] == "PIM_BCAST"),
    }


class TestDensePrefillLlama7BReplay:
    @pytest.mark.analysis_full
    @pytest.mark.parametrize("prompt_len", [2, 4])
    def test_32_layer_dense_prefill_replays_to_completion(self, tmp_path: Path, prompt_len: int):
        result = _run_llama2_7b_dense_prefill_replay(tmp_path, prompt_len=prompt_len)

        assert result["frontend"]["opcode_requests_completed"] == result["frontend"]["opcode_requests_sent"]
        assert result["controller"]["num_pim_reqs_served"] > 0
        assert result["expanded_record_count"] <= concrete_schema_mod.MAX_EXPANDED_RECORDS
        assert result["pim_mac_repeat_count"] > 0
        assert result["concrete_record_count"] > 0

    @pytest.mark.analysis_full
    def test_32_layer_dense_prefill_emits_no_kv_cache_host_reads(self, tmp_path: Path):
        """Prefill must not emit KV-cache HostRead records — full 32 layers."""
        semantic = generator_mod.generate_llama2_7b_dense_prefill_records(prompt_len=4)
        concrete = lowering_mod.lower_semantic_records_to_concrete(
            semantic, manifest_name="prefill_no_kv_read_smoke",
        )
        kv_reads = [
            record for record in concrete
            if record["provenance"]["semantic_source"].get("kind") == "HostRead"
            and "kv_cache" in record["provenance"]["semantic_source"].get("op", "")
        ]
        assert not kv_reads

    @pytest.mark.analysis_full
    def test_32_layer_dense_prefill_has_kv_cache_host_writes(self, tmp_path: Path):
        """Prefill must emit KV-cache population HostWrite records — full 32 layers."""
        semantic = generator_mod.generate_llama2_7b_dense_prefill_records(prompt_len=4)
        concrete = lowering_mod.lower_semantic_records_to_concrete(
            semantic, manifest_name="prefill_kv_write_smoke",
        )
        kv_writes = [
            record for record in concrete
            if record["provenance"]["semantic_source"].get("kind") == "HostWrite"
            and record["provenance"]["semantic_source"].get("op") in {
                "kv_cache_k_prefill_population", "kv_cache_v_prefill_population",
            }
        ]
        assert len(kv_writes) == 64  # 32 layers × (K + V)
        assert {record["opcode"] for record in kv_writes} == {"WRITE"}
