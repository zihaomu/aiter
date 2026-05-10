# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlyDSL fused FP4 tests require CUDA"
)


def _require_flydsl_gfx950():
    from aiter.ops.flydsl.utils import is_flydsl_available

    if get_gfx() != "gfx950":
        pytest.skip("FlyDSL fused FP4 MoE path is currently validated on gfx950")
    if not is_flydsl_available():
        pytest.skip("FlyDSL package is not available")


def _make_fused_fp4_case(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    block_m: int,
    zero_tail_from: int = 0,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    hidden = torch.randn((token, model_dim), dtype=dtype, device=device) / 10
    w1 = torch.randn((experts, inter_dim * 2, model_dim), dtype=dtype, device=device) / 10
    w2 = torch.randn((experts, model_dim, inter_dim), dtype=dtype, device=device) / 10
    if zero_tail_from > 0:
        hidden[:, zero_tail_from:] = 0
        w1[:, :, zero_tail_from:] = 0
    scores = torch.randn((token, experts), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(hidden, scores, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(experts, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(experts, model_dim, inter_dim // 2)

    a1_qt, a1_scale = torch_quant(hidden, quant_dtype=dtypes.fp4x2)
    ref_stage1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )

    ref_fp4, ref_scale = torch_quant(ref_stage1, quant_dtype=dtypes.fp4x2)
    ref_fp4 = ref_fp4.view(token, topk, inter_dim // 2)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, dtype, block_m
    )

    a1_scale_sorted = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )
    return {
        "a1_qt": a1_qt,
        "a1_scale_sorted": a1_scale_sorted,
        "w1_qt": shuffle_weight(w1_qt, (16, 16)),
        "w1_scale": e8m0_shuffle(w1_scale),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "ref_fp4": ref_fp4,
        "ref_scale": ref_scale.view(token, topk, -1),
    }


def _canonicalize_fp4_signed_zero(x: torch.Tensor) -> torch.Tensor:
    bytes_ = x.view(torch.uint8)
    lo = bytes_ & 0xF
    hi = bytes_ >> 4
    lo = torch.where((lo == 0) | (lo == 8), torch.zeros_like(lo), lo)
    hi = torch.where((hi == 0) | (hi == 8), torch.zeros_like(hi), hi)
    return lo | (hi << 4)


def _assert_fp4_bytes_equal(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(
        _canonicalize_fp4_signed_zero(actual).cpu(),
        _canonicalize_fp4_signed_zero(expected).cpu(),
        rtol=0,
        atol=0,
    )


def _assert_tiled_scale_bytes_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token: int,
    topk: int,
):
    actual_flat = actual.view(torch.uint8).flatten()
    expected_flat = expected.view(torch.uint8).view(token * topk, -1)
    scale_cols = expected_flat.shape[1]
    num_valid = int(num_valid_ids.view(-1)[0].item())

    rows = torch.arange(num_valid, device=actual.device, dtype=torch.long)
    ids = sorted_ids[:num_valid].to(torch.long)
    token_ids = ids & 0xFFFFFF
    slot_ids = ids >> 24
    valid = (token_ids < token) & (slot_ids < topk)
    rows = rows[valid]
    token_ids = token_ids[valid]
    slot_ids = slot_ids[valid]

    cols = torch.arange(scale_cols, device=actual.device, dtype=torch.long)
    row_grid = rows[:, None]
    col_grid = cols[None, :]
    byte_offsets = (
        (row_grid // 32) * (scale_cols * 32)
        + (col_grid // 8) * 256
        + (col_grid & 3) * 64
        + (row_grid & 15) * 4
        + ((col_grid >> 2) & 1) * 2
        + ((row_grid >> 4) & 1)
    )
    expected_bytes = expected_flat[token_ids * topk + slot_ids][:, cols]
    torch.testing.assert_close(
        actual_flat[byte_offsets].cpu(),
        expected_bytes.cpu(),
        rtol=0,
        atol=0,
    )


def test_flydsl_moe_stage1_fused_fp4_matches_torch_reference():
    _require_flydsl_gfx950()

    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    token = 8
    inter_dim = 256
    experts = 16
    topk = 4
    block_m = 32
    model_dim = 256

    data = _make_fused_fp4_case(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        block_m=block_m,
    )

    actual_fp4, actual_scale = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="fp4",
        w1_scale=data["w1_scale"],
        a1_scale=data["a1_scale_sorted"],
        sorted_weights=None,
    )
    torch.cuda.synchronize()

    _assert_fp4_bytes_equal(actual_fp4, data["ref_fp4"])
    _assert_tiled_scale_bytes_equal(
        actual_scale,
        data["ref_scale"],
        data["sorted_ids"],
        data["num_valid_ids"],
        token,
        topk,
    )


def test_flydsl_silu_and_mul_fq_fused_fp4_matches_torch_reference():
    _require_flydsl_gfx950()

    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    token = 8
    topk = 4
    inter_dim = 256
    scale_cols = inter_dim // 32
    sorted_rows = token * topk
    padded_rows = (sorted_rows + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    device = torch.device("cuda")

    torch.manual_seed(1)
    x = torch.randn((sorted_rows, inter_dim * 2), dtype=torch.bfloat16, device=device)
    x = x / 4
    row_ids = torch.arange(sorted_rows, device=device, dtype=torch.int32)
    sorted_ids = ((row_ids % topk) << 24) | (row_ids // topk)
    num_valid_ids = torch.tensor([sorted_rows], dtype=torch.int32, device=device)
    out = torch.empty((token, topk, inter_dim // 2), dtype=dtypes.fp4x2, device=device)
    out_scale = torch.empty(
        (padded_rows, padded_cols), dtype=dtypes.fp8_e8m0, device=device
    )

    kernel = build_silu_and_mul_fq_module(
        inter_dim,
        topk,
        quant_mode="fp4",
        gui_layout=False,
        act="silu",
    )
    kernel(
        x,
        out.view(torch.uint8),
        out_scale.view(torch.uint8),
        sorted_ids,
        num_valid_ids,
        torch.empty(0, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        token,
        sorted_rows,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    gate = x[:, :inter_dim].float()
    up = x[:, inter_dim:].float()
    sig = torch.reciprocal(1 + torch.exp2(gate * -1.4426950408889634))
    ref = (gate * sig * up).to(torch.bfloat16).view(token, topk, inter_dim)
    ref_fp4, ref_scale = aiter.get_torch_quant(QuantType.per_1x32)(
        ref, quant_dtype=dtypes.fp4x2
    )

    _assert_fp4_bytes_equal(out, ref_fp4.view(token, topk, inter_dim // 2))
    _assert_tiled_scale_bytes_equal(
        out_scale,
        ref_scale.view(token, topk, -1),
        sorted_ids,
        num_valid_ids,
        token,
        topk,
    )
