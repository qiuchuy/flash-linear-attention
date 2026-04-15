# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs, input_guard

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=2)
        for BV in [16, 32]
        for num_warps in [4, 8]
    ],
    key=["H", "K", "V", "BT", "TRANSPOSE_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_fused_kernel_h_o(
    q,
    w,
    u,
    kg,
    g,
    Aqk,
    o,
    h0,
    ht,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    scale: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = (i_n * T).to(tl.int64)
        T = T.to(tl.int32)

    NT = tl.cdiv(T, BT)

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)

        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)

        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    q += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    u += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    o += (bos * H + i_h) * V

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    for i_t in range(NT):
        b_v = tl.zeros([BT, BV], dtype=tl.float32)
        p_u = tl.make_block_ptr(u, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v += tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)

        p_w1 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w1, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v -= tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v -= tl.dot(b_w, b_h1.to(b_w.dtype))

        if K > 64:
            p_w2 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w2, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v -= tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v -= tl.dot(b_w, b_h2.to(b_w.dtype))

        if K > 128:
            p_w3 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w3, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v -= tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v -= tl.dot(b_w, b_h3.to(b_w.dtype))

        if K > 192:
            p_w4 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w4, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v -= tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v -= tl.dot(b_w, b_h4.to(b_w.dtype))

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_q = tl.load(p_q1, boundary_check=(0, 1))
        b_g = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o += tl.dot(b_qg, b_h1.to(b_qg.dtype))

        if K > 64:
            p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_q = tl.load(p_q2, boundary_check=(0, 1))
            b_g = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))

        if K > 128:
            p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_q = tl.load(p_q3, boundary_check=(0, 1))
            b_g = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))

        if K > 192:
            p_q4 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            p_g4 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_q = tl.load(p_q4, boundary_check=(0, 1))
            b_g = tl.load(p_g4, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))

        p_A = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
        b_o = b_o * scale + tl.dot(b_A, b_v.to(b_A.dtype))

        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        o_k = tl.arange(0, 64)
        b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
        if TRANSPOSE_STATE:
            b_h1 *= exp2(b_g_last)[None, :]
        else:
            b_h1 *= exp2(b_g_last)[:, None]
        p_kg1 = tl.make_block_ptr(kg, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_kg1, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))
        else:
            b_h1 += tl.dot(b_k, b_v.to(b_k.dtype))

        if K > 64:
            o_k = 64 + tl.arange(0, 64)
            b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h2 *= exp2(b_g_last)[None, :]
            else:
                b_h2 *= exp2(b_g_last)[:, None]
            p_kg2 = tl.make_block_ptr(kg, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_kg2, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))
            else:
                b_h2 += tl.dot(b_k, b_v.to(b_k.dtype))

        if K > 128:
            o_k = 128 + tl.arange(0, 64)
            b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h3 *= exp2(b_g_last)[None, :]
            else:
                b_h3 *= exp2(b_g_last)[:, None]
            p_kg3 = tl.make_block_ptr(kg, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_kg3, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))
            else:
                b_h3 += tl.dot(b_k, b_v.to(b_k.dtype))

        if K > 192:
            o_k = 192 + tl.arange(0, 64)
            b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h4 *= exp2(b_g_last)[None, :]
            else:
                b_h4 *= exp2(b_g_last)[:, None]
            p_kg4 = tl.make_block_ptr(kg, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_kg4, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))
            else:
                b_h4 += tl.dot(b_k, b_v.to(b_k.dtype))

    if STORE_FINAL_STATE:
        ht += i_nh * K * V
        if TRANSPOSE_STATE:
            p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))

        if K > 64:
            if TRANSPOSE_STATE:
                p_ht2 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht2 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))

        if K > 128:
            if TRANSPOSE_STATE:
                p_ht3 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht3 = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))

        if K > 192:
            if TRANSPOSE_STATE:
                p_ht4 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht4 = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _solve_tril_16_from_raw(
    b_Araw,
    T,
    i_ti,
    BC: tl.constexpr,
):
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai = -tl.where(m_A, b_Araw, 0.0)
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_Araw, 0.0), axis=0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)

    return b_Ai + m_I


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_fused(
    q,
    k,
    v,
    g,
    beta,
    w,
    u,
    kg,
    Aqk,
    Akk,
    Akkd,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    scale: tl.constexpr,
    LOAD_DIAG_FROM_AKKD: tl.constexpr,
    FUSE_RECOMPUTE: tl.constexpr,
    STORE_AKK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC
    if i_tc0 >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    w += (bos * H + i_h) * K
    u += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    beta += bos * H + i_h
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m0 = i_tc0 + o_i < T
    m1 = i_tc1 + o_i < T
    m2 = i_tc2 + o_i < T
    m3 = i_tc3 + o_i < T

    ################################################################################
    # Aqk. This phase stores Aqk directly and keeps it out of the WY inverse path.
    ################################################################################
    b_Aqk00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk33 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q0 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
        if not LOAD_DIAG_FROM_AKKD:
            b_gn0_diag = tl.load(g + (i_tc0 + min(BC // 2, T - i_tc0 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_Aqk00 += tl.dot(
                (b_q0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_g0 - b_gn0_diag[None, :]), 0.0)).to(tl.bfloat16),
                tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn0_diag[None, :] - b_g0), 0.0)).to(tl.bfloat16),
            )

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_qg1 = b_q1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_g1 - b_gn1[None, :]), 0.0)
            b_Aqk10 += tl.dot(
                b_qg1.to(tl.bfloat16),
                tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn1[None, :] - b_g0), 0.0)).to(tl.bfloat16),
            )
            if not LOAD_DIAG_FROM_AKKD:
                b_gn1_diag = tl.load(g + (i_tc1 + min(BC // 2, T - i_tc1 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_Aqk11 += tl.dot(
                    (b_q1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_g1 - b_gn1_diag[None, :]), 0.0)).to(tl.bfloat16),
                    tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn1_diag[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                )

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_qg2 = b_q2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_g2 - b_gn2[None, :]), 0.0)
                b_Aqk20 += tl.dot(
                    b_qg2.to(tl.bfloat16),
                    tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn2[None, :] - b_g0), 0.0)).to(tl.bfloat16),
                )
                b_Aqk21 += tl.dot(
                    b_qg2.to(tl.bfloat16),
                    tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn2[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                )
                if not LOAD_DIAG_FROM_AKKD:
                    b_gn2_diag = tl.load(g + (i_tc2 + min(BC // 2, T - i_tc2 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_Aqk22 += tl.dot(
                        (b_q2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_g2 - b_gn2_diag[None, :]), 0.0)).to(tl.bfloat16),
                        tl.trans(b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_gn2_diag[None, :] - b_g2), 0.0)).to(tl.bfloat16),
                    )

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_qg3 = b_q3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_g3 - b_gn3[None, :]), 0.0)
                    b_Aqk30 += tl.dot(
                        b_qg3.to(tl.bfloat16),
                        tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g0), 0.0)).to(tl.bfloat16),
                    )
                    b_Aqk31 += tl.dot(
                        b_qg3.to(tl.bfloat16),
                        tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                    )
                    b_Aqk32 += tl.dot(
                        b_qg3.to(tl.bfloat16),
                        tl.trans(b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g2), 0.0)).to(tl.bfloat16),
                    )
                    if not LOAD_DIAG_FROM_AKKD:
                        b_gn3_diag = tl.load(g + (i_tc3 + min(BC // 2, T - i_tc3 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                        b_Aqk33 += tl.dot(
                            (b_q3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_g3 - b_gn3_diag[None, :]), 0.0)).to(tl.bfloat16),
                            tl.trans(b_k3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_gn3_diag[None, :] - b_g3), 0.0)).to(tl.bfloat16),
                        )

    p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    if not LOAD_DIAG_FROM_AKKD:
        p_Aqk00 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_Aqk11 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        p_Aqk22 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        p_Aqk33 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk00, (b_Aqk00 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk11, (b_Aqk11 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk22, (b_Aqk22 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk33, (b_Aqk33 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    ################################################################################
    # Akk raw blocks.
    ################################################################################
    b_Akk00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk33 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
        if not LOAD_DIAG_FROM_AKKD:
            b_gn0_diag = tl.load(g + (i_tc0 + min(BC // 2, T - i_tc0 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_Akk00 += tl.dot(
                (b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_g0 - b_gn0_diag[None, :]), 0.0)).to(tl.bfloat16),
                tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn0_diag[None, :] - b_g0), 0.0)).to(tl.bfloat16),
            )

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_kg1 = b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_g1 - b_gn1[None, :]), 0.0)
            b_Akk10 += tl.dot(
                b_kg1.to(tl.bfloat16),
                tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn1[None, :] - b_g0), 0.0)).to(tl.bfloat16),
            )
            if not LOAD_DIAG_FROM_AKKD:
                b_gn1_diag = tl.load(g + (i_tc1 + min(BC // 2, T - i_tc1 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_Akk11 += tl.dot(
                    (b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_g1 - b_gn1_diag[None, :]), 0.0)).to(tl.bfloat16),
                    tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn1_diag[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                )

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_kg2 = b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_g2 - b_gn2[None, :]), 0.0)
                b_Akk20 += tl.dot(
                    b_kg2.to(tl.bfloat16),
                    tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn2[None, :] - b_g0), 0.0)).to(tl.bfloat16),
                )
                b_Akk21 += tl.dot(
                    b_kg2.to(tl.bfloat16),
                    tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn2[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                )
                if not LOAD_DIAG_FROM_AKKD:
                    b_gn2_diag = tl.load(g + (i_tc2 + min(BC // 2, T - i_tc2 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_Akk22 += tl.dot(
                        (b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_g2 - b_gn2_diag[None, :]), 0.0)).to(tl.bfloat16),
                        tl.trans(b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_gn2_diag[None, :] - b_g2), 0.0)).to(tl.bfloat16),
                    )

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_kg3 = b_k3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_g3 - b_gn3[None, :]), 0.0)
                    b_Akk30 += tl.dot(
                        b_kg3.to(tl.bfloat16),
                        tl.trans(b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g0), 0.0)).to(tl.bfloat16),
                    )
                    b_Akk31 += tl.dot(
                        b_kg3.to(tl.bfloat16),
                        tl.trans(b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g1), 0.0)).to(tl.bfloat16),
                    )
                    b_Akk32 += tl.dot(
                        b_kg3.to(tl.bfloat16),
                        tl.trans(b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_gn3[None, :] - b_g2), 0.0)).to(tl.bfloat16),
                    )
                    if not LOAD_DIAG_FROM_AKKD:
                        b_gn3_diag = tl.load(g + (i_tc3 + min(BC // 2, T - i_tc3 - 1)) * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                        b_Akk33 += tl.dot(
                            (b_k3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_g3 - b_gn3_diag[None, :]), 0.0)).to(tl.bfloat16),
                            tl.trans(b_k3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_gn3_diag[None, :] - b_g3), 0.0)).to(tl.bfloat16),
                        )

    p_b0 = tl.make_block_ptr(beta, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    if not LOAD_DIAG_FROM_AKKD:
        b_Akk00 = b_Akk00 * b_b0[:, None]
        b_Akk11 = b_Akk11 * b_b1[:, None]
        b_Akk22 = b_Akk22 * b_b2[:, None]
        b_Akk33 = b_Akk33 * b_b3[:, None]
    b_Akk10 = b_Akk10 * b_b1[:, None]
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    if LOAD_DIAG_FROM_AKKD:
        p_Akkd00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_Akkd11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
        p_Akkd22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akkd33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
        b_Ai00 = _solve_tril_16_from_raw(tl.load(p_Akkd00, boundary_check=(0, 1)).to(tl.float32), T, i_tc0, BC)
        b_Ai11 = _solve_tril_16_from_raw(tl.load(p_Akkd11, boundary_check=(0, 1)).to(tl.float32), T, i_tc1, BC)
        b_Ai22 = _solve_tril_16_from_raw(tl.load(p_Akkd22, boundary_check=(0, 1)).to(tl.float32), T, i_tc2, BC)
        b_Ai33 = _solve_tril_16_from_raw(tl.load(p_Akkd33, boundary_check=(0, 1)).to(tl.float32), T, i_tc3, BC)
    else:
        b_Ai00 = _solve_tril_16_from_raw(b_Akk00, T, i_tc0, BC)
        b_Ai11 = _solve_tril_16_from_raw(b_Akk11, T, i_tc1, BC)
        b_Ai22 = _solve_tril_16_from_raw(b_Akk22, T, i_tc2, BC)
        b_Ai33 = _solve_tril_16_from_raw(b_Akk33, T, i_tc3, BC)

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11.to(tl.bfloat16), b_Akk10.to(tl.bfloat16)).to(tl.bfloat16),
        b_Ai00.to(tl.bfloat16),
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22.to(tl.bfloat16), b_Akk21.to(tl.bfloat16)).to(tl.bfloat16),
        b_Ai11.to(tl.bfloat16),
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33.to(tl.bfloat16), b_Akk32.to(tl.bfloat16)).to(tl.bfloat16),
        b_Ai22.to(tl.bfloat16),
    )
    b_Ai20 = -tl.dot(
        b_Ai22.to(tl.bfloat16),
        (
            tl.dot(b_Akk20.to(tl.bfloat16), b_Ai00.to(tl.bfloat16)) +
            tl.dot(b_Akk21.to(tl.bfloat16), b_Ai10.to(tl.bfloat16))
        ).to(tl.bfloat16),
    )
    b_Ai31 = -tl.dot(
        b_Ai33.to(tl.bfloat16),
        (
            tl.dot(b_Akk31.to(tl.bfloat16), b_Ai11.to(tl.bfloat16)) +
            tl.dot(b_Akk32.to(tl.bfloat16), b_Ai21.to(tl.bfloat16))
        ).to(tl.bfloat16),
    )
    b_Ai30 = -tl.dot(
        b_Ai33.to(tl.bfloat16),
        (
            tl.dot(b_Akk30.to(tl.bfloat16), b_Ai00.to(tl.bfloat16)) +
            tl.dot(b_Akk31.to(tl.bfloat16), b_Ai10.to(tl.bfloat16)) +
            tl.dot(b_Akk32.to(tl.bfloat16), b_Ai20.to(tl.bfloat16))
        ).to(tl.bfloat16),
    )

    if STORE_AKK:
        p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

        tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    ################################################################################
    # Recompute u from in-register inverse.
    ################################################################################
    if FUSE_RECOMPUTE:
        for i_v in range(tl.cdiv(V, BV)):
            p_v0 = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0))
            p_v1 = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0))
            p_v2 = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0))
            p_v3 = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0))
            b_v0 = tl.load(p_v0, boundary_check=(0, 1))
            b_v1 = tl.load(p_v1, boundary_check=(0, 1))
            b_v2 = tl.load(p_v2, boundary_check=(0, 1))
            b_v3 = tl.load(p_v3, boundary_check=(0, 1))
            b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
            b_vb1 = (b_v1 * b_b1[:, None]).to(b_v1.dtype)
            b_vb2 = (b_v2 * b_b2[:, None]).to(b_v2.dtype)
            b_vb3 = (b_v3 * b_b3[:, None]).to(b_v3.dtype)

            b_A00 = b_Ai00.to(b_v0.dtype)
            b_A10 = b_Ai10.to(b_v0.dtype)
            b_A11 = b_Ai11.to(b_v0.dtype)
            b_A20 = b_Ai20.to(b_v0.dtype)
            b_A21 = b_Ai21.to(b_v0.dtype)
            b_A22 = b_Ai22.to(b_v0.dtype)
            b_A30 = b_Ai30.to(b_v0.dtype)
            b_A31 = b_Ai31.to(b_v0.dtype)
            b_A32 = b_Ai32.to(b_v0.dtype)
            b_A33 = b_Ai33.to(b_v0.dtype)

            b_u0 = tl.dot(b_A00, b_vb0)
            b_u1 = tl.dot(b_A10, b_vb0) + tl.dot(b_A11, b_vb1)
            b_u2 = tl.dot(b_A20, b_vb0) + tl.dot(b_A21, b_vb1) + tl.dot(b_A22, b_vb2)
            b_u3 = tl.dot(b_A30, b_vb0) + tl.dot(b_A31, b_vb1) + tl.dot(b_A32, b_vb2) + tl.dot(b_A33, b_vb3)

            p_u0 = tl.make_block_ptr(u, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0))
            p_u1 = tl.make_block_ptr(u, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0))
            p_u2 = tl.make_block_ptr(u, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0))
            p_u3 = tl.make_block_ptr(u, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))

    ################################################################################
    # Recompute w and kg from in-register inverse.
    ################################################################################
        last_idx = min(i_t * BT + BT, T) - 1
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            b_k0 = tl.load(p_k0, boundary_check=(0, 1))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_k3 = tl.load(p_k3, boundary_check=(0, 1))
            b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)

            b_kb0 = (b_k0 * b_b0[:, None] * exp2(b_g0)).to(b_k0.dtype)
            b_kb1 = (b_k1 * b_b1[:, None] * exp2(b_g1)).to(b_k1.dtype)
            b_kb2 = (b_k2 * b_b2[:, None] * exp2(b_g2)).to(b_k2.dtype)
            b_kb3 = (b_k3 * b_b3[:, None] * exp2(b_g3)).to(b_k3.dtype)

            b_A00 = b_Ai00.to(b_k0.dtype)
            b_A10 = b_Ai10.to(b_k0.dtype)
            b_A11 = b_Ai11.to(b_k0.dtype)
            b_A20 = b_Ai20.to(b_k0.dtype)
            b_A21 = b_Ai21.to(b_k0.dtype)
            b_A22 = b_Ai22.to(b_k0.dtype)
            b_A30 = b_Ai30.to(b_k0.dtype)
            b_A31 = b_Ai31.to(b_k0.dtype)
            b_A32 = b_Ai32.to(b_k0.dtype)
            b_A33 = b_Ai33.to(b_k0.dtype)

            b_w0 = tl.dot(b_A00, b_kb0)
            b_w1 = tl.dot(b_A10, b_kb0) + tl.dot(b_A11, b_kb1)
            b_w2 = tl.dot(b_A20, b_kb0) + tl.dot(b_A21, b_kb1) + tl.dot(b_A22, b_kb2)
            b_w3 = tl.dot(b_A30, b_kb0) + tl.dot(b_A31, b_kb1) + tl.dot(b_A32, b_kb2) + tl.dot(b_A33, b_kb3)

            p_w0 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            p_w1 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_w2 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_w3 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))

            b_gn = tl.load(g + last_idx * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            p_kg0 = tl.make_block_ptr(kg, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
            p_kg1 = tl.make_block_ptr(kg, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_kg2 = tl.make_block_ptr(kg, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_kg3 = tl.make_block_ptr(kg, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            b_kg0 = b_k0 * tl.where(m0[:, None] & m_k[None, :], exp2(b_gn[None, :] - b_g0), 0.0)
            b_kg1 = b_k1 * tl.where(m1[:, None] & m_k[None, :], exp2(b_gn[None, :] - b_g1), 0.0)
            b_kg2 = b_k2 * tl.where(m2[:, None] & m_k[None, :], exp2(b_gn[None, :] - b_g2), 0.0)
            b_kg3 = b_k3 * tl.where(m3[:, None] & m_k[None, :], exp2(b_gn[None, :] - b_g3), 0.0)
            tl.store(p_kg0, b_kg0.to(p_kg0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg1, b_kg1.to(p_kg1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg2, b_kg2.to(p_kg2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg3, b_kg3.to(p_kg3.dtype.element_ty), boundary_check=(0, 1))


def _chunk_kda_fwd_intra_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor, None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = 16
    BK = 64
    BV = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)

    grid = (NT, B * H)
    chunk_kda_fwd_kernel_intra_fused[grid](
        q=q,
        k=k,
        v=v,
        g=gk,
        beta=beta,
        w=w,
        u=u,
        kg=kg,
        Aqk=Aqk,
        Akk=Aqk,
        Akkd=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        scale=scale,
        LOAD_DIAG_FROM_AKKD=False,
        FUSE_RECOMPUTE=True,
        STORE_AKK=False,
        num_warps=2,
        num_stages=2,
    )
    return w, u, None, kg, Aqk, None


def _chunk_kda_fwd_intra_diag_inter_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    BK = 64
    BV = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)

    grid = (NT, B * H)
    chunk_kda_fwd_kernel_intra_fused[grid](
        q=q,
        k=k,
        v=v,
        g=gk,
        beta=beta,
        w=k,
        u=v,
        kg=k,
        Aqk=Aqk,
        Akk=Akk,
        Akkd=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=v.shape[-1],
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        scale=scale,
        LOAD_DIAG_FROM_AKKD=False,
        FUSE_RECOMPUTE=False,
        STORE_AKK=True,
        num_warps=4,
        num_stages=2,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


def _chunk_kda_fwd_intra_inter_recompute_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor, None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = 16
    BK = 64
    BV = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)
    chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        sub_chunk_size=BC,
    )

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)

    grid = (NT, B * H)
    chunk_kda_fwd_kernel_intra_fused[grid](
        q=q,
        k=k,
        v=v,
        g=gk,
        beta=beta,
        w=w,
        u=u,
        kg=kg,
        Aqk=Aqk,
        Akk=Aqk,
        Akkd=Akkd,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        scale=scale,
        LOAD_DIAG_FROM_AKKD=True,
        FUSE_RECOMPUTE=True,
        STORE_AKK=False,
        num_warps=4,
        num_stages=2,
    )
    return w, u, None, kg, Aqk, None


def _chunk_kda_fwd_h_o_fused(
    q: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    kg: torch.Tensor,
    g: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *q.shape, u.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = torch.empty_like(u)
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, H, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * H)
    chunk_kda_fwd_fused_kernel_h_o[grid](
        q=q,
        w=w,
        u=u,
        kg=kg,
        g=g,
        Aqk=Aqk,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        scale=scale,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return o, final_state


@torch.compiler.disable
@input_guard
def chunk_kda_fwd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cu_seqlens_cpu is not None:
        del cu_seqlens_cpu

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be {len(cu_seqlens) - 1}, got {initial_state.shape[0]}."
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num heads)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num heads, head dim)."
    assert q.shape[-1] <= 256, "Currently this fused KDA path supports key headdim <= 256."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    intra_fusion_mode = kwargs.pop("intra_fusion_mode", None) or os.getenv("FLA_KDA_FUSED_INTRA_MODE", "full")

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=kwargs["A_log"],
            dt_bias=kwargs.get("dt_bias"),
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    if safe_gate or chunk_size != 64 or intra_fusion_mode == "none":
        w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            disable_recompute=False,
        )
    elif intra_fusion_mode == "full":
        w, u, _, kg, Aqk, _ = _chunk_kda_fwd_intra_fused(
            q=q,
            k=k,
            v=v,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    elif intra_fusion_mode == "diag_inter":
        w, u, _, kg, Aqk, _ = _chunk_kda_fwd_intra_diag_inter_fused(
            q=q,
            k=k,
            v=v,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    elif intra_fusion_mode == "inter_recompute":
        w, u, _, kg, Aqk, _ = _chunk_kda_fwd_intra_inter_recompute_fused(
            q=q,
            k=k,
            v=v,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(
            "FLA_KDA_FUSED_INTRA_MODE must be one of: full, diag_inter, inter_recompute, none."
        )

    return _chunk_kda_fwd_h_o_fused(
        q=q,
        w=w,
        u=u,
        kg=kg,
        g=g,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        transpose_state_layout=transpose_state_layout,
    )
