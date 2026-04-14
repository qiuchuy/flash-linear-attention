# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, input_guard


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


@triton.jit
def _kda_fused_pair16(
    q,
    k,
    g,
    beta,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    chunk_start,
    ROW: tl.constexpr,
    COL: tl.constexpr,
    scale: tl.constexpr,
):
    offs_m = tl.arange(0, 16)
    offs_k = tl.arange(0, 64)
    row_start = chunk_start + ROW * 16
    col_start = chunk_start + COL * 16
    ref_t = chunk_start + min(ROW * 16 + 8, T - chunk_start - 1)
    m_row = row_start + offs_m < T
    m_col = col_start + offs_m < T

    b_beta = tl.load(beta + (row_start + offs_m) * H, mask=m_row, other=0.0).to(tl.float32)

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (row_start, 0), (16, 64), (1, 0))
    p_kr = tl.make_block_ptr(k, (T, K), (H * K, 1), (row_start, 0), (16, 64), (1, 0))
    p_gr = tl.make_block_ptr(g, (T, K), (H * K, 1), (row_start, 0), (16, 64), (1, 0))
    p_kc = tl.make_block_ptr(k, (T, K), (H * K, 1), (col_start, 0), (16, 64), (1, 0))
    p_gc = tl.make_block_ptr(g, (T, K), (H * K, 1), (col_start, 0), (16, 64), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_kr = tl.load(p_kr, boundary_check=(0, 1)).to(tl.float32)
    b_gr = tl.load(p_gr, boundary_check=(0, 1)).to(tl.float32)
    b_kc = tl.load(p_kc, boundary_check=(0, 1)).to(tl.float32)
    b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
    b_g_ref = tl.load(g + ref_t * H * K + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)[None, :]

    m_rk = m_row[:, None] & (offs_k[None, :] < K)
    m_ck = m_col[:, None] & (offs_k[None, :] < K)
    b_gm_r = b_gr - b_g_ref
    b_qg = tl.where(m_rk, b_q * exp2(b_gm_r), 0.0)
    b_kg = tl.where(m_rk, b_kr * exp2(b_gm_r), 0.0)
    b_kng_t = tl.trans(tl.where(m_ck, b_kc * exp2(b_g_ref - b_gc), 0.0))
    b_qk = tl.dot(b_qg, b_kng_t)
    b_kk = tl.dot(b_kg, b_kng_t)

    if K > 64:
        o_k = 64 + offs_k
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (row_start, 64), (16, 64), (1, 0))
        p_kr = tl.make_block_ptr(k, (T, K), (H * K, 1), (row_start, 64), (16, 64), (1, 0))
        p_gr = tl.make_block_ptr(g, (T, K), (H * K, 1), (row_start, 64), (16, 64), (1, 0))
        p_kc = tl.make_block_ptr(k, (T, K), (H * K, 1), (col_start, 64), (16, 64), (1, 0))
        p_gc = tl.make_block_ptr(g, (T, K), (H * K, 1), (col_start, 64), (16, 64), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_kr = tl.load(p_kr, boundary_check=(0, 1)).to(tl.float32)
        b_gr = tl.load(p_gr, boundary_check=(0, 1)).to(tl.float32)
        b_kc = tl.load(p_kc, boundary_check=(0, 1)).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_g_ref = tl.load(g + ref_t * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)[None, :]

        m_rk = m_row[:, None] & (o_k[None, :] < K)
        m_ck = m_col[:, None] & (o_k[None, :] < K)
        b_gm_r = b_gr - b_g_ref
        b_qg = tl.where(m_rk, b_q * exp2(b_gm_r), 0.0)
        b_kg = tl.where(m_rk, b_kr * exp2(b_gm_r), 0.0)
        b_kng_t = tl.trans(tl.where(m_ck, b_kc * exp2(b_g_ref - b_gc), 0.0))
        b_qk += tl.dot(b_qg, b_kng_t)
        b_kk += tl.dot(b_kg, b_kng_t)

    valid = m_row[:, None] & m_col[None, :]
    if ROW == COL:
        lower = offs_m[:, None] >= offs_m[None, :]
        strict = offs_m[:, None] > offs_m[None, :]
        b_qk = tl.where(valid & lower, b_qk * scale, 0.0)
        b_kk = tl.where(valid & strict, b_kk * b_beta[:, None], 0.0)
    else:
        b_qk = tl.where(valid, b_qk * scale, 0.0)
        b_kk = tl.where(valid, b_kk * b_beta[:, None], 0.0)
    return b_qk, b_kk


@triton.jit
def _kda_inv16(a):
    offs = tl.arange(0, 16)
    eye = offs[:, None] == offs[None, :]
    x = -a
    for i in range(2, 16):
        row = tl.sum(tl.where(offs[:, None] == i, -a, 0.0), axis=0)
        row = tl.where(offs < i, row, 0.0)
        row += tl.sum(row[:, None] * x, axis=0)
        x = tl.where((offs == i)[:, None], row, x)
    return x + eye


@triton.jit
def _kda_load_qexp_kbg16(
    q,
    k,
    g,
    beta,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    chunk_start,
    ROW: tl.constexpr,
    K_OFF: tl.constexpr,
):
    offs_m = tl.arange(0, 16)
    offs_k = tl.arange(0, 64)
    row_start = chunk_start + ROW * 16
    o_k = K_OFF + offs_k
    m_row = row_start + offs_m < T
    m_rk = m_row[:, None] & (o_k[None, :] < K)

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (row_start, K_OFF), (16, 64), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (row_start, K_OFF), (16, 64), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (row_start, K_OFF), (16, 64), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(beta + (row_start + offs_m) * H, mask=m_row, other=0.0).to(tl.float32)
    b_exp = exp2(b_g)
    return tl.where(m_rk, b_q * b_exp, 0.0), tl.where(m_rk, b_k * b_beta[:, None] * b_exp, 0.0)


@triton.jit
def _kda_load_kg16(
    k,
    g,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    chunk_start,
    last_idx,
    ROW: tl.constexpr,
    K_OFF: tl.constexpr,
):
    offs_m = tl.arange(0, 16)
    offs_k = tl.arange(0, 64)
    row_start = chunk_start + ROW * 16
    o_k = K_OFF + offs_k
    m_row = row_start + offs_m < T
    m_rk = m_row[:, None] & (o_k[None, :] < K)
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (row_start, K_OFF), (16, 64), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (row_start, K_OFF), (16, 64), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
    return tl.where(m_rk, b_k * exp2(b_g_last[None, :] - b_g), 0.0), b_g_last


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_fully_fused_block_kernel(
    q,
    k,
    v,
    g,
    beta,
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
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    o += (bos * H + i_h) * V

    offs_m = tl.arange(0, 16)

    for i_t in range(NT):
        chunk_start = i_t * BT

        q00, a00 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 0, 0, scale)
        q10, a10 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 1, 0, scale)
        q11, a11 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 1, 1, scale)
        q20, a20 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 2, 0, scale)
        q21, a21 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 2, 1, scale)
        q22, a22 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 2, 2, scale)
        q30, a30 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 3, 0, scale)
        q31, a31 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 3, 1, scale)
        q32, a32 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 3, 2, scale)
        q33, a33 = _kda_fused_pair16(q, k, g, beta, T, H, K, chunk_start, 3, 3, scale)

        x00 = _kda_inv16(a00)
        x11 = _kda_inv16(a11)
        x22 = _kda_inv16(a22)
        x33 = _kda_inv16(a33)

        x10 = -tl.dot(tl.dot(x11, a10), x00)
        x21 = -tl.dot(tl.dot(x22, a21), x11)
        x32 = -tl.dot(tl.dot(x33, a32), x22)
        x20 = -tl.dot(x22, tl.dot(a20, x00) + tl.dot(a21, x10))
        x31 = -tl.dot(x33, tl.dot(a31, x11) + tl.dot(a32, x21))
        x30 = -tl.dot(x33, tl.dot(a30, x00) + tl.dot(a31, x10) + tl.dot(a32, x20))

        p_v0 = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk_start, i_v * BV), (16, BV), (1, 0))
        p_v1 = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk_start + 16, i_v * BV), (16, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk_start + 32, i_v * BV), (16, BV), (1, 0))
        p_v3 = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk_start + 48, i_v * BV), (16, BV), (1, 0))
        b_v0 = tl.load(p_v0, boundary_check=(0, 1)).to(tl.float32)
        b_v1 = tl.load(p_v1, boundary_check=(0, 1)).to(tl.float32)
        b_v2 = tl.load(p_v2, boundary_check=(0, 1)).to(tl.float32)
        b_v3 = tl.load(p_v3, boundary_check=(0, 1)).to(tl.float32)

        m0 = chunk_start + offs_m < T
        m1 = chunk_start + 16 + offs_m < T
        m2 = chunk_start + 32 + offs_m < T
        m3 = chunk_start + 48 + offs_m < T
        b0 = tl.load(beta + (chunk_start + offs_m) * H, mask=m0, other=0.0).to(tl.float32)
        b1 = tl.load(beta + (chunk_start + 16 + offs_m) * H, mask=m1, other=0.0).to(tl.float32)
        b2 = tl.load(beta + (chunk_start + 32 + offs_m) * H, mask=m2, other=0.0).to(tl.float32)
        b3 = tl.load(beta + (chunk_start + 48 + offs_m) * H, mask=m3, other=0.0).to(tl.float32)

        v0 = tl.dot(x00, b_v0 * b0[:, None])
        v1 = tl.dot(x10, b_v0 * b0[:, None]) + tl.dot(x11, b_v1 * b1[:, None])
        v2 = tl.dot(x20, b_v0 * b0[:, None]) + tl.dot(x21, b_v1 * b1[:, None]) + tl.dot(x22, b_v2 * b2[:, None])
        v3 = (
            tl.dot(x30, b_v0 * b0[:, None])
            + tl.dot(x31, b_v1 * b1[:, None])
            + tl.dot(x32, b_v2 * b2[:, None])
            + tl.dot(x33, b_v3 * b3[:, None])
        )

        o0 = tl.zeros([16, BV], dtype=tl.float32)
        o1 = tl.zeros([16, BV], dtype=tl.float32)
        o2 = tl.zeros([16, BV], dtype=tl.float32)
        o3 = tl.zeros([16, BV], dtype=tl.float32)

        if i_t != 0 or USE_INITIAL_STATE:
            qg0, kbg0 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 0, 0)
            qg1, kbg1 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 1, 0)
            qg2, kbg2 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 2, 0)
            qg3, kbg3 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 3, 0)
            if TRANSPOSE_STATE:
                o0 += tl.dot(qg0, tl.trans(b_h1))
                o1 += tl.dot(qg1, tl.trans(b_h1))
                o2 += tl.dot(qg2, tl.trans(b_h1))
                o3 += tl.dot(qg3, tl.trans(b_h1))
                v0 -= tl.dot(tl.dot(x00, kbg0), tl.trans(b_h1))
                v1 -= tl.dot(tl.dot(x10, kbg0) + tl.dot(x11, kbg1), tl.trans(b_h1))
                v2 -= tl.dot(tl.dot(x20, kbg0) + tl.dot(x21, kbg1) + tl.dot(x22, kbg2), tl.trans(b_h1))
                v3 -= tl.dot(
                    tl.dot(x30, kbg0) + tl.dot(x31, kbg1) + tl.dot(x32, kbg2) + tl.dot(x33, kbg3),
                    tl.trans(b_h1),
                )
            else:
                o0 += tl.dot(qg0, b_h1)
                o1 += tl.dot(qg1, b_h1)
                o2 += tl.dot(qg2, b_h1)
                o3 += tl.dot(qg3, b_h1)
                v0 -= tl.dot(tl.dot(x00, kbg0), b_h1)
                v1 -= tl.dot(tl.dot(x10, kbg0) + tl.dot(x11, kbg1), b_h1)
                v2 -= tl.dot(tl.dot(x20, kbg0) + tl.dot(x21, kbg1) + tl.dot(x22, kbg2), b_h1)
                v3 -= tl.dot(tl.dot(x30, kbg0) + tl.dot(x31, kbg1) + tl.dot(x32, kbg2) + tl.dot(x33, kbg3), b_h1)

            if K > 64:
                qg0, kbg0 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 0, 64)
                qg1, kbg1 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 1, 64)
                qg2, kbg2 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 2, 64)
                qg3, kbg3 = _kda_load_qexp_kbg16(q, k, g, beta, T, H, K, chunk_start, 3, 64)
                if TRANSPOSE_STATE:
                    o0 += tl.dot(qg0, tl.trans(b_h2))
                    o1 += tl.dot(qg1, tl.trans(b_h2))
                    o2 += tl.dot(qg2, tl.trans(b_h2))
                    o3 += tl.dot(qg3, tl.trans(b_h2))
                    v0 -= tl.dot(tl.dot(x00, kbg0), tl.trans(b_h2))
                    v1 -= tl.dot(tl.dot(x10, kbg0) + tl.dot(x11, kbg1), tl.trans(b_h2))
                    v2 -= tl.dot(tl.dot(x20, kbg0) + tl.dot(x21, kbg1) + tl.dot(x22, kbg2), tl.trans(b_h2))
                    v3 -= tl.dot(
                        tl.dot(x30, kbg0) + tl.dot(x31, kbg1) + tl.dot(x32, kbg2) + tl.dot(x33, kbg3),
                        tl.trans(b_h2),
                    )
                else:
                    o0 += tl.dot(qg0, b_h2)
                    o1 += tl.dot(qg1, b_h2)
                    o2 += tl.dot(qg2, b_h2)
                    o3 += tl.dot(qg3, b_h2)
                    v0 -= tl.dot(tl.dot(x00, kbg0), b_h2)
                    v1 -= tl.dot(tl.dot(x10, kbg0) + tl.dot(x11, kbg1), b_h2)
                    v2 -= tl.dot(tl.dot(x20, kbg0) + tl.dot(x21, kbg1) + tl.dot(x22, kbg2), b_h2)
                    v3 -= tl.dot(
                        tl.dot(x30, kbg0) + tl.dot(x31, kbg1) + tl.dot(x32, kbg2) + tl.dot(x33, kbg3),
                        b_h2,
                    )

        v0 = tl.where(m0[:, None], v0, 0.0)
        v1 = tl.where(m1[:, None], v1, 0.0)
        v2 = tl.where(m2[:, None], v2, 0.0)
        v3 = tl.where(m3[:, None], v3, 0.0)

        o0 = o0 * scale + tl.dot(q00, v0)
        o1 = o1 * scale + tl.dot(q10, v0) + tl.dot(q11, v1)
        o2 = o2 * scale + tl.dot(q20, v0) + tl.dot(q21, v1) + tl.dot(q22, v2)
        o3 = o3 * scale + tl.dot(q30, v0) + tl.dot(q31, v1) + tl.dot(q32, v2) + tl.dot(q33, v3)

        p_o0 = tl.make_block_ptr(o, (T, V), (H * V, 1), (chunk_start, i_v * BV), (16, BV), (1, 0))
        p_o1 = tl.make_block_ptr(o, (T, V), (H * V, 1), (chunk_start + 16, i_v * BV), (16, BV), (1, 0))
        p_o2 = tl.make_block_ptr(o, (T, V), (H * V, 1), (chunk_start + 32, i_v * BV), (16, BV), (1, 0))
        p_o3 = tl.make_block_ptr(o, (T, V), (H * V, 1), (chunk_start + 48, i_v * BV), (16, BV), (1, 0))
        tl.store(p_o0, o0.to(p_o0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_o1, o1.to(p_o1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_o2, o2.to(p_o2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_o3, o3.to(p_o3.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(chunk_start + BT, T) - 1

        kg0, g_last1 = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 0, 0)
        kg1, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 1, 0)
        kg2, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 2, 0)
        kg3, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 3, 0)
        if TRANSPOSE_STATE:
            if i_t != 0 or USE_INITIAL_STATE:
                b_h1 *= exp2(g_last1)[None, :]
            b_h1 += tl.trans(tl.dot(tl.trans(kg0), v0) + tl.dot(tl.trans(kg1), v1) + tl.dot(tl.trans(kg2), v2) + tl.dot(tl.trans(kg3), v3))
        else:
            if i_t != 0 or USE_INITIAL_STATE:
                b_h1 *= exp2(g_last1)[:, None]
            b_h1 += tl.dot(tl.trans(kg0), v0) + tl.dot(tl.trans(kg1), v1) + tl.dot(tl.trans(kg2), v2) + tl.dot(tl.trans(kg3), v3)

        if K > 64:
            kg0, g_last2 = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 0, 64)
            kg1, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 1, 64)
            kg2, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 2, 64)
            kg3, _ = _kda_load_kg16(k, g, T, H, K, chunk_start, last_idx, 3, 64)
            if TRANSPOSE_STATE:
                if i_t != 0 or USE_INITIAL_STATE:
                    b_h2 *= exp2(g_last2)[None, :]
                b_h2 += tl.trans(tl.dot(tl.trans(kg0), v0) + tl.dot(tl.trans(kg1), v1) + tl.dot(tl.trans(kg2), v2) + tl.dot(tl.trans(kg3), v3))
            else:
                if i_t != 0 or USE_INITIAL_STATE:
                    b_h2 *= exp2(g_last2)[:, None]
                b_h2 += tl.dot(tl.trans(kg0), v0) + tl.dot(tl.trans(kg1), v1) + tl.dot(tl.trans(kg2), v2) + tl.dot(tl.trans(kg3), v3)

    if STORE_FINAL_STATE:
        ht += i_nh * K * V
        if TRANSPOSE_STATE:
            p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            p_ht2 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        else:
            p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_ht2 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_fully_fused_kernel(
    q,
    k,
    v,
    g,
    beta,
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
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    o += (bos * H + i_h) * V

    offs_t = tl.arange(0, BT)
    offs_b = tl.arange(0, 16)
    offs_k = tl.arange(0, 64)
    m_lower = offs_t[:, None] >= offs_t[None, :]
    m_strict = offs_t[:, None] > offs_t[None, :]
    m_eye = offs_t[:, None] == offs_t[None, :]

    for i_t in range(NT):
        chunk_start = i_t * BT
        m_t = chunk_start + offs_t < T
        m_pair = m_t[:, None] & m_t[None, :]

        p_beta = tl.make_block_ptr(beta, (T,), (H,), (chunk_start,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        b_beta = tl.where(m_t, b_beta, 0.0)

        b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
        b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

        for i_c in range(0, 4):
            row_lo = i_c * 16
            row_hi = row_lo + 16
            ref_t = chunk_start + min(row_lo + 8, T - chunk_start - 1)
            m_q_rows = chunk_start + row_lo + offs_b < T
            m_k_rows = (offs_t < row_hi) & m_t

            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (chunk_start + row_lo, 0), (16, 64), (1, 0))
            p_kr1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start + row_lo, 0), (16, 64), (1, 0))
            p_gr1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start + row_lo, 0), (16, 64), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
            b_q = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_kr = tl.load(p_kr1, boundary_check=(0, 1)).to(tl.float32)
            b_gr = tl.load(p_gr1, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_g_ref = tl.load(g + ref_t * H * K + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)[None, :]
            m_q_tk = m_q_rows[:, None] & (offs_k[None, :] < K)
            m_k_tk = m_k_rows[:, None] & (offs_k[None, :] < K)
            b_qg = tl.where(m_q_tk, b_q * exp2(b_gr - b_g_ref), 0.0)
            b_k_pos = tl.where(m_q_tk, b_kr * exp2(b_gr - b_g_ref), 0.0)
            b_k_neg_t = tl.trans(tl.where(m_k_tk, b_k * exp2(b_g_ref - b_g), 0.0))
            b_Aqk_rows = tl.dot(b_qg, b_k_neg_t)
            b_Akk_rows = tl.dot(b_k_pos, b_k_neg_t)

            if K > 64:
                o_k = 64 + offs_k
                p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (chunk_start + row_lo, 64), (16, 64), (1, 0))
                p_kr2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start + row_lo, 64), (16, 64), (1, 0))
                p_gr2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start + row_lo, 64), (16, 64), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
                b_q = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_kr = tl.load(p_kr2, boundary_check=(0, 1)).to(tl.float32)
                b_gr = tl.load(p_gr2, boundary_check=(0, 1)).to(tl.float32)
                b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_g_ref = tl.load(g + ref_t * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)[None, :]
                m_q_tk = m_q_rows[:, None] & (o_k[None, :] < K)
                m_k_tk = m_k_rows[:, None] & (o_k[None, :] < K)
                b_qg = tl.where(m_q_tk, b_q * exp2(b_gr - b_g_ref), 0.0)
                b_k_pos = tl.where(m_q_tk, b_kr * exp2(b_gr - b_g_ref), 0.0)
                b_k_neg_t = tl.trans(tl.where(m_k_tk, b_k * exp2(b_g_ref - b_g), 0.0))
                b_Aqk_rows += tl.dot(b_qg, b_k_neg_t)
                b_Akk_rows += tl.dot(b_k_pos, b_k_neg_t)

            for i_r in range(0, 16):
                b_aqk_row = tl.sum(tl.where(offs_b[:, None] == i_r, b_Aqk_rows, 0.0), axis=0)
                b_akk_row = tl.sum(tl.where(offs_b[:, None] == i_r, b_Akk_rows, 0.0), axis=0)
                b_Aqk += tl.where(offs_t[:, None] == row_lo + i_r, b_aqk_row[None, :], 0.0)
                b_Akk += tl.where(offs_t[:, None] == row_lo + i_r, b_akk_row[None, :], 0.0)

        b_Aqk = tl.where(m_lower & m_pair, b_Aqk * scale, 0.0)
        b_Akk = tl.where(m_strict & m_pair, b_Akk * b_beta[:, None], 0.0)

        b_Ai = -b_Akk
        for i in range(2, min(BT, T - chunk_start)):
            b_a = tl.sum(tl.where(offs_t[:, None] == i, -b_Akk, 0.0), axis=0)
            b_a = tl.where(offs_t < i, b_a, 0.0)
            b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
            b_Ai = tl.where((offs_t == i)[:, None], b_a, b_Ai)
        b_Ai += m_eye
        b_Ai = tl.where(m_pair, b_Ai, 0.0)

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk_start, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_vnew = tl.dot(b_Ai, (b_v * b_beta[:, None]).to(b_Ai.dtype))
        b_vnew = tl.where(m_t[:, None], b_vnew, 0.0)

        b_o = tl.zeros([BT, BV], dtype=tl.float32)

        p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        b_q = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        m_tk = m_t[:, None] & (offs_k[None, :] < K)
        b_qg = tl.where(m_tk, b_q * exp2(b_g), 0.0)
        b_kbg = tl.where(m_tk, b_k * b_beta[:, None] * exp2(b_g), 0.0)
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_qg, tl.trans(b_h1))
            b_w = tl.dot(b_Ai, b_kbg)
            b_vnew -= tl.dot(b_w, tl.trans(b_h1))
        else:
            b_o += tl.dot(b_qg, b_h1)
            b_w = tl.dot(b_Ai, b_kbg)
            b_vnew -= tl.dot(b_w, b_h1)

        if K > 64:
            o_k = 64 + offs_k
            p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            b_q = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            m_tk = m_t[:, None] & (o_k[None, :] < K)
            b_qg = tl.where(m_tk, b_q * exp2(b_g), 0.0)
            b_kbg = tl.where(m_tk, b_k * b_beta[:, None] * exp2(b_g), 0.0)
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h2))
                b_w = tl.dot(b_Ai, b_kbg)
                b_vnew -= tl.dot(b_w, tl.trans(b_h2))
            else:
                b_o += tl.dot(b_qg, b_h2)
                b_w = tl.dot(b_Ai, b_kbg)
                b_vnew -= tl.dot(b_w, b_h2)

        b_vnew = tl.where(m_t[:, None], b_vnew, 0.0)
        b_o = b_o * scale + tl.dot(b_Aqk, b_vnew)

        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (chunk_start, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(chunk_start + BT, T) - 1

        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_g_last = tl.load(g + last_idx * H * K + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        if TRANSPOSE_STATE:
            b_h1 *= exp2(b_g_last)[None, :]
        else:
            b_h1 *= exp2(b_g_last)[:, None]
        m_tk = m_t[:, None] & (offs_k[None, :] < K)
        b_kg = tl.where(m_tk, b_k * exp2(b_g_last[None, :] - b_g), 0.0)
        b_dh = tl.dot(tl.trans(b_kg), b_vnew)
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(b_dh)
        else:
            b_h1 += b_dh

        if K > 64:
            o_k = 64 + offs_k
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (chunk_start, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_g_last = tl.load(g + last_idx * H * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h2 *= exp2(b_g_last)[None, :]
            else:
                b_h2 *= exp2(b_g_last)[:, None]
            m_tk = m_t[:, None] & (o_k[None, :] < K)
            b_kg = tl.where(m_tk, b_k * exp2(b_g_last[None, :] - b_g), 0.0)
            b_dh = tl.dot(tl.trans(b_kg), b_vnew)
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(b_dh)
            else:
                b_h2 += b_dh

    if STORE_FINAL_STATE:
        ht += i_nh * K * V
        if TRANSPOSE_STATE:
            p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            p_ht2 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        else:
            p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_ht2 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))


def _chunk_kda_fwd_fully_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = torch.empty_like(v)
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, H, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    # A single 128-wide V tile mirrors the cuLA/CUTLASS tile shape, but the
    # Triton version keeps the state, inverse blocks, corrected V, and output
    # fragments live in one program.  Splitting V into two 64-wide programs
    # gives better occupancy and avoids the performance cliff from register
    # pressure on H100.
    BV = 64
    grid = (triton.cdiv(V, BV), N * H)
    chunk_kda_fwd_fully_fused_block_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BV=BV,
        scale=scale,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=4,
        num_stages=1,
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

    fully_fused_supported = q.shape[-1] == 128 and v.shape[-1] == 128 and chunk_size == 64
    if fully_fused_supported:
        # Experimental path for the cuLA benchmark tile.  It carries the full
        # 128x128 state in one Triton program and fuses intra, WY, H, and O.
        return _chunk_kda_fwd_fully_fused(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            transpose_state_layout=transpose_state_layout,
        )

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
