"""Block-sparse flash-decode attention kernel (Triton).

This is the engineering moat of AdaKV. A standard FlashAttention decode kernel
streams the *entire* KV cache; here each query head must attend only to the
*variable-length* list of blocks chosen by the selector, gathering them on the
fly so memory traffic is O(selected_blocks) instead of O(seq_len). Handling a
*ragged* per-head block count in a single launch (no host-side compaction of
the KV tensor) is the part that cannot be obtained by calling an existing
library -- FlashAttention/SDPA assume contiguous KV.

Status: v0 SKELETON. The control flow (flash online-softmax over a gathered
block list, GQA via head->kv mapping) is in place, but it MUST be validated and
autotuned on a CUDA GPU via tests/test_kernel_parity.py before any benchmark
numbers are trusted. Do not report speedups from an unvalidated kernel.

Inputs (decode, one query token per sequence):
    q            : [n_q_heads, head_dim]                fp16/bf16
    k_cache      : [n_kv_heads, seq_len, head_dim]      contiguous (paged variant TODO)
    v_cache      : [n_kv_heads, seq_len, head_dim]
    block_table  : [n_q_heads, max_sel]   int32         selected block ids per head
    sel_lens     : [n_q_heads]            int32         valid entries in block_table
Output:
    out          : [n_q_heads, head_dim]

The numerical ground truth is adakv.reference.sparse_attention (mask form);
this kernel returns the same result while reading only selected blocks.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU/dev box without triton
    HAS_TRITON = False


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["HEAD_DIM", "MAX_SEL"],
    )
    @triton.jit
    def _block_sparse_decode_kernel(
        q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr, out_ptr,
        sm_scale,
        stride_qh, stride_qd,
        stride_kh, stride_ks, stride_kd,
        stride_vh, stride_vs, stride_vd,
        stride_bth, stride_btn,
        stride_oh, stride_od,
        GROUP: tl.constexpr,        # n_q_heads // n_kv_heads
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,      # tokens per KV block (== estimator block_size)
        MAX_SEL: tl.constexpr,      # padded selection width
    ):
        h = tl.program_id(0)            # one program per query head
        kvh = h // GROUP                # GQA: map to kv head

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + h * stride_qh + d * stride_qd).to(tl.float32)

        m_i = -float("inf")             # running max
        l_i = 0.0                       # running denom
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        n_sel = tl.load(lens_ptr + h)
        offs_n = tl.arange(0, BLOCK_N)

        for j in range(0, MAX_SEL):
            if j < n_sel:
                blk = tl.load(bt_ptr + h * stride_bth + j * stride_btn)
                base = blk * BLOCK_N
                k_off = (kvh * stride_kh
                         + (base + offs_n)[:, None] * stride_ks
                         + d[None, :] * stride_kd)
                v_off = (kvh * stride_vh
                         + (base + offs_n)[:, None] * stride_vs
                         + d[None, :] * stride_vd)
                k = tl.load(k_ptr + k_off).to(tl.float32)     # [BLOCK_N, HEAD_DIM]
                v = tl.load(v_ptr + v_off).to(tl.float32)

                s = tl.sum(k * q[None, :], axis=1) * sm_scale  # [BLOCK_N]
                m_new = tl.maximum(m_i, tl.max(s, axis=0))
                p = tl.exp(s - m_new)                          # [BLOCK_N]
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                m_i = m_new

        acc = acc / l_i
        tl.store(out_ptr + h * stride_oh + d * stride_od, acc.to(out_ptr.dtype.element_ty))


def block_sparse_decode(q, k_cache, v_cache, block_table, sel_lens, block_size, sm_scale=None):
    """Host launcher. See module docstring for shapes.

    Raises RuntimeError on a non-Triton box so CPU dev never silently no-ops.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not available. Run on a CUDA GPU; use adakv.reference "
            "for CPU correctness checks."
        )
    import torch

    n_q_heads, head_dim = q.shape
    n_kv_heads = k_cache.shape[0]
    max_sel = block_table.shape[1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (head_dim ** 0.5)
    out = torch.empty_like(q)

    grid = (n_q_heads,)
    _block_sparse_decode_kernel[grid](
        q, k_cache, v_cache, block_table, sel_lens, out,
        sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1),
        GROUP=n_q_heads // n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=block_size,
        MAX_SEL=max_sel,
    )
    return out
