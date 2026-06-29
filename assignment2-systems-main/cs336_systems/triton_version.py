import torch
import math
import triton
import triton.language as tl


configs = [
    triton.Config(
        {'BLOCK_Q': BM, 'BLOCK_K': BN},
        num_stages=s,
        num_warps=w,
    )
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
]


def keep(conf):
    return True


def prune_invalid_configs(configs, named_args, **kwargs):
    N_Q = named_args["N_Q"]
    N_K = named_args["N_K"]
    STAGE = kwargs["STAGE"]

    return [
        conf for conf in configs
        if conf.kwargs.get("BLOCK_Q", 0) <= N_Q
        and conf.kwargs.get("BLOCK_K", 0) <= N_K
        and (
            conf.kwargs.get("BLOCK_Q", 0) >= conf.kwargs.get("BLOCK_K", 0)
            or STAGE == 1
        )
    ]


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_Q", "N_K", "D"],
    prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(q_ptr,
              k_ptr,
              v_ptr,
              o_ptr,
              lse_ptr,
              sm_scale: tl.constexpr,
              stride_qb: tl.constexpr,
              stride_qm: tl.constexpr,
              stride_qd: tl.constexpr,
              stride_kb: tl.constexpr,
              stride_kn: tl.constexpr,
              stride_kd: tl.constexpr,
              stride_vb: tl.constexpr,
              stride_vn: tl.constexpr,
              stride_vd: tl.constexpr,
              stride_ob: tl.constexpr,
              stride_om: tl.constexpr,
              stride_od: tl.constexpr,
              stride_lb: tl.constexpr,
              stride_lm: tl.constexpr,
              N_Q: tl.constexpr,
              N_K: tl.constexpr,
              D: tl.constexpr,
              BLOCK_Q: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_D: tl.constexpr,
              STAGE: tl.constexpr):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + pid_bh * stride_qb + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_q[:, None] < N_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)
    q = (q * sm_scale).to(tl.float16)

    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc,
                                        l_i,
                                        m_i,
                                        q,
                                        k_ptr,
                                        v_ptr,
                                        pid_q,
                                        pid_bh,
                                        stride_kb,
                                        stride_kn,
                                        stride_kd,
                                        stride_vb,
                                        stride_vn,
                                        stride_vd,
                                        N_Q,
                                        N_K,
                                        D,
                                        BLOCK_Q,
                                        BLOCK_K,
                                        BLOCK_D,
                                        4 - STAGE)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc,
                                        l_i,
                                        m_i,
                                        q,
                                        k_ptr,
                                        v_ptr,
                                        pid_q,
                                        pid_bh,
                                        stride_kb,
                                        stride_kn,
                                        stride_kd,
                                        stride_vb,
                                        stride_vn,
                                        stride_vd,
                                        N_Q,
                                        N_K,
                                        D,
                                        BLOCK_Q,
                                        BLOCK_K,
                                        BLOCK_D,
                                        2)

    m_i += tl.math.log(l_i)
    acc = acc / l_i[:, None]

    lse_ptrs = lse_ptr + pid_bh * N_Q + offs_q
    tl.store(lse_ptrs, m_i)

    o_ptrs = o_ptr + pid_bh * stride_ob + offs_q[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16))


@triton.jit
def _attn_fwd_inner(acc,
                    l_i,
                    m_i,
                    q,
                    k_ptr,
                    v_ptr,
                    pid_q,
                    pid_bh,
                    stride_kb: tl.constexpr,
                    stride_kn: tl.constexpr,
                    stride_kd: tl.constexpr,
                    stride_vb: tl.constexpr,
                    stride_vn: tl.constexpr,
                    stride_vd: tl.constexpr,
                    N_Q: tl.constexpr,
                    N_K: tl.constexpr,
                    D: tl.constexpr,
                    BLOCK_Q: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                    BLOCK_D: tl.constexpr,
                    STAGE: tl.constexpr,
                    ):
    if STAGE == 1:
        lo, hi = 0, pid_q * BLOCK_Q
    elif STAGE == 2:
        lo, hi = pid_q * BLOCK_Q, (pid_q + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
    else:
        lo, hi = 0, N_K

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    for start_k in tl.range(lo, hi, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        idxs = start_k + offs_k

        k_ptrs = k_ptr + pid_bh * stride_kb + idxs[None, :] * stride_kn + offs_d[:, None] * stride_kd     # T
        k_mask = (idxs[None, :] < N_K) & (offs_d[:, None] < D)

        v_ptrs = v_ptr + pid_bh * stride_vb + idxs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (idxs[:, None] < N_K) & (offs_d[None, :] < D)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float16)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float16)

        valid_k = (idxs[None, :] < N_K)
        scores = tl.dot(q, k).to(tl.float32)

        if STAGE == 2:
            causal_mask = (offs_q[:, None] >= idxs[None, :])
            valid_k = causal_mask & valid_k

        scores = tl.where(valid_k, scores, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, 1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i

@triton.jit
def _attn_bwd_preprocess(o,
                         do,
                         delta_ptr,
                         B,
                         H,
                         D,
                         N_Q,
                         stride_ob: tl.constexpr,
                         stride_om: tl.constexpr,
                         stride_od: tl.constexpr,
                         BLOCK_D: tl.constexpr,
                         BLOCK_Q: tl.constexpr=128
                         ):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = o + pid_bh * stride_ob + offs_q[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_q[:, None] < N_Q) & (offs_d[None, :] < D)

    do_ptrs = do + pid_bh * stride_ob + offs_q[:, None] * stride_om + offs_d[None, :] * stride_od

    o = tl.load(o_ptrs, mask=o_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=o_mask, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)  # [BLOCK_Q,]
    delta_ptrs = delta_ptr + pid_bh * N_Q + offs_q
    delta_mask = offs_q < N_Q

    tl.store(delta_ptrs, delta, mask=delta_mask)


@triton.jit
def _attn_bwd():
    # TODO


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False) -> torch.Tensor:
        had_heads = q.dim() == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape
            q = q.reshape(B * H, N_q, D)
            k = k.reshape(B * H, N_k, D)
            v = v.reshape(B * H, N_k, D)
        else:
            B, N_q, D = q.shape
            _, N_k, D = k.shape

        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()

        o = torch.empty_like(q)
        lse = torch.empty((q.shape[0], N_q), device=q.device, dtype=torch.float32)

        sm_scale = 1.0 / math.sqrt(q.shape[-1])

        stage = 3 if is_causal else 1

        BLOCK_D = triton.next_power_of_2(D)

        def grid(META):
            return (triton.cdiv(N_q, META["BLOCK_Q"]), q.shape[0])

        _attn_fwd[grid](q,
                        k,
                        v,
                        o,
                        lse,
                        sm_scale,
                        q.stride(0),
                        q.stride(1),
                        q.stride(2),
                        k.stride(0),
                        k.stride(1),
                        k.stride(2),
                        v.stride(0),
                        v.stride(1),
                        v.stride(2),
                        o.stride(0),
                        o.stride(1),
                        o.stride(2),
                        lse.stride(0),
                        lse.stride(1),
                        N_q,
                        N_k,
                        D,
                        BLOCK_D=BLOCK_D,
                        STAGE=stage)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            ctx.B, ctx.H = B, H
            o = o.reshape(B, H, N_q, D)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        if had_heads:
            B, H = ctx.B, ctx.H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]
            do = do.reshape(B*H, N_q, D)
        else:
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        delta = torch.empty_like(lse)
        pre_gird = (triton.cdiv(N_q, 128), B * H)
        _attn_bwd_preprocess[pre_gird](o,
                                       do,
                                       delta,
                                       B,
                                       H,
                                       D,
                                       N_q,
                                       o.stride(0),
                                       o.stride(1),
                                       o.stride(2),
                                       D)

        grid = (triton.cdiv(N_q, 128), 1, B * H)
        _attn_bwd[grid](

        )


        if had_heads:
            dq = dq.reshape(B, H, N_q, D)
            dk = dk.reshape(B, H, N_k, D)
            dv = dv.reshape(B, H, N_k, D)

        return dq, dk, dv, None
