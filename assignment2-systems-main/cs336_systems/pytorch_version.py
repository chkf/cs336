import torch
import math


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False):
        had_heads = q.dim() == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape

            q = q.reshape(B * H, N_q, D)
            k = k.reshape(B * H, N_k, D)
            v = v.reshape(B * H, N_k, D)
            B_H = B * H
        else:
            B, N_q, D = q.shape
            _, N_k, _ = k.shape
            B_H = B

        device = q.device
        softmax_scale = 1.0/math.sqrt(D)

        block_q, block_k = 64, 64

        tile_q, tile_k = math.ceil(N_q / block_q), math.ceil(N_k / block_k)

        O = torch.empty((B_H, N_q, D), device=device, dtype=torch.float32)
        L = torch.empty((B_H, N_q), device=device, dtype=torch.float32)

        q_pos = torch.arange(N_q, device=device)
        k_pos = torch.arange(N_k, device=device)

        for i in range(tile_q):
            q_start, q_end = i * block_q, min((i+1)*block_q, N_q)
            q_len = q_end - q_start

            q_block_i = q[:, q_start:q_end, :]

            m_i = torch.full((B_H, q_len), -float("inf"), device=device, dtype=torch.float32)
            l_i = torch.zeros((B_H, q_len), device=device, dtype=torch.float32)
            O_i = torch.zeros((B_H, q_len, D), device=device, dtype=torch.float32)

            q_idx = q_pos[q_start:q_end]

            for j in range(tile_k):
                k_start, k_end = j * block_k, min((j + 1) * block_k, N_k)
                if is_causal and k_start >= q_end:
                    break

                k_block_j = k[:, k_start:k_end, :]
                v_block_j = v[:, k_start:k_end, :]

                score_ij = (q_block_i.float() @ k_block_j.float().mT) * softmax_scale

                if is_causal:
                    k_idx = k_pos[k_start:k_end]
                    causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)

                    score_ij = score_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                local_m = score_ij.max(dim=-1).values
                m_ij = torch.maximum(m_i, local_m)

                p_local = torch.exp(score_ij - m_ij.unsqueeze(-1))
                alpha = torch.exp(m_i - m_ij)
                
                l_ij = alpha * l_i + p_local.sum(dim=-1).to(l_i.dtype)

                O_ij = alpha.unsqueeze(-1) * O_i + (p_local @ v_block_j.float())

                m_i = m_ij
                l_i = l_ij
                O_i = O_ij
            L[:, q_start:q_end] = m_i + torch.log(l_i)
            O[:, q_start:q_end, :] = O_i / l_i.unsqueeze(-1)

        if had_heads:
            O = O.reshape(B, H, N_q, D)
            L = L.reshape(B, H, N_q)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            ctx.B, ctx.H = B, H
        
        return O
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        if had_heads:
            B, H = ctx.B, ctx.H
            B_H = B * H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]

            grad_O = grad_outputs.reshape(B_H, N_q, D)
            O = O.reshape(B_H, N_q, D)
            L = L.reshape(B_H, N_q)

        else:
            B_H, N_q, D = q.shape
            N_k = k.shape[1]
            grad_O = grad_outputs
        
        device = q.device
        softmax_scale = 1.0 / math.sqrt(D)

        block_q, block_k = 64, 64
        tile_q, tile_k = math.ceil(N_q / block_q), math.ceil(N_k / block_k)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        q_pos = torch.arange(N_q, device=device)
        k_pos = torch.arange(N_k, device=device)

        D_blk = torch.sum(grad_O * O, dim=-1)

        for i in range(tile_q):
            q_start, q_end = i * block_q, min((i + 1) * block_q, N_q)

            q_block_i = q[:, q_start:q_end, :]
            grad_O_block_i = grad_O[:, q_start:q_end, :]
            L_block_i = L[:, q_start:q_end]
            D_block_i = D_blk[:, q_start:q_end]

            q_idx = q_pos[q_start:q_end]

            for j in range(tile_k):
                k_start, k_end = j * block_k, min((j + 1) * block_k, N_k)

                if is_causal and k_start >= q_end:
                    break

                k_block_j = k[:, k_start:k_end, :]
                v_block_j = v[:, k_start:k_end, :]

                scores_ij = (q_block_i.float() @ k_block_j.float().mT) * softmax_scale

                if is_causal:
                    k_idx = k_pos[k_start:k_end]
                    causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
                    scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                P_ij = torch.exp(scores_ij - L_block_i.unsqueeze(-1))

                grad_v[:, k_start:k_end, :] += torch.bmm(P_ij.mT,
                                                         grad_O_block_i.float()).to(grad_v.dtype)
                grad_P = torch.bmm(grad_O_block_i.float(), v_block_j.float().mT)

                grad_S = P_ij * (grad_P - D_block_i.unsqueeze(-1))

                grad_q[:, q_start:q_end, :] += (torch.bmm(grad_S, k_block_j.float()) * softmax_scale).to(grad_q.dtype)

                grad_k[:, k_start:k_end, :] += (torch.bmm(grad_S.mT, q_block_i.float()) * softmax_scale).to(grad_k.dtype)

        if had_heads:
            grad_q = grad_q.reshape(B, H, N_q, D)
            grad_k = grad_k.reshape(B, H, N_k, D)
            grad_v = grad_v.reshape(B, H, N_k, D)

        return grad_q, grad_k, grad_v, None
