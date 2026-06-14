"""
=============================================================================
PyTorch Flash Attention — PyTorch-Autograd-Compatible Reference Implementation
=============================================================================

This file implements the **Flash Attention** algorithm (Dao et al., 2022) using
a pure-Python, block‑wise (tiled) approach that is registered as a custom
`torch.autograd.Function`.  It is a *pedagogical reference* — not the fastest
possible implementation (the real Flash‑Attention kernels are written in
CUDA C++), but it faithfully reproduces the **numerics** and the **I/O‑optimal
tiling strategy** of the original algorithm, while still being fully
differentiable through PyTorch’s autograd engine.

⛔️ IMPORTANT: "Wait, isn't this just a sequential for loop? Where's the parallelism?"
-----------------------------------------------------------------------------

YES — the forward/backward loops in THIS file (`pytorch_version.py`) are pure
Python ``for`` loops over blocks.  They run **sequentially** on a single CPU
thread (with individual block ops dispatched to the GPU one at a time via
PyTorch eager execution).  There is **no block‑level parallelism** here.

So why write it this way?  Two reasons:

1. **Pedagogy**: the nested-loop structure directly mirrors Algorithm 1 of
   the FlashAttention‑2 paper.  It is the clearest way to explain what Flash
   Attention *does* — iterate over Q blocks, inside each iterate over K/V
   blocks, maintain online softmax statistics, never materialise S or P.

2. **Correctness reference**: this implementation is much easier to verify
   against a naive attention (``S = Q@K^T; P = softmax(S); O = P@V``) than a
   CUDA/Triton kernel, so it serves as a ground‑truth for testing.

Where is the real parallelism?  Look at `triton_version.py` in the same
directory.  In Triton, the **outer loop over Q blocks is replaced by a GPU
grid launch**::

    grid = (B_eff, triton.cdiv(N_Q, BLOCK_Q))
    _flash_fwd_kernel[grid](...)

Each (batch‑head, Q‑block) pair becomes a **separate GPU program (kernel)**
that runs **in parallel** across all SMs (streaming multiprocessors).  Inside
each kernel, the inner loop over K/V blocks still exists, but it runs on
GPU SRAM with tensor‑core matmuls (``tl.dot``).  The inner loop is sequential
within one kernel, but thousands of such kernels execute concurrently.

So the architecture is:

- **This file** (pytorch_version.py):  1 CPU thread, loop over Q blocks
  sequentially; each block's inner loop dispatches PyTorch ops to GPU.
  **O(T_q) kernel launches**, no block‑level parallelism.

- **Triton version** (triton_version.py):  T_q parallel GPU kernels, each
  kernel's inner loop runs entirely on‑chip.  **O(1) kernel launches** (one
  grid), all Q blocks computed concurrently.

- **CUDA version** (real FA‑2): same as Triton but hand‑written in CUDA C++
  with even finer control over shared memory and warp scheduling.

-----------------------------------------------------------------------------
MOTIVATION: Why does Flash Attention exist?
-----------------------------------------------------------------------------

Standard scaled dot‑product attention computes::

    S = Q @ K^T / sqrt(d)          # [B, N, N]  ← huge!
    P = softmax(S, dim=-1)         # [B, N, N]  ← still huge
    O = P @ V                      # [B, N, D]

The intermediate matrices **S** and **P** have shape `[batch, N, N]`, where
`N` is the sequence length.  For a sequence of 8K tokens, that is an
8K × 8K = 64M‑element matrix **per head per batch item**.  Storing it in GPU
HBM (high‑bandwidth memory) is extremely expensive and often the **bottleneck**
in transformer training/inference.

**Flash Attention’s key insight:** you don’t ever need to materialise the full
`S` or `P` matrices in HBM.  Instead you can:

1. **Tile** Q, K, V into small blocks that fit in on‑chip SRAM.
2. Use **online softmax** to accumulate the output O incrementally, block‑by‑block.
3. Fuse the entire attention computation (matmul → scale → mask → softmax →
   matmul) into a single CUDA kernel (here a single custom autograd Function).

The result: **O(N²) memory → O(N) memory**, and much faster wall‑clock time
because the bottleneck shifts from HBM bandwidth to compute.

-----------------------------------------------------------------------------
BACKGROUND: torch.autograd.Function vs nn.Module
-----------------------------------------------------------------------------

You are probably familiar with `nn.Module`:

    class MyAttention(nn.Module):
        def forward(self, q, k, v):
            ...
            return output

`nn.Module.forward()` is **eager**: every PyTorch operation (e.g. `@`, `+`,
`softmax`, ...) builds a node in the autograd graph.  The intermediate
activations (`S`, `P`) are saved for backward, which is exactly what we want
to **avoid** in Flash Attention.

`torch.autograd.Function` gives us a way to **define a custom forward and
backward pass as a single "black box" op**.  PyTorch only sees the inputs
(q, k, v) and the output O.  It does **not** record the internal steps
(block‑wise matmuls, online softmax updates) in the autograd graph, so
intermediate tensors are **not** saved — **we** manually implement the backward
pass and recompute whatever we need.

This file contains one such custom Function: **PyTorchFlashAttention**.

-----------------------------------------------------------------------------
STRUCTURE OF THIS FILE
-----------------------------------------------------------------------------

- `forward()`  — tiled Flash Attention forward pass with online softmax.
- `backward()` — manually derived gradients, also tiled, with recomputation.

The logic closely follows Algorithm 1 and Algorithm 2 from the FlashAttention‑2
paper (Dao, 2023).

-----------------------------------------------------------------------------
NUMERICAL NOTE
-----------------------------------------------------------------------------

All accumulation and softmax math is done in **float32** regardless of the
input dtype.  This matches what FA‑2 does internally and guarantees good
numerical stability.  The final output is returned in the **same dtype as the
inputs** (handled implicitly by `ctx.save_for_backward` + subsequent ops).

=============================================================================
"""

import math

import torch


class PyTorchFlashAttention(torch.autograd.Function):
    """
    Custom PyTorch autograd Function that computes **Flash Attention**.

    Usage (like any other PyTorch function)::

        output = PyTorchFlashAttention.apply(q, k, v, is_causal=True)

    `apply()` is inherited from `torch.autograd.Function`; it invokes
    `forward()` during the forward pass and, when gradients are needed,
    `backward()` during the backward pass.

    Shapes
    ------
    q : (B, S, D)   or   (B, H, S, D)   — queries
    k : (B, S, D)   or   (B, H, S, D)   — keys
    v : (B, S, D)   or   (B, H, S, D)   — values
    is_causal : bool — if True, mask out positions where q_idx < k_idx
                      (used in autoregressive/decoder models like GPT)

    Returns
    -------
    O : same shape as q — the attention output
    """

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        """
        Flash Attention forward pass — tiled, online‑softmax version.

        The standard attention formula is::

            S   = Q @ K^T / sqrt(d)       # pre-softmax scores
            P   = softmax(S, dim=-1)       # attention weights (row-wise)
            O   = P @ V                    # output

        We want to compute O **without ever storing the full S or P matrices**.
        We do this by:

        1. Splitting Q into blocks of rows (B_q rows at a time).
        2. For each Q block, iterating over K/V blocks (B_k rows at a time).
        3. Maintaining three running statistics per Q‑row:
           - m : running max of the scores (for numerical stability)
           - l : running sum of exp(scores - m)
           - O : running weighted sum of V

        This is called **online softmax** (or "lazy softmax").  At the end of
        all K/V blocks for a given Q block, we normalise::

            O = O / l          (which equals sum_i P_i @ V_i)

        and store the log‑sum‑exp L = m + log(l) for use in the backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object — the "backpack" where we stash tensors and metadata
            that `backward()` will need later.
        q, k, v : torch.Tensor
        is_causal : bool

        Returns
        -------
        O : torch.Tensor — attention output
        """
        # ================================================================
        # Step 0: Handle optional head dimension + bookkeeping
        # ================================================================
        # We support two input shapes:
        #   (B,  S, D)   — single-head or already merged
        #   (B, H, S, D) — multi-head
        # Internally we always work with 3‑D tensors: (B_eff, S, D),
        # where B_eff = B * H (heads are folded into the batch dimension).
        had_heads = q.dim() == 4
        if had_heads:
            B, H, N_q, D = q.shape
            _, _, N_k, _ = k.shape
            # Fold heads into batch for the block-wise implementation
            q = q.reshape(B * H, N_q, D)
            k = k.reshape(B * H, N_k, D)
            v = v.reshape(B * H, N_k, D)
            B_eff = B * H
        else:
            B, N_q, D = q.shape
            _, N_k, _ = k.shape
            B_eff = B
        device = q.device

        # Scale factor for dot‑product attention: 1 / sqrt(d)
        # This prevents the dot products from growing too large in magnitude,
        # which would push softmax into regimes of extreme saturation (0 or 1).
        scale = 1.0 / math.sqrt(D)

        # ================================================================
        # Step 1: Define block (tile) sizes
        # ================================================================
        # B_q, B_k are hyperparameters chosen to fit in SRAM.
        # Typical values: 64 or 128.  Here we use 64×64 tiles.
        B_q, B_k = 64, 64  # block sizes (rows per tile)
        # Number of query / key blocks along the sequence dimension.
        # We use ceil because the last block may be smaller (padding is not needed).
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)  # number of blocks

        # ================================================================
        # Step 2: Allocate output + log‑sum‑exp buffers
        # ================================================================
        # O: the final attention output, accumulated in fp32 for stability
        # L: per‑token log‑sum‑exp: L_i = m_i + log(l_i)
        #    L is stored for the backward pass because d(softmax) can be
        #    expressed in terms of L without needing the full P matrix.
        # Both are allocated as empty tensors; we fill them block‑by‑block.
        O = torch.empty((B_eff, N_q, D), device=device, dtype=torch.float32)
        L = torch.empty((B_eff, N_q), device=device, dtype=torch.float32)

        # ================================================================
        # Step 3: Precompute position indices for causal masking
        # ================================================================
        # For causal (autoregressive) attention, query position i can only
        # attend to key positions j where j ≤ i.  We pre‑build the position
        # vectors once and slice them per block.
        q_pos = torch.arange(N_q, device=device)  # [0, 1, 2, ..., N_q-1]
        k_pos = torch.arange(N_k, device=device)  # [0, 1, 2, ..., N_k-1]

        # ================================================================
        # Step 4: Outer loop over query blocks  ← THIS IS THE KEY ALGORITHM
        # ================================================================
        # For each query tile (a contiguous range of query positions), we:
        #   1. Load the Q tile into "fast memory" (here, just a variable).
        #   2. Iterate over all K/V tiles.
        #   3. For each K/V tile, compute a partial softmax and update the
        #      running statistics (m, l, O).
        #   4. After all K/V tiles are processed, normalise and write the
        #      result back to the global O and L buffers.
        #
        # This is the "online softmax" / "tiled softmax" algorithm:
        #
        #   Initialise:  m = -inf,  l = 0,  O = 0
        #   For each K/V block j:
        #       S_j   = Q @ K_j^T / sqrt(d)         # tile of scores
        #       m_new = max(m, max(S_j, axis=-1))   # updated running max
        #       P~_j  = exp(S_j - m_new)            # unnormalised weights
        #       α     = exp(m - m_new)              # rescaling factor for old sum
        #       l_new = α * l + sum(P~_j, axis=-1)  # updated running sum
        #       O_new = α * O + P~_j @ V_j          # updated running output
        #       m, l, O = m_new, l_new, O_new
        #   Output:   O = O / l                     # normalise
        #             L = m + log(l)                # log‑sum‑exp for backward

        for i in range(T_q):  # For each query block
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
            q_len = q_end - q_start  # actual size (last block may be smaller)

            # ---- Load Q Tile ----
            # Shape: (B_eff, q_len, D)
            Q_blk_i = q[:, q_start:q_end, :]

            # ---- Initialise online softmax states for this Q block ----
            # m_i: running maximum of attention scores (per query row)
            #      Initialised to -inf so the first block's max becomes the initial m.
            # l_i: running sum of exp(scores - m) (per query row), i.e. the
            #      denominator of softmax before final division.
            #      Initialised to 0.
            # O_i: running weighted sum of V (per query row), i.e. the numerator of
            #      attention output before final division.
            #      Initialised to 0.
            m_i = torch.full(
                (B_eff, q_len), -float("inf"), device=device, dtype=torch.float32
            )
            l_i = q.new_zeros((B_eff, q_len), dtype=torch.float32)
            O_i = q.new_zeros((B_eff, q_len, D), dtype=torch.float32)

            # ---- Precompute query position indices for this block ----
            # These are used to build the causal mask inside the inner loop.
            q_idx = q_pos[q_start:q_end]  # shape: (q_len,)

            # ============================================================
            # Inner loop over key/value blocks
            # ============================================================
            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

                # ---- Causal short‑circuit ----
                # For causal attention, if the earliest key position in this
                # K/V block (k_start) is ≥ the last query position (q_end),
                # then the entire block is fully masked (all scores = -inf).
                # Since K blocks are processed in increasing order of position,
                # all subsequent blocks will also be fully masked — we can break.
                if is_causal and k_start >= q_end:
                    break

                # ---- Load K, V Tiles ----
                # Each tile is a contiguous slice of K/V along the sequence dim.
                # Shape: (B_eff, k_len, D)
                k_blk_j = k[:, k_start:k_end, :]
                v_blk_j = v[:, k_start:k_end, :]

                # ---- Compute tile of pre-softmax attention scores ----
                # scores_ij[b, p, r] = <Q_blk_i[b, p, :], k_blk_j[b, r, :]> / sqrt(d)
                #
                # Step breakdown:
                #   k_blk_j.float().transpose(-1, -2)  → (B_eff, D, k_len)
                #   Q_blk_i.float() @ ...               → (B_eff, q_len, k_len)
                #   * scale                              → scaled scores
                #
                # We cast to float32 before matmul for numerical precision.
                scores_ij = (
                    Q_blk_i.float() @ k_blk_j.float().transpose(-1, -2)
                ) * scale  # (B_eff, q_len, k_len)

                # ---- Apply causal mask (if needed) ----
                # Causal constraint: query position p can only attend to key
                # position r if p ≥ r (i.e., the query looks "backward" only).
                # We build a boolean mask of shape (q_len, k_len) where
                # mask[p, r] = True (keep) iff q_idx[p] >= k_idx[r].
                # Then we fill masked positions with -inf so that exp(-inf) = 0
                # in the subsequent softmax step.
                if is_causal:
                    k_idx = k_pos[k_start:k_end]  # shape: (k_len,)
                    # Broadcasting: (q_len, 1) >= (1, k_len) → (q_len, k_len)
                    causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
                    # ~causal_mask inverts so masked positions get -inf
                    # unsqueeze(0) adds the batch dimension for broadcasting
                    scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # ---- ONLINE SOFTMAX UPDATE ----
                # At this point we have a new tile of scores S_j.
                # We want to incorporate S_j into our running softmax statistics.

                # **Step A: Update running maximum**
                # m_ij[b, p] = max(m_i[b, p], max_{r}(scores_ij[b, p, r]))
                # This is element‑wise maximum: for each query row, take the
                # larger of the old running max and the new tile's max.
                m_ij = torch.maximum(m_i, scores_ij.max(dim=-1).values)  # (B_eff, q_len) 

                # **Step B: Compute unnormalised attention weights for this tile**
                # p~_ij = exp(scores_ij - m_ij)
                # Subtracting m_ij (the updated max) from scores before exp
                # is the standard "safe softmax" trick: it prevents overflow
                # and keeps the largest value at exp(0) = 1.
                # At this stage these are UNNORMALISED — we still need to
                # combine them with previous tiles' contributions.
                p_tilde_ij = torch.exp(scores_ij - m_ij.unsqueeze(-1))  # (B_eff, q_len, k_len)

                # **Step C: Compute rescaling factor α**
                # α = exp(m_old - m_new)
                # Because m_new = max(m_old, max(S_new)), we have:
                #   - If m_old == m_new:  α = exp(0) = 1 (no rescaling needed)
                #   - If m_old < m_new:   α = exp(negative) ∈ (0, 1)
                # This rescales the OLD running sum l_i to account for the
                # fact that we subtracted a larger m_ij in the exp of the
                # previous blocks.
                alpha = torch.exp(m_i - m_ij)  # (B_eff, q_len)

                # **Step D: Update running sum of exp (softmax denominator)**
                # l_ij = α * l_old + sum(p~_ij, axis=-1)
                #   - α * l_old: rescale the old sum to the new m
                #   - sum(p~_ij, axis=-1): add the new tile's contribution
                l_ij = alpha * l_i + p_tilde_ij.sum(dim=-1).to(l_i.dtype)  # (B_eff, q_len)

                # **Step E: Update running output (softmax numerator)**
                # O_ij = α * O_old + p~_ij @ V_j
                #   - α * O_old: rescale the old weighted sum of V
                #   - p~_ij @ V_j: add the new tile's weighted V contribution
                #     Shape: (B_eff, q_len, k_len) @ (B_eff, k_len, D) → (B_eff, q_len, D)
                O_ij = alpha.unsqueeze(-1) * O_i + (p_tilde_ij @ v_blk_j.float())  # (B_eff, q_len, D)

                # **Step F: Update running states for the next K/V block**
                m_i = m_ij
                l_i = l_ij
                O_i = O_ij

            # ================================================================
            # End of inner loop: all K/V blocks for this Q block are processed
            # ================================================================
            # Now O_i contains the numerator, l_i contains the denominator.
            # Final O for this Q block:  O = O_i / l_i
            # (broadcast l_i over the feature dimension D with unsqueeze(-1))
            #
            # L (log‑sum‑exp) is stored for backward:  L = m + log(l)
            # This is because softmax(x)_i = exp(x_i - L), and we need L
            # in the backward pass to recompute P without storing it.
            L[:, q_start:q_end] = m_i + torch.log(l_i)  # (B_eff, q_len)
            O[:, q_start:q_end, :] = O_i / l_i.unsqueeze(-1)  # (B_eff, q_len, D)

        # ================================================================
        # Step 5: Restore original shape + save for backward
        # ================================================================
        if had_heads:
            O = O.reshape(B, H, N_q, D)
            L = L.reshape(B, H, N_q)
        # `save_for_backward` is the mechanism by which `torch.autograd.Function`
        # preserves tensors from forward to be used in backward.  Only tensors
        # saved this way are accessible via `ctx.saved_tensors` in backward().
        # Critically, we do NOT save the full attention matrix P — only O and L,
        # which have much smaller memory footprints (O(N) vs O(N²)).
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        ctx.had_heads = had_heads
        if had_heads:
            ctx.B, ctx.H = B, H  # stash original batch/head counts for reshape

        return O

    # ------------------------------------------------------------------
    # BACKWARD PASS
    # ------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_O: torch.Tensor):
        """
        Manually compute gradients of Flash Attention with respect to q, k, v.

        The backward pass of Flash Attention also uses **tiling and recomputation**.
        We do NOT have the full attention matrix P stored from the forward pass.
        Instead, we **recompute** each P tile on the fly from the saved Q, K, V
        and the log‑sum‑exp L.

        ------------------------------------------------------------------
        Mathematical derivation (simplified):
        ------------------------------------------------------------------
        Forward:  O = softmax(S) @ V,   where S = Q @ K^T / sqrt(d)

        Let P = softmax(S) = exp(S - L) where L is the log‑sum‑exp per row.

        Given upstream gradient dO (same shape as O), the gradients are::

            dV = P^T @ dO                         (1)
            dP = dO @ V^T                          (2)
            dS = P * (dP - diag(sum(dP * P, -1))) (3)
            dQ = dS @ K  / sqrt(d)                (4)
            dK = dS^T @ Q / sqrt(d)               (5)

        Equation (3) is the gradient of the softmax function.  The term
        `D = sum(dP * P, dim=-1) = sum(dO * O, dim=-1)` is precomputed as `D_blk`.

        Again, everything is tiled: we loop over Q blocks and K blocks,
        recompute P_ij = exp(S_ij - L_i) for each tile, and accumulate
        contributions to dQ, dK, dV.

        Parameters
        ----------
        ctx : FunctionCtx — carries saved tensors and metadata from forward.
        grad_O : torch.Tensor — upstream gradient of the loss w.r.t. O.
                 Same shape as O.

        Returns
        -------
        grad_q, grad_k, grad_v : torch.Tensor
            Gradients for q, k, v respectively.  Same shape as the inputs.
        None : for `is_causal` (a bool flag, no gradient needed).
        """
        # ---- Retrieve saved tensors and metadata from forward ----
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        had_heads = ctx.had_heads

        # ---- Handle multi‑head shape bookkeeping ----
        if had_heads:
            # q/k/v were saved in flattened (B_eff, S, D) form by forward
            B, H = ctx.B, ctx.H
            B_eff = B * H
            N_q, D = q.shape[1], q.shape[2]
            N_k = k.shape[1]

            # grad_O arrives as (B, H, S, D); O and L also as (B, H, S, ...)
            # Flatten heads into batch dimension to match our tiling loops.
            grad_O = grad_O.reshape(B_eff, N_q, D)
            O = O.reshape(B_eff, N_q, D)
            L = L.reshape(B_eff, N_q)
        else:
            B_eff, N_q, D = q.shape
            N_k = k.shape[1]

        device = q.device
        scale = 1.0 / math.sqrt(D)

        # ---- Block sizes (same as forward) ----
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        # ---- Allocate gradient tensors (initialised to zero) ----
        # Gradients for q/k/v must be the same shape as the inputs.
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # ---- Precompute position indices (same as forward) ----
        q_pos = torch.arange(N_q, device=device)
        k_pos = torch.arange(N_k, device=device)

        # ---- Precompute D = sum(dO * O, dim=-1) ----
        # This is a key quantity in the softmax backward: the sum over the
        # feature dimension of the element‑wise product of grad_O and O.
        # Shape: (B_eff, N_q)
        # It measures the "alignment" of the gradient with the output, and
        # appears in the dS formula:
        #   dS = P * (dP - D)   where D acts as a per‑row constant shift.
        D_blk = torch.sum(grad_O * O, dim=-1)  # (B_eff, N_q)

        # ================================================================
        # Outer loop over query blocks
        # ================================================================
        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)

            # ---- Load Q tile and corresponding gradient slices ----
            Q_blk_i = q[:, q_start:q_end, :]           # (B_eff, q_len, D)
            grad_O_blk_i = grad_O[:, q_start:q_end, :]  # (B_eff, q_len, D)
            L_blk_i = L[:, q_start:q_end]                # (B_eff, q_len)
            D_blk_i = D_blk[:, q_start:q_end]            # (B_eff, q_len)

            # Query position indices for causal mask in this block
            q_idx = q_pos[q_start:q_end]  # (q_len,)

            # ============================================================
            # Inner loop over key/value blocks
            # ============================================================
            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

                # ---- Causal short‑circuit (same logic as forward) ----
                if is_causal and k_start >= q_end:
                    break

                # ---- Load K, V tiles ----
                k_blk_j = k[:, k_start:k_end, :]  # (B_eff, k_len, D)
                v_blk_j = v[:, k_start:k_end, :]  # (B_eff, k_len, D)

                # ---- Recompute the score tile S_ij ----
                # This is the same computation as in forward.  We could
                # alternatively save scores and avoid recomputation, but
                # that would defeat the purpose — saving scores costs O(N²)
                # memory, which is exactly what Flash Attention avoids.
                scores_ij = (
                    Q_blk_i.float() @ k_blk_j.float().transpose(-1, -2)
                ) * scale  # (B_eff, q_len, k_len)
                if is_causal:
                    k_idx = k_pos[k_start:k_end]
                    causal_mask = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)
                    scores_ij = scores_ij.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # ---- Recompute softmax probabilities P_ij ----
                # P_ij = softmax(S_ij) = exp(S_ij - L_i)
                # L_i is the log‑sum‑exp we stored during forward.  Because
                # L_i is the normalisation constant for the FULL row (over all
                # K blocks), using it here correctly recovers the true P_ij.
                P_ij = torch.exp(scores_ij - L_blk_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

                # ---- (1) Compute gradient w.r.t. V:  dV += P^T @ dO ----
                # P_ij:        (B_eff, q_len, k_len)
                # P_ij^T:      (B_eff, k_len, q_len)   via .transpose(1,2)
                # dO:          (B_eff, q_len, D)
                # dV_contrib:  (B_eff, k_len, D)
                # We accumulate into grad_v at the positions [k_start:k_end].
                # Because the same K/V block is visited in multiple Q‑block
                # iterations, we use `+=` to sum contributions.
                grad_v[:, k_start:k_end, :] += torch.bmm(
                    P_ij.transpose(1, 2),
                    grad_O_blk_i.float(),
                ).to(grad_v.dtype)

                # ---- (2) Compute gradient w.r.t. softmax probs:  dP = dO @ V^T ----
                # grad_O_blk_i: (B_eff, q_len, D)
                # V^T:          (B_eff, D, k_len)
                # grad_P:       (B_eff, q_len, k_len)
                # This is the upstream gradient for the softmax operation:
                # d(loss)/dP[p, r] for each query position p and key position r.
                grad_P = torch.bmm(
                    grad_O_blk_i.float(),
                    v_blk_j.float().transpose(1, 2),
                )  # (B_eff, q_len, k_len)

                # ---- (3) Compute gradient w.r.t. pre‑softmax scores ----
                # This is the gradient of softmax:
                #   dS = P * (dP - D)
                # where D is the per‑row constant sum(dP * P, dim=-1).
                #
                # Intuition: softmax is over‑parameterised along each row
                # (adding a constant to all elements doesn't change the output),
                # so the gradient subtracts the row‑wise mean (weighted by P).
                # D_blk_i = sum(dO * O, dim=-1) = sum(dP * P, dim=-1) per row.
                grad_S = P_ij * (grad_P - D_blk_i.unsqueeze(-1))  # (B_eff, q_len, k_len)

                # ---- (4) Compute gradient w.r.t. Q:  dQ += dS @ K * scale ----
                # S = Q @ K^T / sqrt(d), so by the chain rule:
                #   dQ = dS @ K / sqrt(d)
                # grad_S:      (B_eff, q_len, k_len)
                # K:           (B_eff, k_len, D)
                # result:      (B_eff, q_len, D)
                # We accumulate using `+=` into the [q_start:q_end] slice.
                grad_q[:, q_start:q_end, :] += (torch.bmm(grad_S, k_blk_j.float()) * scale).to(grad_q.dtype)

                # ---- (5) Compute gradient w.r.t. K:  dK += dS^T @ Q * scale ----
                # By symmetry: S = Q @ K^T / sqrt(d), so dK = dS^T @ Q / sqrt(d)
                # grad_S^T:    (B_eff, k_len, q_len)
                # Q:           (B_eff, q_len, D)
                # result:      (B_eff, k_len, D)
                # Accumulate into the [k_start:k_end] slice of grad_k.
                grad_k[:, k_start:k_end, :] += (
                    torch.bmm(grad_S.transpose(1, 2), Q_blk_i.float()) * scale
                ).to(grad_k.dtype)

            # End of inner loop over key, value blocks
        # End of outer loop over query blocks

        # ---- Restore original multi‑head shape if needed ----
        if had_heads:
            grad_q = grad_q.reshape(B, H, N_q, D)
            grad_k = grad_k.reshape(B, H, N_k, D)
            grad_v = grad_v.reshape(B, H, N_k, D)

        # Return gradients in the same order as forward()'s non‑ctx arguments:
        # (q, k, v, is_causal).  `is_causal` is a bool → gradient is None.
        return grad_q, grad_k, grad_v, None