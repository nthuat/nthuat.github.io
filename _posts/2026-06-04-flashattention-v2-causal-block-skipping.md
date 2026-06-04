---
layout: post
title: FlashAttention v2 - Where the 1.7x Speedup Actually Comes From
---

Two weeks ago I [built FlashAttention v1 from scratch in Triton]({% post_url 2026-05-24-flash-attention-from-scratch %}) and got 22% of peak bandwidth on an A10G. The v1 to v2 paper claims ~2x faster on causal models. I wanted to know exactly where that speedup comes from - so I added the one change v2 actually needs and measured.

Here's what I found.

## The one change that matters for causal models

In v1, every Q block iterates over every K block, then masks out future positions with `-inf`:

```python
for k_block_idx in range(seq_len // BLOCK_K):
    # ... load K, V, compute S
    if IS_CAUSAL:
        S = tl.where(q_idx[:, None] >= k_idx[None, :], S, float('-inf'))
    # ... online softmax
```

For Q block 0 at seq_len=4096, BLOCK_K=64, there are 64 K blocks. Block 0 only needs the first 1 (positions 0-63 can only see positions 0-63). The other 63 K blocks load, compute QK^T, set everything to `-inf`, then contribute zero. Pure wasted work.

v2's fix is one line:

```python
k_blocks = tl.cdiv((q_block_idx + 1) * BLOCK_Q, BLOCK_K) if IS_CAUSAL else seq_len // BLOCK_K

for k_block_idx in range(k_blocks):
    # ...
```

For Q block `i`, only iterate over K blocks whose last token index `<= (i+1) * BLOCK_Q - 1`. Everything past that is guaranteed to be masked.

That's it. The online softmax stays identical, the load logic stays identical, the autotune configs stay identical.

## Numbers on A10G

seq=4096, batch=2, heads=4, head_dim=64, causal=True:

| seq_len | v1 ms | v2 ms | speedup |
|---------|-------|-------|---------|
| 512     | 0.097 | 0.092 | 1.06x   |
| 1024    | 0.231 | 0.182 | 1.27x   |
| 2048    | 0.660 | 0.394 | 1.68x   |
| 4096    | 2.347 | 1.378 | **1.70x** |

The speedup grows with seq_len. At seq=512 it barely helps - block 0 still skips a few K blocks, but block 7 (the last) still processes everything. At seq=4096 you skip half the K work on average, which is exactly the theoretical max for causal attention.

Why not 2x? The **last Q block always processes all K blocks**. It can see every position in the sequence, so no K blocks are skippable. As the number of Q blocks grows, the unskippable last block becomes a smaller fraction of total work - which is why the speedup approaches but never quite reaches 2x.

## Then I went chasing occupancy

After confirming the speedup, I wanted to know why even v2 still leaves performance on the table. The answer turned out to be the same thing that limited v1: register pressure.

Each SM on the A10G has 65,536 registers, shared across all resident warps. Maximum theoretical occupancy is 48 warps per SM. Triton showed me what we actually get:

```
BLOCK_Q=64:  138 regs/thread -> 14 warps/SM -> 29.2% occupancy
BLOCK_Q=128: 247 regs/thread -> 8  warps/SM -> 16.7% occupancy
```

To hit full 48-warp occupancy you'd need ≤42 registers per thread. FA2 with BLOCK_Q=64 needs 138 - more than 3x the budget. The `O` accumulator alone (BLOCK_Q × HEAD_DIM × float32) costs a lot of register state, and it has to stay live across the entire K-block loop because online softmax keeps rescaling it.

The takeaway: SRAM stopped being the limiter on Ampere, and v2 didn't change that. What it changed is the *amount of compute* per Q block in causal mode. Both bottlenecks - low occupancy and partial work - hit simultaneously in v1. v2 removes the work-side bottleneck, occupancy stays the same.

## What v3 does that v2 doesn't

v3 (Hopper-only) explicitly attacks the occupancy problem with **warp specialization**:

- Producer warps just handle memory loads (low register pressure, lots of them resident)
- Consumer warps just do compute (high register pressure, fewer of them, but the producers keep the data flowing)

Two roles instead of one means you can have many lightweight memory-loading warps hiding latency for a few heavy compute warps. On H100 this gets you to ~75% of peak. On A10G you can't do this - no async warp specialization, no TMA. You're stuck with v2.

That gap - 29% vs 75% - is the difference between consumer GPUs and datacenter GPUs that AI infra teams actually care about.

## What I got out of this

The 1.7x came from one line of code. The interesting part wasn't the change itself - it was running the speedup curve and seeing it asymptote at 2x for the structural reason (last Q block).

Occupancy analysis was the bigger lesson. I went in expecting v2 to also help with register pressure. It doesn't. The algorithm changes between v1 and v2 are all about *what work you do*, not *how many warps stay resident*. Solving occupancy on attention requires a different architecture entirely - which is what v3 brings, but only on hardware that supports it.

Next: I want to port the same causal block skipping logic into the prefill path inside SGLang's `context_attention_fwd` Triton kernel and measure end-to-end on Llama-3 prefill.

## References

- Tri Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- Jay Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. [https://arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)
- NVIDIA Ampere whitepaper, register file and warp scheduling. [https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
