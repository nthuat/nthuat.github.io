---
layout: post
title: Building FlashAttention from Scratch on an A10G - What the Numbers Actually Say
---

I spent the last two weeks building FlashAttention from scratch in Triton. Not to use it in production - vLLM already ships a better one. I built it to understand what "IO-aware" actually means, why the online softmax trick works, and what autotuning does on a real GPU.

Here's what I found out.

## The problem with naive attention

Standard attention looks simple:

```
O = softmax(Q @ K.T / sqrt(d)) @ V
```

The issue is that NxN matrix. At seq_len=8192 and float16, that's 8192² × 2 bytes = 128 MB just for the attention weights - per head. With 32 heads you're at 4 GB. The bottleneck isn't the math, it's reading and writing that matrix to and from GPU memory (HBM).

## The FlashAttention idea

The paper (Tri Dao et al., 2022) is 26 pages. After reading it I could explain the algorithm - but I couldn't tell you why BLOCK_Q=128 might be slower than BLOCK_Q=64, or why we're only hitting 22% of peak bandwidth. Writing the code forced me to actually figure that out.

Instead of computing the full NxN matrix and writing it to HBM, FlashAttention tiles Q into blocks and iterates over K/V blocks, computing the output incrementally. The Q block stays in SRAM (fast on-chip memory) the whole time. No NxN materialization.

![FlashAttention tiling memory layout](/assets/flashattention_tiling.svg)

The trick that makes this work is **online softmax** - you normally need two passes (one to find the max for numerical stability, one to compute exp and normalize). Online softmax does it in one pass with running accumulators:

```python
# for each K/V block:
m_new = max(m, row_max(S))       # update running max
alpha  = exp(m - m_new)          # rescale factor for previous blocks
P      = exp(S - m_new)          # softmax numerator for this block
l      = l * alpha + row_sum(P)  # running denominator
O      = alpha * O + P @ V       # accumulate output
m      = m_new
```

At the end: `O = O / l`. Same result as standard softmax, never wrote the NxN matrix anywhere.

![FlashAttention online softmax per K-block iteration](/assets/flashattention_online_softmax.svg)

## Implementation

Full Triton kernel:

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['seq_len', 'HEAD_DIM'],
)
@triton.jit
def flash_attention_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        seq_len,
        stride_b, stride_h,
        num_heads,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
):
    bh_idx      = tl.program_id(0)
    q_block_idx = tl.program_id(1)
    batch_idx   = bh_idx // num_heads
    head_idx    = bh_idx % num_heads
    bh_offset   = batch_idx * stride_b + head_idx * stride_h

    q_offset   = q_block_idx * BLOCK_Q * HEAD_DIM
    q_row_offs = tl.arange(0, BLOCK_Q)
    q_col_offs = tl.arange(0, HEAD_DIM)
    q_offsets  = q_row_offs[:, None] * HEAD_DIM + q_col_offs[None, :]
    Q = tl.load(Q_ptr + bh_offset + q_offset + q_offsets)

    m = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l = tl.zeros([BLOCK_Q],              dtype=tl.float32)
    O = tl.zeros([BLOCK_Q, HEAD_DIM],    dtype=tl.float32)

    for k_block_idx in range(seq_len // BLOCK_K):
        k_offset   = k_block_idx * BLOCK_K * HEAD_DIM
        k_row_offs = tl.arange(0, BLOCK_K)
        k_col_offs = tl.arange(0, HEAD_DIM)
        k_offsets  = k_row_offs[:, None] * HEAD_DIM + k_col_offs[None, :]
        K = tl.load(K_ptr + bh_offset + k_offset + k_offsets)
        V = tl.load(V_ptr + bh_offset + k_offset + k_offsets)

        S = tl.dot(Q, tl.trans(K)) / tl.sqrt(float(HEAD_DIM))

        if IS_CAUSAL:
            q_idx = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
            k_idx = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            S = tl.where(q_idx[:, None] >= k_idx[None, :], S, float('-inf'))

        m_new = tl.maximum(m, tl.max(S, axis=1))
        alpha  = tl.exp(m - m_new)
        P      = tl.exp(S - m_new[:, None])
        l      = l * alpha + tl.sum(P, axis=1)
        O      = alpha[:, None] * O + tl.dot(P, V)
        m      = m_new

    O = O / l[:, None]
    o_offset   = q_block_idx * BLOCK_Q * HEAD_DIM
    o_row_offs = tl.arange(0, BLOCK_Q)
    o_col_offs = tl.arange(0, HEAD_DIM)
    o_offsets  = o_row_offs[:, None] * HEAD_DIM + o_col_offs[None, :]
    tl.store(O_ptr + bh_offset + o_offset + o_offsets, O)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = Q.shape
    assert head_dim in (32, 64, 128), f"head_dim must be 32, 64, or 128"
    grid = lambda meta: (batch * num_heads, triton.cdiv(seq_len, meta['BLOCK_Q']))
    O = torch.empty_like(Q)
    flash_attention_kernel[grid](
        Q, K, V, O, seq_len,
        Q.stride(0), Q.stride(1),
        num_heads,
        HEAD_DIM=head_dim, IS_CAUSAL=causal,
    )
    return O
```

Correctness check against `F.scaled_dot_product_attention`:

```python
import torch.nn.functional as F

batch, num_heads, seq_len, head_dim = 2, 4, 256, 64
Q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
K = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
V = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")

out = flash_attention(Q, K, V, causal=False)
expected = F.scaled_dot_product_attention(Q, K, V)
print("causal=False max error:", (out - expected).abs().max().item())
assert torch.allclose(out, expected, atol=1e-2)

out_causal = flash_attention(Q, K, V, causal=True)
expected_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print("causal=True  max error:", (out_causal - expected_causal).abs().max().item())
assert torch.allclose(out_causal, expected_causal, atol=1e-2)
```

```
causal=False max error: 1.5e-03  ✓
causal=True  max error: 2.4e-03  ✓
```

Larger than float64 precision but well within `atol=1e-2` - expected for float32 online softmax.

## Causal mask

For decoder attention, each token can only see earlier positions. One extra check per K block:

```python
q_idx = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
k_idx = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
S = tl.where(q_idx[:, None] >= k_idx[None, :], S, float('-inf'))
```

Setting future positions to `-inf` means `exp(-inf) = 0`, so they contribute nothing. Online softmax handles this correctly without any special casing.

One thing I noticed: when an entire K block is in the future (all positions masked), the kernel still runs the full computation. FlashAttention v2 skips those blocks entirely - that's a big part of why v2 is ~2x faster on causal models.

## Autotuning

I tested 6 configs. The A10G picked BLOCK_Q=64, BLOCK_K=64, num_warps=4 every time.

| BLOCK_Q | BLOCK_K | num_warps | result   |
|---------|---------|-----------|----------|
| 64      | 64      | 4         | **best** |
| 128     | 64      | 4         | slower   |
| 64      | 128     | 4         | slower   |
| 128     | 128     | 4         | slower   |
| 128     | 64      | 8         | slower   |
| 128     | 128     | 8         | slower   |

I expected BLOCK_Q=128, BLOCK_K=128 to win on the A10G - it has 96 KB of shared memory per SM, so larger blocks fit:

```
BLOCK_Q=128, BLOCK_K=128:
  Q: 128 x 64 x 4 bytes = 32 KB
  K: 128 x 64 x 4 bytes = 32 KB
  V: 128 x 64 x 4 bytes = 32 KB
  total = 96 KB  fits exactly
```

But it still lost. The bottleneck shifted from SRAM to **registers**.

Each SM has 65,536 registers total, shared across all resident warps. The `O` accumulator - the running output that stays live across the entire K-loop - eats most of it:

```
BLOCK_Q=128: O is [128, 64] float32 = 8,192 values = 8,192 registers per warp
BLOCK_Q=64:  O is [64,  64] float32 = 4,096 values = 4,096 registers per warp
```

Adding other live values (Q block, m, l, loop counters, pointers):

```
BLOCK_Q=128: ~10,000 registers per warp → 65,536 / 10,000 ≈ 6 warps fit on SM
BLOCK_Q=64:  ~5,000  registers per warp → 65,536 / 5,000  ≈ 13 warps fit on SM
```

Fewer warps means less **latency hiding**. HBM (the GPU's main memory) has ~300 cycle latency. Every time a warp issues a memory load for a K or V block, it stalls for those 300 cycles. The GPU hides this by switching to another ready warp - but only if there are enough warps resident on the SM.

With 6 warps, when one stalls on a memory load there are fewer warps to switch to. The SM sits idle waiting. With 13 warps, the SM stays busy with other warps' compute while the first warp waits.

Now you might think: BLOCK_Q=128 does 2x more output computation per warp, so shouldn't it break even? The problem is the dominant cost isn't the output accumulation - it's the K and V memory loads every inner loop iteration. Both block sizes pay the same K/V load cost per output row. BLOCK_Q=128 gets 2x the compute but also 2x the register pressure, and the latency hiding loss wins.

So even with 96 KB SRAM, the winner is still BLOCK_Q=64. The constraint just moved from SRAM to registers.

## Numbers

A10G, float32, 25 warmup + 100 rep:

| batch | heads | seq  | causal | ms    | GB/s  |
|-------|-------|------|--------|-------|-------|
| 1     | 1     | 256  | no     | 0.013 |  19.5 |
| 1     | 1     | 1024 | no     | 0.036 |  29.2 |
| 2     | 4     | 256  | no     | 0.016 | 132.3 |
| 2     | 4     | 1024 | no     | 0.077 | 109.3 |
| 2     | 4     | 256  | yes    | 0.016 | 128.2 |
| 2     | 4     | 1024 | yes    | 0.078 | 106.9 |

Peak: 132.3 GB/s out of 600 GB/s - 22% of theoretical max.

## Why only 22%?

Two things.

**Tensors are too small.** batch=2, heads=4, seq=256, head_dim=64 in float32 is about 1 MB total. The A10G can push 600 GB/s with large sustained transfers. With 1 MB, you're mostly paying for kernel launch overhead and HBM latency. seq=1024 is better (109 GB/s) but still not enough data to saturate the bus.

**Register pressure limits occupancy.** Inside the K loop, `Q` and `O` are both [64, 64] float32 - that's 32 KB of registers just for those two. With that much register pressure, only a few warps can live on each SM at once. When a warp stalls on a memory load, there aren't enough other warps to fill the gap.

FlashAttention v2 fixes this by changing the loop structure to reduce the live register set. v3 on H100 goes further - separate "producer" warps handle memory loads while "consumer" warps do math, both running in parallel. That's where the other 78% lives.

## What I got out of this

Getting the kernel correct took a day. Getting it to pass the numerical tests took another day (the causal masking had an off-by-one I kept missing). Then a week of reading to understand why the numbers are what they are.

The SRAM vs register insight was the most useful thing - I went in thinking "bigger blocks = better" and came out understanding that on Ampere, SRAM is no longer the constraint. It shifted to registers, and that requires a different algorithm (v2), not just different block sizes.

132 GB/s sounds decent until you realize vLLM's FlashAttention hits much closer to peak. The gap is real, and now I understand where it comes from.

## Reproducing

```python
configs = [
    (1, 1,  256, 64, False),
    (1, 1, 1024, 64, False),
    (2, 4,  256, 64, False),
    (2, 4, 1024, 64, False),
    (2, 4,  256, 64, True),
    (2, 4, 1024, 64, True),
]

print(f"{'batch':>5} {'heads':>5} {'seq':>6} {'dim':>5} {'causal':>7} {'ms':>8} {'GB/s':>8}")
print("-" * 55)
for batch, heads, seq_len, head_dim, causal in configs:
    Q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    K = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    V = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    ms = triton.testing.do_bench(lambda: flash_attention(Q, K, V, causal=causal), warmup=25, rep=100)
    total_bytes = 4 * batch * heads * seq_len * head_dim * 4  # 4 tensors, float32 = 4 bytes
    gb_s = (total_bytes / 1e9) / (ms / 1e3)
    print(f"{batch:>5} {heads:>5} {seq_len:>6} {head_dim:>5} {str(causal):>7} {ms:>8.3f} {gb_s:>8.1f}")
```

Needs Triton >= 2.0, CUDA GPU, head_dim in (32, 64, 128), seq_len divisible by 64.

## References

- Tri Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- Tri Dao et al. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- Triton documentation. [https://triton-lang.org](https://triton-lang.org)
