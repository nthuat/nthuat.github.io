---
layout: post
title: Building FlashAttention from Scratch on an A10G - What the Numbers Actually Say
---

I spent the last two weeks building FlashAttention from scratch in Triton. Not to use it in production - vLLM already ships a better one. I built it to understand what "IO-aware" actually means, why the online softmax trick works, and what autotuning does on a real GPU.

Here's what I learned and what the numbers actually say - benchmarked on an A10G.

## Why build it yourself?

The FlashAttention paper (Dao et al., 2022) is 26 pages. After reading it I could explain the algorithm. But I couldn't tell you *why* BLOCK_Q=128 might be worse than BLOCK_Q=64 on an A10G, or what 17% occupancy means, or why bandwidth is 17% of peak even on fast hardware.

The only way to know those things is to run the code and stare at the numbers.

## 1. The problem with naive attention

Standard attention computes:

```
O = softmax(Q @ K.T / sqrt(d)) @ V
```

For sequence length N and head dimension d, this materializes an N×N attention matrix in HBM (GPU DRAM). At N=8192 and float16, that's 8192² × 2 bytes = 128 MB **per head**. With 32 heads you're at 4 GB just for attention weights.

The bottleneck isn't compute - it's memory bandwidth. Reading and writing that N×N matrix dominates runtime.

## 2. Tiled FlashAttention: the key idea

FlashAttention avoids materializing the full N×N matrix. Instead it:

1. Tiles Q into blocks of BLOCK_Q rows
2. For each Q block, iterates over all K/V blocks
3. Computes the output incrementally using **online softmax**

Online softmax is the trick that makes this possible. Normal softmax needs two passes: one to find the max (for numerical stability), one to compute exp and normalize. FlashAttention does it in one pass by tracking a running max `m` and running sum `l`:

```python
# For each K/V block:
m_new = max(m, row_max(S))       # update running max
alpha  = exp(m - m_new)          # rescale factor for previous blocks
P      = exp(S - m_new)          # softmax numerator for this block
l      = l * alpha + row_sum(P)  # update running denominator
O      = alpha * O + P @ V       # update output
m      = m_new
```

At the end: `O = O / l`. The final output is numerically identical to standard softmax but computed without ever writing the full N×N matrix to HBM.

## 3. Triton implementation

Here's the full implementation:

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

    q_offset     = q_block_idx * BLOCK_Q * HEAD_DIM
    q_row_offs   = tl.arange(0, BLOCK_Q)
    q_col_offs   = tl.arange(0, HEAD_DIM)
    q_offsets    = q_row_offs[:, None] * HEAD_DIM + q_col_offs[None, :]
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
    o_offset  = q_block_idx * BLOCK_Q * HEAD_DIM
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

**Correctness check** against `F.scaled_dot_product_attention`:

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

The errors are larger than float64 precision but well within `atol=1e-2` - expected for float32 with online softmax accumulation across blocks.

## 4. Causal mask

For decoder attention, each token can only attend to tokens at its position or earlier. The mask is straightforward:

```python
q_idx = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
k_idx = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
causal_mask = q_idx[:, None] >= k_idx[None, :]
S = tl.where(causal_mask, S, float('-inf'))
```

Setting masked positions to `-inf` before the softmax means `exp(-inf) = 0`, so they contribute nothing to the output. The online softmax handles this correctly without any special casing.

One subtlety: when `q_block_idx * BLOCK_Q < k_block_idx * BLOCK_K`, the entire block is masked. The kernel still runs the computation - a real implementation (FlashAttention v2) skips these blocks entirely, which is part of why v2 is ~2× faster for causal models.

## 5. Autotuning

`@triton.autotune` benchmarks all configs at the first call for a given `(seq_len, HEAD_DIM)` pair and caches the best one.

I tested 6 configs on A10G:

| BLOCK_Q | BLOCK_K | num_warps | A10G result |
|---------|---------|-----------|-------------|
| 64      | 64      | 4         | **best**    |
| 128     | 64      | 4         | slower      |
| 64      | 128     | 4         | slower      |
| 128     | 128     | 4         | slower      |
| 128     | 64      | 8         | slower      |
| 128     | 128     | 8         | slower      |

**Why BLOCK_Q=128 still loses on A10G:**

A10G has 96 KB of shared memory per SM - double the T4's 48 KB. So BLOCK_Q=128 now fits in SRAM:

```
BLOCK_Q=128: 128 x 64 x 4 bytes = 32 KB (Q)
             + 32 KB (K) + 32 KB (V) = 96 KB  fits exactly
```

But autotune still picked BLOCK_Q=64. The bottleneck shifted from SRAM to **register pressure**.

With BLOCK_Q=128, the kernel keeps a [128, 64] `O` accumulator live across the entire K-loop - that's 8,192 float32 values = 32 KB of registers per warp. Each A10G SM has 65,536 registers total. Larger Q blocks mean fewer warps can be resident simultaneously, which reduces the GPU's ability to hide memory latency. SRAM was the hard constraint on T4; registers are the hard constraint on A10G.

## 6. Benchmark results

Measured on Modal A10G (float32, `do_bench` with 25 warmup + 100 rep):

| config                              | ms    | GB/s  |
|-------------------------------------|-------|-------|
| batch=1, heads=1, seq=256           | 0.018 |  15.0 |
| batch=1, heads=1, seq=1024          | 0.052 |  20.4 |
| batch=2, heads=4, seq=256           | 0.020 | 106.4 |
| batch=2, heads=4, seq=1024          | 0.105 |  79.8 |
| batch=2, heads=4, seq=256  (causal) | 0.020 | 104.7 |
| batch=2, heads=4, seq=1024 (causal) | 0.106 |  78.8 |

A10G peak HBM bandwidth: **600 GB/s**. Peak measured: **106.4 GB/s** - about 17.7% of peak.

## 7. Why 17.7% of peak bandwidth?

Better than I expected, but still far from peak. Two factors explain the gap.

**Factor 1: Small tensors**

At batch=2, heads=4, seq=256, head_dim=64, float32:

```
tensor size = 2 x 4 x 256 x 64 x 4 bytes = 1 MB
```

1 MB is tiny. The A10G can saturate its 600 GB/s bandwidth only with large, sustained transfers. For 1 MB, kernel launch overhead and HBM latency dominate actual transfer time. Notice that seq=1024 drops to 79.8 GB/s - more data doesn't fully compensate because the working set is still small.

**Factor 2: 17% SM occupancy**

Nsight reports ~17% occupancy - only 17% of A10G's SMs are active simultaneously. The root cause is **register pressure**.

Inside the K-block loop, the kernel keeps these live simultaneously:
- `Q`: [64, 64] float32 = 4,096 floats = 16 KB of registers
- `O`: [64, 64] float32 = 16 KB
- `m`, `l`: [64] float32 = negligible
- `K`, `V`, `S`, `P`: loaded and used each iteration

Each SM has 65,536 registers total. With this register footprint, only a small number of warps can be resident per SM - so even when one warp stalls waiting for memory, there aren't enough other warps to keep the SM busy.

This is what FlashAttention v2 fixes: it restructures the outer loop to reduce the live register set. The v3 paper goes further with warp specialization on H100 - dedicated "producer" warps load data while "consumer" warps compute, overlapping both completely. That's where the remaining 82% of peak lives.

## 8. What I actually learned

**The algorithm is elegant but the numbers are humbling.**

I can reproduce the online softmax. I understand why tiling reduces HBM traffic. But getting from "correctness" to "production-grade performance" requires:

1. Skipping fully-masked causal blocks (v2)
2. Warp specialization for overlapping compute and memory (v3)
3. Larger sequences to amortize kernel launch overhead
4. float16/bfloat16 for 2x bandwidth improvement

The gap between my 106 GB/s and what vLLM ships isn't a bug - it's the rest of the engineering.

## Reproducing the benchmark

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

Requires Triton >= 2.0, CUDA GPU, head_dim in (32, 64, 128), seq_len divisible by 64.
