# CPU weight repack and the q2_0 MoE performance investigation

This document explains what CPU weight repack is, how it engages, and the
outcome of a focused investigation into why repack q2_0 MoE weights appeared to
lose to the no-repack path during prefill. The short conclusion: repack is a
CPU-only optimization that works as designed. The apparent "regression" was an
artifact of dynamic GPU op-offload, not a real CPU slowdown.

## 1. What repack is

Repack is an extra CPU buffer type (`CPU_REPACK`) that reorganizes quantized
weight blocks into a layout that AVX-512 dot-product kernels can consume
directly, without unpacking the original quant block inside the inner loop.

The layout is defined by the `block<K, N>` template in
`ggml/src/ggml-cpu/repack.h`:

```cpp
template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};
```

For q2_0, the repack form is `block_q2_0x8 = block<2, 8>` (repack.h ~45). It
interleaves 8 q2_0 blocks so a `vpermb`/`vpshufb` + `vpdpbusd` chain can compute
8 column dots at a time. The AVX-512 tensor traits that select this layout are
registered in `ggml_repack_get_optimal_repack_type` (repack.cpp ~5097), which
lists every supported repack instance, including `q2_0_8x8_q8_0`.

Repack is CPU-only. The repack format is not readable by GPU kernels, so a
repacked tensor cannot be offloaded to CUDA. That constraint is intentional and
is central to the investigation below.

## 2. How repack engages

Repack engages only when several conditions are met.

### Flags

- `--repack` / `--no-repack` (common/arg.cpp ~2419) sets `params.no_extra_bufts`.
  Repack is enabled by default.
- `--no-host` (common/arg.cpp ~2427) sets `params.no_host = true`. It bypasses
  the host buffer so that the extra (repack) buffer type can be chosen. The arg
  description is "bypass host buffer allowing extra buffers to be used".
- `--load-mode none` disables mmap loading. mmap weights are not repacked.

### Buffer-type list ordering

`make_cpu_buft_list` (src/llama-model.cpp ~1075) builds the CPU buffer list:

```cpp
// add extra buffer types (e.g. CPU_REPACK)
// must come before the host buffer so that eligible tensors get repacked in CPU memory
if (use_extra_bufts) {
    ... // append ggml_backend_dev_get_extra_bufts(cpu_dev) -> CPU_REPACK
}

// add a host buffer type
if (!no_host) {
    ... // append host buffer of the first offload device
}
```

The `CPU_REPACK` buft is placed **before** the host buffer, so eligible tensors
prefer repack over a host buffer. `select_weight_buft` (src/llama-model-loader.cpp
~1067) returns the first buffer type in that list that can support the op.

### Per-op support check

The repack buffer type's `supports_op` (repack.cpp ~5376) accepts `MUL_MAT`
(dims == 2) and `MUL_MAT_ID` (dims == 3) only when:

- `src0` lives in the `CPU_REPACK` buffer and has an optimal repack type, and
- `src1` (the activations) is host-visible, and
- `src1` is `F32` or `Q8_0`.

### All-or-nothing across prefill and gen

Repack is a **load-time** format decision. The weight tensor is converted once,
into q2_0x8 bytes inside a `CPU_REPACK` buffer. Every `MUL_MAT_ID` on that tensor
then uses the repack path. There is no runtime switch. Prefill and decode share
the same repacked tensor, so repack cannot apply to only one phase. It is either
both prefill and gen, or nothing.

The MoE path in repack is gemv-only (no gemm batching): `forward_mul_mat_id`
(repack.cpp ~5040+) runs `gemv<BLOC_TYPE, ...>` per selected expert row. This is
true for both a large prefill batch and a single-token decode batch.

### Nuance: truly CPU-only runs

When CUDA is absent (for example `CUDA_VISIBLE_DEVICES=""`) there is no GPU
device and therefore no host buffer to bypass, so repack engages regardless of
`--no-host`. In that case `--no-host` is redundant, but the result is identical.

## 3. The apparent regression vs no-repack

Initial comparisons looked like repack was losing prefill badly:

```
                 prompt    gen
GPU present      no-repack   ~124     7.5
GPU present      repack       ~83     7.8
```

That looked like a 50% prefill loss. It was misleading.

### The mechanism: dynamic GPU op-offload

llama.cpp has dynamic op-offload. When `op_offload` is on, the scheduler can
move a compute op from the CPU backend to a higher-priority backend (GPU) at
graph-build time. The gate is in `ggml/src/ggml-backend.cpp` ~970:

```cpp
if (sched->op_offload && src_backend_id == sched->n_backends - 1
        && ggml_backend_buffer_is_host(src->buffer)) {
    ... // move op to GPU if it supports_op and offload_op
}
```

Key facts:

- `cparams.op_offload` defaults to **true** (src/llama-context.cpp ~3728).
- The gate requires `ggml_backend_buffer_is_host(src->buffer)`. A standard CPU
  weight buffer wrapped in a host buffer passes this test, so its `MUL_MAT_ID`
  gets offloaded to CUDA. The GPU reads the weights streamed over PCIe from pinned
  host memory.
- The CUDA offload test accepts `MUL_MAT_ID` when the op batch size is at least
  `op_offload_min_batch_size` (ggml/src/ggml-cuda/ggml-cuda.cu ~5626, env
  `GGML_OP_OFFLOAD_MIN_BATCH`, default 32 at ~5797). Prefill batch (for example
  4096) is well above the threshold. Decode batch is 1, so decode is not offloaded.
- The repack buffer sets `.is_host = nullptr` (repack.cpp ~5420). So
  `ggml_backend_buffer_is_host()` returns false for a repacked tensor, and the
  offload gate never fires. Repack weights are trapped on the CPU.

### What the GPU-present comparison really measured

With a GPU present and `op_offload` on:

- no-repack MoE weights are in host buffers -> `MUL_MAT_ID` prefill is offloaded
  and streamed to the GPU -> very fast prefill (~124 t/s).
- repack MoE weights are in `CPU_REPACK` buffers -> `MUL_MAT_ID` cannot be
  offloaded -> prefill runs on the CPU gemv (~83 t/s).

So repack lost prefill not because its kernel is slow, but because it blocked
GPU streaming that no-repack enjoyed.

### Perf window evidence

`perf report` restricted to the prefill window (`--time '50%-90%'`) showed the
asymmetry directly:

- no-repack prefill window: dense CPU BLAS `sgemm_kernel` plus GPU streaming
  symbols (`cudaMemcpyAsync`, `cudaStreamSynchronize`,
  `ggml_backend_cuda_set_tensor_async`), and **no** CPU q2_0 kernel at all ->
  the q2_0 MoE work was on the GPU.
- repack prefill window: `ggml_gemv_q2_0_8x8_q8_0` dominant (~26%) -> the q2_0
  MoE work was on the CPU.

### Decisive forced CPU-only comparison

Removing dynamic offload makes both paths CPU-only. Two methods agree:
`CUDA_VISIBLE_DEVICES=""` (removes the CUDA backend so `op_offload` has no target)
and `GGML_OP_OFFLOAD_MIN_BATCH=999999` (raises the offload threshold so nothing
offloads). Under both:

```
                     prompt    gen
CPU-only  no-repack   ~36-37    7.1-7.5
CPU-only  repack      ~48       7.9-8.0
```

Once GPU streaming is removed, no-repack drops to ~36 t/s (flat
`ggml_vec_dot_q2_0_q8_0`) and repack (~48 t/s, repack `gemv`) is genuinely faster
on both prefill and decode. This confirms repack is a real CPU-only win.

## 4. Kernel-level numbers

Standalone microbenchmarks (not GPU-contaminated):

```
repack gemv q2_0x8   ~60 GF   (commit f01a5126f, lane-dot rewrite, ~38 -> ~60 GF)
flat vec_dot q2_0    ~21 GF   (no-repack CPU path)
repack gemv q4_0     ~134 GF
```

The lane-dot rewrite (commit `f01a5126f`) accumulates column dots in int32 lanes
across K, folds the q2_0 bias by `sub_epi8` (no separate sum_a), and converts per-block
`dw` with `cvtph_ps`. It is exact versus the generic path.

Decode is batch=1 and bandwidth-bound, so the higher kernel GF does not translate
into a large decode gain. The decode benefit is small (~10%) and comes mainly
from fewer bytes touched per repack block, not from compute throughput.

## 5. Conclusions

- Repack is a CPU-only optimization. Its x8 format is intentionally not
  CUDA-readable (`.is_host = nullptr`), so it never participates in GPU streaming.
- With a GPU present and `op_offload` on, no-repack offloads MoE prefill to the
  GPU and beats repack. That is why repack looked like a regression.
- On truly CPU-only inference (GPU removed or offload disabled), repack is faster
  on both prefill (+~32%) and decode (~+10%). There is no regression.
- Repack engagement requires the weights to be loaded into CPU memory: enabled
  extra bufts (default), `--load-mode none`, and `--no-host` to bypass the host
  buffer. When CUDA is absent, repack engages regardless of `--no-host`.
- Repack is all-or-nothing across prefill and gen. There is no way to get the
  decode benefit without also changing prefill.

## 6. Code references

- `ggml/src/ggml-cpu/repack.h`: `block<K, N>` template (~29), `block_q2_0x8` (~45).
- `ggml/src/ggml-cpu/repack.cpp`:
  - `ggml_repack_get_optimal_repack_type` (~5097) lists the repack traits, incl. `q2_0_8x8_q8_0`.
  - `forward_mul_mat_id` (~5040+) is the gemv-only MoE path.
  - repack `extra_buffer_type::supports_op` (~5376) requires host-visible `src1`.
  - `CPU_REPACK` buffer type `.is_host = nullptr` (~5420).
- `ggml/src/ggml-cpu/arch/x86/repack.cpp`: AVX-512 `ggml_gemv_q2_0_8x8_q8_0` kernel (lane-dot rewrite in commit `f01a5126f`).
- `ggml/src/ggml-cpu/quants.c`: flat `ggml_vec_dot_q2_0_q8_0_generic` (no-repack CPU baseline).
- `src/llama-model.cpp`: `make_cpu_buft_list` (~1075), extra bufts before host (~1092), host added if `!no_host` (~1116).
- `src/llama-model-loader.cpp`: `select_weight_buft` (~1067), avoid host buffer when mmap (~1271).
- `ggml/src/ggml-backend.cpp`: dynamic op-offload gate requiring `is_host(src->buffer)` (~970).
- `src/llama-context.cpp`: `op_offload` default true (~3728).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `offload_op` batch >= `op_offload_min_batch_size` (~5626), env `GGML_OP_OFFLOAD_MIN_BATCH` default 32 (~5797).
- `common/arg.cpp`: `--repack` / `--no-repack` (~2419), `--no-host` (~2427).
