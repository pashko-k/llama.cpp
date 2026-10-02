# MoE expert cache (CUDA)

Dynamic VRAM cache for CPU-resident MoE expert weights (`-cmoe`). The cache
engages inside the CPU `mul_mat_id` kernel: cached expert rows are computed on
the GPU in one batched launch while the threadpool computes the miss rows on
the CPU. Expert weights are inserted into VRAM asynchronously from decode-path
misses. The slab of VRAM is carved from the idle tail of the monolithic
compute buffer (see `ggml_gallocr_expand_buffer_chunk` in `ggml-alloc.c`).

Implementation: `moe-cache.cu` / `moe-cache.cuh`. Cross-backend API table:
`ggml/include/ggml-backend-moe-cache.h`.

## Enable

The cache is **opt-in** in this fork:

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE` | off | Set to `1` to enable. Off means the backend registers nothing and every MoE node runs the pure-CPU path. |

All variables below are read once at backend registration (except where noted
as lazily read on first use) and apply per process.

## Memory budget

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_NDEV` | all GPUs | Limit the cache to the first N CUDA devices. |
| `GGML_CUDA_MOE_CACHE_BUDGET_MB` | 0 (auto) | Cap the total cache size per device. 0 = auto: free VRAM at init minus the reserve, or the slab size when a compute-buffer tail slab was carved. |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | 512 | Free VRAM to leave untouched per device. Also the safety margin llama-context keeps when expanding the compute buffer before slab carving. Lowering it too far lets the lazily growing CUDA pool starve mid-decode and crash the process. |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | 256 | Experts smaller than this (KB) are never cached; too small to amortize the dispatch cost. |

## Cache engagement

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | 8 | Largest decode batch (`n_tokens`) that engages the cache. Range 1..8. Above it the node runs pure-CPU (prefill). |
| `GGML_CUDA_MOE_CACHE_MTP_BLK` | 1024 (off) | Skip cache engagement for transformer blocks with index >= N. Use to exclude MTP draft layers. |

## Inserts and eviction

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_INSERTS` | 8 | Max async insert jobs enqueued per `plan()` call. |
| `GGML_CUDA_MOE_CACHE_THROTTLE` | 8 | When pools are full, admit only 1-in-N misses for insertion. |
| `GGML_CUDA_MOE_CACHE_WORKERS` | 4 | Insert worker thread count. Range 1..16. |
| `GGML_CUDA_MOE_CACHE_FREQ_FILTER` | 1 | Frequency-based admission filter. `0` admits every miss. |
| `GGML_CUDA_MOE_CACHE_MIN_FREQ` | 2 | Admit an expert only after it is seen this many decode times. |
| `GGML_CUDA_MOE_CACHE_DECAY_TOKENS` | 512 | Halve all frequency counters every N decode tokens. |

## Warm start and phase transitions

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_PREFETCH` | 1 | Background backfill: when the demand queue is empty, workers walk the (blk, expert) space and preload. `0` disables. |
| `GGML_CUDA_MOE_CACHE_SEQUENTIAL_BACKFILL` | 0 | Sweep (blk, expert) in strict order instead of hot-set-first then sweep. |
| `GGML_CUDA_MOE_CACHE_HOTSET` | 1 | Persist per-layer residency bitmaps so the next run with the same model starts warm. File: `$HOME/.cache/llama.cpp/moe-cache-hotset-<fingerprint>.bin` (or `%LOCALAPPDATA%` on Windows). |
| `GGML_CUDA_MOE_CACHE_TAIL_SEED` | 1 | On the prefill-to-decode transition, preload experts activated by the last prompt tokens as top-priority inserts. |

## Fast paths

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_FUSE` | 1 | Paired gate+up slots and fused `silu(gate)*up` dispatch on the GPU. |
| `GGML_CUDA_MOE_CACHE_REUSE` | 1 | Reuse the quantized activation across hits in one node. |
| `GGML_CUDA_MOE_CACHE_REDIRECT` | 1 | Scatter GPU-computed result rows directly into the GPU-side copy of the CPU node's dst, skipping the host round-trip. |

## Bail-out

The cache samples its own wall time against a pure-CPU baseline. If engaged
nodes stay slower, it disables itself and frees its VRAM.

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_BAIL` | 1 | `0` disables the judge (cache always stays on once pools exist). |
| `GGML_CUDA_MOE_CACHE_BAIL_RATIO` | 1.25 | Trip when sustained engaged time exceeds baseline x this ratio. |
| `GGML_CUDA_MOE_CACHE_BAIL_STRIKES` | 16 | Consecutive over-threshold checks before tripping. |
| `GGML_CUDA_MOE_CACHE_BAIL_WARM` | 500 | Samples ignored after pools build (first-touch effects). |
| `GGML_CUDA_MOE_CACHE_BAIL_SAMPLE` | 2750 | Node count where the pure-CPU baseline sampling window ends. |

## Stats and debug

These read the environment lazily (on first use), not at registration.

| Variable | Default | Description |
| :-- | :-- | :-- |
| `GGML_CUDA_MOE_CACHE_STATS` | 0 (off) | Log hit/miss/insert stats every N `collect()` calls. |
| `GGML_CUDA_MOE_CACHE_DEBUG` | 0 (off) | Trace the first API calls to locate stalls. `>= 10` traces that many lines instead of the default 400. |
| `GGML_CUDA_MOE_CACHE_DEBUG_GATE` | 0 (off) | Log up to N `begin()` gate decisions: why a node engages or stays CPU. |
| `GGML_CUDA_MOE_CACHE_SHADOW` | 0 (off) | Verification mode: compute hits on both CPU and GPU, compare rows, log divergence. Slows the hot path; diagnostics only. |
| `GGML_CUDA_MOE_CACHE_SELFTEST` | 0 (off) | Run the kernel self-test at registration (needs a GPU). |
