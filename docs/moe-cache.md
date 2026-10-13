# MoE Cache Documentation

The MoE Cache keeps frequently accessed expert weights in GPU VRAM while storing the majority of Mixture of Experts (MoE) weights in host CPU RAM. This enables running large MoE models (such as DeepSeek-V2/V3, Mixtral, and Qwen-MoE) on memory-constrained single-GPU and multi-GPU systems.

---

## Key Features

1. **Zero-VRAM Ephemeral Pre-fill Headroom Reuse (`--moe-cache-reuse-prefill`)**:
   - Token generation uses drastically less activation memory ($S_{\text{tg}}$) than prompt processing ($S_{\text{pp}}$).
   - During token generation, the idle memory region $[S_{\text{tg}}, S_{\text{pp}}]$ is reclaimed as an ephemeral probationary cache at **0 extra bytes of VRAM**.
   - During prefill batches (`n_tokens > 32`), ephemeral slots are automatically invalidated so prefill activations execute without interference.

2. **Generic Buffer Tail Padding (`ggml_gallocr_set_tail_padding`)**:
   - Allows allocating persistent cache slots beyond the prefill graph footprint ($[S_{\text{pp}}, S_{\text{pp}} + S_{\text{persist}}]$).
   - Keeps all cache slots in a single contiguous memory block so a single `ggml_mul_mat_id` kernel executes per layer without graph splitting.
   - Guaranteed isolation: graph tensor allocations planned by `ggml_gallocr` never enter the tail padding region.

3. **Segmented LRU (2-Tier SLRU) Admission and Eviction**:
   - **Probationary Tier**: Newly accessed experts enter the probationary queue.
   - **Protected Tier**: Experts that receive repeat hits (frequency $\ge 2$) are promoted to the protected tier.
   - **Asynchronous Promotion**: In hybrid mode (ephemeral + persistent), promotions execute via asynchronous device-to-device (D2D) copies into persistent VRAM.
   - **Periodic Frequency Decay**: Demotes the oldest protected entries back to probationary every 512 operations to adapt to context changes.

4. **Localized Pinned Memory**:
   - Registers host expert weight buffers with `cudaHostRegister` for line-rate PCIe DMA transfers without CPU bounce-buffering.
   - Uses localized registration so global memory allocations remain unpinned by default.
   - Thread-safe refcounted registry (`pinned_buffer_registry`) manages page-aligned bounds across devices and contexts.

5. **Multi-GPU Layer-Split Mode (`LLAMA_SPLIT_MODE_LAYER`)**:
   - Each GPU hosting CPU MoE layers instantiates an isolated `llama_moe_cache` instance.
   - Prefill headroom and tail padding are sized and applied per GPU backend independently.
   - Asynchronous transfers run concurrently on each device's stream.

---

## Command Line Arguments

| Parameter | Default | Description |
|---|---|---|
| `--moe-cache-reuse-prefill` | `false` | Reuse idle prefill compute buffer tail for ephemeral MoE cache (0 extra VRAM). |
| `--moe-cache-mib N0,N1,...` | `0` | Persistent GPU cache size in MiB for MoE experts. Either a single value split like `--tensor-split`, or a comma-separated list of MiB per GPU device. |
| `-cmoe`, `--cpu-moe` | `false` | Keep all Mixture of Experts (MoE) weights in CPU memory. |
| `-ncmoe N`, `--n-cpu-moe N` | `0` | Keep MoE weights of the first N layers in CPU memory. |
| `-sm layer`, `--split-mode layer` | `layer` | Layer-split mode across multiple GPUs. Note: tensor-parallel modes (`-sm row` / `-sm tensor`) are not supported with MoE cache. |
| `-ts`, `--tensor-split` | Proportional | Proportional VRAM fraction across GPUs (e.g. `-ts 12,8`). |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_MOE_CACHE_LAYER_BUDGET` | `1` | Enable layer-adaptive persistent budgeting (1 = enabled, 0 = disabled). Prevents high-entropy layers from evicting repeat-hit experts of middle layers. |

---

## Memory Layout

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              Unified GPU Backend Buffer                                                 |
|                                                                                                                         |
|  [0 .............. S_tg]  [S_tg ................................. S_pp]  [S_pp .................. S_total]              |
|      Decode Compute       | <---------- Ephemeral Headroom -----------> | | <------ Persistent Pool ------> |              |
|       Active Memory       |         (Reused from Prefill Tail)          | |    (Tail Padding via gallocr)   |              |
|      (~100-200 MiB)       |             (~2-3 GiB, 0 Extra VRAM)        | |          (--moe-cache-mib)      |              |
|                           |                                             | |                                 |              |
|                           | [Slot 0 ................... Slot N_ephem-1] | | [Slot N_ephem ... Slot N_total] |              |
|                           | <------- Probationary SLRU Segment -------> | | <--- Protected SLRU Segment --> |              |
+---------------------------+---------------------------------------------+-+---------------------------------+-----------+
```

---

## Example Usage

### Pure Zero-VRAM Mode (No Persistent Allocation)
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -cmoe \
  --moe-cache-reuse-prefill \
  -p "Explain the MoE cache architecture."
```

### Hybrid Mode (Ephemeral Headroom + 1 GiB Persistent Cache)
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -cmoe \
  --moe-cache-reuse-prefill \
  --moe-cache-mib 1024 \
  -p "Explain the MoE cache architecture."
```

### Multi-GPU Layer-Split Mode (Per-Device Cache Sizes)
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -sm layer \
  -ts 10,1 \
  -cmoe \
  --moe-cache-reuse-prefill \
  --moe-cache-mib 1400,512 \
  -p "Explain the MoE cache architecture."
```

---

## Verification Tests

```bash
# Verify generic tail padding allocator bounds
./build/bin/test-alloc

# Verify backend matrix multiplication and expert routing
./build/bin/test-backend-ops -o MUL_MAT_ID

# Verify single-GPU and multi-GPU end-to-end MoE cache accuracy
./build/bin/test-llama-archs -a deepseek2

# Verify MoE cache SLRU, invalidation, and layer-adaptive budgeting
./build/bin/test-moe-cache-lru
```
