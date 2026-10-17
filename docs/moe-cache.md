# MoE Cache Documentation

The MoE Cache keeps frequently accessed expert weights in GPU VRAM while storing the majority of Mixture of Experts (MoE) weights in host CPU RAM. This enables running large MoE models (such as DeepSeek-V2/V3, Mixtral, and Qwen-MoE) on memory-constrained single-GPU and multi-GPU systems.

---

## Key Features

1. **Zero-VRAM Ephemeral Pre-fill Headroom Reuse (`--moe-cache-reuse-prefill`)**:
   - Token generation uses drastically less activation memory ($S_{\text{tg}}$) than prompt processing ($S_{\text{pp}}$).
   - During token generation, the idle memory region $[S_{\text{tg}}, S_{\text{pp}}]$ is reclaimed as an ephemeral probationary cache at **0 extra bytes of VRAM**.
   - During prefill batches (`n_tokens > 32`), ephemeral slots are automatically invalidated so prefill activations execute without interference.

2. **Dedicated Persistent Cache (`--moe-cache-mib`)**:
   - Allocates dedicated persistent cache slots for hot experts in GPU VRAM.
   - Keeps all cache slots in a single contiguous 3D memory bank per layer so a single `ggml_mul_mat_id` kernel executes without graph splitting.
   - Persistent slots remain intact across both prefill and decode passes.

3. **Segmented LRU (2-Tier SLRU) Admission and Eviction**:
   - **Probationary Tier**: Newly accessed experts enter the probationary queue.
   - **Protected Tier**: Experts that receive repeat hits (frequency >= 2) are promoted to the protected tier within persistent VRAM.
   - **Periodic Frequency Decay**: Demotes the oldest 25% of protected entries back to probationary every 512 copy operations per device group (preventing cold experts from squatting in persistent VRAM across topic shifts).

4. **Compute Buffer Boundary Querying (`ggml_gallocr_get_tail_padding_offset`)**:
   - Used in ephemeral mode to safely query the graph allocation boundary within the compute buffer.
   - Ensures ephemeral cache tensors never collide with planned decode activations.

5. **Localized Pinned Memory**:
   - Registers host expert weight buffers with `cudaHostRegister` for line-rate PCIe DMA transfers without CPU bounce-buffering.
   - Uses localized registration so global memory allocations remain unpinned by default.
   - Thread-safe refcounted registry (`pinned_buffer_registry`) manages page-aligned bounds across devices and contexts.

6. **Multi-GPU Layer-Split Mode (`LLAMA_SPLIT_MODE_LAYER`)**:
   - Each GPU hosting CPU MoE layers instantiates an isolated `llama_moe_cache` instance.
   - Prefill headroom and dedicated cache budgets are sized and applied per GPU backend independently.
   - Asynchronous transfers run concurrently on each device's stream.

---

## Command Line Arguments

| Parameter | Default | Description |
|---|---|---|
| `--moe-cache-reuse-prefill` | `false` | Reuse idle prefill compute buffer tail for ephemeral MoE cache (0 extra VRAM). Can be combined with `--moe-cache-mib` for hybrid mode. |
| `--moe-cache-mib N0,N1,...` | `0` | Persistent GPU cache size in MiB for MoE experts. Either a single value split like `--tensor-split`, or a comma-separated list of MiB per GPU device (in device priority order as specified by `-devd` / `--device`). |
| `-cmoe`, `--cpu-moe` | `false` | Keep all Mixture of Experts (MoE) weights in CPU memory. |
| `-ncmoe N`, `--n-cpu-moe N` | `0` | Keep MoE weights of the first N layers in CPU memory. |
| `-sm layer`, `--split-mode layer` | `layer` | Layer-split mode across multiple GPUs. Note: tensor parallelism (`-sm tensor`) is not supported with MoE cache. |
| `-ts`, `--tensor-split` | Proportional | Proportional VRAM fraction across GPUs (e.g. `-ts 12,8`). |

---

## Memory Layout

The MoE cache supports three operating modes:

### 1. Hybrid Mode (`--moe-cache-reuse-prefill` + `--moe-cache-mib`)

Combines zero-VRAM prefill activation headroom with dedicated persistent tail padding:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                            Unified GPU Compute Buffer                                             |
|                                                                                                                   |
|  [0 ......... S_tg]  [S_tg ................................. S_pp]  [S_pp .............................. S_total] |
|     Decode Compute   | <---------- Ephemeral Headroom -----------> | | <--------- Persistent Tail Padding --------> | |
|      Active Graph    |        (Reused from Prefill Graph)          | |        (Allocated via gallocr tail_pad)    | |
|     (~100-200 MiB)   |           (~2-3 GiB, 0 Extra VRAM)          | |                 (--moe-cache-mib)          | |
|                      |                                             | |                                            | |
|                      | [Slot 0] [Slot 1] ... [Slot N_ephem-1]      | | [Slot N_ephem] ... [Slot N_slots-1] [Guard]| |
|                      | <------- Ephemeral Probationary Tier -----> | | <--- Persistent Probationary + Protected ->| |
+----------------------+---------------------------------------------+-+--------------------------------------------+-+
```

To make hybrid caching possible, expert projection matrices are arranged with an **interleaved slot layout**:
- Each slot $s$ groups all projections (`gate`, `up`, `down`) adjacent in memory.
- All banks share a uniform slot stride $S_{\text{slot}}$ aligned to the least common multiple (LCM) of the backend alignment and quantization block sizes.
- Ephemeral slots live strictly in $[S_{\text{tg}}, S_{\text{pp}})$, while persistent slots and the guard slot live strictly in $[S_{\text{pp}}, S_{\text{total}})$.
- During prefill (`n_tokens > 32`), ephemeral slots are invalidated while persistent slots survive intact. Hot experts promoted to the persistent tier remain cached across prefill passes.

> [!NOTE]
> Interleaved 3D banks with non-contiguous slot strides (`nb[2] > ne[1] * nb[1]`) are supported on CUDA, CPU, and Metal backends.

### 2. Zero-VRAM Ephemeral Mode (`--moe-cache-reuse-prefill`)

```
+-----------------------------------------------------------------------------------+
|                            Unified GPU Backend Buffer                             |
|                                                                                   |
|  [0 .............. S_tg]  [S_tg ................................. S_pp]           |
|      Decode Compute       | <---------- Ephemeral Headroom -----------> |         |
|       Active Memory       |         (Reused from Prefill Tail)          |         |
|      (~100-200 MiB)       |             (~2-3 GiB, 0 Extra VRAM)        |         |
|                           |                                             |         |
|                           | [Slot 0 ................... Slot N_slots-1] |         |
|                           | <---------------- LRU Cache --------------> |         |
+---------------------------+---------------------------------------------+---------+
```

During token generation, the idle memory region `[S_tg, S_pp]` is temporarily reused as an ephemeral cache at zero additional VRAM cost. On prefill batches (`n_tokens > 32`), ephemeral slots are invalidated to allow compute activations to proceed.

### 3. Dedicated Persistent Mode (`--moe-cache-mib`)

```
+-----------------------------------------------------------------------------------+
|                          Dedicated GPU Memory Buffer                              |
|                                                                                   |
|  [Slot 0 ........................................................... Slot N-1]    |
|  <------------------------------ 2-Tier SLRU Cache -----------------------------> |
|  <------- Probationary Tier -------> | <------------ Protected Tier ------------> |
+--------------------------------------+--------------------------------------------+
```

Allocates a dedicated GPU memory buffer sized via `--moe-cache-mib`. Persistent slots remain intact across prefill and decode passes, surviving throughout the lifetime of the context.

---

## Example Usage

### Hybrid Mode (Recommended)
Combines prefill activation headroom with 1 GiB of persistent cache:
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -cmoe \
  --moe-cache-reuse-prefill \
  --moe-cache-mib 1024 \
  -p "Explain the MoE cache architecture."
```

### Zero-VRAM Ephemeral Mode
Reuses prefill activation headroom with zero additional VRAM overhead:
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -cmoe \
  --moe-cache-reuse-prefill \
  -p "Explain the MoE cache architecture."
```

### Dedicated Persistent Mode
Allocates a dedicated persistent VRAM pool (e.g. 1 GiB):
```bash
./build/bin/llama-cli \
  -m model.gguf \
  -ngl 99 \
  -cmoe \
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
  --moe-cache-mib 1400,512 \
  -p "Explain the MoE cache architecture."
```

### Line-Rate DMA Transfers (Optional)
On systems with sufficient physical RAM where all host expert weights remain memory-resident, setting `GGML_CUDA_REGISTER_HOST=1` enables CUDA page-locked memory registration for maximum PCIe DMA transfer throughput.

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
