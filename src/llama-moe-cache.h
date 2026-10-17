#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache for the layers assigned to it
    // if device_sizes is set, it holds the per-device budget indexed by the model device, otherwise size is split among the devices
    // if reuse_prefill is set, the cache banks are bound to suballocated slices of the compute buffer tail
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * device_sizes, bool reuse_prefill = false);
    ~llama_moe_cache();

    // maximum batch size for ephemeral cache reuse (covers single-token decode and speculative verify batches)
    static constexpr int32_t max_batch = 32;

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the share of the cache budget assigned to backend, 0 if it has no cached layers
    size_t cache_size(ggml_backend_t backend) const;

    // bind the cache bank tensors of backend to suballocated slices of its compute buffer,
    // returns true if banks are bound
    bool bind_compute_buffer(ggml_backend_t backend, ggml_backend_buffer_t compute_buf, size_t base_offset, size_t ephem_bytes, size_t persist_bytes);

    // unbind the cache bank tensors from the compute buffer
    void unbind();

    void set_ephem_enabled(bool enabled);

    // true if the cache can accept queries for the current batch
    bool is_enabled() const;

    // true if the ephemeral tier is currently enabled
    bool is_ephem_enabled() const;

    // maximum batch size safe for ephemeral tier without exhausting any group's slot budget
    int32_t max_ephem_tokens() const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    void prepare_reserve();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void print_stats() const;
    void get_stats(size_t & hits, size_t & misses, size_t & bytes, int64_t & time_us, int32_t & n_slots) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
