#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

// Segmented LRU (SLRU) of the experts of a group of layers
// Protected segment holds hot anchor experts, probationary segment handles transient traffic
struct moe_cache_lru {
    static constexpr uint32_t DECAY_INTERVAL = 512;
    static constexpr int32_t  DECAY_DIV      = 4; // demote oldest 25% of protected tier
    static constexpr int32_t  MAX_FLOOR_SHARE = 8;
    static constexpr int32_t  VICTIM_SEARCH_MAX_HOPS = 32;

    int32_t n_expert   = 0;
    int32_t n_slots    = 0;
    int32_t n_ephem    = 0;
    int32_t n_prot_max = 0;
    int32_t n_prot     = 0;
    int32_t n_active_layers = 0;
    uint64_t copy_count = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    std::vector<bool>      is_protected; // [n_slots] true if in protected segment
    std::vector<int32_t>   n_prot_layer; // [n_layer] protected slots per layer
    bool layer_budget_enabled = true;

    // doubly linked list of the slots
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head_prob = -1;
    int32_t tail_prob = -1;
    int32_t head_prot = -1;
    int32_t tail_prot = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> miss_slots;

    bool ephem_enabled = true;

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    struct promotion {
        int32_t expert;
        int32_t src_slot;
        int32_t dst_slot;
    };

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_ephem = 0, int32_t n_active = 0);
    void set_ephem_enabled(bool enabled);
    void invalidate_ephemeral();
    void touch(int32_t s);
    void decay();
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, std::vector<promotion> & promotions, size_t & n_hit);

    void remove_node(int32_t s, int32_t & head, int32_t & tail);
    void push_tail_node(int32_t s, int32_t & head, int32_t & tail);
};

// number of slots of size nb2, starting at offset, whose bytes overlap [0, size)
// a slot that straddles size is counted as well
static inline int32_t moe_cache_n_overlap(size_t offset, size_t size, size_t nb2) {
    if (offset >= size || nb2 == 0) {
        return 0;
    }
    const size_t n = (size - offset + nb2 - 1)/nb2;
    return n > (size_t) INT32_MAX ? INT32_MAX : (int32_t) n;
}

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

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void print_stats() const;
    void get_stats(size_t & hits, size_t & misses, size_t & bytes, int64_t & time_us, int32_t & n_slots) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
