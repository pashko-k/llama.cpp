#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <vector>

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
