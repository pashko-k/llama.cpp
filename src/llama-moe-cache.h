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
    int32_t n_expert   = 0;
    int32_t n_slots    = 0;
    int32_t n_ephem    = 0;
    int32_t n_prot_max = 0;
    int32_t n_prot     = 0;
    int32_t n_active_layers = 0;

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

    void remove_node(int32_t s, int32_t & head, int32_t & tail) {
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
    }

    void push_tail_node(int32_t s, int32_t & head, int32_t & tail) {
        prev[s] = tail;
        next[s] = -1;
        if (tail >= 0) {
            next[tail] = s;
        } else {
            head = s;
        }
        tail = s;
    }

    bool ephem_enabled = true;

    void set_ephem_enabled(bool enabled) {
        ephem_enabled = enabled;
        if (!enabled) {
            invalidate_ephemeral();
        }
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_ephem = 0, int32_t n_active = 0) {
        this->n_expert        = n_expert;
        this->n_slots         = n_slots;
        this->n_ephem         = n_ephem;
        this->n_active_layers = n_active > 0 ? n_active : n_layer;
        const int32_t n_persist = n_slots - n_ephem;

        const char * env_budget = getenv("LLAMA_MOE_CACHE_LAYER_BUDGET");
        this->layer_budget_enabled = env_budget == nullptr || atoi(env_budget) != 0;

        if (n_ephem > 0 && n_persist > 0) {
            this->n_prot_max = n_persist;
            this->n_prot     = 0;
        } else if (n_ephem > 0) {
            this->n_prot_max = 0;
            this->n_prot     = 0;
        } else {
            this->n_prot_max = std::max(0, (n_slots * 3) / 4); // 75% protected, 25% probationary
            this->n_prot     = 0;
        }

        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        is_protected.assign(n_slots, false);
        n_prot_layer.assign(n_layer, 0);
        prev.resize(n_slots);
        next.resize(n_slots);

        if (n_ephem > 0 && n_persist > 0) {
            for (int32_t s = 0; s < n_ephem; ++s) {
                prev[s] = s - 1;
                next[s] = s + 1 < n_ephem ? s + 1 : -1;
                is_protected[s] = false;
            }
            head_prob = 0;
            tail_prob = n_ephem - 1;

            // empty persistent slots initialize in protected chain as promotion victims
            for (int32_t s = n_ephem; s < n_slots; ++s) {
                prev[s] = s - 1 >= n_ephem ? s - 1 : -1;
                next[s] = s + 1 < n_slots ? s + 1 : -1;
                is_protected[s] = true;
            }
            head_prot = n_ephem;
            tail_prot = n_slots - 1;
        } else {
            for (int32_t s = 0; s < n_slots; ++s) {
                prev[s] = s - 1;
                next[s] = s + 1 < n_slots ? s + 1 : -1;
            }
            head_prob = 0;
            tail_prob = n_slots - 1;
            head_prot = -1;
            tail_prot = -1;
        }

        seen.assign(n_expert, 0);
    }

    void invalidate_ephemeral() {
        if (n_ephem <= 0) {
            return;
        }

        // preserve persistent-region slots currently sitting in the probationary chain
        std::vector<int32_t> persistent_prob;
        for (int32_t s = head_prob; s >= 0; s = next[s]) {
            if (s >= n_ephem) {
                persistent_prob.push_back(s);
            }
        }

        for (int32_t s = 0; s < n_ephem; ++s) {
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                key_of[s] = -1;
            }
            prev[s] = s - 1;
            next[s] = s + 1 < n_ephem ? s + 1 : -1;
            is_protected[s] = false;
        }
        head_prob = 0;
        tail_prob = n_ephem - 1;

        if (!persistent_prob.empty()) {
            next[tail_prob] = persistent_prob[0];
            for (size_t i = 0; i < persistent_prob.size(); ++i) {
                int32_t s = persistent_prob[i];
                prev[s] = (i == 0) ? tail_prob : persistent_prob[i - 1];
                next[s] = (i + 1 < persistent_prob.size()) ? persistent_prob[i + 1] : -1;
            }
            tail_prob = persistent_prob.back();
        }
    }

    // touch slot s on hit: promote probationary to protected, or refresh protected
    void touch(int32_t s) {
        if (is_protected[s]) {
            if (s != tail_prot) {
                remove_node(s, head_prot, tail_prot);
                push_tail_node(s, head_prot, tail_prot);
            }
        } else {
            remove_node(s, head_prob, tail_prob);
            push_tail_node(s, head_prot, tail_prot);
            is_protected[s] = true;
            n_prot++;
            if (key_of[s] >= 0) {
                n_prot_layer[key_of[s] / n_expert]++;
            }

            if (n_prot > n_prot_max && head_prot >= 0) {
                int32_t demoted = head_prot;
                remove_node(demoted, head_prot, tail_prot);
                push_tail_node(demoted, head_prob, tail_prob);
                is_protected[demoted] = false;
                n_prot--;
                if (key_of[demoted] >= 0) {
                    n_prot_layer[key_of[demoted] / n_expert]--;
                }
            }
        }
    }

    // periodic decay demotes oldest protected slots to probationary
    void decay() {
        int32_t to_demote = n_prot / 4;
        while (to_demote-- > 0 && head_prot >= 0) {
            if (key_of[head_prot] < 0) {
                break;
            }
            int32_t s = head_prot;
            remove_node(s, head_prot, tail_prot);
            push_tail_node(s, head_prob, tail_prob);
            is_protected[s] = false;
            n_prot--;
            if (key_of[s] >= 0) {
                n_prot_layer[key_of[s] / n_expert]--;
            }
        }
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    struct promotion {
        int32_t expert;
        int32_t src_slot;
        int32_t dst_slot;
    };

    // plan slot assignments for layer il
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, std::vector<promotion> & promotions, size_t & n_hit) {
        GGML_ASSERT(il >= 0 && il < (int32_t) slot_map.size());

        fills.clear();
        promotions.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];
        const int32_t n_persist = n_slots - n_ephem;
        const int32_t eff_ephem = ephem_enabled ? n_ephem : 0;

        // hits are touched and promoted first
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                int32_t s = slots[e];
                n_hit++;
                if (eff_ephem > 0 && n_persist > 0 && s < eff_ephem) {
                    if (head_prot >= 0) {
                        int32_t s_prot = head_prot;

                        // layer-adaptive victim selection when persistent capacity is full
                        if (layer_budget_enabled && key_of[head_prot] >= 0 && n_active_layers > 1) {
                            const int32_t fair_share = std::max(1, n_persist / n_active_layers);
                            const int32_t floor_min  = std::max(1, std::min(8, fair_share / 2));
                            const bool over_share    = n_prot_layer[il] > fair_share;

                            int32_t s_best = -1;
                            int32_t max_victim_count = -1;

                            int32_t cur = head_prot;
                            for (int32_t hops = 0; hops < 32 && cur >= 0; ++hops) {
                                if (key_of[cur] < 0) {
                                    s_best = cur;
                                    break;
                                }
                                const int32_t vl = key_of[cur] / n_expert;
                                const int32_t ve = key_of[cur] % n_expert;
                                if (vl == il && seen[ve] == seen_gen) {
                                    cur = next[cur];
                                    continue;
                                }
                                if (over_share && vl == il) {
                                    s_best = cur;
                                    break;
                                }
                                if (!over_share && n_prot_layer[vl] > fair_share) {
                                    s_best = cur;
                                    break;
                                }
                                if (n_prot_layer[vl] > floor_min && n_prot_layer[vl] > max_victim_count) {
                                    max_victim_count = n_prot_layer[vl];
                                    s_best = cur;
                                }
                                cur = next[cur];
                            }

                            if (s_best >= 0) {
                                s_prot = s_best;
                            }
                        } else if (key_of[s_prot] >= 0) {
                            int32_t cur = head_prot;
                            for (int32_t hops = 0; hops < 32 && cur >= 0; ++hops) {
                                if (key_of[cur] < 0) {
                                    s_prot = cur;
                                    break;
                                }
                                const int32_t vl = key_of[cur] / n_expert;
                                const int32_t ve = key_of[cur] % n_expert;
                                if (vl == il && seen[ve] == seen_gen) {
                                    cur = next[cur];
                                    continue;
                                }
                                s_prot = cur;
                                break;
                            }
                        }

                        if (key_of[s_prot] >= 0) {
                            const int32_t vl = key_of[s_prot] / n_expert;
                            const int32_t ve = key_of[s_prot] % n_expert;
                            if (vl == il && seen[ve] == seen_gen) {
                                // all accessible victims are active batch experts; keep in ephemeral
                                if (s != tail_prob && head_prob >= 0) {
                                    remove_node(s, head_prob, tail_prob);
                                    push_tail_node(s, head_prob, tail_prob);
                                }
                                continue;
                            }
                        }

                        remove_node(s_prot, head_prot, tail_prot);
                        push_tail_node(s_prot, head_prot, tail_prot);
                        if (key_of[s_prot] >= 0) {
                            const int32_t victim_layer = key_of[s_prot] / n_expert;
                            slot_map[victim_layer][key_of[s_prot] % n_expert] = -1;
                            n_prot_layer[victim_layer]--;
                        } else {
                            n_prot++;
                        }
                        n_prot_layer[il]++;
                        key_of[s_prot] = il*n_expert + e;
                        slots[e] = s_prot;

                        remove_node(s, head_prob, tail_prob);
                        push_tail_node(s, head_prob, tail_prob);
                        key_of[s] = -1;

                        promotions.push_back({ e, s, s_prot });
                    }
                } else if (eff_ephem > 0 && n_persist > 0 && s >= eff_ephem) {
                    if (is_protected[s]) {
                        if (s != tail_prot && head_prot >= 0) {
                            remove_node(s, head_prot, tail_prot);
                            push_tail_node(s, head_prot, tail_prot);
                        }
                    } else {
                        remove_node(s, head_prob, tail_prob);
                        push_tail_node(s, head_prot, tail_prot);
                        is_protected[s] = true;
                        n_prot++;
                        n_prot_layer[il]++;
                    }
                } else if (n_persist == 0) {
                    if (s != tail_prob && head_prob >= 0) {
                        remove_node(s, head_prob, tail_prob);
                        push_tail_node(s, head_prob, tail_prob);
                    }
                } else {
                    touch(s);
                }
            }
        }

        // sorted misses usually get consecutive slots, so uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        miss_slots.clear();
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }

            int32_t s = head_prob;
            if (s >= 0) {
                remove_node(s, head_prob, tail_prob);
            } else if (head_prot >= 0) {
                s = head_prot;
                remove_node(s, head_prot, tail_prot);
                is_protected[s] = false;
                if (key_of[s] >= 0) {
                    n_prot--;
                    n_prot_layer[key_of[s] / n_expert]--;
                }
            }

            if (s < 0) {
                return false;
            }

            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e]  = s;

            miss_slots.push_back(s);
            fills.push_back({ e, s });
        }

        // push newly assigned miss slots to tail of probationary list
        for (int32_t s : miss_slots) {
            push_tail_node(s, head_prob, tail_prob);
            is_protected[s] = false;
        }

        return true;
    }
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
