#include "llama-moe-cache.h"
#include "llama-moe-cache-lru.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>

#if !defined(_WIN32)
#include <unistd.h>
#endif

//
// moe_cache_lru implementation
//

void moe_cache_lru::remove_node(int32_t s, int32_t & head, int32_t & tail) {
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

void moe_cache_lru::push_tail_node(int32_t s, int32_t & head, int32_t & tail) {
    prev[s] = tail;
    next[s] = -1;
    if (tail >= 0) {
        next[tail] = s;
    } else {
        head = s;
    }
    tail = s;
}

void moe_cache_lru::set_ephem_enabled(bool enabled) {
    ephem_enabled = enabled;
    if (!enabled) {
        invalidate_ephemeral();
    }
}

void moe_cache_lru::init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_ephem, int32_t n_active) {
    this->n_expert        = n_expert;
    this->n_slots         = n_slots;
    this->n_ephem         = n_ephem;
    this->n_active_layers = n_active > 0 ? n_active : n_layer;
    this->copy_count      = 0;
    const int32_t n_persist = n_slots - n_ephem;

    this->layer_budget_enabled = true;

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

void moe_cache_lru::invalidate_ephemeral() {
    if (n_ephem == 0) {
        return;
    }
    // collect demoted persistent slots in recency order from the probationary chain
    std::vector<int32_t> persistent_prob;
    for (int32_t cur = head_prob; cur >= 0; cur = next[cur]) {
        if (cur >= n_ephem) {
            persistent_prob.push_back(cur);
        }
    }
    // safe to clear host slot map: called between graphs when no backend is executing
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

void moe_cache_lru::touch(int32_t s) {
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

void moe_cache_lru::decay() {
    int32_t to_demote = n_prot / DECAY_DIV;
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

bool moe_cache_lru::plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, std::vector<promotion> & promotions, size_t & n_hit) {
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
    const int32_t active_slots = ephem_enabled ? n_slots : n_slots - n_ephem;
    if (active_slots <= 0 || uniq.size() > (size_t) active_slots) {
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
                int32_t s_prob_candidate = -1;
                for (int32_t cur = head_prob; cur >= 0; cur = next[cur]) {
                    if (cur >= eff_ephem) {
                        if (key_of[cur] >= 0 && key_of[cur] / n_expert == il && seen[key_of[cur] % n_expert] == seen_gen) {
                            continue;
                        }
                        s_prob_candidate = cur;
                        break;
                    }
                }
                if (s_prob_candidate >= 0) {
                    int32_t s_prot = s_prob_candidate;
                    remove_node(s_prot, head_prob, tail_prob);
                    push_tail_node(s_prot, head_prot, tail_prot);
                    is_protected[s_prot] = true;
                    if (key_of[s_prot] >= 0) {
                        const int32_t victim_layer = key_of[s_prot] / n_expert;
                        slot_map[victim_layer][key_of[s_prot] % n_expert] = -1;
                    }
                    n_prot++;
                    n_prot_layer[il]++;
                    key_of[s_prot] = il*n_expert + e;
                    slots[e] = s_prot;

                    remove_node(s, head_prob, tail_prob);
                    push_tail_node(s, head_prob, tail_prob);
                    key_of[s] = -1;

                    promotions.push_back({ e, s, s_prot });
                } else if (head_prot >= 0) {
                    int32_t s_prot = head_prot;

                    // layer-adaptive victim selection when persistent capacity is full
                    if (layer_budget_enabled && key_of[head_prot] >= 0 && n_active_layers > 1) {
                        const int32_t fair_share = std::max(1, n_persist / n_active_layers);
                        const int32_t floor_min  = std::max(1, std::min(MAX_FLOOR_SHARE, fair_share / 2));
                        const bool over_share    = n_prot_layer[il] > fair_share;

                        int32_t s_best = -1;
                        int32_t max_victim_count = -1;

                        int32_t cur = head_prot;
                        for (int32_t hops = 0; hops < VICTIM_SEARCH_MAX_HOPS && cur >= 0; ++hops) {
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
                        for (int32_t hops = 0; hops < VICTIM_SEARCH_MAX_HOPS && cur >= 0; ++hops) {
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

    std::sort(uniq.begin(), uniq.end());
    miss_slots.clear();
    for (int32_t e : uniq) {
        if (slots[e] >= 0) {
            continue;
        }

        // scan probationary queue for a victim not active in the current batch (usually exits on first hop)
        int32_t s = -1;
        if (eff_ephem > 0 && n_persist > 0) {
            // first prefer empty ephemeral slots
            for (int32_t cur = head_prob; cur >= 0; cur = next[cur]) {
                if (cur < eff_ephem && key_of[cur] < 0) {
                    s = cur;
                    remove_node(s, head_prob, tail_prob);
                    break;
                }
            }
            // if ephemeral slots are saturated, admit misses to empty persistent slots before evicting
            if (s < 0 && head_prot >= 0) {
                for (int32_t cur = head_prot; cur >= 0; cur = next[cur]) {
                    if (key_of[cur] < 0) {
                        s = cur;
                        remove_node(s, head_prot, tail_prot);
                        break;
                    }
                }
            }
            // if all slots are occupied, evict an ephemeral slot from probationary queue
            if (s < 0) {
                for (int32_t cur = head_prob; cur >= 0; cur = next[cur]) {
                    if (cur < eff_ephem && (key_of[cur] / n_expert != il || seen[key_of[cur] % n_expert] != seen_gen)) {
                        s = cur;
                        remove_node(s, head_prob, tail_prob);
                        break;
                    }
                }
            }
        }
        if (s < 0) {
            for (int32_t cur = head_prob; cur >= 0; cur = next[cur]) {
                if (!ephem_enabled && cur < n_ephem) {
                    continue;
                }
                if (key_of[cur] < 0 || key_of[cur] / n_expert != il || seen[key_of[cur] % n_expert] != seen_gen) {
                    s = cur;
                    remove_node(s, head_prob, tail_prob);
                    break;
                }
            }
        }
        // fallback to protected queue if all probationary slots are occupied by the current batch
        if (s < 0) {
            for (int32_t cur = head_prot; cur >= 0; cur = next[cur]) {
                if (!ephem_enabled && cur < n_ephem) {
                    continue;
                }
                if (key_of[cur] < 0 || key_of[cur] / n_expert != il || seen[key_of[cur] % n_expert] != seen_gen) {
                    s = cur;
                    remove_node(s, head_prot, tail_prot);
                    is_protected[s] = false;
                    if (key_of[s] >= 0) {
                        n_prot--;
                        n_prot_layer[key_of[s] / n_expert]--;
                    }
                    break;
                }
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

namespace {

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

typedef bool (*ggml_backend_register_host_buffer_t)(void * buffer, size_t size);
typedef void (*ggml_backend_unregister_host_buffer_t)(void * buffer);

struct pinned_buffer_registry {
    std::mutex mu;
    struct entry {
        int32_t ref_count = 0;
        size_t size = 0;
        ggml_backend_unregister_host_buffer_t unregister_fn = nullptr;
    };
    std::map<void *, entry> entries;

    bool add(void * page_base, size_t page_size, ggml_backend_register_host_buffer_t reg_fn, ggml_backend_unregister_host_buffer_t unreg_fn) {
        if (!reg_fn || !page_base || page_size == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu);
        auto & e = entries[page_base];
        if (e.ref_count == 0) {
            if (!reg_fn(page_base, page_size)) {
                entries.erase(page_base);
                return false;
            }
            e.size = page_size;
            e.unregister_fn = unreg_fn;
        }
        e.ref_count++;
        return true;
    }

    void remove(void * page_base) {
        std::lock_guard<std::mutex> lock(mu);
        auto it = entries.find(page_base);
        if (it != entries.end()) {
            if (--it->second.ref_count <= 0) {
                if (it->second.unregister_fn) {
                    it->second.unregister_fn(page_base);
                }
                entries.erase(it);
            }
        }
    }
};

static pinned_buffer_registry g_pinned_registry;

static void moe_cache_zero_tensor_bytes(ggml_tensor * tensor, size_t offset, size_t size) {
    if (tensor == nullptr || tensor->data == nullptr || size == 0) {
        return;
    }
    ggml_backend_tensor_memset(tensor, 0, offset, size);
}

template<typename T>
static inline T round_up(T x, T n) {
    return n == 0 ? x : ((x + n - 1) / n) * n;
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget, when it is split among the devices
        size_t budget = 0;     // the cache budget of the device in bytes

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf; // owned device buffer (when reuse_prefill == false)
        ggml_backend_buffer_t compute_buf = nullptr; // non-owning compute buffer (when reuse_prefill == true)
        size_t base_offset   = 0;
        size_t ephem_bytes   = 0;
        size_t persist_bytes = 0;
        size_t buf_size = 0;

        // slot maps in host memory
        ggml_context_ptr ctx_host;
        ggml_backend_buffer_ptr buf_host;
        size_t buf_host_size = 0;

        std::vector<void *> pinned_pages;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int32_t n_ephem = 0;
        std::vector<size_t> bank_offsets;
        size_t slot_stride = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t  hits    = 0;
        size_t  misses  = 0;
        size_t  bytes   = 0;
        int64_t time_us = 0;
    };

    static constexpr int64_t max_batch = llama_moe_cache::max_batch;

    const llama_model & model;
    size_t size;
    bool reuse_prefill;
    int32_t n_expert_used;
    double split_sum = 0.0;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;

    // views used by copy_experts and promotions
    ggml_context_ptr ctx_views;

    static ggml_context_ptr init_ctx(size_t n_tensors) {
        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr res(ggml_init(params));
        if (!res) {
            throw std::runtime_error("failed to create the MoE cache context");
        }
        return res;
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * device_sizes, bool reuse_prefill) :
            model(model), size(size), reuse_prefill(reuse_prefill),
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        std::vector<std::unordered_set<ggml_backend_buffer_t>> host_bufs(devices.size());
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
                if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                    host_bufs[id].insert(t->buffer);
                }
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // register host weight buffers for line-rate PCIe DMA transfers
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[id].backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto register_fn   = (ggml_backend_register_host_buffer_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
            auto unregister_fn = (ggml_backend_unregister_host_buffer_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
            if (register_fn && unregister_fn) {
                for (ggml_backend_buffer_t b : host_bufs[id]) {
                    void * raw_base = ggml_backend_buffer_get_base(b);
                    size_t raw_size = ggml_backend_buffer_get_size(b);
                    if (!raw_base || raw_size == 0) {
                        continue;
                    }
                    size_t page_sz = 4096;
#if defined(_WIN32)
                    SYSTEM_INFO si;
                    GetSystemInfo(&si);
                    page_sz = si.dwPageSize;
#elif defined(_SC_PAGESIZE)
                    const long sc_sz = sysconf(_SC_PAGESIZE);
                    if (sc_sz > 0) {
                        page_sz = (size_t) sc_sz;
                    }
#endif
                    uintptr_t ubase = (uintptr_t) raw_base;
                    uintptr_t page_base = ubase & ~(uintptr_t)(page_sz - 1);
                    size_t page_offset = ubase - page_base;
                    size_t page_size   = ((raw_size + page_offset + page_sz - 1) & ~(uintptr_t)(page_sz - 1));
                    if (g_pinned_registry.add((void *) page_base, page_size, register_fn, unregister_fn)) {
                        devices[id].pinned_pages.push_back((void *) page_base);
                    }
                }
                if (!devices[id].pinned_pages.empty()) {
                    LLAMA_LOG_INFO("%s: registered %zu pinned host buffers for line-rate DMA\n", __func__, devices[id].pinned_pages.size());
                }
            }
        }

        try {
            // if device_sizes is set, it holds the per-device budget indexed by the model device,
            // otherwise the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
            if (device_sizes == nullptr) {
                const float * tensor_split = model.tensor_split();
                const bool split_by_free = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
                split_sum = 0.0;
                for (device & d : devices) {
                    if (d.host_bytes == 0) {
                        continue;
                    }
                    ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                    if (split_by_free) {
                        size_t free;
                        size_t total;
                        ggml_backend_dev_memory(dev, &free, &total);
                        d.split = (double) free;
                    } else {
                        const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                        GGML_ASSERT(it != model.devices.end());
                        d.split = (double) tensor_split[it - model.devices.begin()];
                    }
                    split_sum += d.split;
                }
                if (split_sum == 0.0) {
                    // the devices do not report their free memory
                    for (device & d : devices) {
                        d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                        split_sum += d.split;
                    }
                }
                for (device & d : devices) {
                    if (d.host_bytes > 0) {
                        d.budget = (size_t) ((double) size*d.split/split_sum);
                    }
                }
            } else {
                for (device & d : devices) {
                    if (d.host_bytes == 0) {
                        continue;
                    }
                    const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == ggml_backend_get_device(d.backend); });
                    d.budget = it != model.devices.end() ? device_sizes[it - model.devices.begin()] : 0;
                }
            }

            ctx_views = init_ctx(2);

            if (!reuse_prefill) {
                size_t n_tensors = 0;
                for (size_t id = 0; id < devices.size(); ++id) {
                    if (devices[id].host_bytes == 0) {
                        continue;
                    }
                    n_tensors += allocate_device(id, nullptr, 0, 0, devices[id].budget);
                }
                if (n_tensors == 0) {
                    throw std::runtime_error("MoE cache is too small to hold the experts of one token");
                }
            }
        } catch (...) {
            for (device & d : devices) {
                for (void * page_base : d.pinned_pages) {
                    g_pinned_registry.remove(page_base);
                }
                d.pinned_pages.clear();
            }
            throw;
        }
    }

    ~impl() {
        for (device & d : devices) {
            for (void * page_base : d.pinned_pages) {
                g_pinned_registry.remove(page_base);
            }
        }
    }

    void unbind_device(size_t id) {
        device & d = devices[id];
        d.compute_buf = nullptr;
        for (group & g : groups) {
            if (g.id != (int32_t) id) {
                continue;
            }
            for (int32_t il : g.layers) {
                if (layers[il].slot_map != nullptr) {
                    layer_of.erase(layers[il].slot_map);
                }
                for (const ggml_tensor * t : layers[il].experts) {
                    bindings.erase(t);
                }
            }
            g.banks.clear();
            g.bank_offsets.clear();
            g.slot_stride = 0;
            g.n_slots = 0;
            g.n_ephem = 0;
            g.lru.init(model.layers.size(), g.ref[0]->ne[2], 0, 0, (int32_t) g.layers.size());
        }
        d.ctx.reset();
        d.buf.reset();
        d.buf_size = 0;
        d.ctx_host.reset();
        d.buf_host.reset();
        d.buf_host_size = 0;
    }

    void unbind() {
        for (size_t id = 0; id < devices.size(); ++id) {
            unbind_device(id);
        }
    }

    // (re)allocate the banks of the device, either in an owned buffer (ext_buf == nullptr)
    // or in a slice of the compute buffer starting at base_offset
    size_t allocate_device(size_t id, ggml_backend_buffer_t ext_buf, size_t base_offset, size_t ephem_bytes, size_t persist_bytes) {
        unbind_device(id);

        device & d = devices[id];
        d.compute_buf   = ext_buf;
        d.base_offset   = base_offset;
        d.ephem_bytes   = ephem_bytes;
        d.persist_bytes = persist_bytes;

        const size_t alignment = ggml_backend_buft_get_alignment(d.buft);

        // calculate bank offsets and slot stride for each group
        for (group & g : groups) {
            if (g.id != (int32_t) id) {
                continue;
            }
            size_t L = alignment;
            for (const ggml_tensor * t : g.ref) {
                L = std::lcm(L, ggml_type_size(t->type));
            }
            g.bank_offsets.resize(g.ref.size());
            g.bank_offsets[0] = 0;
            for (size_t ip = 0; ip + 1 < g.ref.size(); ++ip) {
                const size_t align_next = std::lcm(alignment, ggml_type_size(g.ref[ip + 1]->type));
                g.bank_offsets[ip + 1] = round_up(g.bank_offsets[ip] + g.ref[ip]->nb[2], align_next);
            }

            size_t max_extra = 0;
            for (const ggml_tensor * t : g.ref) {
                if (ggml_is_quantized(t->type) && (t->ne[0] % 512 != 0)) {
                    max_extra = std::max(max_extra, (size_t) ggml_row_size(t->type, 512 - (t->ne[0] % 512)));
                }
            }
            g.slot_stride = round_up(g.bank_offsets.back() + g.ref.back()->nb[2] + max_extra, L);

            for (size_t ip = 0; ip < g.ref.size(); ++ip) {
                const ggml_tensor * t = g.ref[ip];
                const size_t ts = ggml_type_size(t->type);
                GGML_ASSERT(g.slot_stride % ts == 0);
                GGML_ASSERT(g.bank_offsets[ip] % ts == 0);
                GGML_ASSERT(g.bank_offsets[ip] % alignment == 0);
            }
            GGML_ASSERT(g.slot_stride % alignment == 0);
        }

        group * primary_g = nullptr;
        if (reuse_prefill && persist_bytes > 0) {
            for (group & g : groups) {
                if (g.id == (int32_t) id && (primary_g == nullptr || g.host_bytes > primary_g->host_bytes)) {
                    primary_g = &g;
                }
            }
        }

        size_t n_tensors      = 0;
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            if (g.id != (int32_t) id) {
                continue;
            }
            g.n_slots = 0;
            g.n_ephem = 0;
            const int32_t n_expert  = g.ref[0]->ne[2];
            const int32_t max_slots = g.layers.size() * n_expert;

            if (reuse_prefill) {
                const size_t group_persist_bytes = (size_t) ((double) persist_bytes * g.host_bytes / d.host_bytes);

                if (persist_bytes > 0) {
                    // Case A: Hybrid mode
                    // Only primary group straddles S_pp and uses ephemeral headroom; secondary groups are pure persistent
                    if (&g == primary_g) {
                        int32_t n_persist = group_persist_bytes >= (size_t) g.slot_stride
                            ? std::max(0, (int32_t)(group_persist_bytes / g.slot_stride) - 1)
                            : 0;
                        n_persist = std::min(max_slots, n_persist);
                        const int32_t max_ephem = n_persist == 0
                            ? std::max(0, (int32_t)(ephem_bytes / g.slot_stride) - 1)
                            : (int32_t)(ephem_bytes / g.slot_stride);
                        g.n_ephem = std::min(max_slots - n_persist, max_ephem);
                        g.n_slots = g.n_ephem + n_persist;
                    } else {
                        g.n_ephem = 0;
                        int32_t n_persist = group_persist_bytes >= (size_t) g.slot_stride
                            ? std::max(0, (int32_t)(group_persist_bytes / g.slot_stride) - 1)
                            : 0;
                        if (n_persist > max_slots) {
                            n_persist = max_slots;
                        }
                        g.n_slots = n_persist;
                    }
                } else {
                    // Case B: Pure ephemeral mode
                    const size_t group_ephem_bytes = (size_t) ((double) ephem_bytes * g.host_bytes / d.host_bytes);
                    g.n_ephem = group_ephem_bytes >= (size_t) g.slot_stride
                        ? std::max(0, (int32_t)(group_ephem_bytes / g.slot_stride) - 1)
                        : 0;
                    if (g.n_ephem > max_slots) {
                        g.n_ephem = max_slots;
                    }
                    g.n_slots = g.n_ephem;
                }
            } else {
                // Standalone dedicated buffer mode
                const size_t group_persist_bytes = (size_t) ((double) persist_bytes * g.host_bytes / d.host_bytes);
                g.n_ephem = 0;
                int32_t n_persist = std::max(0, (int32_t)(group_persist_bytes / g.slot_stride) - 1);
                if (n_persist > max_slots) {
                    n_persist = max_slots;
                }
                g.n_slots = n_persist;
            }

            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers (need %d slots), they are not cached\n",
                    __func__, g.layers.size(), n_expert_used);
                g.n_slots = 0;
                g.n_ephem = 0;
                continue;
            }

            g.lru.init(model.layers.size(), n_expert, g.n_slots, g.n_ephem, (int32_t) g.layers.size());
            n_tensors      += g.ref.size();
            n_tensors_host += g.layers.size();
        }

        if (n_tensors == 0) {
            if (reuse_prefill) {
                LLAMA_LOG_WARN("%s: MoE cache is too small to hold the experts of one token, cache disabled\n", __func__);
            }
            return 0;
        }

        d.ctx      = init_ctx(n_tensors);
        d.ctx_host = init_ctx(n_tensors_host);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = d.ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                bank->nb[2] = g.slot_stride;
                bank->nb[3] = (g.n_slots + 1) * g.slot_stride;
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    bindings[l.experts[ip]] = { il, (int32_t) ip, g.banks[ip] };
                }
                l.slot_map = ggml_new_tensor_2d(d.ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                d.buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            d.buf_size += (g.n_slots + 1) * g.slot_stride;
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
            d.buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                t->buffer = d.buf.get();
            }
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_host.get(), t)) {
                t->buffer = d.buf_host.get();
            }
        } else if (ext_buf == nullptr) {
            d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, d.buf_size));
            d.buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_host.get(), buft_host));
            if (!d.buf || !d.buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(d.buf.get(), 0);
            ggml_backend_buffer_clear(d.buf_host.get(), 0xff); // all slots are -1
            d.buf_host_size = ggml_backend_buffer_get_size(d.buf_host.get());

            size_t group_offset = 0;
            for (group & g : groups) {
                if (g.id != (int32_t) id || g.n_slots == 0) {
                    continue;
                }
                for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    void * bank_addr = (char *) ggml_backend_buffer_get_base(d.buf.get()) + group_offset + g.bank_offsets[ip];
                    ggml_backend_tensor_alloc(d.buf.get(), bank, bank_addr);

                    const size_t matrix_bytes = g.ref[ip]->nb[2];
                    moe_cache_zero_tensor_bytes(bank, g.n_slots * bank->nb[2], matrix_bytes);
                }
                group_offset += (g.n_slots + 1) * g.slot_stride;
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
            ggml_backend_buffer_set_usage(d.buf.get(),      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_set_usage(d.buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        } else {
            d.buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_host.get(), buft_host));
            if (!d.buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache host buffer");
            }
            ggml_backend_buffer_clear(d.buf_host.get(), 0xff);
            d.buf_host_size = ggml_backend_buffer_get_size(d.buf_host.get());

            const size_t s_pp = base_offset + ephem_bytes;
            size_t cur_persist = 0;
            size_t cur_ephem_used = 0;

            auto alloc_group_banks = [&](group & g, size_t group_base) {
                for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    void * bank_addr = (char *) ggml_backend_buffer_get_base(ext_buf) + group_base + g.bank_offsets[ip];
                    ggml_backend_tensor_alloc(ext_buf, bank, bank_addr);

                    const size_t matrix_bytes = g.ref[ip]->nb[2];
                    moe_cache_zero_tensor_bytes(bank, g.n_slots * bank->nb[2], matrix_bytes);
                }
                for (int32_t il : g.layers) {
                    layer & l = layers[il];
                    if (l.slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) l.slot_map->data;
                    }
                }
            };

            if (persist_bytes > 0) {
                // Case A: Hybrid mode
                // First allocate primary group straddling S_pp
                if (primary_g != nullptr && primary_g->n_slots > 0) {
                    const int32_t n_persist = primary_g->n_slots - primary_g->n_ephem;
                    if (n_persist > 0) {
                        const size_t group_base = s_pp - (size_t) primary_g->n_ephem * primary_g->slot_stride;
                        GGML_ASSERT(group_base >= base_offset);
                        alloc_group_banks(*primary_g, group_base);
                        cur_persist += (size_t) (n_persist + 1) * primary_g->slot_stride;
                    } else {
                        const size_t group_base = s_pp - (size_t) (primary_g->n_ephem + 1) * primary_g->slot_stride;
                        GGML_ASSERT(group_base >= base_offset);
                        alloc_group_banks(*primary_g, group_base);
                    }
                }
                // Then allocate secondary groups sequentially in persistent space
                for (group & g : groups) {
                    if (g.id != (int32_t) id || g.n_slots == 0 || &g == primary_g) {
                        continue;
                    }
                    const size_t group_base = s_pp + cur_persist;
                    alloc_group_banks(g, group_base);
                    cur_persist += (size_t) (g.n_slots + 1) * g.slot_stride;
                }
                GGML_ASSERT(cur_persist <= persist_bytes);
                GGML_ASSERT(s_pp + cur_persist <= ggml_backend_buffer_get_size(ext_buf));
            } else {
                // Case B: Pure ephemeral mode
                for (group & g : groups) {
                    if (g.id != (int32_t) id || g.n_slots == 0) {
                        continue;
                    }
                    const size_t group_span = (size_t) (g.n_ephem + 1) * g.slot_stride;
                    cur_ephem_used += group_span;
                    GGML_ASSERT(cur_ephem_used <= ephem_bytes);
                    const size_t group_base = s_pp - cur_ephem_used;
                    GGML_ASSERT(group_base >= base_offset);
                    alloc_group_banks(g, group_base);
                }
            }
            ggml_backend_buffer_set_usage(d.buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        if (reuse_prefill) {
            if (persist_bytes > 0) {
                LLAMA_LOG_INFO("%s: device %zu (%s): hybrid MoE cache (ephemeral = %.2f MiB, persistent = %.2f MiB) for %.2f MiB host experts\n", __func__,
                    id, ggml_backend_buft_name(d.buft), ephem_bytes/1024.0/1024.0, persist_bytes/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            } else {
                LLAMA_LOG_INFO("%s: device %zu (%s): ephemeral MoE cache size = %8.2f MiB for %.2f MiB host experts\n", __func__,
                    id, ggml_backend_buft_name(d.buft), ephem_bytes/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            }
        } else {
            LLAMA_LOG_INFO("%s: device %zu (%s): MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                id, ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.id != (int32_t) id) {
                continue;
            }
            if (g.n_ephem > 0 && g.n_slots > g.n_ephem) {
                LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%d ephem + %d persist, %.1f%%)\n", __func__,
                    g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots,
                    g.n_ephem, g.n_slots - g.n_ephem,
                    100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
            } else {
                LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                    g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots,
                    100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
            }
        }

        return n_tensors;
    }

    void prepare_reserve() {
        unbind();
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            size_t n_tensors      = 0;
            size_t n_tensors_host = 0;
            for (group & g : groups) {
                if (g.id != (int32_t) id) {
                    continue;
                }
                const int32_t n_expert = g.ref[0]->ne[2];
                g.n_slots = n_expert;
                g.n_ephem = g.n_slots;
                g.lru.init(model.layers.size(), n_expert, g.n_slots, g.n_ephem, (int32_t) g.layers.size());
                g.lru.ephem_enabled = true;
                n_tensors      += g.ref.size();
                n_tensors_host += g.layers.size();
            }
            if (n_tensors == 0) {
                continue;
            }
            d.ctx      = init_ctx(n_tensors);
            d.ctx_host = init_ctx(n_tensors_host);
            ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
            d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
            d.buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                if (g.id != (int32_t) id || g.n_slots == 0) {
                    continue;
                }
                ggml_context * ctx = d.ctx.get();
                for (const ggml_tensor * t : g.ref) {
                    ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                    bank->buffer = d.buf.get();
                    g.banks.push_back(bank);
                }
                for (int32_t il : g.layers) {
                    layer & l = layers[il];
                    l.ig      = (int32_t) ig;
                    l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        bindings[l.experts[ip]] = { il, (int32_t) ip, g.banks[ip] };
                    }
                    l.slot_map = ggml_new_tensor_2d(d.ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.slot_map->buffer = d.buf_host.get();
                    layer_of[l.slot_map] = il;
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    size_t cache_size(ggml_backend_t backend) const {
        const auto it = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        if (it == devices.end() || it->host_bytes == 0) {
            return 0;
        }
        return it->budget;
    }

    bool bind_compute_buffer(ggml_backend_t backend, ggml_backend_buffer_t compute_buf, size_t base_offset, size_t ephem_bytes, size_t persist_bytes) {
        const auto it = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.backend == backend; });
        if (it == devices.end() || it->host_bytes == 0) {
            return false;
        }
        if (!model.hparams.no_alloc && compute_buf == nullptr) {
            return false;
        }
        if (compute_buf != nullptr) {
            GGML_ASSERT(base_offset + ephem_bytes + persist_bytes <= ggml_backend_buffer_get_size(compute_buf));
        }
        return allocate_device((size_t) (it - devices.begin()), compute_buf, base_offset, ephem_bytes, persist_bytes) > 0;
    }

    void set_ephem_enabled(bool enabled) {
        for (group & g : groups) {
            const bool was_enabled = g.lru.ephem_enabled;
            g.lru.set_ephem_enabled(enabled);
            // Re-zero guard slot only when transitioning into decode in pure ephemeral mode
            if (enabled && !was_enabled && devices[g.id].persist_bytes == 0) {
                for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    if (bank && bank->data) {
                        const size_t matrix_bytes = g.ref[ip]->nb[2];
                        moe_cache_zero_tensor_bytes(bank, g.n_slots * bank->nb[2], matrix_bytes);
                    }
                }
            }
        }
    }

    bool is_enabled() const {
        if (groups.empty()) {
            return false;
        }
        if (reuse_prefill) {
            bool has_slots = false;
            for (const auto & g : groups) {
                if (g.n_slots > 0) {
                    has_slots = true;
                    if (g.n_slots > g.n_ephem) {
                        return true;
                    }
                }
            }
            if (!has_slots) {
                return false;
            }
            return is_ephem_enabled();
        }
        return true;
    }

    bool is_ephem_enabled() const {
        return !groups.empty() && groups[0].lru.ephem_enabled;
    }

    int32_t max_ephem_tokens() const {
        int32_t max_tok = max_batch;
        if (reuse_prefill) {
            for (const auto & g : groups) {
                if (g.n_slots > 0 && n_expert_used > 0) {
                    max_tok = std::min(max_tok, g.n_slots / (int32_t) n_expert_used);
                }
            }
        }
        return std::max(0, max_tok);
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (!is_enabled() || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.ig];

        const int32_t active_slots = g.lru.ephem_enabled ? g.n_slots : g.n_slots - g.n_ephem;
        if (n_tokens == 0 || n_tokens > max_batch || active_slots <= 0 || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > active_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        if (!is_enabled()) {
            return nullptr;
        }
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        GGML_UNUSED(last);
        if (groups.empty()) {
            return 0;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend || g.n_slots == 0 || b.ip >= (int32_t) g.banks.size()) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < g.lru.n_ephem) {
            return 0;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_2d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_2d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  dst->nb[1],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += 1;
        stats_copy.bytes += ggml_nbytes(src_view);

        return 1;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, promotions, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        if (++g.lru.copy_count % moe_cache_lru::DECAY_INTERVAL == 0) {
            g.lru.decay();
        }

        // execute promotions via D2D copy
        for (const auto & p : promotions) {
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                ggml_tensor * bank = g.banks[ip];
                ggml_reset(ctx_views.get());
                ggml_tensor * src_view = ggml_view_2d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], p.src_slot*bank->nb[2]);
                ggml_tensor * dst_view = ggml_view_2d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], p.dst_slot*bank->nb[2]);
                ggml_backend_view_init(src_view);
                ggml_backend_view_init(dst_view);
                ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
            }
        }

        // upload the missing experts individually to respect interleaved slot strides
        const int64_t t_upload_start = ggml_time_us();
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size(); ++i) {
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*bank->nb[2], expert_size);
                bytes += expert_size;
            }
        }
        if (!fills.empty()) {
            ggml_backend_synchronize(backend);
        }
        const int64_t t_upload = ggml_time_us() - t_upload_start;

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits    += n_hit;
        st.misses  += fills.size();
        st.bytes   += bytes;
        st.time_us += t_upload;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void print_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            const double throughput = st.time_us > 0 ? (double) st.bytes / (double) st.time_us * 1e6 / 1024.0 / 1024.0 / 1024.0 : 0.0;
            if (throughput > 0.0) {
                LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB (%.2f GB/s)\n",
                    name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0, throughput);
            } else {
                LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                    name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
            }
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }

    void get_stats(size_t & out_hits, size_t & out_misses, size_t & out_bytes, int64_t & out_time_us, int32_t & out_slots) const {
        out_hits    = stats_small.hits + stats_large.hits;
        out_misses  = stats_small.misses + stats_large.misses;
        out_bytes   = stats_small.bytes + stats_large.bytes;
        out_time_us = stats_small.time_us + stats_large.time_us;
        out_slots   = 0;
        for (const auto & g : groups) {
            out_slots += g.n_slots;
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * device_sizes, bool reuse_prefill) :
    pimpl(new impl(model, backends, bufts, size, device_sizes, reuse_prefill)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

size_t llama_moe_cache::cache_size(ggml_backend_t backend) const {
    return pimpl->cache_size(backend);
}

bool llama_moe_cache::bind_compute_buffer(ggml_backend_t backend, ggml_backend_buffer_t compute_buf, size_t base_offset, size_t ephem_bytes, size_t persist_bytes) {
    return pimpl->bind_compute_buffer(backend, compute_buf, base_offset, ephem_bytes, persist_bytes);
}

void llama_moe_cache::unbind() {
    pimpl->unbind();
}

void llama_moe_cache::set_ephem_enabled(bool enabled) {
    pimpl->set_ephem_enabled(enabled);
}

bool llama_moe_cache::is_enabled() const {
    return pimpl->is_enabled();
}

bool llama_moe_cache::is_ephem_enabled() const {
    return pimpl->is_ephem_enabled();
}

int32_t llama_moe_cache::max_ephem_tokens() const {
    return pimpl->max_ephem_tokens();
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

void llama_moe_cache::prepare_reserve() {
    pimpl->prepare_reserve();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.compute_buf != nullptr || pimpl->reuse_prefill) {
            if (d.persist_bytes > 0 && d.buft != nullptr) {
                res[d.buft] += d.persist_bytes;
            }
        } else if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
        if (d.buf_host) {
            res[ggml_backend_buffer_get_type(d.buf_host.get())] += d.buf_host_size;
        }
    }
    return res;
}

void llama_moe_cache::print_stats() const {
    pimpl->print_stats();
}

void llama_moe_cache::get_stats(size_t & hits, size_t & misses, size_t & bytes, int64_t & time_us, int32_t & n_slots) const {
    pimpl->get_stats(hits, misses, bytes, time_us, n_slots);
}
