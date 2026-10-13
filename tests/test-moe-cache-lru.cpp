#include "../src/llama-moe-cache.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void validate_lru_invariants(const moe_cache_lru & lru) {
    int count_prob = 0;
    std::vector<bool> in_prob(lru.n_slots, false);
    if (lru.head_prob >= 0) {
        GGML_ASSERT(lru.prev[lru.head_prob] == -1);
        int cur = lru.head_prob;
        int prev_node = -1;
        while (cur >= 0) {
            GGML_ASSERT(!in_prob[cur]);
            in_prob[cur] = true;
            GGML_ASSERT(!lru.is_protected[cur]);
            GGML_ASSERT(lru.prev[cur] == prev_node);
            prev_node = cur;
            if (lru.next[cur] == -1) {
                GGML_ASSERT(cur == lru.tail_prob);
            }
            cur = lru.next[cur];
            count_prob++;
        }
    } else {
        GGML_ASSERT(lru.tail_prob == -1);
    }

    int count_prot = 0;
    std::vector<bool> in_prot(lru.n_slots, false);
    if (lru.head_prot >= 0) {
        GGML_ASSERT(lru.prev[lru.head_prot] == -1);
        int cur = lru.head_prot;
        int prev_node = -1;
        while (cur >= 0) {
            GGML_ASSERT(!in_prot[cur]);
            in_prot[cur] = true;
            GGML_ASSERT(!in_prob[cur]);
            GGML_ASSERT(lru.is_protected[cur]);
            GGML_ASSERT(lru.prev[cur] == prev_node);
            prev_node = cur;
            if (lru.next[cur] == -1) {
                GGML_ASSERT(cur == lru.tail_prot);
            }
            cur = lru.next[cur];
            count_prot++;
        }
    } else {
        GGML_ASSERT(lru.tail_prot == -1);
    }

    for (int s = 0; s < lru.n_slots; ++s) {
        GGML_ASSERT(in_prob[s] || in_prot[s]);
        GGML_ASSERT(!(in_prob[s] && in_prot[s]));
    }

    for (int s = 0; s < lru.n_slots; ++s) {
        int key = lru.key_of[s];
        if (key >= 0) {
            int il = key / lru.n_expert;
            int e = key % lru.n_expert;
            GGML_ASSERT(lru.slot_map[il][e] == s);
        }
    }

    int occupied_prot = 0;
    std::vector<int32_t> expected_layer_counts(lru.slot_map.size(), 0);
    for (int s = 0; s < lru.n_slots; ++s) {
        if (in_prot[s] && lru.key_of[s] >= 0) {
            occupied_prot++;
            expected_layer_counts[lru.key_of[s] / lru.n_expert]++;
        }
    }
    GGML_ASSERT(occupied_prot == lru.n_prot);
    for (size_t il = 0; il < lru.slot_map.size(); ++il) {
        GGML_ASSERT(lru.n_prot_layer[il] == expected_layer_counts[il]);
    }
}

static void validate_batch_slots(const moe_cache_lru & lru, int32_t il, const int32_t * ids, size_t n_ids) {
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t e = ids[i];
        int32_t s = lru.slot_map[il][e];
        GGML_ASSERT(s >= 0 && s < lru.n_slots);
        GGML_ASSERT(lru.key_of[s] == il * lru.n_expert + e);
    }
}

static void test_b1_overflow_and_invalidation() {
    printf("Running test_b1_overflow_and_invalidation...\n");
    const int32_t n_layer = 1;
    const int32_t n_expert = 16;
    const int32_t n_ephem = 4;
    const int32_t n_slots = 8; // 4 ephem, 4 persistent

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem);

    std::vector<int32_t> map_data(n_expert, -1);
    lru.slot_map[0] = map_data.data();

    validate_lru_invariants(lru);

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    // Step 1: Miss overflow with 6 distinct experts (> n_ephem=4)
    int32_t batch1[] = { 0, 1, 2, 3, 4, 5 };
    bool ok = lru.plan(0, batch1, 6, fills, promotions, n_hit);
    GGML_ASSERT(ok);
    GGML_ASSERT(fills.size() == 6);
    GGML_ASSERT(promotions.empty());
    validate_lru_invariants(lru);

    // Two persistent slots (s >= 4) must now be in prob list
    int persistent_in_prob = 0;
    for (int s = lru.head_prob; s >= 0; s = lru.next[s]) {
        if (s >= n_ephem) {
            persistent_in_prob++;
        }
    }
    GGML_ASSERT(persistent_in_prob == 2);

    // Step 2: Ephemeral invalidation (prefill occurs)
    lru.invalidate_ephemeral();
    validate_lru_invariants(lru);

    // Persistent slots in prob must survive and still be present in prob list
    persistent_in_prob = 0;
    for (int s = lru.head_prob; s >= 0; s = lru.next[s]) {
        if (s >= n_ephem) {
            persistent_in_prob++;
        }
    }
    GGML_ASSERT(persistent_in_prob == 2);

    // Step 3: Hits on the surviving persistent slots in prob must promote to prot
    int32_t hit_expert = -1;
    for (int e = 0; e < n_expert; ++e) {
        if (lru.slot_map[0][e] >= n_ephem) {
            hit_expert = e;
            break;
        }
    }
    GGML_ASSERT(hit_expert >= 0);
    int32_t batch2[] = { hit_expert };
    ok = lru.plan(0, batch2, 1, fills, promotions, n_hit);
    GGML_ASSERT(ok);
    GGML_ASSERT(n_hit == 1);
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.is_protected[lru.slot_map[0][hit_expert]]);

    printf("  Passed.\n");
}

static void test_b2_slru_accounting_and_decay() {
    printf("Running test_b2_slru_accounting_and_decay...\n");
    const int32_t n_layer = 1;
    const int32_t n_expert = 16;
    const int32_t n_ephem = 4;
    const int32_t n_slots = 8; // 4 ephem, 4 persistent

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem);

    std::vector<int32_t> map_data(n_expert, -1);
    lru.slot_map[0] = map_data.data();

    GGML_ASSERT(lru.n_prot == 0);
    GGML_ASSERT(lru.n_prot_max == 4);

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    // Fill ephemeral slots with experts 0, 1, 2, 3
    int32_t batch1[] = { 0, 1, 2, 3 };
    lru.plan(0, batch1, 4, fills, promotions, n_hit);
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot == 0);

    // Hit expert 0: promotes from ephemeral to persistent, n_prot must become 1
    int32_t batch2[] = { 0 };
    lru.plan(0, batch2, 1, fills, promotions, n_hit);
    validate_lru_invariants(lru);
    GGML_ASSERT(n_hit == 1);
    GGML_ASSERT(promotions.size() == 1);
    GGML_ASSERT(lru.n_prot == 1);

    // Hit experts 1, 2, 3: promote them all
    for (int32_t e : { 1, 2, 3 }) {
        int32_t b[] = { e };
        lru.plan(0, b, 1, fills, promotions, n_hit);
        validate_lru_invariants(lru);
    }
    GGML_ASSERT(lru.n_prot == 4);

    // Decay demotes n_prot / 4 = 1 slot to prob
    lru.decay();
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot == 3);

    // Invalidate ephemeral: demoted persistent slot must survive in prob
    lru.invalidate_ephemeral();
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot == 3);

    printf("  Passed.\n");
}

static void test_stress_cycles() {
    printf("Running test_stress_cycles...\n");
    const int32_t n_layer = 2;
    const int32_t n_expert = 32;
    const int32_t n_ephem = 8;
    const int32_t n_slots = 16;

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem);

    std::vector<std::vector<int32_t>> maps(n_layer, std::vector<int32_t>(n_expert, -1));
    for (int32_t il = 0; il < n_layer; ++il) {
        lru.slot_map[il] = maps[il].data();
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> layer_dist(0, n_layer - 1);
    std::uniform_int_distribution<int> expert_dist(0, n_expert - 1);
    std::uniform_int_distribution<int> batch_size_dist(1, n_slots);
    std::uniform_int_distribution<int> action_dist(0, 9);

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    for (int iter = 0; iter < 10000; ++iter) {
        int action = action_dist(rng);
        if (action == 0) {
            // Invalidate ephemeral
            lru.invalidate_ephemeral();
        } else if (action == 1) {
            // Decay
            lru.decay();
        } else {
            // Plan batch
            int il = layer_dist(rng);
            int batch_size = batch_size_dist(rng);
            std::vector<int32_t> ids(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                ids[i] = expert_dist(rng);
            }
            bool ok = lru.plan(il, ids.data(), ids.size(), fills, promotions, n_hit);
            GGML_ASSERT(ok);
            validate_batch_slots(lru, il, ids.data(), ids.size());
        }
        validate_lru_invariants(lru);
    }

    printf("  Passed.\n");
}

static void test_n_overlap() {
    printf("Running test_n_overlap...\n");
    // no overlap
    GGML_ASSERT(moe_cache_n_overlap(100, 50, 8) == 0);
    GGML_ASSERT(moe_cache_n_overlap(0, 0, 8) == 0);
    // exact fit: slots [0,8), [8,16), [16,24) fit, slot 3 starts exactly at 24 and does not overlap
    GGML_ASSERT(moe_cache_n_overlap(0, 24, 8) == 3);
    // straddling: [0, 25) additionally contains 1 byte of slot [24, 32)
    GGML_ASSERT(moe_cache_n_overlap(0, 25, 8) == 4);
    // nonzero offset
    GGML_ASSERT(moe_cache_n_overlap(10, 25, 8) == 2);
    GGML_ASSERT(moe_cache_n_overlap(24, 25, 8) == 1); // only the straddling slot
    // clamp to int32
    GGML_ASSERT(moe_cache_n_overlap(0, SIZE_MAX, 1) == INT32_MAX);
}

static void test_layer_adaptive_budgeting() {
    printf("Running test_layer_adaptive_budgeting...\n");
    const int32_t n_layer = 2;
    const int32_t n_expert = 16;
    const int32_t n_ephem = 4;
    const int32_t n_slots = 8; // 4 ephem, 4 persistent (fair_share = 4 / 2 = 2)

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem, n_layer);

    std::vector<std::vector<int32_t>> maps(n_layer, std::vector<int32_t>(n_expert, -1));
    for (int32_t il = 0; il < n_layer; ++il) {
        lru.slot_map[il] = maps[il].data();
    }
    validate_lru_invariants(lru);

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    // Fill persistent slots with 3 experts from layer 0 and 1 from layer 1
    int32_t b0[] = { 0, 1, 2 };
    lru.plan(0, b0, 3, fills, promotions, n_hit);
    for (int32_t e : { 0, 1, 2 }) {
        int32_t hit[] = { e };
        lru.plan(0, hit, 1, fills, promotions, n_hit);
    }
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot_layer[0] == 3);

    int32_t b1[] = { 0 };
    lru.plan(1, b1, 1, fills, promotions, n_hit);
    lru.plan(1, b1, 1, fills, promotions, n_hit);
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot_layer[0] == 3);
    GGML_ASSERT(lru.n_prot_layer[1] == 1);
    GGML_ASSERT(lru.n_prot == 4);

    // Layer 0 is over fair share (3 > 2), so promoting another layer 0 expert must recycle layer 0's own slot
    int32_t b0_new[] = { 3 };
    lru.plan(0, b0_new, 1, fills, promotions, n_hit);
    lru.plan(0, b0_new, 1, fills, promotions, n_hit);
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot_layer[0] == 3);
    GGML_ASSERT(lru.n_prot_layer[1] == 1);
    GGML_ASSERT(lru.slot_map[1][0] >= 0); // layer 1 slot preserved
    GGML_ASSERT(lru.slot_map[0][0] == -1); // layer 0 expert 0 self-recycled
    GGML_ASSERT(lru.slot_map[0][3] >= 0); // layer 0 expert 3 now in persistent

    // Layer 1 is under fair share (1 < 2), so promoting layer 1 must evict from over-budget layer 0
    int32_t b1_new[] = { 1 };
    lru.plan(1, b1_new, 1, fills, promotions, n_hit);
    lru.plan(1, b1_new, 1, fills, promotions, n_hit);
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot_layer[0] == 2);
    GGML_ASSERT(lru.n_prot_layer[1] == 2);
    GGML_ASSERT(lru.slot_map[1][1] >= 0);

    printf("  Passed.\n");
}

static void test_multi_expert_batch_promotions() {
    printf("Running test_multi_expert_batch_promotions...\n");
    const int32_t n_layer = 2;
    const int32_t n_expert = 16;
    const int32_t n_ephem = 6;
    const int32_t n_slots = 12; // 6 ephem, 6 persistent (fair_share = 6 / 2 = 3)

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem, n_layer);

    std::vector<std::vector<int32_t>> maps(n_layer, std::vector<int32_t>(n_expert, -1));
    for (int32_t il = 0; il < n_layer; ++il) {
        lru.slot_map[il] = maps[il].data();
    }
    validate_lru_invariants(lru);

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    // Fill persistent slots: Layer 0 gets 4 slots (over-share), Layer 1 gets 2 slots (total 6)
    int32_t b0_init[] = { 0, 1, 2, 3 };
    lru.plan(0, b0_init, 4, fills, promotions, n_hit);
    for (int32_t e : { 0, 1, 2, 3 }) {
        int32_t hit[] = { e };
        lru.plan(0, hit, 1, fills, promotions, n_hit);
    }

    int32_t b1_init[] = { 0, 1 };
    lru.plan(1, b1_init, 2, fills, promotions, n_hit);
    for (int32_t e : { 0, 1 }) {
        int32_t hit[] = { e };
        lru.plan(1, hit, 1, fills, promotions, n_hit);
    }
    validate_lru_invariants(lru);
    GGML_ASSERT(lru.n_prot_layer[0] == 4);
    GGML_ASSERT(lru.n_prot_layer[1] == 2);
    GGML_ASSERT(lru.n_prot == 6);

    // Put experts 4, 5, 6 of Layer 0 into ephemeral
    int32_t b0_eph[] = { 4, 5, 6 };
    lru.plan(0, b0_eph, 3, fills, promotions, n_hit);
    validate_lru_invariants(lru);

    // Multi-expert query with K=4:
    // Expert 0 (in prot) and 4, 5, 6 (in ephemeral, promoted in same plan call)
    int32_t b0_multi[] = { 0, 4, 5, 6 };
    bool ok = lru.plan(0, b0_multi, 4, fills, promotions, n_hit);
    GGML_ASSERT(ok);
    GGML_ASSERT(n_hit == 4);
    GGML_ASSERT(promotions.size() == 3);

    // Verify all promotions have distinct destination slots
    std::vector<int32_t> dst_slots;
    for (const auto & p : promotions) {
        GGML_ASSERT(std::find(dst_slots.begin(), dst_slots.end(), p.dst_slot) == dst_slots.end());
        dst_slots.push_back(p.dst_slot);
    }

    // Verify all 4 active experts have distinct, valid slots and none were self-evicted
    std::vector<int32_t> active_slots;
    for (int32_t e : b0_multi) {
        int32_t s = lru.slot_map[0][e];
        GGML_ASSERT(s >= 0);
        GGML_ASSERT(std::find(active_slots.begin(), active_slots.end(), s) == active_slots.end());
        active_slots.push_back(s);
    }
    validate_lru_invariants(lru);

    printf("  Passed.\n");
}

static void test_batch_expert_fallback_preservation() {
    printf("Running test_batch_expert_fallback_preservation...\n");
    // 1 layer, 16 experts, 2 ephem, 2 persistent (total 4 slots)
    const int32_t n_layer = 1;
    const int32_t n_expert = 16;
    const int32_t n_ephem = 2;
    const int32_t n_slots = 4;

    moe_cache_lru lru;
    lru.init(n_layer, n_expert, n_slots, n_ephem, n_layer);

    std::vector<int32_t> map_data(n_expert, -1);
    lru.slot_map[0] = map_data.data();

    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::promotion> promotions;
    size_t n_hit = 0;

    // Fill persistent slots with experts 0 and 1
    int32_t b_init[] = { 0, 1 };
    lru.plan(0, b_init, 2, fills, promotions, n_hit);
    for (int32_t e : { 0, 1 }) {
        int32_t hit[] = { e };
        lru.plan(0, hit, 1, fills, promotions, n_hit);
    }
    // Put expert 2 into ephemeral
    int32_t b_eph[] = { 2 };
    lru.plan(0, b_eph, 1, fills, promotions, n_hit);

    // Multi-expert query requesting {0, 1, 2}:
    // 0 and 1 are hits in protected.
    // 2 is a hit in ephemeral. Promotion needs a persistent victim.
    // Since all persistent slots are active batch experts,
    // fallback must skip promotion so neither 0 nor 1 is evicted.
    int32_t b_all[] = { 0, 1, 2 };
    bool ok = lru.plan(0, b_all, 3, fills, promotions, n_hit);
    GGML_ASSERT(ok);
    GGML_ASSERT(n_hit == 3);
    GGML_ASSERT(promotions.empty());
    validate_batch_slots(lru, 0, b_all, 3);
    validate_lru_invariants(lru);

    printf("  Passed.\n");
}

int main() {
    test_n_overlap();
    test_b1_overflow_and_invalidation();
    test_b2_slru_accounting_and_decay();
    test_layer_adaptive_budgeting();
    test_multi_expert_batch_promotions();
    test_batch_expert_fallback_preservation();
    test_stress_cycles();
    printf("All moe_cache_lru tests passed successfully!\n");
    return 0;
}
