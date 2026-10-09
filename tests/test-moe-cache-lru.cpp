#include "../src/llama-moe-cache.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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

int main() {
    test_n_overlap();
    printf("All moe_cache_lru tests passed successfully!\n");
    return 0;
}
