// WARN: This is simulated for now.

#include "fastest/custom_tests.h"
#include <time.h>
#include <stdlib.h>

#define STEP 1000     // µs per index step
#define OVERHEAD_1 1  // multiplier for bine_bcast
#define OVERHEAD_2 2  // multiplier for ring_bcast

__attribute__((constructor))
static void seed_rng(void) {
    srand(time(NULL));
}

void sleep_us(int us) {
    struct timespec ts;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (us % 1000000) * 1000;
    nanosleep(&ts, NULL);
}

void sleep_idx(int idx, int overhead) {
    int to_sleep = STEP * idx * overhead + (rand() % STEP) - (STEP / 2);
    sleep_us(to_sleep);
}

FASTEST_CUSTOMTEST_INLINE("scaling/bine_bcast/0", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(0, OVERHEAD_1);
})

FASTEST_CUSTOMTEST_INLINE("scaling/bine_bcast/1", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(1, OVERHEAD_1);
})

FASTEST_CUSTOMTEST_INLINE("scaling/bine_bcast/2", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(2, OVERHEAD_1);
})

FASTEST_CUSTOMTEST_INLINE("scaling/bine_bcast/3", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(3, OVERHEAD_1);
})

FASTEST_CUSTOMTEST_INLINE("scaling/bine_bcast/4", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(4, OVERHEAD_1);
})

FASTEST_CUSTOMTEST_INLINE("scaling/ring_bcast/0", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(0, OVERHEAD_2);
})

FASTEST_CUSTOMTEST_INLINE("scaling/ring_bcast/1", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(1, OVERHEAD_2);
})

FASTEST_CUSTOMTEST_INLINE("scaling/ring_bcast/2", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(2, OVERHEAD_2);
})

FASTEST_CUSTOMTEST_INLINE("scaling/ring_bcast/3", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(3, OVERHEAD_2);
})

FASTEST_CUSTOMTEST_INLINE("scaling/ring_bcast/4", FASTEST_TIME_NS, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    sleep_idx(4, OVERHEAD_2);
})
