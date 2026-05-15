#include <stdio.h>

extern "C" {
#include "fastest/tests.h"
#include "fastest/custom_tests.h"
}

#include "logging.h"
#include "bine.h"
#include "device/bine_utils.h"

/**
 * INFO: We don't need to test 20 steps each time (>1M gpus).
 * This reduces test scope: 12 steps = 4096 GPUs, sufficient for regular testing.
 */
#ifdef BINE_MAX_STEPS
    #undef BINE_MAX_STEPS
#endif /* ifdef BINE_MAX_STEPS */
#define BINE_MAX_STEPS 12

void print_butterfly(int nRanks, int steps, int *partners, int *index, int *order) {
    // Virtual index mapping
    printf(ANSI_BOLD "=== Virtual index mapping ===\n" ANSI_RESET);
    printf(ANSI_DIM "  rank |  virt\n  -----|------\n" ANSI_RESET);
    for (int r = 0; r < nRanks; ++r)
        printf(ANSI_DIM "  r%-3d |" ANSI_RESET "  %d\n", r, index[r]);
    printf("\n");

    // Partner table
    printf(ANSI_BOLD "=== Partner table ===\n" ANSI_RESET);
    printf(ANSI_DIM "  rank |");
    for (int s = 0; s < steps; ++s) printf("  s%-2d  ", s);
    printf("\n  -----|");
    for (int s = 0; s < steps; ++s) printf("-------");
    printf("\n" ANSI_RESET);

    for (int r = 0; r < nRanks; ++r) {
        printf(ANSI_DIM "  r%-3d |" ANSI_RESET, r);
        for (int step = 0; step < steps; ++step) {
            int p = partners[idx(r, step, steps)];
            if (p < 0)
                printf(ANSI_IDLE "   .   " ANSI_RESET);
            else
                printf(ANSI_BOLD " ↔ r%-2d " ANSI_RESET, p);
        }
        printf("\n");
    }
}

// 1. Partner table is symmetric: partners[r][s] = q  =>  partners[q][s] = r.
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/partners_symmetric", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        for (int _r = 0; _r < nranks; _r++) {
            for (int _s = 0; _s < steps; _s++) {
                int q = partners[idx(_r, _s, steps)];
                if (q < 0) continue;
                int back = partners[idx(q, _s, steps)];
                if (back != _r) {
                    printf("  [butterfly/partners_symmetric] FAIL nranks=%d r=%d s=%d: partners[%d][%d]=%d but partners[%d][%d]=%d\n",
                           nranks, _r, _s, _r, _s, q, q, _s, back);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(partners); free(index); free(order);
    }
});

// 2. No rank partners with itself.
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/no_self_partner", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        for (int _r = 0; _r < nranks; _r++) {
            for (int _s = 0; _s < steps; _s++) {
                if (partners[idx(_r, _s, steps)] == _r) {
                    printf("  [butterfly/no_self_partner] FAIL nranks=%d r=%d s=%d: rank partners with itself\n",
                           nranks, _r, _s);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(partners); free(index); free(order);
    }
});

// 3. Virtual indices are a permutation of [0, nranks).
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/virt_index_is_permutation", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        int *seen = (int*) calloc(nranks, sizeof(int));
        for (int _r = 0; _r < nranks; _r++) {
            int v = index[_r];
            if (v < 0 || v >= nranks) {
                printf("  [butterfly/virt_index_is_permutation] FAIL nranks=%d r=%d: virt index %d out of range\n",
                       nranks, _r, v);
                out->exit_status = FASTEST_ERROR_ASSERT;
                goto fail;
            } else {
                seen[v]++;
            }
        }
        for (int _r = 0; _r < nranks; _r++) {
            if (seen[_r] != 1) {
                printf("  [butterfly/virt_index_is_permutation] FAIL nranks=%d: virt index %d appears %d times\n",
                       nranks, _r, seen[_r]);
                out->exit_status = FASTEST_ERROR_ASSERT;
                goto fail;
            }
        }
        fail:
        free(seen); free(partners); free(index); free(order);
    }
});

// 4. At each step, partners differ by exactly bit `step` in their virtual index.
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/partners_differ_by_one_bit", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        for (int _r = 0; _r < nranks; _r++) {
            for (int _s = 0; _s < steps; _s++) {
                int q = partners[idx(_r, _s, steps)];
                if (q < 0) continue;
                int diff = index[_r] ^ index[q];
                if (diff != (1 << _s)) {
                    printf("  [butterfly/partners_differ_by_one_bit] FAIL nranks=%d r=%d s=%d: virt xor=0x%x expected 0x%x\n",
                           nranks, _r, _s, diff, 1 << _s);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(partners); free(index); free(order);
    }
});

// 5. order[] is the inverse of index[]: order[index[r]] == r for all r.
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/order_is_inverse_of_index", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        for (int _r = 0; _r < nranks; _r++) {
            if (order[index[_r]] != _r) {
                printf("  [butterfly/order_is_inverse_of_index] FAIL nranks=%d r=%d: order[index[%d]]=%d expected %d\n",
                       nranks, _r, _r, order[index[_r]], _r);
                out->exit_status = FASTEST_ERROR_ASSERT;
                goto fail;
            }
        }
        fail:
        free(partners); free(index); free(order);
    }
});

// 6. No idle slots in power-of-two communicator: every rank has a partner at every step.
FASTEST_CUSTOMTEST_INLINE("standalone/butterfly/no_idle_slots", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *partners = (int*) malloc(nranks * steps * sizeof(int));
        int *index    = (int*) malloc(nranks * sizeof(int));
        int *order    = (int*) malloc(nranks * sizeof(int));
        ncclGetBineButterflyDdbl(nranks, steps, partners, index, order);

        for (int _r = 0; _r < nranks; _r++) {
            for (int _s = 0; _s < steps; _s++) {
                if (partners[idx(_r, _s, steps)] < 0) {
                    printf("  [butterfly/no_idle_slots] FAIL nranks=%d r=%d s=%d: idle slot\n",
                           nranks, _r, _s);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(partners); free(index); free(order);
    }
});
