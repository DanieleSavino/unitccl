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

void print_tree(int *sendTable, int *recvTable, int nRanks, int steps) {
    for (int root = 0; root < nRanks; ++root) {
        printf(ANSI_BOLD "=== Root %d ===\n" ANSI_RESET, root);

        // Header
        printf(ANSI_DIM "  rank |");
        for (int s = 0; s < steps; ++s) printf("  s%-2d  ", s);
        printf("\n  -----|");
        for (int s = 0; s < steps; ++s) printf("-------");
        printf("\n" ANSI_RESET);

        for (int rank = 0; rank < nRanks; ++rank) {
            printf(ANSI_DIM "  r%-3d |" ANSI_RESET, rank);
            for (int step = 0; step < steps; ++step) {
                const size_t i = (size_t)root * nRanks * steps + (size_t)rank * steps + step;
                int s = sendTable[i], r = recvTable[i];
                if (s < 0 && r < 0)
                    printf(ANSI_IDLE "   .   " ANSI_RESET);
                else if (s >= 0 && r < 0)
                    printf(ANSI_SEND " S→%-2d  " ANSI_RESET, s);
                else if (s < 0 && r >= 0)
                    printf(ANSI_RECV " R←%-2d  " ANSI_RESET, r);
                else
                    printf(ANSI_BOTH "S→%d R←%d" ANSI_RESET, s, r);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// 1. Every rank receives data exactly once per root (except root itself).
FASTEST_CUSTOMTEST_INLINE("standalone/tree/recv_exactly_once", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 4; nranks <= (4); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            int *recv_count = (int*) calloc(nranks, sizeof(int));
            for (int _r = 0; _r < nranks; _r++)
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    if (recv_table[i] >= 0) recv_count[_r]++;
                }
            for (int _r = 0; _r < nranks; _r++) {
                int expected = (_r == _root) ? 0 : 1;
                if (recv_count[_r] != expected) {
                    printf("  [tree/recv_exactly_once] FAIL nranks=%d root=%d r=%d: recv_count=%d expected=%d\n",
                           nranks, _root, _r, recv_count[_r], expected);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
            free(recv_count);
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 2. send[root][a][s] = b  iff  recv[root][b][s] = a  (send/recv are symmetric).
FASTEST_CUSTOMTEST_INLINE("standalone/tree/send_recv_symmetric", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            for (int _r = 0; _r < nranks; _r++) {
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    int peer = send_table[i];
                    if (peer < 0) continue;
                    size_t j = (size_t)_root * nranks * steps + (size_t)peer * steps + _s;
                    if (recv_table[j] != _r) {
                        printf("  [tree/send_recv_symmetric] FAIL nranks=%d root=%d: send[%d][%d]=%d but recv[%d][%d]=%d\n",
                               nranks, _root, _r, _s, peer, peer, _s, recv_table[j]);
                        out->exit_status = FASTEST_ERROR_ASSERT;
                        goto fail;
                    }
                }
            }
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 3. A rank can only send after it has received (causal ordering).
//    Root sends from step 0. Others send only after their recv step.
FASTEST_CUSTOMTEST_INLINE("standalone/tree/causal_ordering", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            int *recv_step = (int*) malloc(nranks * sizeof(int));
            for (int _r = 0; _r < nranks; _r++) recv_step[_r] = (_r == _root) ? -1 : steps;
            for (int _r = 0; _r < nranks; _r++)
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    if (recv_table[i] >= 0) { recv_step[_r] = _s; break; }
                }
            for (int _r = 0; _r < nranks; _r++) {
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    if (send_table[i] < 0) continue;
                    int must_have_recv_by = (_r == _root) ? -1 : recv_step[_r];
                    if (_s <= must_have_recv_by) {
                        printf("  [tree/causal_ordering] FAIL nranks=%d root=%d r=%d: sends at step %d but recv_step=%d\n",
                               nranks, _root, _r, _s, recv_step[_r]);
                        out->exit_status = FASTEST_ERROR_ASSERT;
                        goto fail;
                    }
                }
            }
            free(recv_step);
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 4. No rank sends to itself.
FASTEST_CUSTOMTEST_INLINE("standalone/tree/no_self_send", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            for (int _r = 0; _r < nranks; _r++) {
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    if (send_table[i] == _r) {
                        printf("  [tree/no_self_send] FAIL nranks=%d root=%d r=%d s=%d: rank sends to itself\n",
                               nranks, _root, _r, _s);
                        out->exit_status = FASTEST_ERROR_ASSERT;
                        goto fail;
                    }
                    if (recv_table[i] == _r) {
                        printf("  [tree/no_self_send] FAIL nranks=%d root=%d r=%d s=%d: rank recvs from itself\n",
                               nranks, _root, _r, _s);
                        out->exit_status = FASTEST_ERROR_ASSERT;
                        goto fail;
                    }
                }
            }
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 5. All ranks are reachable (every non-root rank has recv_step < steps).
FASTEST_CUSTOMTEST_INLINE("standalone/tree/all_ranks_reached", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            for (int _r = 0; _r < nranks; _r++) {
                if (_r == _root) continue;
                bool found = false;
                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    if (recv_table[i] >= 0) { found = true; break; }
                }
                if (!found) {
                    printf("  [tree/all_ranks_reached] FAIL nranks=%d root=%d r=%d: rank never receives\n",
                           nranks, _root, _r);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 6. Root never receives.
FASTEST_CUSTOMTEST_INLINE("standalone/tree/root_never_recvs", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            for (int _s = 0; _s < steps; _s++) {
                size_t i = (size_t)_root * nranks * steps + (size_t)_root * steps + _s;
                if (recv_table[i] >= 0) {
                    printf("  [tree/root_never_recvs] FAIL nranks=%d root=%d s=%d: root recvs\n",
                           nranks, _root, _s);
                    out->exit_status = FASTEST_ERROR_ASSERT;
                    goto fail;
                }
            }
        }
        fail:
        free(send_table); free(recv_table);
    }
});

// 7. At step s, send/recv peer's negabinary repr differs from rank's in exactly
//    the least significant (steps-s) bits (all ones), per Eq. 1 of the paper.
FASTEST_CUSTOMTEST_INLINE("standalone/tree/peer_differ_lsbs", FASTEST_FAIL_ERROR, NULL, {
    out->exit_status = FASTEST_SUCCESS;
    for (int nranks = 2; nranks <= (1<<BINE_MAX_STEPS); nranks *= 2) {
        int steps = __builtin_ctz(nranks);
        int *send_table = (int*) malloc(nranks * nranks * steps * sizeof(int));
        int *recv_table = (int*) malloc(nranks * nranks * steps * sizeof(int));

        ncclGetBineTreeDhlv(nranks, steps, send_table, recv_table);

        for (int _root = 0; _root < nranks; _root++) {
            for (int _r = 0; _r < nranks; _r++) {
                int vrank = (_r - _root + nranks) % nranks;
                int nb_vr = rank2nb(vrank, steps);

                for (int _s = 0; _s < steps; _s++) {
                    size_t i = (size_t)_root * nranks * steps + (size_t)_r * steps + _s;
                    int expected_diff = (1 << (steps - _s)) - 1;

                    int s_peer = send_table[i];
                    int r_peer = recv_table[i];
                    if (s_peer >= 0) {
                        int vpeer = (s_peer - _root + nranks) % nranks;
                        int nb_vpeer = rank2nb(vpeer, steps);
                        int diff = nb_vr ^ nb_vpeer;

                        if (diff != expected_diff) {
                            printf("  [tree/peer_differ_lsbs] FAIL nranks=%d root=%d r=%d s=%d: send_peer=%d nb_vr=0x%x nb_vpeer=0x%x diff=0x%x expected=0x%x\n",
                                   nranks, _root, _r, _s, s_peer, nb_vr, nb_vpeer, diff, expected_diff);
                            out->exit_status = FASTEST_ERROR_ASSERT;
                            goto fail;
                        }
                    }
                    if (r_peer >= 0) {
                        int vpeer = (r_peer - _root + nranks) % nranks;
                        int nb_vpeer = rank2nb(vpeer, steps);
                        int diff = nb_vr ^ nb_vpeer;

                        if (diff != expected_diff) {
                            printf("  [tree/peer_differ_lsbs] FAIL nranks=%d root=%d r=%d s=%d: recv_peer=%d nb_vr=0x%x nb_vpeer=0x%x diff=0x%x expected=0x%x\n",
                                   nranks, _root, _r, _s, r_peer, nb_vr, nb_vpeer, diff, expected_diff);
                            out->exit_status = FASTEST_ERROR_ASSERT;
                            goto fail;
                        }
                    }
                }
            }
        }
        fail:
        free(send_table); free(recv_table);
    }
});
