#include "nccl.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <time.h>

extern "C" {
#include "fastest/custom_tests.h"
}

#define NRANKS 4

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void run_bcast(const char *algo, size_t nbytes, FASTEST_TestOutput_t *out) {
    setenv("NCCL_ALGO", algo, 1);
    setenv("NCCL_ALLGATHERV_ENABLE", "0", 1);

    ncclComm_t   comms[NRANKS];
    void        *sendbuf[NRANKS], *recvbuf[NRANKS];
    cudaStream_t streams[NRANKS];
    int          devs[NRANKS] = {0, 1, 2, 3};

    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&sendbuf[i], nbytes);
        cudaMalloc(&recvbuf[i], nbytes);
        cudaStreamCreate(&streams[i]);
    }
    ncclCommInitAll(comms, NRANKS, devs);

    // warmup
    ncclGroupStart();
    for (int i = 0; i < NRANKS; i++)
        ncclBroadcast(sendbuf[i], recvbuf[i], nbytes, ncclChar, 0, comms[i], streams[i]);
    ncclGroupEnd();
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // measured run
    uint64_t t0 = now_ns();
    ncclGroupStart();
    for (int i = 0; i < NRANKS; i++)
        ncclBroadcast(sendbuf[i], recvbuf[i], nbytes, ncclChar, 0, comms[i], streams[i]);
    ncclGroupEnd();
    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    uint64_t t1 = now_ns();

    out->time_ns = t1 - t0;

    for (int i = 0; i < NRANKS; i++) {
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuf[i]);
        cudaFree(recvbuf[i]);
        cudaStreamDestroy(streams[i]);
    }
}

#define DEFINE_BCAST_TEST(algo, idx, nbytes)                          \
FASTEST_CUSTOMTEST_INLINE("scaling/" algo "_bcast/" #idx,             \
                           FASTEST_TIME_NS, NULL,                     \
{ run_bcast(algo, nbytes, out); out->exit_status = FASTEST_SUCCESS; })

DEFINE_BCAST_TEST("BINE", 0, 1024)
DEFINE_BCAST_TEST("BINE", 1, 16 * 1024)
DEFINE_BCAST_TEST("BINE", 2, 256 * 1024)
DEFINE_BCAST_TEST("BINE", 3, 4 * 1024 * 1024)
DEFINE_BCAST_TEST("BINE", 4, 64 * 1024 * 1024)

DEFINE_BCAST_TEST("RING", 0, 1024)
DEFINE_BCAST_TEST("RING", 1, 16 * 1024)
DEFINE_BCAST_TEST("RING", 2, 256 * 1024)
DEFINE_BCAST_TEST("RING", 3, 4 * 1024 * 1024)
DEFINE_BCAST_TEST("RING", 4, 64 * 1024 * 1024)
