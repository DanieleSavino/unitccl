extern "C" {
#include "fastest/custom_tests.h"
}

#include "nccl.h"
#include <cuda_runtime.h>
#include <cstdlib>

#define NRANKS 4

static void run_bcast(const char *algo, size_t nbytes) {
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

    ncclGroupStart();
    for (int i = 0; i < NRANKS; i++)
        ncclBroadcast(sendbuf[i], recvbuf[i], nbytes, ncclChar, 0, comms[i], streams[i]);
    ncclGroupEnd();

    for (int i = 0; i < NRANKS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
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
{ run_bcast(algo, nbytes); out->exit_status = FASTEST_SUCCESS; })

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
