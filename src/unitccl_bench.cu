/**
 * unitccl_bench.cu  —  1 MPI rank per GPU collective benchmark
 *
 * Usage:
 *   NCCL_ALGO=RING mpirun -n 4 ./unitccl_bench bcast 1048576
 *   NCCL_ALGO=BINE mpirun -n 4 ./unitccl_bench bcast 1048576
 *
 * argv: <bcast|allreduce|allgather|reduce> <vec_size_floats>
 *
 * Notes:
 *   - rank i owns GPU i (cudaSetDevice(world_rank))
 *   - ncclCommInitRank via MPI_Bcast of unique ID
 *   - allgather recv buffer is count*world_size elements — allocated correctly
 *     per rank since every rank participates symmetrically
 *   - timing: cudaEvent pair on the single stream, all ranks sync via
 *     MPI_Barrier before each iteration so we measure the collective cleanly
 *   - stats: each rank computes its own median/min/max; rank 0 gathers and
 *     reports global median (across ranks) and true straggler (global max)
 */

#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define WARMUP  10
#define ITERS   50

/* ── error macros ───────────────────────────────────────────────────────── */

#define CUDA_CHECK(cmd) do {                                               \
    cudaError_t _e = (cmd);                                                \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "[rank %d] CUDA %s:%d  %s\n",                     \
                world_rank, __FILE__, __LINE__, cudaGetErrorString(_e));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while (0)

#define NCCL_CHECK(cmd) do {                                               \
    ncclResult_t _r = (cmd);                                               \
    if (_r != ncclSuccess) {                                               \
        fprintf(stderr, "[rank %d] NCCL %s:%d  %s\n",                     \
                world_rank, __FILE__, __LINE__, ncclGetErrorString(_r));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while (0)

/* ── types ──────────────────────────────────────────────────────────────── */

typedef enum { COLL_BCAST, COLL_ALLREDUCE, COLL_ALLGATHER, COLL_REDUCE } CollType;

/* ── helpers ────────────────────────────────────────────────────────────── */

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static void fill_device(float *d, size_t count, float val) {
    float *h = (float *)malloc(count * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (size_t i = 0; i < count; i++) h[i] = val;
    cudaMemcpy(d, h, count * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

/* ── collective dispatch ────────────────────────────────────────────────── */

/*
 * send_count: number of floats in d_send (== argv vec_size for all collectives)
 * recv_count: number of floats in d_recv (== send_count * world_size for allgather,
 *             else send_count)
 *
 * For allgather, NCCL expects the per-rank send count, not the total.
 * For reduce/bcast, root is rank 0.
 */
static void run_coll(CollType coll, size_t send_count,
                     ncclComm_t comm, cudaStream_t stream,
                     float *d_send, float *d_recv,
                     int world_rank) {
    switch (coll) {
        case COLL_BCAST:
            /* in-place on root (d_recv == d_send on rank 0 is fine;
             * on non-root ranks d_recv is the destination)            */
            NCCL_CHECK(ncclBroadcast(d_send, d_recv, send_count,
                                     ncclFloat, /*root=*/0, comm, stream));
            break;

        case COLL_ALLREDUCE:
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, send_count,
                                     ncclFloat, ncclSum, comm, stream));
            break;

        case COLL_ALLGATHER:
            /* ncclAllGather takes the PER-RANK send count.
             * d_recv must be world_size * send_count floats.          */
            NCCL_CHECK(ncclAllGather(d_send, d_recv, send_count,
                                     ncclFloat, comm, stream));
            break;

        case COLL_REDUCE:
            NCCL_CHECK(ncclReduce(d_send, d_recv, send_count,
                                  ncclFloat, ncclSum, /*root=*/0,
                                  comm, stream));
            break;
    }
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    if (argc != 3) {
        /* print on stderr before MPI init so it always shows */
        fprintf(stderr,
            "%d args found, usage: NCCL_ALGO=<ALGO> mpirun -n <ngpus> %s"
            " <Bcast|AllReduce|AllGather|Reduce> <vec_size_floats>\n",
            argc, argv[0]);

        fprintf(stderr, "args: %s %s %s %s\n", argv[0], argv[1], argv[2], argv[3]);
        return EXIT_FAILURE;
    }

    const char *coll_str = argv[1];
    size_t      count    = (size_t)atoll(argv[2]);

    CollType coll;
    // NCCL Converts Bcast to Scatter + AllGatherv
    if      (!strcmp(coll_str, "Bcast"))     {coll = COLL_BCAST; setenv("NCCL_ALLGATHERV_ENABLE", "0", 1);}
    else if (!strcmp(coll_str, "AllReduce")) coll = COLL_ALLREDUCE;
    else if (!strcmp(coll_str, "AllGather")) coll = COLL_ALLGATHER;
    else if (!strcmp(coll_str, "Reduce"))    coll = COLL_REDUCE;
    else {
        fprintf(stderr, "unknown collective: %s\n", coll_str);
        return EXIT_FAILURE;
    }

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* ── GPU assignment ───────────────────────────────────────────────── */

    int n_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (world_rank >= n_gpus) {
        fprintf(stderr,
            "[rank %d] world_size %d > n_gpus %d — reduce -n\n",
            world_rank, world_size, n_gpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(world_rank));

    /* ── buffer sizing ────────────────────────────────────────────────── */

    /*
     * send buffer: always `count` floats.
     * recv buffer:
     *   allgather → count * world_size floats (each rank contributes count)
     *   all others → count floats
     *
     * reduce: only rank 0 needs a meaningful recv buffer, but we allocate
     * it on every rank to keep the code uniform; NCCL ignores the pointer
     * on non-root ranks for ncclReduce.
     */
    const size_t send_count = count;
    const size_t recv_count = (coll == COLL_ALLGATHER)
                              ? count * (size_t)world_size
                              : count;
    const size_t send_bytes = send_count * sizeof(float);
    const size_t recv_bytes = recv_count * sizeof(float);

    float *d_send, *d_recv;
    CUDA_CHECK(cudaMalloc(&d_send, send_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv, recv_bytes));

    /* fill send buffer with 1.0f — ncclSum gives predictable integer results */
    fill_device(d_send, send_count, 1.0f);

    /* ── stream and events ────────────────────────────────────────────── */

    cudaStream_t stream;
    cudaEvent_t  ev0, ev1;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    /* ── NCCL communicator ────────────────────────────────────────────── */

    /*
     * Standard pattern: rank 0 generates a unique ID, broadcasts it via MPI,
     * every rank calls ncclCommInitRank with the same ID.
     * NCCL_ALGO is already set in the environment by the caller.
     */
    ncclUniqueId nccl_id;
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, world_rank));

    MPI_Barrier(MPI_COMM_WORLD);

    /* ── warmup ───────────────────────────────────────────────────────── */

    for (int iter = 0; iter < WARMUP; iter++) {
        run_coll(coll, send_count, comm, stream, d_send, d_recv, world_rank);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ── measurement ──────────────────────────────────────────────────── */

    /*
     * MPI_Barrier before each iteration ensures all ranks start the
     * collective at the same wall-clock moment, so the event-measured
     * time includes any real load-imbalance straggler effect.
     */
    float iter_ms[ITERS];

    for (int iter = 0; iter < ITERS; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        run_coll(coll, send_count, comm, stream, d_send, d_recv, world_rank);
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        iter_ms[iter] = ms;
    }

    /* ── per-rank stats ───────────────────────────────────────────────── */

    float sorted[ITERS];
    memcpy(sorted, iter_ms, sizeof(sorted));
    qsort(sorted, ITERS, sizeof(float), cmp_float);

    float rank_min    = sorted[0];
    float rank_max    = sorted[ITERS - 1];
    float rank_median = (ITERS & 1) ? sorted[ITERS / 2]
                        : (sorted[ITERS/2 - 1] + sorted[ITERS/2]) * 0.5f;

    /* ── global reduction ─────────────────────────────────────────────── */

    /*
     * Gather all per-rank medians to rank 0, compute median-of-medians.
     * True straggler = max single-rank max across all ranks.
     * True best      = min single-rank min across all ranks.
     */
    float *all_medians = NULL;
    if (world_rank == 0)
        all_medians = (float *)malloc(world_size * sizeof(float));

    MPI_Gather(&rank_median, 1, MPI_FLOAT,
               all_medians, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float global_min, global_straggler;
    MPI_Reduce(&rank_min, &global_min,       1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rank_max, &global_straggler, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── report (rank 0 only) ─────────────────────────────────────────── */

    if (world_rank == 0) {
        /* median-of-medians over all ranks */
        qsort(all_medians, world_size, sizeof(float), cmp_float);
        float global_median = (world_size & 1)
            ? all_medians[world_size / 2]
            : (all_medians[world_size/2 - 1] + all_medians[world_size/2]) * 0.5f;
        free(all_medians);

        printf("collective: %s\n",           coll_str);
        printf("vec_size:   %zu floats\n",   count);
        printf("ranks:      %d\n",           world_size);
        printf("warmup:     %d  iters: %d\n", WARMUP, ITERS);
        printf("median:     %llu ns\n",
               (unsigned long long)(uint64_t)(global_median    * 1e6f));
        printf("min:        %llu ns\n",
               (unsigned long long)(uint64_t)(global_min       * 1e6f));
        printf("straggler:  %llu ns\n",
               (unsigned long long)(uint64_t)(global_straggler * 1e6f));
    }

    /* ── cleanup ──────────────────────────────────────────────────────── */

    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    MPI_Finalize();
    return EXIT_SUCCESS;
}
