/**
 * unitccl_bench.cu  —  1 MPI rank per GPU collective benchmark
 *
 * Usage:
 *   NCCL_ALGO=RING mpirun -n 4 ./unitccl_bench Bcast 1048576
 *   NCCL_ALGO=BINE mpirun -n 4 ./unitccl_bench Bcast 1048576
 *
 * Correctness check (iteration 0 only):
 *   UNITCCL_CHECK_BINE=1 NCCL_ALGO=BINE mpirun -n 4 ./unitccl_bench AllReduce 1048576
 *
 * argv: <Bcast|AllReduce|AllGather|Reduce|ReduceScatter> <vec_size_floats>
 *
 * Notes:
 *   - rank i owns GPU i (cudaSetDevice(world_rank))
 *   - ncclCommInitRank via MPI_Bcast of unique ID
 *   - send buffer fill: send[i] = (rank+1) * i  for every rank and collective;
 *     using (rank+1) so rank 0 also sends non-zero data
 *   - buffer sizing:
 *       AllGather:     send = count,             recv = count * world_size
 *       ReduceScatter: send = count * world_size, recv = count
 *       all others:    send = count,             recv = count
 *   - timing: cudaEvent pair on the single stream, all ranks sync via
 *     MPI_Barrier before each iteration so we measure the collective cleanly
 *   - stats: each rank computes its own median/min/max; rank 0 gathers and
 *     reports global median (across ranks) and true straggler (global max)
 *   - correctness: enabled when env UNITCCL_CHECK_BINE=1, checked on iter 0
 *     only; each rank prints "status: success|error assert|error cuda/nccl"
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

#define CHECK_ENV "UNITCCL_CHECK"
#define ITERS_ENV "UNITCCL_ITERS"
#define WARMUP_ENV "UNITCCL_WARMUP"

#define DEFAULT_WARMUP 10
#define DEFAULT_ITERS  40

/* ── status tracking ────────────────────────────────────────────────────── */

typedef enum { STATUS_OK, STATUS_ERROR_ASSERT, STATUS_ERROR_RUNTIME } Status;

/* Per-rank global — set by macros or check, read for the final status line. */
static Status g_status = STATUS_OK;

/* ── error macros ───────────────────────────────────────────────────────── */

#define CUDA_CHECK(cmd) do {                                               \
    cudaError_t _e = (cmd);                                                \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "[rank %d] CUDA %s:%d  %s\n",                     \
                world_rank, __FILE__, __LINE__, cudaGetErrorString(_e));   \
        g_status = STATUS_ERROR_RUNTIME;                                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while (0)

#define NCCL_CHECK(cmd) do {                                               \
    ncclResult_t _r = (cmd);                                               \
    if (_r != ncclSuccess) {                                               \
        fprintf(stderr, "[rank %d] NCCL %s:%d  %s\n",                     \
                world_rank, __FILE__, __LINE__, ncclGetErrorString(_r));   \
        g_status = STATUS_ERROR_RUNTIME;                                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while (0)

/* ── types ──────────────────────────────────────────────────────────────── */

typedef enum {
    COLL_BCAST,
    COLL_ALLREDUCE,
    COLL_ALLGATHER,
    COLL_REDUCE,
    COLL_REDUCESCATTER,
} CollType;

/* ── helpers ────────────────────────────────────────────────────────────── */

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/*
 * Fill device buffer: send[i] = (rank+1) * i.
 * Using (rank+1) so every rank — including rank 0 — sends non-zero data,
 * which gives a meaningful correctness signal for all collectives.
 */
static void fill_device_ranked(float *d, size_t count, int rank) {
    float *h = (float *)malloc(count * sizeof(float));
    if (!h) { fprintf(stderr, "malloc failed\n"); exit(1); }
    float scale = (float)(rank + 1);
    for (size_t i = 0; i < count; i++) h[i] = scale * (float)i;
    cudaMemcpy(d, h, count * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

/* ── collective dispatch ────────────────────────────────────────────────── */

static void run_coll(CollType coll, size_t count,
                     ncclComm_t comm, cudaStream_t stream,
                     float *d_send, float *d_recv,
                     int world_rank) {
    switch (coll) {
        case COLL_BCAST:
            NCCL_CHECK(ncclBroadcast(d_send, d_recv, count,
                                     ncclFloat, /*root=*/0, comm, stream));
            break;
        case COLL_ALLREDUCE:
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count,
                                     ncclFloat, ncclSum, comm, stream));
            break;
        case COLL_ALLGATHER:
            NCCL_CHECK(ncclAllGather(d_send, d_recv, count,
                                     ncclFloat, comm, stream));
            break;
        case COLL_REDUCE:
            NCCL_CHECK(ncclReduce(d_send, d_recv, count,
                                  ncclFloat, ncclSum, /*root=*/0,
                                  comm, stream));
            break;
        case COLL_REDUCESCATTER:
            /*
             * ncclReduceScatter's third argument is recvcount — the number of
             * elements each rank receives (== count).  d_send holds
             * count * world_size elements; NCCL derives the total from
             * recvcount * comm_size internally.
             */
            NCCL_CHECK(ncclReduceScatter(d_send, d_recv, count,
                                         ncclFloat, ncclSum, comm, stream));
            break;
    }
}

/* ── correctness check ──────────────────────────────────────────────────── */

/*
 * check_correctness — called after iteration 0 when UNITCCL_CHECK_BINE=1.
 *
 * All collectives use send[i] = (rank+1)*i, so expected recv values are:
 *
 *   Bcast:         recv[i]           == i                      (root=0, scale=1)
 *   AllReduce:     recv[i]           == i * W*(W+1)/2
 *   AllGather:     recv[r*count + i] == (r+1)*i
 *   Reduce:        recv[i]           == i * W*(W+1)/2          (root rank 0 only)
 *   ReduceScatter: recv[i]           == (rank*count + i) * W*(W+1)/2
 *                  (rank r holds chunk r of the full AllReduce result)
 */
static void check_correctness(CollType coll, size_t count,
                               float *d_recv, int world_rank, int world_size) {
    size_t recv_count = (coll == COLL_ALLGATHER)
                        ? count * (size_t)world_size : count;

    float *h = (float *)malloc(recv_count * sizeof(float));
    if (!h) {
        fprintf(stderr, "[rank %d] check malloc failed\n", world_rank);
        g_status = STATUS_ERROR_RUNTIME;
        return;
    }
    cudaMemcpy(h, d_recv, recv_count * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_scales = (float)(world_size * (world_size + 1)) / 2.0f;

    int   first_bad = -1;
    float got_bad   = 0.f, exp_bad = 0.f;

    switch (coll) {

        case COLL_BCAST:
            for (size_t i = 0; i < count; i++) {
                float expected = (float)i;          /* root=0, scale=1 */
                if (h[i] != expected) {
                    first_bad = (int)i; got_bad = h[i]; exp_bad = expected;
                    goto done;
                }
            }
            break;

        case COLL_ALLREDUCE:
            for (size_t i = 0; i < count; i++) {
                float expected = (float)i * sum_scales;
                float rel_err  = fabsf(h[i] - expected) / (fabsf(expected) + 1e-6f);
                if (rel_err > 1e-3f) {
                    first_bad = (int)i; got_bad = h[i]; exp_bad = expected;
                    goto done;
                }
            }
            break;

        case COLL_ALLGATHER:
            for (int r = 0; r < world_size; r++) {
                float scale = (float)(r + 1);
                for (size_t i = 0; i < count; i++) {
                    float expected = scale * (float)i;
                    float got      = h[(size_t)r * count + i];
                    if (got != expected) {
                        first_bad = (int)((size_t)r * count + i);
                        got_bad = got; exp_bad = expected;
                        goto done;
                    }
                }
            }
            break;

        case COLL_REDUCE:
            if (world_rank != 0) break;     /* non-root: undefined, skip */
            for (size_t i = 0; i < count; i++) {
                float expected = (float)i * sum_scales;
                float rel_err  = fabsf(h[i] - expected) / (fabsf(expected) + 1e-6f);
                if (rel_err > 1e-3f) {
                    first_bad = (int)i; got_bad = h[i]; exp_bad = expected;
                    goto done;
                }
            }
            break;

        case COLL_REDUCESCATTER:
            /*
             * Rank r owns chunk r of the full AllReduce result.
             * Global index of recv[i] is (world_rank * count + i), so:
             *   expected = (world_rank * count + i) * sum_scales
             *
             * When world_rank == 0 and i == 0 the expected value is 0;
             * the relative-error formula handles this via the +1e-6 epsilon.
             */
            for (size_t i = 0; i < count; i++) {
                float expected = (float)((size_t)world_rank * count + i) * sum_scales;
                float rel_err  = fabsf(h[i] - expected) / (fabsf(expected) + 1e-6f);
                if (rel_err > 1e-3f) {
                    first_bad = (int)i; got_bad = h[i]; exp_bad = expected;
                    goto done;
                }
            }
            break;
    }

done:
    if (first_bad >= 0) {
        fprintf(stderr,
            "[rank %d] MISMATCH at index %d: got %.6g expected %.6g\n",
            world_rank, first_bad, (double)got_bad, (double)exp_bad);
        g_status = STATUS_ERROR_ASSERT;
    }
    free(h);
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    if (argc != 3) {
        fprintf(stderr,
            "usage: %s=<check> %s=<iters> %s=<warmup> NCCL_ALGO=<ALGO> mpirun -n <ngpus> %s"
            " <Bcast|AllReduce|AllGather|Reduce|ReduceScatter> <vec_size_floats>\n",
            CHECK_ENV, ITERS_ENV, WARMUP_ENV, argv[0]
        );
        MPI_Abort(MPI_COMM_WORLD, 1);
        return EXIT_FAILURE;
    }

    const char *coll_str = argv[1];
    size_t      count    = (size_t)atoll(argv[2]);

    CollType coll;
    if      (!strcmp(coll_str, "Bcast"))        { coll = COLL_BCAST;
                                                   setenv("NCCL_ALLGATHERV_ENABLE", "0", 1); }
    else if (!strcmp(coll_str, "AllReduce"))      coll = COLL_ALLREDUCE;
    else if (!strcmp(coll_str, "AllGather"))      coll = COLL_ALLGATHER;
    else if (!strcmp(coll_str, "Reduce"))         coll = COLL_REDUCE;
    else if (!strcmp(coll_str, "ReduceScatter"))  coll = COLL_REDUCESCATTER;
    else {
        fprintf(stderr, "unknown collective: %s\n", coll_str);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return EXIT_FAILURE;
    }

    const char *check_env = getenv(CHECK_ENV);
    int do_check = (check_env &&
                    (check_env[0] == '1' || strcmp(check_env, "on") == 0));

    const char *warmup_env = getenv(WARMUP_ENV);
    int warmup = (warmup_env && (atoi(warmup_env) > 0)) ? atoi(warmup_env) : DEFAULT_WARMUP;

    const char *iters_env = getenv(ITERS_ENV);
    int iters = (iters_env && (atoi(iters_env) > 0)) ? atoi(iters_env) : DEFAULT_ITERS;

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* ── GPU assignment ───────────────────────────────────────────────── */

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &local_comm);

    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    int n_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (local_rank >= n_gpus) {
        fprintf(stderr, "[rank %d] local_rank %d >= n_gpus %d on this node\n",
                world_rank, local_rank, n_gpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank));

    MPI_Comm_free(&local_comm);

    /* ── buffer sizing ────────────────────────────────────────────────── */

    /*
     * ReduceScatter is the dual of AllGather:
     *   AllGather:     send = count,             recv = count * world_size
     *   ReduceScatter: send = count * world_size, recv = count
     */
    const size_t send_count = (coll == COLL_REDUCESCATTER)
                              ? count * (size_t)world_size : count;
    const size_t recv_count = (coll == COLL_ALLGATHER)
                              ? count * (size_t)world_size : count;
    const size_t send_bytes = send_count * sizeof(float);
    const size_t recv_bytes = recv_count * sizeof(float);

    float *d_send, *d_recv;
    CUDA_CHECK(cudaMalloc(&d_send, send_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv, recv_bytes));

    fill_device_ranked(d_send, send_count, world_rank);

    /* ── stream and events ────────────────────────────────────────────── */

    cudaStream_t stream;
    cudaEvent_t  ev0, ev1;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    /* ── NCCL communicator ────────────────────────────────────────────── */

    ncclUniqueId nccl_id;
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_id, world_rank));

    MPI_Barrier(MPI_COMM_WORLD);

    /* ── warmup ───────────────────────────────────────────────────────── */

    for (int iter = 0; iter < warmup; iter++) {
        run_coll(coll, count, comm, stream, d_send, d_recv, world_rank);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (iter == 0 && do_check)
            check_correctness(coll, count, d_recv, world_rank, world_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ── measurement ──────────────────────────────────────────────────── */

    float iter_ms[iters];
    float iter_stragglers_ms[iters];

    for (int iter = 0; iter < iters; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        run_coll(coll, count, comm, stream, d_send, d_recv, world_rank);
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        iter_ms[iter] = ms;

        if (warmup == 0 && iter == 0 && do_check)
            check_correctness(coll, count, d_recv, world_rank, world_size);

        if(world_rank == 0) {
            MPI_Reduce(&iter_ms[iter], &iter_stragglers_ms[iter], 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&iter_ms[iter], NULL, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        }
    }

    /* ── report (rank 0 only) ─────────────────────────────────────────── */

    /*
     * STATUS_OK=0 < STATUS_ERROR_ASSERT=1 < STATUS_ERROR_RUNTIME=2, so
     * MPI_MAX across all ranks gives the worst status seen anywhere.
     * Rank 0 prints a single line; no per-rank noise.
     */
    int local_status  = (int)g_status;
    int global_status = STATUS_OK;
    MPI_Reduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        const char *status_str;
        switch (global_status) {
            case STATUS_OK:            status_str = "success";         break;
            case STATUS_ERROR_ASSERT:  status_str = "error assert";    break;
            case STATUS_ERROR_RUNTIME: status_str = "error cuda/nccl"; break;
            default:                   status_str = "unknown";          break;
        }
        printf("status:     %s\n", status_str);

        qsort(iter_stragglers_ms, world_size, sizeof(float), cmp_float);
        float global_median = (world_size & 1)
            ? iter_stragglers_ms[world_size / 2]
            : (iter_stragglers_ms[world_size/2 - 1] + iter_stragglers_ms[world_size/2]) * 0.5f;

        printf("collective: %s\n",            coll_str);
        printf("vec_size:   %zu floats\n",    count);
        printf("ranks:      %d\n",            world_size);
        printf("warmup:     %d  iters: %d\n", warmup, iters);
        printf("median:     %llu ns\n",
               (unsigned long long)(uint64_t)(global_median    * 1e6f));
        printf("min:        %llu ns\n",
               (unsigned long long)(uint64_t)(iter_stragglers_ms[0]       * 1e6f));
        printf("straggler:  %llu ns\n",
               (unsigned long long)(uint64_t)(iter_stragglers_ms[iters - 1] * 1e6f));
    }

    /* ── cleanup ──────────────────────────────────────────────────────── */

    // NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    MPI_Finalize();
    return (g_status == STATUS_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
