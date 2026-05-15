/**
 * unitccl_bench.cu
 * Usage: ./unitccl_bench <coll> <algo> <vec_size>
 *   coll:     bcast | allreduce | allgather | reduce
 *   algo:     RING | BINE | TREE | ...
 *   vec_size: number of floats
 *
 * Timing methodology:
 *   - All GPU workers complete WARMUP iterations and join.
 *   - A global MPI_Barrier synchronises all nodes before measurement starts.
 *   - Each of the ITERS measurement iterations is timed individually with
 *     its own cudaEvent pair, giving a full latency distribution.
 *   - Reported metric: median across GPUs and ranks (most representative),
 *     plus min (best-case) and max (stragglers).
 *   - MPI_THREAD_MULTIPLE is required so worker threads can call MPI_Barrier
 *     between phases without a full rejoin to main.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define N_GPUS_PER_NODE 4
#define WARMUP          50
#define ITERS           100

#define CUDA_CHECK(cmd) do {                                              \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
        fprintf(stderr, "[rank %d gpu %d] CUDA %s:%d '%s'\n",            \
                g->world_rank, g->gpu_idx,                                \
                __FILE__, __LINE__, cudaGetErrorString(e));               \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

#define NCCL_CHECK(cmd) do {                                              \
    ncclResult_t r = (cmd);                                               \
    if (r != ncclSuccess) {                                               \
        fprintf(stderr, "[rank %d gpu %d] NCCL %s:%d '%s'\n",            \
                g->world_rank, g->gpu_idx,                                \
                __FILE__, __LINE__, ncclGetErrorString(r));               \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

#define CUDA_CHECK_MAIN(cmd) do {                                         \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
        fprintf(stderr, "[rank %d] CUDA %s:%d '%s'\n",                   \
                world_rank, __FILE__, __LINE__, cudaGetErrorString(e));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

#define NCCL_CHECK_MAIN(cmd) do {                                         \
    ncclResult_t r = (cmd);                                               \
    if (r != ncclSuccess) {                                               \
        fprintf(stderr, "[rank %d] NCCL %s:%d '%s'\n",                   \
                world_rank, __FILE__, __LINE__, ncclGetErrorString(r));   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float median_f(float *arr, int n) {
    qsort(arr, n, sizeof(float), cmp_float);
    return (n & 1) ? arr[n / 2] : (arr[n / 2 - 1] + arr[n / 2]) * 0.5f;
}

typedef enum { COLL_BCAST, COLL_ALLREDUCE, COLL_ALLGATHER, COLL_REDUCE } CollType;

typedef struct {
    int               world_rank;
    int               world_size;
    int               gpu_idx;
    int               total_gpus;
    ncclComm_t        comm;
    cudaStream_t      stream;
    float            *d_send;
    float            *d_recv;
    size_t            count;
    CollType          coll;
    float             iter_ms[ITERS];
    float             median_ms;
    float             min_ms;
    pthread_barrier_t *node_barrier;
} GpuCtx;

static void run_coll(GpuCtx *g) {
    switch (g->coll) {
        case COLL_BCAST:
            NCCL_CHECK(ncclBroadcast(g->d_send, g->d_recv, g->count,
                                     ncclFloat, 0, g->comm, g->stream));
            break;
        case COLL_ALLREDUCE:
            NCCL_CHECK(ncclAllReduce(g->d_send, g->d_recv, g->count,
                                     ncclFloat, ncclSum, g->comm, g->stream));
            break;
        case COLL_ALLGATHER:
            NCCL_CHECK(ncclAllGather(g->d_send, g->d_recv, g->count,
                                     ncclFloat, g->comm, g->stream));
            break;
        case COLL_REDUCE:
            NCCL_CHECK(ncclReduce(g->d_send, g->d_recv, g->count,
                                  ncclFloat, ncclSum, 0, g->comm, g->stream));
            break;
    }
}

static void *gpu_worker(void *arg) {
    GpuCtx *g = (GpuCtx *)arg;

    size_t send_bytes = g->count * sizeof(float);
    size_t recv_bytes = (g->coll == COLL_ALLGATHER)
                        ? send_bytes * g->total_gpus
                        : send_bytes;

    CUDA_CHECK(cudaSetDevice(g->gpu_idx));
    CUDA_CHECK(cudaMalloc(&g->d_send, send_bytes));
    CUDA_CHECK(cudaMalloc(&g->d_recv, recv_bytes));
    CUDA_CHECK(cudaMemset(g->d_send, 1, send_bytes));
    CUDA_CHECK(cudaStreamCreate(&g->stream));

    /* warmup */
    for (int i = 0; i < WARMUP; i++) {
        run_coll(g);
        CUDA_CHECK(cudaStreamSynchronize(g->stream));
    }

    /* synchronise all GPUs on this node, then GPU 0 does global MPI barrier */
    pthread_barrier_wait(g->node_barrier);
    if (g->gpu_idx == 0)
        MPI_Barrier(MPI_COMM_WORLD);
    pthread_barrier_wait(g->node_barrier);

    /* measurement */
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(t0, g->stream));
        run_coll(g);
        CUDA_CHECK(cudaEventRecord(t1, g->stream));
        CUDA_CHECK(cudaStreamSynchronize(g->stream));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        g->iter_ms[i] = ms;
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    float sorted[ITERS];
    memcpy(sorted, g->iter_ms, sizeof(sorted));
    g->median_ms = median_f(sorted, ITERS);
    g->min_ms    = sorted[0];

    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <bcast|allreduce|allgather|reduce> <algo> <vec_size>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *coll_str = argv[1];
    const char *algo     = argv[2];
    size_t      count    = (size_t)atoll(argv[3]);

    CollType coll;
    if      (!strcmp(coll_str, "bcast"))     coll = COLL_BCAST;
    else if (!strcmp(coll_str, "allreduce")) coll = COLL_ALLREDUCE;
    else if (!strcmp(coll_str, "allgather")) coll = COLL_ALLGATHER;
    else if (!strcmp(coll_str, "reduce"))    coll = COLL_REDUCE;
    else {
        fprintf(stderr, "unknown coll: %s\n", coll_str);
        return EXIT_FAILURE;
    }

    setenv("NCCL_ALGO", algo, 1);
    setenv("NCCL_ALLGATHERV_ENABLE", "0", 1);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int total_gpus = world_size * N_GPUS_PER_NODE;

    ncclUniqueId nccl_id;
    if (world_rank == 0) NCCL_CHECK_MAIN(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    pthread_barrier_t node_barrier;
    pthread_barrier_init(&node_barrier, NULL, N_GPUS_PER_NODE);

    GpuCtx ctx[N_GPUS_PER_NODE];
    memset(ctx, 0, sizeof(ctx));
    for (int i = 0; i < N_GPUS_PER_NODE; i++) {
        ctx[i].world_rank   = world_rank;
        ctx[i].world_size   = world_size;
        ctx[i].gpu_idx      = i;
        ctx[i].total_gpus   = total_gpus;
        ctx[i].count        = count;
        ctx[i].coll         = coll;
        ctx[i].node_barrier = &node_barrier;
    }

    ncclGroupStart();
    for (int i = 0; i < N_GPUS_PER_NODE; i++) {
        cudaSetDevice(i);
        ncclCommInitRank(&ctx[i].comm, total_gpus, nccl_id,
                         world_rank * N_GPUS_PER_NODE + i);
    }
    ncclGroupEnd();

    MPI_Barrier(MPI_COMM_WORLD);

    pthread_t threads[N_GPUS_PER_NODE];
    for (int i = 0; i < N_GPUS_PER_NODE; i++)
        pthread_create(&threads[i], NULL, gpu_worker, &ctx[i]);
    for (int i = 0; i < N_GPUS_PER_NODE; i++)
        pthread_join(threads[i], NULL);

    pthread_barrier_destroy(&node_barrier);

    /* aggregate: gather per-node medians to rank 0 */
    float local_medians[N_GPUS_PER_NODE], local_mins[N_GPUS_PER_NODE];
    for (int i = 0; i < N_GPUS_PER_NODE; i++) {
        local_medians[i] = ctx[i].median_ms;
        local_mins[i]    = ctx[i].min_ms;
    }

    float sorted_med[N_GPUS_PER_NODE];
    memcpy(sorted_med, local_medians, sizeof(sorted_med));
    float node_median = median_f(sorted_med, N_GPUS_PER_NODE);

    float node_min = local_mins[0];
    for (int i = 1; i < N_GPUS_PER_NODE; i++)
        if (local_mins[i] < node_min) node_min = local_mins[i];

    float node_max_median = local_medians[0];
    for (int i = 1; i < N_GPUS_PER_NODE; i++)
        if (local_medians[i] > node_max_median) node_max_median = local_medians[i];

    float *all_node_medians = NULL;
    if (world_rank == 0)
        all_node_medians = (float *)malloc(world_size * sizeof(float));
    MPI_Gather(&node_median, 1, MPI_FLOAT,
               all_node_medians, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float global_min_ms, global_max_median_ms;
    MPI_Reduce(&node_min,        &global_min_ms,        1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&node_max_median, &global_max_median_ms, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    /* correctness check (BINE only) */
    int correct = 1;
    if (strcmp(algo, "BINE") == 0) {
        setenv("NCCL_ALGO", "RING", 1);

        size_t recv_count = (coll == COLL_ALLGATHER) ? count * total_gpus : count;
        size_t recv_bytes = recv_count * sizeof(float);

        float *d_ref = NULL;
        if (world_rank == 0) {
            CUDA_CHECK_MAIN(cudaSetDevice(0));
            CUDA_CHECK_MAIN(cudaMalloc(&d_ref, recv_bytes));
        }

        ncclGroupStart();
        for (int i = 0; i < N_GPUS_PER_NODE; i++) {
            float *dst = (world_rank == 0 && i == 0) ? d_ref : ctx[i].d_recv;
            switch (coll) {
                case COLL_BCAST:
                    NCCL_CHECK_MAIN(ncclBroadcast(ctx[i].d_send, dst, count,
                                                  ncclFloat, 0, ctx[i].comm, ctx[i].stream));
                    break;
                case COLL_ALLREDUCE:
                    NCCL_CHECK_MAIN(ncclAllReduce(ctx[i].d_send, dst, count,
                                                  ncclFloat, ncclSum, ctx[i].comm, ctx[i].stream));
                    break;
                case COLL_ALLGATHER:
                    NCCL_CHECK_MAIN(ncclAllGather(ctx[i].d_send, dst, count,
                                                  ncclFloat, ctx[i].comm, ctx[i].stream));
                    break;
                case COLL_REDUCE:
                    NCCL_CHECK_MAIN(ncclReduce(ctx[i].d_send, dst, count,
                                               ncclFloat, ncclSum, 0, ctx[i].comm, ctx[i].stream));
                    break;
            }
        }
        ncclGroupEnd();
        for (int i = 0; i < N_GPUS_PER_NODE; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(ctx[i].stream);
        }

        if (world_rank == 0) {
            float *h_bine = (float *)malloc(recv_bytes);
            float *h_ring = (float *)malloc(recv_bytes);
            cudaSetDevice(0);
            cudaMemcpy(h_bine, ctx[0].d_recv, recv_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ring, d_ref,          recv_bytes, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < recv_count; i++) {
                if (h_bine[i] != h_ring[i]) { correct = 0; break; }
            }
            free(h_bine);
            free(h_ring);
            cudaFree(d_ref);
        }
    }

    if (world_rank == 0) {
        float global_median_ms = median_f(all_node_medians, world_size);
        free(all_node_medians);

        uint64_t median_ns   = (uint64_t)(global_median_ms     * 1e6f);
        uint64_t min_ns      = (uint64_t)(global_min_ms        * 1e6f);
        uint64_t straggler_ns= (uint64_t)(global_max_median_ms * 1e6f);

        printf("status:    %s\n",      correct ? "success" : "error assert");
        printf("median:    %llu ns\n", (unsigned long long)median_ns);
        printf("min:       %llu ns\n", (unsigned long long)min_ns);
        printf("straggler: %llu ns\n", (unsigned long long)straggler_ns);
        printf("(warmup=%d iters=%d gpus=%d)\n", WARMUP, ITERS, total_gpus);
    }

    for (int i = 0; i < N_GPUS_PER_NODE; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(ctx[i].comm);
        cudaFree(ctx[i].d_send);
        cudaFree(ctx[i].d_recv);
        cudaStreamDestroy(ctx[i].stream);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
