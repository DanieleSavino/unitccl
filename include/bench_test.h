#pragma once

extern "C" {
#include "fastest/tests.h"
#include "fastest/custom_tests.h"
}
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define kB 1024
#define MB kB * 1024

static int run_bench(const char *coll, const char *algo, size_t nbytes,
                     FASTEST_TestOutput_t *out) {

    int world_size = 0;
    const char *ntasks = getenv("SLURM_NTASKS");
    if (ntasks) {
	world_size = atoi(ntasks);
    } else {
        // fallback if not running under Slurm (e.g. manual mpirun)
        fprintf(stderr, "SLURM_NTASKS not set\n");
    }

    int ppn = 1;
    const char *tpn = getenv("SLURM_NTASKS_PER_NODE");
    if (tpn && atoi(tpn) > 0) ppn = atoi(tpn);

    /* a failed/stale previous repeat must not leak its numbers into this one */
    out->time_ns   = 0;
    out->avg_ns    = 0.0;
    out->stddev_ns = 0.0;
    out->n_iters   = 0;

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "NCCL_ALGO=%s mpirun -np %d --map-by ppr:%d:node -x NCCL_ALGO "
        "build/unitccl_bench %s %zu 2>&1",
        algo, world_size, ppn, coll, nbytes / sizeof(float));

    printf("[bench] cmd: %s\n", cmd);

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[bench] popen failed\n");
        out->exit_status = FASTEST_ERROR_RESOURCE;
        return -1;
    }

    char     line[256];
    int      got_median   = 0;
    int      got_avg      = 0;
    int      got_stddev   = 0;
    int      got_iters    = 0;
    unsigned long long n_iters = 0;
    int      got_status   = 0;
    uint64_t median_ns    = 0;
    double   avg_ns       = 0.0;
    double   stddev_ns    = 0.0;
    int      bench_status = FASTEST_SUCCESS;
    while (fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "[bench] %s", line);
        if (strncmp(line, "median:", 7) == 0) {
            got_median = 1;
            sscanf(line, "median: %llu ns", (unsigned long long *)&median_ns);
        } else if (strncmp(line, "avg:", 4) == 0) {
            got_avg = (sscanf(line, "avg: %lf ns", &avg_ns) == 1);
        } else if (strncmp(line, "iters:", 6) == 0) {
            got_iters = (sscanf(line, "iters: %llu", &n_iters) == 1);
        } else if (strncmp(line, "stddev:", 7) == 0) {
            got_stddev = (sscanf(line, "stddev: %lf ns", &stddev_ns) == 1);
        } else if (strncmp(line, "status:", 7) == 0) {
            got_status = 1;
            if      (strstr(line, "error assert"))    bench_status = FASTEST_ERROR_ASSERT;
            else if (strstr(line, "error cuda/nccl")) bench_status = FASTEST_ERROR_RESOURCE;
            /* "success" → bench_status stays FASTEST_SUCCESS */
        }
    }
    int ret = pclose(fp);
    if (ret != 0) {
        fprintf(stderr, "[bench] pclose failed: %d\n", ret);
        out->exit_status |= FASTEST_ERROR_RESOURCE;
        //return -1;
    }
    /* avg/stddev/iters are mandatory: a missing line means a stale unitccl_bench binary */
    if (!got_median || !got_avg || !got_stddev || !got_iters || !got_status) {
        out->exit_status |= FASTEST_ERROR_UNEXPECTED;
        return -1;
    }
    if (bench_status != FASTEST_SUCCESS) {
        out->exit_status = bench_status;
        return -1;
    }
    out->time_ns     = median_ns;   /* median of the per-iteration slowest-rank time */
    out->avg_ns      = avg_ns;      /* mean   of the same samples */
    out->stddev_ns   = stddev_ns;   /* population stddev of the same samples */
    out->n_iters     = n_iters;     /* how many samples that is (needed for CIs) */
    out->exit_status = FASTEST_SUCCESS;
    return 0;
}

#define DEFINE_BENCH_TEST(coll, algo, name, nbytes)                    \
FASTEST_CUSTOMTEST_INLINE("scaling/1kB_64MB/" algo "_" coll "/" name,          \
                           FASTEST_FAIL_ERROR, NULL,                                  \
{ run_bench(coll, algo, nbytes, out); })

#define DEFINE_TEST_1KB_64MB(coll, algo) \
    DEFINE_BENCH_TEST(coll, algo, "1kB",   1 * kB) \
    DEFINE_BENCH_TEST(coll, algo, "16kB",  16 * kB) \
    DEFINE_BENCH_TEST(coll, algo, "256kB", 256 * kB) \
    DEFINE_BENCH_TEST(coll, algo, "1MB", 1 * MB) \
    DEFINE_BENCH_TEST(coll, algo, "4MB",   4 * MB) \
    DEFINE_BENCH_TEST(coll, algo, "64MB",  64 * MB)
