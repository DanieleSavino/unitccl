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
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "NCCL_ALGO=%s mpirun build/unitccl_bench %s %zu 2>&1",
             algo, coll, nbytes / sizeof(float));

    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[bench] popen failed\n");
        out->exit_status = FASTEST_ERROR_RESOURCE;
        return -1;
    }
    char     line[256];
    int      got_median = 0;
    uint64_t median_ns = 0;
    while (fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "[bench] %s", line);
        if (strncmp(line, "median:", 7) == 0) {
            got_median = 1;
            sscanf(line, "median: %llu ns", (unsigned long long *)&median_ns);
        }
    }
done:
    int ret = pclose(fp);
    if (ret != 0) {
        out->exit_status |= FASTEST_ERROR_RESOURCE;
        return -1;
    }
    if (!got_median) {
        out->exit_status |= FASTEST_ERROR_UNEXPECTED;
        return -1;
    }
    out->time_ns     = median_ns;
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
    DEFINE_BENCH_TEST(coll, algo, "4MB",   4 * MB) \
    DEFINE_BENCH_TEST(coll, algo, "64MB",  64 * MB)
