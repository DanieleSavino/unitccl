extern "C" {
#include "fastest/tests.h"
#include "fastest/custom_tests.h"
}
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

static int run_bench(const char *coll, const char *algo, size_t nbytes,
                     FASTEST_TestOutput_t *out) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "mpirun build/unitccl_bench %s %s %zu 2>&1",
             coll, algo, nbytes / sizeof(float));
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[bench] popen failed\n");
        out->exit_status = FASTEST_ERROR_RESOURCE;
        return -1;
    }
    char     line[256];
    int      got_status = 0, got_median = 0, success = 0;
    uint64_t median_ns = 0;
    while (fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "[bench] %s", line);
        if (strncmp(line, "status:", 7) == 0) {
            got_status = 1;
            if      (strstr(line, "success"))      success = 1;
            else if (strstr(line, "error assert")) { out->exit_status = FASTEST_ERROR_ASSERT;    goto done; }
            else if (strstr(line, "error cuda"))   { out->exit_status = FASTEST_ERROR_CUDA;      goto done; }
        }
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
    if (!got_status || !got_median) {
        out->exit_status |= FASTEST_ERROR_UNEXPECTED;
        return -1;
    }
    if (!success) {
        out->exit_status |= FASTEST_ERROR_ASSERT;
        return -1;
    }
    out->time_ns     = median_ns;
    out->exit_status = FASTEST_SUCCESS;
    return 0;
}

#define DEFINE_BENCH_TEST(coll, algo, name, nbytes)                    \
FASTEST_CUSTOMTEST_INLINE("scaling/" algo "_" coll "/" name,          \
                           FASTEST_FAIL_ERROR, NULL,                                  \
{ run_bench(coll, algo, nbytes, out); })

DEFINE_BENCH_TEST("bcast", "BINE", "1kB", 1024)
DEFINE_BENCH_TEST("bcast", "BINE", "4kB", 16 * 1024)
DEFINE_BENCH_TEST("bcast", "BINE", "256kB", 256 * 1024)
DEFINE_BENCH_TEST("bcast", "BINE", "4MB", 4 * 1024 * 1024)
DEFINE_BENCH_TEST("bcast", "BINE", "64MB", 64 * 1024 * 1024)
DEFINE_BENCH_TEST("bcast", "RING", "1kB", 1024)
DEFINE_BENCH_TEST("bcast", "RING", "16kB", 16 * 1024)
DEFINE_BENCH_TEST("bcast", "RING", "256kB", 256 * 1024)
DEFINE_BENCH_TEST("bcast", "RING", "4MB", 4 * 1024 * 1024)
DEFINE_BENCH_TEST("bcast", "RING", "64MB", 64 * 1024 * 1024)
