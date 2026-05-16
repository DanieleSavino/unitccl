#include "bench_test.h"

// Bine
DEFINE_TEST_1KB_64MB("AllGather", "BINE")

// Ring
DEFINE_TEST_1KB_64MB("AllGather", "RING")

// Collnet Direct
DEFINE_TEST_1KB_64MB("AllGather", "COLLNET_DIRECT")

// Nvls
DEFINE_TEST_1KB_64MB("AllGather", "NVLS")

// Pat
DEFINE_TEST_1KB_64MB("AllGather", "PAT")
