#include "bench_test.h"

// Bine
DEFINE_TEST_1KB_64MB("AllReduce", "BINE")

// Ring
DEFINE_TEST_1KB_64MB("AllReduce", "RING")

// Tree
DEFINE_TEST_1KB_64MB("AllReduce", "TREE")
