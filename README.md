# unitccl

> Unit tests and scaling benchmarks for Bine tree collective algorithms in NCCL.

Part of a Bachelor's thesis at Sapienza Università di Roma (HLC Lab, supervisor: Daniele De Sensi), implementing [Bine trees](https://doi.org/10.1145/3712285.3759835) inside a fork of NCCL and validating correctness and performance across European HPC systems.

---

## Overview

`unitccl` sits on top of two components:

- **[fastest](https://github.com/DanieleSavino/fastest)** — a macro-based C test runner with Python orchestration, nanosecond-resolution timing, and a `fastest_py` package for comparison plots.
- **[NCCL fork](https://github.com/DanieleSavino/nccl)** (`nccl/`) — extends NCCL with Bine tree graph generation (`src/graph/bine.cc`) for distance-halving broadcast trees and distance-doubling butterfly patterns.

The test suite validates the mathematical properties of the generated communication schedules and benchmarks them against NCCL's built-in Ring and other algorithms.

---

## Repository Layout

```
unitccl/
├── include/
│   └── logging.h             # ANSI color macros (ANSI_SEND, ANSI_RECV, …)
├── src/
│   ├── nccl_bridge/
│   │   └── bine_graph.cpp    # symlink → nccl/src/graph/bine.cc
│   ├── scaling/
│   │   └── bcast.cpp         # fastest-registered scaling benchmarks
│   ├── standalone/
│   │   ├── test_bine_tree.cpp      # 7 correctness tests for the tree schedule
│   │   └── test_bine_butterfly.cpp # 6 correctness tests for the butterfly schedule
│   └── unitccl_bench.cu      # GPU benchmark binary (NCCL bcast / allreduce)
├── nccl/                     # NCCL fork submodule
├── vendor/
│   └── fastest/              # fastest submodule
├── plots/
│   └── bcast_scaling.png
├── Makefile
├── build_wheel.sh            # builds fastest Python wheel
└── run_tests.sh
```

---

## Building

### Prerequisites

| Dependency | Notes |
|---|---|
| GCC ≥ 9 / G++ (C++17) | Host compiler |
| CUDA toolkit | `nvcc` for `unitccl_bench.cu` |
| NCCL fork | Built into `nccl/build/` before linking |
| pybind11 ≥ 2.11 | Python bindings (optional) |
| matplotlib | Scaling plots (optional) |

### Static library (correctness tests only)

```bash
# Build NCCL fork first (sets HAVE_NCCL flag automatically)
make -C nccl src.build NVCC_GENCODE="-gencode arch=compute_80,code=sm_80"

# Build unitccl
make
# → build/libunitccl.a
```

Without a built NCCL the Makefile still compiles the standalone tests; `HAVE_NCCL` is left undefined and the bridge symlink is excluded.

### Python wheel (for fastest_py orchestration)

```bash
./build_wheel.sh
pip install vendor/fastest/fastest_py
```

---

## Running Tests

`tests.py` is the single entry point. `run_tests.sh` wraps it with `FASTEST_PROJECT` set:

```bash
./run_tests.sh                   # all suites
./run_tests.sh standalone        # correctness only
./run_tests.sh scaling           # benchmarks only (no plot)
./run_tests.sh scaling plot      # benchmarks + PNG output per NCCL protocol
```

Scaling tests call `mpirun build/unitccl_bench <coll> <algo> <nelems>` internally and record the median latency. When `plot` is passed, a comparison plot is saved for each of `SIMPLE`, `LL`, and `LL128` under `plots/bcast_scaling_<PROTO>.png`.

---

## Correctness Tests

### `standalone/tree` — Bine Distance-Halving Tree

Tests call `ncclGetBineTreeDhlv(nRanks, steps, send_table, recv_table)` and verify the schedule for all power-of-two communicator sizes from 2 to 4096 ranks.

| Test | Property verified |
|---|---|
| `recv_exactly_once` | Every non-root rank receives data exactly once per root |
| `send_recv_symmetric` | `send[root][a][s] = b ⟺ recv[root][b][s] = a` |
| `causal_ordering` | A rank only sends after its own receive step |
| `no_self_send` | No rank sends to or receives from itself |
| `all_ranks_reached` | Every non-root rank has a valid receive step |
| `root_never_recvs` | Root has no receive entries |
| `peer_differ_lsbs` | At step `s`, peers' negabinary representations differ in exactly the `steps−s` least significant bits (Eq. 1 of the paper) |

### `standalone/butterfly` — Bine Distance-Doubling Butterfly

Tests call `ncclGetBineButterflyDdbl(nRanks, steps, partners, index, order)`.

| Test | Property verified |
|---|---|
| `partners_symmetric` | `partners[r][s] = q ⟹ partners[q][s] = r` |
| `no_self_partner` | No rank partners with itself |
| `virt_index_is_permutation` | `index[]` is a bijection on `[0, nRanks)` |
| `partners_differ_by_one_bit` | At step `s`, `index[r] XOR index[q] == 1 << s` |
| `order_is_inverse_of_index` | `order[index[r]] == r` for all `r` |
| `no_idle_slots` | In a power-of-two communicator, every rank has a partner at every step |

---

## Scaling Benchmarks

Defined in `src/scaling/bcast.cpp` via the `DEFINE_BENCH_TEST` macro. Each test spawns `unitccl_bench` through MPI and records median latency.

| Suite | Algo | Sizes |
|---|---|---|
| `scaling/BINE_bcast` | BINE | 1 kB, 4 kB, 256 kB, 4 MB, 64 MB |
| `scaling/RING_bcast` | RING | 1 kB, 16 kB, 256 kB, 4 MB, 64 MB |

---

## Background: Bine Trees

Bine (*binomial negabinary*) trees are a family of collective algorithms that optimize communication locality by assigning each rank a negabinary representation and selecting communication partners that minimize modular rank distance. Compared to standard binomial trees and butterflies they reduce global-link traffic by up to 33%.

This project implements and validates the NCCL graph-generation layer for Bine trees as part of a Bachelor's thesis.

**Reference:** De Sensi et al., [*Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality*](https://doi.org/10.1145/3712285.3759835), SC '25.

---

## License

MIT License.
