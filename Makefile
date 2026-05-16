# Makefile — unitccl/
ROOT         := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FASTEST_ROOT := $(ROOT)vendor/fastest
NCCL_ROOT    ?= $(ROOT)nccl
CUDA_HOME ?= $(shell dirname $(shell dirname $(shell which nvcc 2>/dev/null)) 2>/dev/null)
MPI_HOME     ?= $(shell dirname $(shell dirname $(shell which mpicc)))

NCCL_LIB := $(NCCL_ROOT)/build/lib/libnccl_static.a

CC      := g++
NVCC    := $(CUDA_HOME)/bin/nvcc
AR      := ar

CFLAGS  := -std=c++17 -fPIC -I$(FASTEST_ROOT)/include
NVFLAGS := -std=c++17 -I$(FASTEST_ROOT)/include

ifneq ($(wildcard $(NCCL_LIB)),)
  CFLAGS  += -I$(NCCL_ROOT)/build/include -I$(NCCL_ROOT)/src/include -I$(NCCL_ROOT)/src -I$(CUDA_HOME)/include -DHAVE_NCCL
  NVFLAGS += -I$(NCCL_ROOT)/build/include -I$(NCCL_ROOT)/src/include -I$(NCCL_ROOT)/src -DHAVE_NCCL
endif

SRCS := $(shell find src -name '*.cpp')
OBJS := $(SRCS:src/%.cpp=build/%.o)
LIB  := build/libunitccl.a
BENCH := build/unitccl_bench

all: $(LIB) $(BENCH)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BENCH): src/unitccl_bench.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -ccbin mpicxx -Iinclude $< -o $@ \
		$(shell mpicc --showme:incdirs | sed 's/^/-I/;s/ / -I/g') \
		-L$(NCCL_ROOT)/build/lib -lnccl \
		-Xlinker -rpath,$(NCCL_ROOT)/build/lib \
		-L$(CUDA_HOME)/lib64 -lcudart \
		-lmpi -lrt -ldl

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

clean:
	rm -rf build/

.PHONY: all clean
