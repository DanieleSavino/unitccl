# Makefile — unitccl/
ROOT         := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FASTEST_ROOT := $(ROOT)vendor/fastest
NCCL_ROOT    := $(ROOT)nccl
CUDA_HOME    ?= /usr/local/cuda

NCCL_LIB := $(NCCL_ROOT)/build/lib/libnccl_static.a

CC      := g++
AR      := ar

CFLAGS  := -std=c++17 -fPIC -I$(FASTEST_ROOT)/include

ifneq ($(wildcard $(NCCL_LIB)),)
  CFLAGS += -I$(NCCL_ROOT)/build/include -I$(NCCL_ROOT)/src/include -I$(CUDA_HOME)/include -DHAVE_NCCL
endif

SRCS := $(shell find src -name '*.cpp')
OBJS := $(SRCS:src/%.cpp=build/%.o)
LIB  := build/libunitccl.a

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build/

.PHONY: all clean
