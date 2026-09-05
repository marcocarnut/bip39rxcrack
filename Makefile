# bip39rxcrack -- native CUDA BIP39 seed cracker (Phase 1: crypto gates).
#
# Correctness authority: the byte-exact browser reference in reseed39
# (estimator/bip39crypto.js, estimator/bip39.js) + published BIP vectors.
# The gate harness NVRTC-compiles cuda/gate_kernels.cu to compute_90 PTX and
# the sm_120 driver JIT-forwards it (bip38rxcrack's proven route on this box).
#
#   make gate     # generate oracle vectors + build harness + run all 4 gates
#   make vectors  # (re)generate vectors from the reseed39 oracle
#   make build    # build the gate harness binary only
#   make clean
#
# RESEED39_DIR points the generator at the reseed39 clone (default: sibling).
# RXE_DIR is reserved for Phase 2 (librxe enumerator), matching bip38rxcrack.

CC        ?= cc
CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter
CUDA_HOME ?= /usr/local/cuda
RESEED39_DIR ?= /root/bip39rxcrack
RXE_DIR   ?= ../rxe

INC  = -I$(CUDA_HOME)/include
LIBS = -L$(CUDA_HOME)/lib64 -lnvrtc -lcuda

BIN = phase1gate

.PHONY: all gate vectors build clean
all: build

# Full Phase-1 deliverable: regenerate vectors, build, run every gate.
gate: vectors build
	@echo "== running Phase-1 crypto gates =="
	@RESEED39_DIR=$(RESEED39_DIR) ./$(BIN) vectors cuda/gate_kernels.cu

vectors: gate/gen_vectors.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/gen_vectors.js

build: $(BIN)

$(BIN): gate/gate.c cuda/gate_kernels.cu
	$(CC) $(CFLAGS) $(INC) -o $(BIN) gate/gate.c $(LIBS)

clean:
	rm -f $(BIN) vectors/*.txt
