# bip39rxcrack -- native CUDA BIP39 seed cracker.
#
# Correctness authority: the byte-exact browser reference in reseed39
# (estimator/bip39crypto.js, estimator/bip39.js) + published BIP vectors.
# Kernels are NVRTC-compiled at runtime to the DEVICE's own compute capability
# (auto-detected: compute_86 on a 3060 Ti, compute_120 on a 5090); the driver JITs
# the PTX to the GPU. NVRTC_HOME just needs to be new enough for your GPU.
#
#   make          # build the gate harness + the cracker
#   make gate     # generate oracle vectors + run the crypto gates
#   make ec-gate / addr-gate / decode-gate / nth-gate   # further gates
#   make clean
#
# RESEED39_DIR points the generators at the reseed39 clone (default: sibling).
# RXE_DIR points at the sibling rxe repo; librxe.a is the candidate enumerator.

CC        ?= cc
CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter
# CUDA_HOME: for the driver-API header cuda.h (stable API). Point at the versioned
# toolkit dir (the generic /usr/local/cuda symlink only exists with the full toolkit).
CUDA_HOME ?= /usr/local/cuda-12.5
# NVRTC_HOME: the NVRTC toolkit for runtime codegen. Arch is auto-detected from the
# device, so this only needs to be new enough for your GPU (12.5 covers Ampere/sm_86;
# a 5090/sm_120 needs 12.8+). Override on the command line for a different toolkit.
NVRTC_HOME ?= /usr/local/cuda-12.5
# DRIVER_LIB: libcuda.so from the installed NVIDIA driver.
DRIVER_LIB ?= /usr/lib/x86_64-linux-gnu
RESEED39_DIR ?= /root/bip39rxcrack
RXE_DIR   ?= ../rxe
LIBRXE     = $(RXE_DIR)/librxe.a

INC  = -I$(CUDA_HOME)/include -I$(NVRTC_HOME)/include -I$(RXE_DIR) -DNVRTC_LIBDIR='"$(NVRTC_HOME)/lib64"'
# libnvrtc from NVRTC_HOME (rpath so libnvrtc.so.N resolves at runtime). For -lcuda:
# prefer a real driver libcuda.so if present, else the toolkit stub (NEEDED soname is
# libcuda.so.1 either way; the real driver lib loads at runtime from the default path).
LIBS = -L$(NVRTC_HOME)/lib64 -lnvrtc -Wl,-rpath,$(NVRTC_HOME)/lib64 -L$(DRIVER_LIB) -L$(NVRTC_HOME)/lib64/stubs -lcuda
RXELIBS = $(LIBRXE) -lgmp -lm -lpthread

BIN = phase1gate
CRACK = bip39rxcrack

.PHONY: all gate vectors build cracker clean miss-gate multigpu-gate addr-e2e workqueue-gate missing-e2e bloom-selftest bloom-e2e xpubs-e2e pass-e2e account-e2e pass-pattern-e2e pass-alt-e2e hive-e2e
all: build cracker

# librxe.a comes from the sibling rxe repo (built there on demand).
miss-gate: $(CRACK)
	./$(CRACK) --miss-gate 64

# multi-GPU fan-out parity: plant -> fan out over both GPUs -> assert one FOUND
# at the global librxe rank (needs >=2 CUDA devices).
multigpu-gate: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_multigpu.js

# address-target crack (mode_crack_addr, compacted + fused paths) vs a planted winner
addr-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_addr.js

# work-queue fan-out: fine shards + every ordering policy find the planted winner
# at the global librxe rank (needs >=1 CUDA device; uses all visible GPUs).
workqueue-gate: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_workqueue.js

# missing-word ([:Nth:]) + address crack (mode_missing + its work-queue worker)
missing-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_missing.js

# blocked-bloom construction: no false negatives + FPR-vs-bits/key sweep (host-only)
bloom-selftest: $(CRACK)
	./$(CRACK) --bloom-selftest 1000000

# bloom target SET crack: winner hidden among decoys -> FOUND at rank; decoys-only -> NOT FOUND
bloom-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_bloom.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_bloom_missing.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_bloom_mixed.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_bloom_file.js

# multi-xpub via a chaincode bloom (EC-free): winner xpub among a decoy -> FOUND at rank
xpubs-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_xpubs.js

# regime A (fixed mnemonic + passphrase [0-9]{N}) with a bloom SET + --exhaustive:
# the PIN/passphrase-confusion case -- several PINs each deriving a funded address,
# ALL reported (not just the first). Plus single-target regression + negative.
pass-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_pass_bloom.js

# --account scan: an address funded under account' 1 is missed by default (account 0)
# and found with --account 3, at m/49'/0'/1'/0/0.
account-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_account.js

# generic passphrase patterns: charsets/literals/mixed-radix ([a-z]{4}, pass[0-9]{2}, ...)
pass-pattern-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_pass_pattern.js

# passphrase alternation (a|b|c), POSIX [:digit:], and external [:dict:] files via -D
pass-alt-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_pass_alternation.js

# SSH hive over loopback: workers reached via `ssh host --worker`; found at the
# global rank through --hosts localhost/2 and localhost/1,localhost/1. SKIPs if
# passwordless `ssh localhost` isn't available.
hive-e2e: $(CRACK)
	@RESEED39_DIR=$(RESEED39_DIR) node gate/e2e_hive.js

$(LIBRXE):
	$(MAKE) -C $(RXE_DIR) librxe.a

# Full Phase-1 deliverable: regenerate vectors, build, run every gate.
gate: vectors build
	@echo "== running Phase-1 crypto gates =="
	@RESEED39_DIR=$(RESEED39_DIR) ./$(BIN) vectors cuda/gate_kernels.cu

vectors: gate/gen_vectors.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/gen_vectors.js

build: $(BIN)

$(BIN): gate/gate.c cuda/gate_kernels.cu
	$(CC) $(CFLAGS) $(INC) -o $(BIN) gate/gate.c $(LIBS)

# Embed the kernel source into the binary (self-contained; a copied binary needs
# no cuda/ dir, and its g_ptx_key matches every copy of the same binary).
KERNEL_SRCS = cuda/crack_kernels.cu cuda/bip39_device.cuh cuda/secp256k1_device.cuh cuda/bloom_common.h
cuda/kernels_embed.h: $(KERNEL_SRCS) cuda/gen_embed.py
	python3 cuda/gen_embed.py $(KERNEL_SRCS) > $@

# Phase-2 self-enumerate cracker (links librxe as the canonical enumerator).
cracker: $(CRACK)
$(CRACK): src/bip39rxcrack.c cuda/kernels_embed.h $(KERNEL_SRCS) $(LIBRXE)
	$(CC) $(CFLAGS) $(INC) -o $(CRACK) src/bip39rxcrack.c $(RXELIBS) $(LIBS)

# EC / seed->address correctness gate (secp256k1 + hash160 + programs vs oracle).
ecvectors: gate/gen_ec.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/gen_ec.js 4000
ec-gate: cracker ecvectors
	@./$(CRACK) --words "abandon ability able" --ec-gate vectors/vec_ec.txt

clean:
	rm -f $(BIN) $(CRACK) vectors/*.txt

addrvectors: gate/gen_addr.js
	@RESEED39_DIR=$(RESEED39_DIR) node gate/gen_addr.js 300
addr-gate: cracker addrvectors
	@./$(CRACK) --words "abandon ability able" --addr-gate vectors/vec_addr.txt

decode-gate: cracker
	@RESEED39_DIR=$(RESEED39_DIR) node gate/decode_gate.js

nth-gate: cracker
	@RESEED39_DIR=$(RESEED39_DIR) node gate/nth_gate.js
