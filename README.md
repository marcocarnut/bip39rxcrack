# bip39rxcrack

A native CUDA seed cracker for BIP39 (and later Electrum): takes an rxe
mnemonic/passphrase pattern + a public target (xpub or address) and searches the
keyspace on GPU. Sibling of [`bip38rxcrack`](https://github.com/marcocarnut/bip38rxcrack);
shares `librxe` (`RXE_DIR=../rxe`) and the **reseed39** browser crypto as the
byte-exact correctness oracle. See `cli/PLAN.md` in the reseed39 repo for the design.

**Status: Phase 1 — crypto gates only.** No enumeration or crack loop yet. The
CUDA kernels are proven byte-exact against the browser reference before any
cracking code is written (that is the collaboration law).

## Correctness law

Every kernel is gated **byte-exact against the browser reference**
(`reseed39/estimator/bip39crypto.js`, `estimator/bip39.js`) and the published
BIP39/BIP32 test vectors — **never against a sibling GPU model.** The gate
vectors are produced by running that exact JS oracle (`gate/gen_vectors.js`);
the CUDA kernels must reproduce them.

## Toolchain (the Blackwell box)

The box has **CUDA 11.8** only (`libnvrtc.so.11.8.89`) but a **driver 595 / CUDA
13.2 runtime**, on an **RTX 5090 (sm_120, Blackwell)**. NVRTC 11.8 tops out at
`compute_90` PTX, so the gate harness compiles the kernels to **`compute_90`
PTX** and lets the **sm_120 driver JIT-forward** it at module load — the same
route `bip38rxcrack` uses here, no toolkit install required. (A native CUDA 13
install is deferred to the performance phase.)

Requires Node (to run the JS oracle vector generator) — install any recent
Node 18/20; the box was seeded with v20 from the official tarball.

## Build & run the gates

```
make gate        # generate oracle vectors + build harness + run all 4 gates
```

`RESEED39_DIR` points the generator at the reseed39 clone (default
`/root/bip39rxcrack`). Individual steps: `make vectors`, `make build`,
`./phase1gate vectors cuda/gate_kernels.cu`. Exit code 0 iff every stage passes
with zero mismatches.

### The four gates

| stage | kernel | gated against |
|------|--------|---------------|
| 1 | SHA-512 (streaming) | SHA-512 KAT + random multi-length messages |
| 2 | PBKDF2-HMAC-SHA512(2048) | `mnemonicToSeed`: Trezor vector + random mnemonics 12/15/18/21/24w, incl. the 24-word >128-byte password (HMAC key pre-hash) boundary |
| 3 | BIP39 checksum (SHA-256, top `cs` bits) | `sha256_1blk_h0`, `cs`=4/5/6/7/8 masked per word count |
| 4 | BIP32 master + hardened EC-FREE CKDpriv → account node | `accountNode` (chaincode+priv), incl. BIP32 vector-1 (`m`, `m/0'`) |

The EC-free BIP32 path (SHA-512 + mod-n add only, no secp256k1) is the flagship
xpub-crack lane: derive to the account node and compare its 32-byte chain code.

### Latest result (RTX 5090 D, sm_120)

```
  [PASS] SHA-512            : 79/79 match
  [PASS] PBKDF2-HMAC-SHA512 : 47/47 match  (2048 iters; 20 vectors >128B key/pre-hash)
  [PASS] BIP39 checksum     : 80/80 match  (cs bits: 12w=4 15w=5 18w=6 21w=7 24w=8)
  [PASS] BIP32 EC-free deriv: 70/70 match  (master + hardened CKDpriv; incl. BIP32 vector-1)
  ==== GATE PASSED: 0 total mismatches ====
```

## Layout

```
cuda/gate_kernels.cu   device kernels (SHA-512/256, HMAC, PBKDF2, EC-free BIP32)
gate/gen_vectors.js    oracle-driven vector generator (the reference)
gate/gate.c            NVRTC harness: compile -> run -> byte-exact diff
Makefile               `make gate`
```

## Collaboration model

Same as `bip38rxcrack`: the box-agent builds/measures on the 5090 and commits
**branch-only**; the laptop session `rxe-b5` is the correctness authority and
sole git integrator (reviews, gates, and establishes/merges `main`).

## Roadmap (see `cli/PLAN.md`)

Phase 2: xpub crack, single GPU, GPU **self-enumerates** its candidates
(odometer digits → checksum sieve → PBKDF2 → hardened derive → chain-code
compare). Then `[:Nth:]` last-word construction, the address path (secp256k1),
multi-GPU fork/exec sharding, and the reseed39 **job-file parity contract**
(`--job`/`--link`).
