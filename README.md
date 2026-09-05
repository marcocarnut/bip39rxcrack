# bip39rxcrack

A native CUDA seed cracker for BIP39: takes an rxe mnemonic/passphrase pattern +
a public target (xpub or address) and searches the keyspace on GPU. Sibling of
[`bip38rxcrack`](https://github.com/marcocarnut/bip38rxcrack); shares `librxe`
(`RXE_DIR=../rxe`) as the canonical enumerator and the **reseed39** browser crypto
as the byte-exact correctness oracle. Design: `cli/PLAN.md` in the reseed39 repo.

The GPU **self-enumerates** its own candidates from a global index (no CPU-sweep
hybrid): each thread unranks its permutation/passphrase index on-die, runs the
whole pipeline, and reports hits. Nothing is fed over PCIe.

## Correctness law

Every kernel is gated **byte-exact against the browser reference**
(`reseed39/estimator/bip39crypto.js`, `estimator/bip39.js`) + published
BIP39/BIP32 vectors and public constants — never against a sibling GPU model.
Vector "expected" values are produced by running that exact JS oracle.

## What works (Phase 1 + Phase 2)

- **xpub target (EC-free):** permutation self-enumerate → checksum sieve → PBKDF2
  → hardened CKDpriv to the account node → 32-byte chain-code compare.
- **address target (secp256k1):** … → non-hardened derive to `change/index` →
  `k·G` → p2pkh (BIP44) / p2sh-p2wpkh (BIP49) / p2wpkh (BIP84) program compare.
- **Regime A** (fixed mnemonic, passphrase `[0-9]{N}` varies; no sieve) and
  **Regime B** (words permutation `{{N!}}`, checksum sieve).

### The two reseed39 preset examples (both recovered, oracle-verified)

```
# A — "Forgotten PIN" (passphrase, BIP49)
bip39rxcrack --mnemonic "laundry very receive soldier town age monkey already senior cereal vocal split" \
             --passphrase '[0-9]{4}' --address 36TaauZymS8sLXYh9HqYeqMvShJcMajK7j --purpose 49
  → pin 1705,  m/49'/0'/0'/0/0

# B — "Unknown order" (words permutation, BIP44)
bip39rxcrack --words "slow edge build grunt acid rich garment address open health voyage frozen" \
             --address 144uEF4Yc4TuWFSBttRYDXDFD44BCDi7Hc --purpose 44
  → "slow grunt garment health edge acid address voyage build rich open frozen",  m/44'/0'/0'/0/0
```

## Toolchain (the Blackwell box)

RTX 5090 (sm_120), driver 595 / CUDA 13.2. Kernels are NVRTC-compiled at runtime
to **native `compute_120`** using **CUDA 13.2 NVRTC** (installed from NVIDIA's
network repo); the driver JITs `compute_120` PTX to sm_120 with no forward-compat
gap. `cuda.h` (driver API) comes from the 11.8 headers, `libcuda` from the driver.
The binary self-reexecs once to set `LD_LIBRARY_PATH` for NVRTC's builtins.

> Note: CUDA 11.8 NVRTC (the box default) miscompiles a lone inlined 64-bit
> integer divide on the compute_90→sm_120 JIT path (folds to 0); `decode_perm`
> keeps a `volatile` divide guard. Native CUDA-13 codegen removes the class.

Requires Node (to run the JS oracle vector generators) — any recent Node 18/20.

## Build, gate, run

```
make            # build gate harness + cracker (NVRTC 13.2, compute_120)
make gate       # Phase-1 crypto gates: SHA-512, PBKDF2==mnemonicToSeed,
                #   BIP39 checksum, BIP32 EC-free CKDpriv  (vs oracle + vectors)
make ec-gate    # secp256k1 privToPub / hash160 / p2pkh / p2sh vs oracle (4000+ scalars)
make addr-gate  # full seed→address (incl. non-hardened ckd) vs oracle, purposes 44/49/84
node gate/e2e.js  # xpub end-to-end: plant → crack → assert index/mnemonic/path
```

Latest gate results (RTX 5090, native sm_120): Phase-1 79/79·47/47·80/80·70/70;
recon-vs-librxe 5000/5000; EC 4006/4006; seed→address 100/100 ×(44/49/84).

Cracker flags: `--words` | `--mnemonic`+`--passphrase`; target `--xpub` |
`--address` | `--target-chaincode`; `--purpose --change --index --no-checksum`;
engine `--start --count --limit --kernels`; gates `--ec-gate --addr-gate
--recon-gate --dump-valid`; `--rank` (librxe index of an arrangement).

## Performance (1× RTX 5090, correctness-only kernels)

| run | rate |
|-----|------|
| xpub (EC-free): PBKDF2 + hardened derive, every candidate | ~0.32 Mseed/s |
| address: PBKDF2 + full derive + EC, every candidate | ~0.31 Mcand/s |
| checksum-ON effective raw (12w, sieve → PBKDF2+EC on ~1/16) | ~0.55 Mcand/s |

The EC-free and with-EC per-candidate rates are nearly identical, confirming the
tool is **PBKDF2-bound** (the whole point of the architecture). Deep-winner
wall-clock: T7 `[0-9]{7}` (10M) → pin 9317864 in **31.9s**. Compute-bound at
450 W. Headroom is in occupancy (the per-thread PBKDF2 + EC state is large) and,
secondarily, the correctness-only EC (fixed-base comb `k·G` + batch inversion) —
the next levers toward the PLAN's 1–2 M seeds/s.

## Layout

```
cuda/bip39_device.cuh      SHA-512/256, HMAC (ipad/opad midstate), PBKDF2,
                           BIP32 EC-free, decode_perm, checksum, build_mnemonic
cuda/secp256k1_device.cuh  fe mod-p, Jacobian ops, k·G, RIPEMD-160, hash160,
                           address programs, non-hardened ckd, derive_address
cuda/gate_kernels.cu       Phase-1 gate kernels
cuda/crack_kernels.cu      g_recon, g_crack (xpub), g_crack_addr (regime B),
                           g_crack_pass (regime A), g_ec / g_addr (gates)
src/bip39rxcrack.c         librxe-linked host (enumerate, decode, launch, report)
gate/gen_*.js, gate.c, e2e.js   oracle-driven vector generators + harnesses
```

All four script types are supported end to end, base58 **and** bech32 targets:
p2pkh (BIP44, `1…`), p2sh-p2wpkh (BIP49, `3…`), p2wpkh (BIP84, `bc1q…`), p2tr
(BIP86, `bc1p…`, TapTweak). `make decode-gate` checks target decode vs the
oracle. (NVRTC compile of the full EC+taproot module is slow, ~2–3 min, so the
harness caches the PTX under `/tmp` keyed by source hash — repeat runs are
instant; `CRACK_NOCACHE=1` forces a recompile.)

## Known follow-ups

1. **EC / occupancy perf.** secp256k1 is correctness-only (double-and-add `k·G` +
   Fermat inverse). Fixed-base comb `k·G`, batch/Montgomery inversion, regime-A
   fixed-key HMAC-midstate precompute, and occupancy tuning are the levers toward
   the PLAN's 1–2 M seeds/s.
2. **Feistel keyed-shuffle** (`{{N!?}}`, librxe `permute.c`) — **dropped**: no
   real use case (a recovery sweeps the full space; a contiguous shard samples an
   unknown permutation as uniformly as a shuffle would), reported-index parity
   already holds (canonical rank 8952072 == reseed39's found_index), and it would
   force a per-candidate factorial decode that conflicts with cheap incremental
   odometer sharding.

Further out: `[:Nth:]` last-word checksum construction; Electrum; multi-GPU
fork/exec (range-shard machinery is in place via `--start/--count`); full
(re)seed39 `--job`/`--link` parity (PLAN §13).

## Collaboration model

Box-agent builds/measures on the 5090 and commits **branch-only**; the laptop
session `rxe-b5` is the correctness authority and sole git integrator (reviews,
gates, and establishes/merges `main`).
