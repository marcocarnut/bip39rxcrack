# bip39rxcrack

A native CUDA seed cracker for **BIP39**. You give it an [rxe](https://github.com/marcocarnut/rxe)
pattern describing what you know (and don't know) about your seed phrase, plus a
public target you control — an address or an account xpub — and it searches the
keyspace on the GPU.

It's the **sibling of [(re)seed39](https://github.com/marcocarnut/reseed39)** and a
**cousin of [bip38rxcrack](https://github.com/marcocarnut/bip38rxcrack)**. (re)seed39
does everything this tool does, but **in your browser** — try it on your own seed
with nothing to install (with GPU acceleration too, if your browser supports
WebGPU). It's the friendliest place to build your pattern and get a feel for the
problem. Reach for `bip39rxcrack` when the run gets big: talking straight to the
hardware, it's far faster, which is what matters for long, difficult recoveries.
In fact `bip39rxcrack` is gated **byte-exact against (re)seed39's crypto** (below),
so the two agree candidate-for-candidate — the browser is the reference, the CLI
is the speed.

It links `librxe` (`RXE_DIR=../rxe`) as the candidate enumerator.

> **Intended use — recovering your *own* wallets.** This is a recovery tool for a
> seed you own where you've lost a word, the word order, or the passphrase. Point
> it only at your own funds.

## How it works

The GPU **self-enumerates** its own candidates from a global index: each thread
unranks its permutation / passphrase / word-position index on-die, runs the whole
pipeline (checksum → PBKDF2-HMAC-SHA512 → BIP32 derive → address/xpub compare) and
reports hits. No candidates cross PCIe, so the tool is GPU-compute-bound rather
than limited by how fast a host can feed it.

## What it does

- **Targets:** an account **xpub** (EC-free — a chain-code compare, no secp256k1)
  or an **address** — p2pkh (BIP44, `1…`), p2sh-p2wpkh (BIP49, `3…`), p2wpkh
  (BIP84, `bc1q…`), p2tr (BIP86, `bc1p…`, TapTweak). Both base58 and bech32/bech32m.
  The script type and derivation purpose are read from the address.
- **Unknown passphrase:** fixed mnemonic, passphrase pattern varies (e.g. a
  forgotten numeric PIN, `[0-9]{N}`).
- **Unknown word order:** all permutations of a known word set (`{{N!}}`), with the
  BIP39 checksum sieving the ~15/16 (12-word) invalid orderings on-GPU.
- **Missing words:** known words plus `K` unknown `[:bip39-en:]` positions. When the
  **last** (checksum-bearing) word is one of the unknowns, the **`[:Nth:]`
  construction** builds only the checksum-valid finals directly — enumerate the
  free entropy bits, SHA-256 the entropy, append the checksum bits — instead of
  sweeping all 2048. That's `2^cs`× fewer candidates (16× at 12 words, 256× at 24)
  with zero rejected work (auto-detected; `--nth` / `--no-nth`).
- **Stream compaction** (default; `--no-compact` to disable): sieve into a dense
  survivor array, then run a full-warp PBKDF2 kernel over it. This recovers the GPU
  throughput otherwise lost to warp divergence when only a fraction of candidates
  survive the checksum (**9.1×** on the permutation sweep — see Performance). The
  sweep is chunked so the survivor buffer is bounded (a chunk's worst case is every
  candidate surviving, so it can't overflow), ~1 GiB by default (`COMPACT_BUDGET_MB`
  to tune); it falls back to the fused path if that buffer won't allocate.

### Examples

```sh
# Unknown passphrase — a forgotten 4-digit PIN, p2sh-p2wpkh (BIP49) address
bip39rxcrack --mnemonic "laundry very receive soldier town age monkey already senior cereal vocal split" \
             --passphrase '[0-9]{4}' --address 36TaauZymS8sLXYh9HqYeqMvShJcMajK7j
#   → passphrase 1705,  m/49'/0'/0'/0/0

# Unknown word order — 12 known words, any order, p2pkh (BIP44) address
bip39rxcrack --words "slow edge build grunt acid rich garment address open health voyage frozen" \
             --address 144uEF4Yc4TuWFSBttRYDXDFD44BCDi7Hc
#   → slow grunt garment health edge acid address voyage build rich open frozen,  m/44'/0'/0'/0/0
```

## Correctness

Every kernel is gated **byte-exact against the reseed39 reference**
(`estimator/bip39crypto.js`, `estimator/bip39.js`) plus published BIP39 / BIP32 /
BIP86 vectors and public constants — never against a sibling GPU model. The
"expected" side of every gate is produced by running that exact JS oracle.

Gate suite (RTX 5090, native sm_120), all byte-exact, zero mismatches:

- **crypto:** SHA-512 · PBKDF2 == `mnemonicToSeed` (incl. the 24-word >128-byte key
  pre-hash) · BIP39 checksum · BIP32 master + hardened EC-free CKDpriv
- **enumeration:** unrank + reconstruction == librxe, 5000/5000
- **secp256k1:** privToPub / hash160 / p2pkh / p2sh-p2wpkh, 4006/4006 (incl. edge
  scalars 1, 2, n−1, n−2)
- **seed → address:** 100/100 each for purposes 44 / 49 / 84 / 86 (full chain,
  incl. the non-hardened ckd EC path and BIP86 TapTweak)
- **`[:Nth:]` construction:** the constructed valid-last-word set == the brute +
  sieve survivor set, byte-identical

## Requirements & build

- An **NVIDIA GPU**. Blackwell (sm_120, e.g. RTX 5090) needs the **CUDA 12.8+/13**
  NVRTC toolkit — kernels are NVRTC-compiled at runtime to native `compute_120`
  and the driver JITs the PTX to sm_120. (Older architectures build with their
  matching toolkit; paths are set in the `Makefile`.)
- **Node 18/20** — the gate vector generators run the reseed39 JS oracle.
- **librxe** from the sibling [`rxe`](https://github.com/marcocarnut/rxe) repo
  (`RXE_DIR=../rxe`) and a **reseed39** checkout for the oracle (`RESEED39_DIR`).

```sh
make            # build the gate harness + the cracker
make gate       # crypto gates (SHA-512, PBKDF2, checksum, BIP32) vs the oracle
make ec-gate    # secp256k1 / hash160 / address programs vs the oracle
make addr-gate  # full seed → address vs the oracle
make decode-gate  # address target decode vs the oracle
make nth-gate   # [:Nth:] construction == brute + sieve
node gate/e2e.js  # xpub end-to-end: plant → crack → assert index / mnemonic / path
```

Flags: `--words` | `--mnemonic` + `--passphrase`; target `--address` | `--xpub` |
`--target-chaincode`; `--purpose --change --index --no-checksum`; `--nth` /
`--no-nth`; `--no-compact` (compaction is on by default); sharding
`--start --count --limit`; `--rank` (librxe index of an arrangement).

**Progress & logging.** `-p` prints a live status line (~1/s: elapsed, swept/total,
rate, ETA — the ETA uses a recent-window rate so it tracks the real throughput):

```
[    6.2s] 4.2/10.0M (41.9%) 0.68 Mc/s  hashed 4.19M  ETA 9s
```

`--loginterval MS[:FILE]` writes a CSV
(`t_ms,swept,swept_total,pct,rate,hashed,eta_s`, with a `#`-comment header/footer)
— `hashed` is the number of candidates that reached PBKDF2 (the real work: it
equals `swept` for a passphrase search, ≈`swept`/16 for a 12-word checksum-sieved
search). Every mode is windowed, so a run also **early-exits** as soon as it finds
a hit instead of sweeping the whole space.

## Performance

Measured on 1× RTX 5090 (native sm_120), correctness-only kernels:

| run | rate |
|-----|------|
| words, checksum-ON — compaction, dense PBKDF2 (default) | **~10.6 Mcand/s** |
| words, checksum-ON — fused sieve→PBKDF2 (`--no-compact`) | ~1.17 Mcand/s |
| passphrase (every candidate: PBKDF2 + full derive + EC) | ~0.67 Mcand/s (p2wpkh) |
| PBKDF2 seed rate | ~0.69 Mseed/s |

The full 479M-permutation example recovers in **~46 s** by default (411 s with
`--no-compact`); with an actual winner it early-exits far sooner. A `[0-9]{7}`
(10M) passphrase deep-winner is found in ~15 s.

**What moves the needle:** the tool is **PBKDF2-bound**, not EC-bound — the EC-free
(xpub) and with-EC (address) per-candidate rates are nearly identical. The two
biggest wins reflect that:

- **A fixed-block HMAC** that builds each SHA-512 block straight from the ipad/opad
  midstate (the general streaming path was padding one byte at a time): ~2.2×.
- **Stream compaction** de-diverging the checksum-sieved path: 9.1× on the
  permutation sweep, because the fused kernel wasted most PBKDF2 lanes on the
  ~15/16 candidates that fail the checksum.

A fixed-base comb for `k·G` and a fixed-key HMAC-midstate precompute are correct
and gated but give ~0 speedup here — an honest null result of being PBKDF2-bound,
not EC-bound. The remaining ceiling is SHA-512 itself: it's 64-bit-integer-bound,
and consumer GeForce silicon runs 64-bit ops at a reduced rate, so ~0.69 M
seeds/s is near this card's practical PBKDF2 limit. A full-rate-INT64 datacenter
card, or a second GPU, is the real throughput lever from here.

## Layout

```
cuda/bip39_device.cuh      SHA-512/256, HMAC (ipad/opad midstate), PBKDF2,
                           BIP32 EC-free, decode_perm, checksum, build_mnemonic
cuda/secp256k1_device.cuh  field mod-p, Jacobian ops, k·G, RIPEMD-160, hash160,
                           address programs, non-hardened ckd, BIP86 taproot
cuda/gate_kernels.cu       crypto gate kernels
cuda/crack_kernels.cu      crack + sieve/compaction + gate kernels
src/bip39rxcrack.c         host: enumerate, decode target, launch, report
gate/gen_*.js, gate.c,     oracle-driven vector generators + gate harnesses
  e2e.js, *_gate.js
```

## Roadmap

- **EC / occupancy tuning.** secp256k1 is correctness-only (double-and-add `k·G` +
  Fermat inverse); a fixed-base comb, batch inversion and occupancy work are the
  levers for the EC-heavier paths.
- **Multi-GPU** fork/exec — the range-shard machinery (`--start/--count`) is
  already in place.
- **Electrum** seeds; full reseed39 job-file (`--job`/`--link`) parity.

The `{{N!?}}` keyed-shuffle is intentionally **not** implemented: a full-space
recovery is order-independent, the reported index already matches reseed39's
canonical rank, and a per-candidate shuffle would block cheap incremental sharding.
