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
- **missing-word order:** the `[:Nth:]` and baseline unrank index == librxe's
  canonical rank (leftmost unknown = most-significant digit), 128/128 sample
  indices byte-identical for both spaces (`--miss-gate`)

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
make miss-gate  # missing-word/[:Nth:] unrank index == librxe canonical rank
node gate/e2e.js  # xpub end-to-end: plant → crack → assert index / mnemonic / path
node gate/e2e_multigpu.js  # multi-GPU: plant → fan out → assert one GPU FOUND at global rank
```

Flags: `--words` | `--mnemonic` + `--passphrase`; target `--address` | `--xpub` |
`--target-chaincode`; `--purpose --change --index --no-checksum`; `--nth` /
`--no-nth`; `--no-compact` (compaction is on by default); sharding
`--start --count --limit`; `--rank` (librxe index of an arrangement);
`--resume LOG` (continue a killed run from its progress log); multi-GPU
`--device D`, `--devices 0,1` / `--gpus N`, `--print-total`.

**Multi-GPU.** `--devices 0,1` (or `--gpus 2`) turns the process into a supervisor:
it computes the job size once (`--print-total`, no GPU), warms the PTX cache, then
forks one crack child per GPU over a **contiguous slice of the canonical index
space** (reusing `--device`/`--start`/`--count`). The first child to find a hit exits
0 and the supervisor kills the siblings; Ctrl-C kills the whole group. The reported
index stays the **global** librxe rank regardless of which GPU found it, and an outer
`--start/--count` still applies (so a cluster can split first, then fan out locally).
Speedup is linear — a 100M-candidate sweep drops from 50.4 s on one 5090 to 25.4 s on
two (**1.99×**). Each child logs its own `-p`/CSV (`[devN]` tag, `<file>.devN.csv`).

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

The CSV header records **every parameter** of the run as explicit `# key: value`
lines (`input_kind`, `wp`, `pp`, `target`, `purpose`, `gap`, `change`, `start`,
`total`, …). **`--resume LOG`** reads that header, reconstructs the run, reads the
last progress row to see how far it got, and continues from there — appending to
the same log so the `swept`/`total` columns stay continuous (they are global, so a
resumed log can itself be resumed). It refuses gracefully if the log already found
a hit or already swept its whole window.

```sh
# a long run, killed (Ctrl-C / power loss) — the CSV is all you need to pick up
bip39rxcrack --pattern "…" --address 1Ap… --gap 5 --loginterval 1000:run.csv
bip39rxcrack --resume run.csv        # continues from the last logged position
```

## Performance

Measured on 1× RTX 5090 (native sm_120), correctness-only kernels:

| run | rate |
|-----|------|
| words, checksum-ON — compaction, dense PBKDF2 (default) | **~20.9 Mcand/s** |
| words, checksum-ON — fused sieve→PBKDF2 (`--no-compact`) | ~1.61 Mcand/s |
| passphrase (every candidate: PBKDF2 + full derive + EC) | ~1.38 Mcand/s |
| PBKDF2 seed rate (dense) | ~1.4 Mseed/s |

Multi-address scans cost more EC per seed: a 24-word `[:Nth:]` run does **~1.40
Mcand/s at `--gap 1`** and **~1.18 Mcand/s at `--gap 5`** (five address derivations
per candidate).

The full 479M-permutation example recovers in **~23 s** by default (~5 min with
`--no-compact`); with an actual winner it early-exits far sooner. A `[0-9]{7}`
(10M) passphrase deep-winner is found in ~7 s. These are **per-GPU** figures;
`--devices` scales them linearly (measured **1.99×** on 2× RTX 5090).

**What moves the needle:** at `--gap 1` the tool is **PBKDF2-bound** (SHA-512
dominates); at `--gap > 1` the extra secp256k1 derivations per seed make EC a real
factor too. The biggest wins, in order:

- **Unrolled SHA-512 with a register-resident 16-word schedule** (~1.83×). The full
  `w[80]` message schedule was indexed by a runtime loop counter, so it lived in
  local memory; fully unrolling the 80 rounds makes every `w[i&15]` a compile-time
  index (registers, no spill) and exposes instruction-level parallelism. Counter-
  intuitively this is an **ILP** win, not occupancy: forcing higher occupancy (fewer
  registers) actually *slowed* the kernel — a serial-dependency hash hides its
  latency better by overlapping one thread's instructions than by adding warps.
- **Inlined + unrolled `fe_mul`** (secp256k1): the 512-bit product `t[8]` was passed
  by pointer into `__noinline__` helpers and indexed by non-unrolled loops → local
  memory on every limb. Inlining and unrolling makes it register-resident; up to
  **+68%** on EC-heavy (`--gap 5`) runs.
- **Stream compaction** de-diverging the checksum-sieved path: it keeps the dense
  PBKDF2 kernel busy instead of wasting ~15/16 lanes on checksum-failing candidates.
- **A fixed-block HMAC** built straight from the ipad/opad midstate (~2.2×).

The remaining ceiling is SHA-512's arithmetic itself. NVIDIA GPUs have no native
64-bit integer ALU — 64-bit ops are synthesised from 32-bit — so the compression
function is INT-throughput-bound even on this card; ~1.4 M seeds/s is close to the
practical limit for this implementation. Batch inversion (amortising the Fermat
`fe_inv` across many points) would help EC-heavy scans further; a second GPU
(`--start/--count` sharding) scales linearly.

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
