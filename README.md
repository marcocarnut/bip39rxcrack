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
  The script type and derivation purpose are read from the address. A **set** of
  targets works too: `--addresses a,b,…` / `--addresses-file`, `--xpubs`, or a
  prebuilt **`--bloom` filter of every funded address** (see *Any funded address*).
- **Unknown passphrase:** fixed mnemonic, unknown BIP39 passphrase whose *shape*
  you know — a `--passphrase PATTERN` of literals, char classes (`[0-9]`, `[a-z]`),
  POSIX classes (`[:digit:]`, `[:alnum:]`, …), inline alternation `(spring|summer)`,
  and external dictionaries `[:name:]` (a `name.dict`, one word/line, via `-D DIR`).
  e.g. `[0-9]{4}` (a PIN), `bike[0-9]{2}`, `(spring|summer)[0-9]{2}`, `prefix[:words:]`.
  See `docs/PASSPHRASE_PATTERNS.md`.
- **Unknown account / gap / index:** scan `--account N`, `--change N`, `--gap N`
  over `m/purpose'/0'/account'/change/index`. With a bloom of your funded addresses
  you rarely need a wide `--gap` (any funded address in a small range matches), but
  accounts are independent hardened subtrees — scan them with `--account`.
- **All matches, not just the first:** `--exhaustive` reports every hit in a set
  (e.g. several PINs that each derive one of your funded addresses).
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

# Missing words, ANY funded address — no target address needed, hit any address
# that ever transacted (a prebuilt bloom of the whole chain; see below)
bip39rxcrack --pattern "trial [:bip39:] gloom dragon … receive [:bip39:] [:bip39:] [:bip39:]" \
             --bloom ../data/alladdrs.blf --purpose 44

# Every funded passphrase under one mnemonic (the PIN-confusion case)
bip39rxcrack --mnemonic "laundry very receive … vocal split" \
             --passphrase '[0-9]{4}' --bloom ../data/alladdrs.blf --exhaustive

# SSH hive — split the sweep across machines (2 local GPUs + a rented box)
bip39rxcrack --words "…" --address 1Ap… \
             --hosts 'local/2,root@box.example.com:48851/1' \
             --remote-bin /root/bip39rxcrack-cli/bip39rxcrack
```

## Any funded address (bloom filter)

For "I lost words/passphrase **and** don't know which address held the funds", build
a membership filter of **every address that ever transacted** and match against it —
no target address needed. `--bloom-build IN OUT --bloom-n N` streams an address list
(one per line; `IN='-'` reads stdin, so pipe a full-node/indexer dump straight in,
nothing hits disk but the filter) into a **dual bloom**: filter 1 is a *blocked*
bloom over the raw program (the GPU prefilter, one cache-line per probe); filter 2 is
a *classic* bloom over `sha256(program)` (the host cull, k≈32). The two are
independent, so their false-positive rates multiply. At the whole-chain scale
(~1.5e9 addresses) that's **8 GiB + 8 GiB = 16 GiB** with a **combined FPR ~1e-15** —
effectively zero spurious hits across a trillion-candidate sweep. `--bloom FILE` mmaps
it (filter 1 → each GPU, filter 2 paged on the host), derives each candidate under a
purpose list (mixed script types), and reports the matched address + full path.

**Always `--bloom FILE --bloom-stat` after building** — it *measures* the true
per-filter and combined FPR (a blocked bloom's real rate is far above the naive
`fill^k`; the estimator was wrong once, and `--bloom-stat` is the ground truth).
`--bloom-sizes N` dry-runs the sizing; `--bloom-check ADDR` probes one address through
each filter. Details + the FPR analysis: `docs/BLOOM_PLAN.md`.

```sh
# build the filter from a streamed address dump, then verify it
curl -s https://…/all_addresses.txt.gz | zcat \
  | bip39rxcrack --bloom-build - ../data/alladdrs.blf --bloom-n 1500000000
bip39rxcrack --bloom ../data/alladdrs.blf --bloom-stat     # measure the real FPR
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

Flags: `--words` | `--mnemonic` + `--passphrase PATTERN` (`-D DIR` for `[:dict:]`s) |
`--template`/`--pattern`; target `--address` | `--addresses`/`--addresses-file` |
`--xpub`/`--xpubs` | `--bloom FILE` | `--target-chaincode`; derivation scan
`--purpose --account --change --gap --no-checksum`; `--exhaustive` (all matches in a
set); `--nth` / `--no-nth`; `--no-compact` (compaction is on by default); sharding
`--start --count --limit --shards N --order`; `--rank` (librxe index of an
arrangement); `--resume LOG` (continue a killed run); multi-GPU `--device D`,
`--devices 0,1` / `--gpus N`, `--print-total`; SSH hive `--hosts`, `--remote-bin`,
`--ssh`; bloom tooling `--bloom-build`, `--bloom-stat`, `--bloom-sizes`,
`--bloom-check`; `--kernel-key` (kernels-version hash, for checking hive hosts are in
sync). `bip39rxcrack -h` lists them all.

**Multi-GPU work-queue.** For the `--words`+`--address` and `--template`+`--address`
(missing-word / `[:Nth:]`) paths, the supervisor owns a queue of **fine shards** (~8M
candidates each, or `--shards N`) handed to persistent per-GPU workers over a line
protocol, and shows **one consolidated live line** summing all GPUs. `--order first|ends|center|random[:seed]` sets the sweep order so you can
exploit a prior on where the key is (a work queue with no prior has the same *expected*
time as contiguous halves, but ordering wins when the key isn't uniform). A worker that
dies has its in-flight shard **re-queued** to a survivor, so the run tolerates a GPU —
or a whole machine — falling over. Design: `docs/WORKQUEUE_HIVE_PLAN.md`. Gate: `make workqueue-gate`.

**SSH hive.** The *same* protocol runs over SSH: **`--hosts SPEC`** spreads workers
across machines, where `SPEC` is a comma list of `[user@]host[:port][/ngpu]` (host
`local` = in-process). Each remote GPU is a peer worker in the one shard queue,
reached as `ssh dest --remote-bin … --worker`. `--remote-bin PATH` (absolute → also
the remote working dir, so relative `--bloom`/`-D` paths resolve there); `--ssh "CMD"`
for jump hosts / ports / keys. Assumes passwordless key auth and that the tool + any
`.blf`/dict files already exist on each box. Scaling is linear across the network
(coarse shards + rate-limited progress make the SSH hop negligible against
PBKDF2-bound compute — measured **1.5× for +50% GPUs** across two rented boxes). A
machine dropping off mid-run has its shards re-queued to survivors. Gate: `make hive-e2e`
(loopback; skips without passwordless `ssh localhost`).

**Self-contained binary.** The kernel source is **embedded in the binary**, so a copied
binary runs with no `cuda/` dir — handy for the hive (copy just the binary to each box).
The binary carries a kernels-version key (`--kernel-key`); the hive supervisor **refuses
a worker whose kernels don't match** rather than letting a stale copy fail silently. (A
`cuda/` dir on disk still wins for in-place kernel edits during development.)

**Multi-GPU (other modes).** Crack jobs **fan out over all visible GPUs by default**
(pin a single card with `--device D`; `CUDA_VISIBLE_DEVICES` is honoured). `--devices
0,1` (or `--gpus 2`) selects an explicit set. Either way the process becomes a supervisor:
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
cuda/crack_kernels.cu      crack + sieve/compaction + bloom + passphrase kernels
cuda/gen_embed.py          bakes the 4 kernel files into cuda/kernels_embed.h (make)
src/bip39rxcrack.c         host: enumerate, bloom, patterns, work-queue, hive, report
gate/gen_*.js, gate.c,     oracle-driven vector generators + gate harnesses
  e2e.js, *_gate.js
```

## More docs

- `docs/BLOOM_PLAN.md` — the "any funded address" dual-bloom (blocked GPU prefilter +
  classic host cull), the FPR analysis, and `--bloom-*` tooling.
- `docs/PASSPHRASE_PATTERNS.md` — the passphrase pattern grammar (classes / POSIX /
  alternation / dictionaries) and the mixed-radix on-GPU unranking.
- `docs/WORKQUEUE_HIVE_PLAN.md` — the fine-shard work-queue and the SSH hive.
- `docs/MULTIGPU_PLAN.md` — the contiguous multi-GPU fan-out (other modes).

## Roadmap

- **EC / occupancy tuning.** secp256k1 is correctness-only (double-and-add `k·G` +
  Fermat inverse); a fixed-base comb, batch inversion and occupancy work are the
  levers for the EC-heavier paths.
- **Passphrase:** `{m,n}` variable length; nested alternation; and a generic
  streaming-`rxe` fallback for patterns no closed-form unranker can express.
- **Hive:** two-level machine chunking (one SSH connection per box, not per GPU);
  bounded per-host reconnect; `--account` scan for the xpub path.
- **Electrum** seeds; full reseed39 job-file (`--job`/`--link`) parity.

The `{{N!?}}` keyed-shuffle is intentionally **not** implemented: a full-space
recovery is order-independent, the reported index already matches reseed39's
canonical rank, and a per-candidate shuffle would block cheap incremental sharding.
