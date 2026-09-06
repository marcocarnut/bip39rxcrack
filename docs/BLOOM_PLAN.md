# Bloom target set — design & plan

Status: **planned 2026-09-06.** Unifies single-address, multi-address, "any funded
address ever," and (by generalization) multi-xpub matching into ONE on-GPU membership
test. Successor context: `main`@`ebd509c` (work-queue).

## Goal

Replace "compare the derived program to one `target_prog`" with "probe a membership
filter, then CPU-cull the rare hits." The filter holds a *set* of target fingerprints,
so every target cardinality collapses to the same mechanism:

- `--address A`            -> the existing exact compare (degenerate case, unchanged)
- `--addresses A,B,...` / `--addresses-file F` -> filter built at startup from N programs
- `--bloom F.blf`          -> a prebuilt filter (e.g. every funded address, ~1.5e9)
- `--xpubs X,...`          -> filter of account chaincodes (EC-free match); falls out free

One construction, N from 1 to ~1.5e9, no per-cardinality special-casing and no binary
search: the filter degrades gracefully to near-zero FPR for small N.

## Key insight: the target is already a hash — slice, don't re-hash

An address program (hash160, or a taproot x-only key) and an account chaincode (HMAC
output) are already uniform, independent bits. A generic bloom hashes its key because it
can't assume that; we can, so we **slice bit-fields straight out of the program** as the
filter's hash outputs. Free, and provably as good as a real hash.

## Construction: blocked (split-block) bloom, one fetch per lookup

Filter = an array of **32-byte blocks** (= one GPU memory sector = one DRAM transaction).
Each key maps to exactly one block; all k bits live inside it. A lookup is a single
32-byte aligned load — **1 fetch, vs k scattered fetches** for a naive bloom.

For a program/chaincode `P` (uniform bytes; use the first 20):

```
block  = u32(P[0..3]) & (NBLOCKS-1)     // NBLOCKS = 2^b ; b bits pick the 32-byte block
for i in 0..k-1:                         // k bits, all within that one block
    pos_i = P[4+i] & 0xFF                // 8 bits -> which of the block's 256 bits
    set / test  block bit pos_i
```

- **Insert** = OR the k bits; **test** = AND-check them. All k tests are register ops on
  the one loaded block; the cost is the single load.
- **Bit budget**: b + 8k bits. 4 GiB -> b=27; k=16 -> 27+128 = 155 of 160 (hash160) — fits,
  with independent (not double-hashed) positions. Chaincodes/taproot (256 b) fit trivially.
- Chosen **k=16** (bit position j from byte `P[4+j]`, `0..255` in the 256-bit block):
  uses `P[0..19]` = exactly one hash160, and lands near optimal k for our 16-32 bits/key
  range. (k=8-one-bit-per-lane, the Parquet/Impala SBBF, is a simpler variant but far from
  optimal at high bits/key — measured FPR ~1e-3 vs k=16's ~1e-5 at 33 bits/key.)

**Empirically validated** (`make bloom-selftest`, host-only, `cuda/bloom_common.h`):
zero false negatives, and FPR tracks theory once probed with *uniform* keys (a real
hash160 is uniform; an early test using a weak xorshift low-byte inflated FPR ~370x and
was the test's bug, not the filter's):

| bits/key | filter (1M keys) | measured FPR (k=16) |
|---|---|---|
| ~8  | 1 MiB | 1e-1  (k too high for so few bits) |
| ~17 | 2 MiB | 2.2e-3 |
| ~33 | 4 MiB | ~1.5e-5 |

At the real target (1.5e9 keys, 4 GiB = ~23 bits/key) this interpolates to **FPR ~1e-4**,
matching the estimate below; 6 GiB (~33 bits/key) buys ~1e-5.

## Sizing & FPR — and why we have huge latitude

4 GiB = 2^35 bits, 2^27 blocks, n=1.5e9 -> ~23 bits/key, ~11 keys/block -> **FPR ~1e-4**
(blocked imbalance included). 8 GiB -> ~1e-6. Either is fine, because:
- **true positives ~1** (one wallet in a vast space), and
- the **cull is cheap**: FPR x candidate-rate = 1e-4 x ~2e6/s = ~200/s. Nothing.

So run the filter loose; the cull mops up. Trade memory for FPR freely (we use ~2 GiB of
32; a 4-8 GiB filter is comfortable — each GPU/process holds its own copy).

## The cull (exact, CPU-side)

A bloom hit is emitted to the existing hit buffer; the host culls it against an exact set:
- **small N** (`--addresses`/`--xpubs`): the N programs in RAM (hash set) — trivial, zero FP.
- **"any address"** (`--bloom`): an exact backing set for ~1.5e9 programs. A sorted array
  of the programs, mmap'd, binary-searched. Full 20-byte = ~30 GB; a **sorted 10-byte
  prefix** (~15 GB, collision prob ~2^-something negligible at 1.5e9) is enough for a
  near-exact cull, with the full program only needed to *report* the hit. Built alongside
  the filter (see below). False-hit volume at FPR 1e-4 x 2e6/s = ~200/s -> the cull must be
  automatic (too many to hand-check over a long run), but 200 exact lookups/s is nothing.

The cull is the final arbiter, which is *why* we can drop script-type handling: different
purposes derive different keys (different programs); a same-key cross-type match is a true
positive; a different-key 20-byte collision is 2^-160; anything else the cull kills.

## Build & load paths

- **Startup build** (`--addresses` / `--addresses-file` / `--xpubs`): decode each target to
  its program (or chaincode), insert into a filter sized to N (`m = ceil(1.44*k*N/blockfill)`,
  rounded to a power-of-two block count), upload to the GPU. Cull set = the N entries in RAM.
- **Prebuilt** (`--bloom F.blf`): a separate builder tool `bloom-build` reads an address /
  UTXO dump (full node `dumptxoutset`, electrs, or an address list), decodes each to its
  program, inserts, and writes:
  - `F.blf`  — the filter: a small header (magic, version, NBLOCKS, k, b, kernels-note) +
    the raw block array (mmap-and-upload).
  - `F.cull` — the sorted exact backing set (10-byte prefixes + an index to full bytes),
    for the host cull.
  The cracker mmaps `F.blf`, uploads the block array to each worker's GPU once at setup.

## GPU integration

- In `pub_to_program` we already have the raw program (no encoding). The match step in
  `derive_address_match` changes from `memcmp(prog,target_prog,tlen)` to **`bloom_test(prog)`**;
  on a probe hit, emit `(global_index, program, purpose, change, index)` to the hit buffer
  exactly like today. Host culls; a surviving hit is a real find (re-encode the address for
  display from the known purpose).
- The filter is a device buffer (4-8 GiB) built once in `crack_addr_setup` /
  `crack_missing_setup` (and the chaincode setup for `--xpubs`). Each **work-queue worker**
  holds its own copy on its GPU; for the SSH hive, each box loads/ships its own `F.blf`.
- "Any address" mode probes MULTIPLE programs per seed (across purposes x change x gap), so
  it is heavier per seed than a single fixed target — inherent to scanning derivation paths.

## Performance note (honest)

At our PBKDF2-bound ~1-2 Mc/s, the filter is essentially **free** — one 32-byte random read
per candidate against a GPU that sustains tens of billions/s; even a naive 17-read bloom
would be free. The blocked/slice design isn't bought for speed today; it's bought for
**elegance, no extra hash code, a proven construction, and future-proofing** — and it's no
more work than the naive one.

## Flags (proposed)

```
--addresses A,B,...        build the filter from these target addresses
--addresses-file PATH      ... one address per line (large lists ok)
--xpubs X,...              filter of account chaincodes (multi-xpub; EC-free match)
--bloom PATH               load a prebuilt filter (+ sibling .cull for the exact cull)
--bloom-fpr P              target FPR for a startup-built filter (default 1e-4)
```
`bloom-build` (separate tool): `bloom-build --in addrs.txt --out funded.blf [--fpr P]`.

## Staging

1a. **DONE** — blocked-bloom core (`cuda/bloom_common.h`, slice-don't-hash, k=16, one
    fetch) + host self-test (`make bloom-selftest`): zero false negatives, FPR ~1.5e-5 at
    33 bits/key, extrapolating to ~1e-4 at the 1.5e9 / 4 GiB target.
1b. **DONE** — GPU wiring for `--words + --addresses`/`--addresses-file`: `g_crack_addr_bloom`
    (fused) probes each derived program + appends to a device hit buffer;
    `crack_addr_sweep_bloom` culls per chunk against the sorted RAM set (first chunk with a
    culled-true hit holds the lowest rank → min + stop). Single-target path untouched
    (byte-exact). Gate `make bloom-e2e`: winner among decoys → FOUND at rank; decoys-only →
    NOT FOUND. v1 limits: one script type per set; in-process (no fan-out yet); fused only.
1c. (next) compact bloom path (`g_sieve_perm` → `g_pbkdf2_perm_bloom`) for ~20 Mc/s; the
    missing-word bloom (`g_crack_nth/_missing`); and fold `--addresses` into the work-queue.
2. **`--xpubs`** (chaincode filter; trivial once the above exist — different key length + EC-free).
3. **`bloom-build` + `--bloom` + the sorted `.cull`** (the "any funded address" mode). Needs
   an address source (full-node dump) and the on-disk formats.
4. (Later) fold into the work-queue/hive so each worker/box loads its own filter.

## Multi-currency (forward-looking — design so we don't trap ourselves)

The bloom/cull/work-queue/hive are **coin-agnostic**: they operate on uniform 20/32-byte
fingerprints and don't care what produced them. That's the payoff of slice-don't-hash.
What differs per coin is only the *derivation* and the *encoding*:

- **UTXO family (LTC coin=2, BCH=145, DOGE=3, …)** are secp256k1 + hash160, so the program
  bytes are the SAME construction. Only (1) the `coin'` level in `m/purpose'/coin'/…` (a
  different key -> different program) and (2) the address encoding (version byte / bech32
  HRP / BCH CashAddr, which touches only target decode and find-time re-encode, never the
  hot path) differ. One **filter per coin** is the right unit — contents + derivation are
  per-coin; merging coins works but loses which-coin identity and buys nothing.
- **Ethereum (coin=60)** is a different *derive*: Keccak-256 of the *uncompressed* pubkey,
  last 20 bytes — no hash160, no compression, no script types. It needs its own
  derive-program function, but the fingerprint is still 20 uniform bytes, so the bloom,
  cull, work-queue and hive apply unchanged. Size is NOT the blocker: ETH has ~10^8-10^9
  addresses (same order as BTC or smaller) and fits the same ~4 GiB filter; the cost is the
  algorithm, not capacity.

**Two concrete traps to avoid (fix when we touch derivation, not necessarily now):**
1. `derive_address_match` hardcodes `coin'=0` (`m/purpose'/0'/0'`). Thread a `coin` value
   through the derive (default 0) so adding LTC/BCH is a flag + version table, not surgery.
2. `decode_address` assumes Bitcoin version bytes / HRP. Make target decode + address
   encode **table-driven per coin** (versions, HRP, CashAddr), isolated from the core.

**The abstraction:** a *coin profile* `{coin_type, derive_program_fn, decode_target_fn,
encode_address_fn}` — UTXO coins share one `derive_program_fn` (only `coin'` varies), ETH
plugs its own. Everything downstream consumes the profile's uniform fingerprint and stays
coin-blind (same shape as the `Sweeper` abstraction, one level lower).

**Testing caveat:** the reseed39 oracle is Bitcoin-only. Each coin needs its own SLIP-0044
reference vectors (path + known seed->address) before it can be gated — a validation cost,
not a code trap.

## Open questions

- Exact `bloom-build` input source & cadence (full-node `dumptxoutset` vs a maintained list).
- `.cull` format: 10-byte-prefix sorted array vs a compact exact hash; and whether to keep
  it host-only or also mmap on remote hive boxes.
- Whether single `--address` should route through the (size-1) filter for one code path, or
  keep the current direct compare (recommended: keep it — it's free and already correct).
