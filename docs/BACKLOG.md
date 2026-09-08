# Backlog

The feature roadmap Kiko laid out is shipped: multi-GPU → work-queue → dual-bloom
(BLF3) → passphrase integration → patterns → alternation/dictionaries → account scan →
SSH hive → self-contained binary → kernel-version check → bloom byte-identity check are
all in `main`, gated, and the big ones proven on real hardware (two-box hive, real
four-`[:bip39:]` recovery).

This file is the single place for what's deliberately *not* done yet — the items were
scattered across `BLOOM_PLAN.md`, `PASSPHRASE_PATTERNS.md`, `WORKQUEUE_HIVE_PLAN.md`,
`MULTIGPU_PLAN.md`, and the README Roadmap. Nothing here is a bug or a correctness risk;
it's all "someday" and "nice-to-have." Each item says *why* it's deferred.

## Features (the big rocks)

- **Generic streaming librxe passphrases.** The mixed-radix grammar (classes / POSIX /
  `(a|b|c)` / `[:dict:]` / `{N}`) covers the passphrase-recovery cases in closed form.
  The fallback for *arbitrary* patterns no closed-form unranker can express is a generic
  streaming path: enumerate candidates on the host with librxe, feed the GPU. Costs
  throughput (host-bound enumeration), but it's the "handles anything" escape hatch.
  Spec: `PASSPHRASE_PATTERNS.md` (the "generic streaming path" note).

- **Multicoin.** `coin'` is hardcoded to Bitcoin; per-coin address decode/encode are the
  only real code traps. **Blocked on the oracle, not the kernel:** the reseed39
  correctness authority is BTC-only, so multicoin would ship *unverified* against a
  reference. Prerequisite = extend the oracle (reseed39) to the target coin(s) first;
  only then wire `coin'` + the address codec. Plan's multi-currency section in
  `BLOOM_PLAN.md`.

## Smaller feature gaps

All of these are **rejected cleanly today** (a clear error, never a silent wrong answer):

- **`{m,n}` variable-length** passphrase repeats — only fixed `{N}` is supported.
- **Nested alternation** — `(a(b|c)|d)`; the current parser is one level.
- **xpub account-scan asymmetry.** `--account N` scans the `--words`/chaincode paths, but
  `--xpub`/`--xpubs` don't sweep accounts (an xpub already pins one account node). An
  `--account` sweep for the xpub path would remove the asymmetry. Low priority.
- **Electrum seeds** (a different mnemonic scheme) and **full reseed39 job-file parity**
  (`--job`/`--link` to run a reseed39 estimator job directly).

## Rough edges worth polishing

1. **Hive preflight / `--hive-check`.** *(Highest value.)* The kernel-key and bloom-key
   checks fire at READY time — early, but only after a job is launched and workers spin
   up. A one-shot that SSHes every host up front and prints a table
   (`host | reachable | kernel-key | bloom-key | match?`) would let you *verify before
   committing* to a long run and fail fast on a stale binary/filter. The plumbing already
   exists (`--kernel-key`, `--bloom-key`); this just fans them out. Serves the multi-box
   workflow directly.
2. **Bounded per-host reconnect.** A machine that drops mid-run has its in-flight shards
   re-queued to survivors (correct, proven), but is **never retried on that host** — you
   silently lose that box's throughput for the rest of the run. Fine for a short job,
   wasteful for a long one. Add a bounded reconnect/re-add.
3. **Two-level hive fan-out.** The flat model opens one SSH connection *per GPU*; fine for
   a handful, chatty for a box with dozens. A per-machine sub-supervisor (one connection
   per box, local fan-out) is the fix. Deferred — not needed at current scale.

## Build / memory

- **mmap-based filter construction (build filters larger than RAM).** The *only* piece
  that requires the whole filter in RAM is the BUILD: `build_bloom_file` `calloc`s
  filter1 + filter2, inserts, then `fwrite`s (peak RAM = full filter size, e.g. 14.5 GiB
  for a 6.5+8 build). Everything else already `mmap`s and so already works on a filter
  larger than RAM: `load_bloom_file` maps the `.blf` `PROT_READ, MAP_SHARED` and hands the
  pointers straight to the crack (filter1 -> VRAM via a sequential `cuMemcpyHtoD` from the
  map; filter2 host cull reads the map directly) and to `--bloom-stat`'s Monte-Carlo FPR
  measurement (random probes page-fault through the cache — slow but correct). So the fix
  is just the build: `ftruncate` the output to its final size, `mmap(MAP_SHARED)` it, and
  insert into the file-backed region. Caveat: bloom inserts are maximally random (k
  scattered bytes/address), so past RAM the dirty-page writeback thrashes to disk-random-IO
  speed — mmap makes oversized builds *possible*, not fast. Not needed on the 256 GiB box;
  worthwhile for low-RAM hosts or very large filters. (Kiko's observation, 2026-09-08.)

## EC / performance

- **secp256k1 is correctness-only** — double-and-add `k·G` + a Fermat inverse. A
  fixed-base comb, batch (Montgomery) inversion, and occupancy tuning are the levers for
  the EC-heavier paths (the address/xpub derivations). The PBKDF2-bound paths (the common
  case) don't gain from this; the EC-heavy ones would. Diagnostics already present:
  `--profile`, `CRACK_MAXREG`, `CRACK_DEF`.

## Operational (not code)

- **Address data source** for rebuilding `alladdrs.blf` at scale. Today: the
  alladdresses.loyce.club sorted list (~36 GB gz, ~72-min download on the box — a single
  point of friction). Alternatives: a full node's `dumptxoutset`, or a maintained indexer
  (electrs). See `BLOOM_PLAN.md` Open questions. Also open there: the `.cull` format
  question is now moot (BLF3 stores no exact cull), and whether single `--address` should
  route through a size-1 filter (recommendation stands: keep the direct compare).

## Explicitly *not* doing

- **`{{N!?}}` keyed-shuffle.** A full-space recovery is order-independent, the reported
  index already matches reseed39's canonical rank, and a per-candidate shuffle would block
  cheap incremental sharding. Intentionally omitted.
