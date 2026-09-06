# Multi-GPU fan-out — implementation plan

Status: **not started** (written 2026-09-06, for the dual-RTX-5090 session).
Baseline: single 5090 at ~1.43 Mc/s (gap=1) / ~1.23 Mc/s (gap=5), 24-word `[:Nth:]`,
after the SHA-512 + secp256k1 unroll wins. `main` @ `8c902ff`.

## Goal

Run one crack job across N GPUs with ~linear speedup, one command:

```sh
bip39rxcrack --devices 0,1 --pattern '…' --address 1Ap… --gap 5
#   -> forks one child per device over disjoint index slices,
#      first child to FOUND wins, siblings are killed, result printed.
```

Keep single-GPU behaviour byte-identical (same found index == librxe rank).

## What already exists (lean on it)

- **Index sharding**: `--start IDX --count N` windows the canonical index space; every
  mode honours it. This IS the slicing primitive — no new enumeration work.
- **Canonical index**: after phase8, a found index == librxe rank globally, so a
  child's reported index is already the global answer (no offset math needed).
- **Per-shard resume**: `--resume LOG` works per child (each slice has its own log),
  so a killed/rebooted child resumes its slice independently.
- **`total` is computed in-mode**: e.g. `mode_missing` computes
  `total = 2048^kU << freebits`; the permutation/passphrase modes likewise. We just
  need to surface it before cracking.

## Design decisions (decided; revisit only if a test says so)

1. **Process fan-out, not threads.** One child process per GPU (`fork`+`exec` of
   `/proc/self/exe`, or `posix_spawn`). Rationale: CUDA contexts are per-process and
   clean; a crash/OOM on one GPU can't take down the others; matches the article's
   `-startp/-endp` model and our existing `--start/--count`.
2. **Contiguous slices**, not interleaved. Child g gets
   `[g*total/N, (g+1)*total/N)`. Simpler (reuses `--start/--count` verbatim), and the
   *average* find time is still ~single/N. Downside: worst case (winner at the end of
   its slice) is ~a full slice. If that ever bites, switch to **strided** slicing
   (child g does indices ≡ g mod N) — but that needs a `--stride` kernel arg, so defer.
3. **Early-exit at process granularity.** First child to print `FOUND` (exit 0) wins;
   parent SIGTERMs the siblings. Children already check `hit_found` every chunk and
   exit within ~one chunk of a hit, so cross-GPU stop latency is one chunk (~0.1–0.5 s).
   No shared-memory flag needed for v1.
4. **Each GPU holds its own everything** (comb table, survivor buffer, later the bloom).
   No inter-GPU sharing; no NVLink assumptions.

## Implementation phases

### Phase A — `--device D` (foundation, ~20 min)
`build_module()` hardcodes `cuDeviceGet(&dev,0)`. Add:
- global `static int g_device=0;`
- CLI: `else if(!strcmp(argv[i],"--device")&&i+1<argc) g_device=atoi(argv[++i]);`
- `cuDeviceGet(&dev,g_device);`
- Print the device index in the `device: …` banner so logs are unambiguous.
Test: `--device 1 --profile` shows the second card; a crack on `--device 1` runs there
(watch `nvidia-smi`).

### Phase B — surface the job size (`--print-total`, ~30 min)
The supervisor must split `[0,total)` without re-implementing the enumeration math.
Add a `--print-total` flag: parse the pattern exactly as the normal path (route through
the same `--pattern`/`--words`/`--template`/`--mnemonic` logic), compute `total` the way
each mode does, print it as a single decimal, and exit **before** `build_module()` (no
GPU needed). Factor the per-mode `total` computation into a small helper the crack modes
and `--print-total` share, so the two can never disagree.
Test: `--print-total` for the 24-word `[:Nth:]` example prints `68719476736`.

### Phase C — the fan-out supervisor (`--devices`, the meat)
Add `--devices 0,1` (or `--gpus N` meaning `0..N-1`). When present, `main` becomes a
supervisor instead of cracking directly:
1. Compute `total` (call the Phase-B helper in-process).
2. Honour an outer `--start/--count` too (so a *cluster* of machines can each take a
   third, then fan out locally — mirrors the article's two-level split). Effective range
   = the outer window; slice THAT across the local devices.
3. For each device d, slice → `(start_d, count_d)`; build the child argv = this process's
   argv with `--devices` removed and `--device d --start start_d --count count_d` added
   (plus `--loginterval …:job_dev{d}.csv` if the user asked for logging, per-child file).
4. `posix_spawn` / fork+exec each child; record pids.
5. **Supervise**: `wait()` in a loop. On a child exiting 0 with a FOUND (parse its stdout,
   captured via pipe, or a `--result-file` each child writes on hit), record the result,
   `kill(SIGTERM)` the siblings, reap them, print the winner, exit 0.
   If all children exit non-zero (NOT_FOUND), print NOT FOUND, exit 1.
Edge cases to handle: a child dies on a CUDA error (treat as failure, don't hang);
Ctrl-C on the parent must SIGTERM all children (install a SIGINT handler that kills the
group — spawn children in a new process group and signal the group).

### Phase D — progress aggregation (nice-to-have)
Each child already emits `-p`/CSV for its slice. For a **global** view, the supervisor
can sum children's rates: simplest is each child writes its CSV (`job_dev{d}.csv`) and the
supervisor periodically reads the last row of each and prints a combined
`swept_total / Σrate / global-pct` line. Reuses the CSV format we already emit. Optional
for v1 — per-child `-p` to stderr is already informative.

### Phase E — verification / gating (do NOT skip)
- **Parity test**: plant a known winner at global index J (e.g. the 24-word
  `awful clap arrow` @ 6142642800). Run `--devices 0,1` over a window straddling J and
  assert: exactly one child reports FOUND, reported index == J == librxe rank, mnemonic
  matches, and the other child was killed. This is the multi-GPU analogue of `e2e.js`.
- **Throughput test**: sum of two children's Mc/s ≈ 2× single-GPU (allow ~5% for
  supervisor/imbalance). Log if it isn't (points to slicing or contention).
- **Resume test**: SIGTERM a child mid-slice, `--resume job_dev0.csv`, confirm it
  continues its slice from the last row.

## Gotchas noted in advance
- **PTX cache race**: children share `/tmp/bip39rxcrack_ptx_*.ptx`. First-run concurrent
  builds could race the file write. Mitigation: have the supervisor warm the cache once
  (a throwaway `--print-total`/tiny build) before forking, OR write the PTX atomically
  (tmp + rename). Prefer warm-once.
- **Device enumeration**: `--devices` indices are CUDA ordinals; honour
  `CUDA_VISIBLE_DEVICES` if the user sets it (don't fight it).
- **Uneven GPUs**: dual 5090 → equal slices. If a mixed rig ever appears, add optional
  weights `--devices 0:1.0,1:0.7`; skip for now.
- **Found-index semantics under an outer window**: keep everything in the GLOBAL
  canonical index so a found index is directly comparable to `--rank`/librxe regardless
  of how it was sliced. (We already have this; don't regress it.)

## Rough sizing
Phases A–C are the real work; A is trivial, B is small, C is a few hundred lines of host
code (spawn/supervise/slice) with no kernel changes. Target: A+B+C+E in one session.

---

## Future: bloom "no-target" mode (separate, later)

Return any derived address that has ever transacted, instead of matching one target —
useful when a partial-seed recovery has *also* forgotten which address/derivation index
was theirs.

- **Size** (n=1.5e9 addrs, p=1e-5): m/n = -ln p/(ln2)² ≈ 23.96 bits → **~4.2 GiB**, k≈17.
  Fits easily in 32 GiB alongside our ~2 GiB working set. Each GPU holds its own copy.
- **Use a *blocked* bloom** (all k bits within one cache-line block) → **1 DRAM read per
  probe** instead of 17 scattered ones. Costs ~20–40% more memory (~5–6 GiB, still fine)
  but keeps the latency hit small (est. <15%; bandwidth is a non-issue at ~3 GB/s of probe
  traffic vs ~1.8 TB/s). Naive 17-scattered-probe bloom would be ~20–40% slower — avoid.
- **Two-stage**: bloom (fast, on-GPU) → exact verify of the rare hits host-side (kills the
  1e-5 false positives). Build the bloom offline from a UTXO/address dump; ship/generate
  separately.
- **Scope guard**: only useful when the search space is already constrained (partial
  seed); random-seed scanning against all funded addresses is computationally hopeless
  (~2^160 keyspace), so this stays a recovery aid, not a theft path. Keep the README's
  "recover your own wallet" framing.

## Pointers into the code
- Device select + build: `src/bip39rxcrack.c` `build_module()` (~line 108), banner ~line 145.
- CLI parse loop: `src/bip39rxcrack.c` ~line 800.
- Per-mode `total`: `mode_missing` (`total=2048^kU<<freebits`), `mode_crack*`,
  `mode_crack_pass`. Factor these into a shared `job_total()` for `--print-total`.
- Found-index/report: each mode's FOUND block already prints the canonical index.
