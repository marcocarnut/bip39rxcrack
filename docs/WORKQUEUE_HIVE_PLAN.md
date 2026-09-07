# Work-queue + ordering + hive — design & plan

Status: **planned 2026-09-06** (supersedes the contiguous fan-out in `MULTIGPU_PLAN.md`
for crack jobs). Successor to `main`@`68fa373` (contiguous halves, all-GPUs-default).

## Why (the honest case)

The shipped fan-out splits the space into N contiguous slices. For a **uniform** key
that's already optimal on average (E[time]=T/4r, same as any front-loaded scheme) — but
it wastes a GPU when the key sits on one side, has no way to exploit a **prior** about
where the key is, and can't load-balance uneven/contended GPUs. Fine shards pulled from a
**supervisor-owned queue** fix all three, and — the reason this shape specifically — the
same "hand a worker a shard range, stream results back" protocol runs over **SSH** to a
hive of rented boxes with no rearchitecture.

Measured cost that rules out the naive approach: a fresh process costs **~0.7 s** of
startup (CUDA context + module JIT-to-SASS + comb table) *before* the first candidate. So
**process-per-shard is out**; workers must be **persistent** (context built once) and loop
over shards.

## The math (settled, so we build for the right reason)

| key at fraction *p* | contiguous halves | fine front-loaded | fine + correct prior |
|---|---|---|---|
| E[time], uniform key | T/4r | T/4r | **≪ T/4r** |
| worst case | T/2r | T/2r | depends |

Ordering only helps when the key distribution is **non-uniform and you can order toward
the mass**. That — plus load-balancing and crash-recovery — is the payoff, not average
uniform time-to-find.

## Core abstraction: a worker is a stream you send shards to

Transport-agnostic line protocol over a duplex stream (local pipe today, `ssh host … --worker`
later). ASCII lines, one per message:

```
S->W:  HELLO <proto> <shard-hint>              (optional; or supervisor just sends SHARD)
W->S:  READY <ngpu> <binhash> <kernelshash> <version>
S->W:  SHARD <id> <start> <count>              (a unit of work; global canonical indices)
W->S:  PROG  <id> <swept> <hashed>             (streamed, ~2–4/s)
W->S:  DONE  <id>                              (shard exhausted, no hit)
W->S:  FOUND <id> <index> <mnemonic...>        (hit; supervisor stops the run)
S->W:  STOP                                    (found elsewhere / user abort -> worker exits)
```

- **Supervisor owns the queue + ordering policy.** It decides which shard id maps to which
  `[start,count)` and in what order it hands them out. Policies are just orderings:
  `first` (front-to-back), `ends` (ends-to-center), `center` (center-out), `random[:seed]`,
  and later adaptive.
- **Load-balancing is automatic** — a worker asks for the next shard when it finishes one.
- **Crash/partition recovery** — a shard handed out but not `DONE`/`FOUND` before its worker
  dies (EOF on the stream) goes back on the queue for someone else. At-least-once delivery.
- **Version invariant** — `READY` reports `binhash`/`kernelshash`/`version`; the supervisor
  refuses a worker that doesn't match, because "global index J" must mean the same mnemonic
  on every machine or the canonical-rank contract breaks.

## Shard sizing

Target each shard at ~1–4 s of GPU work so PROG cadence is smooth and re-queue on death
loses little: `shard_count ≈ ceil(total / (rate·target_secs·safety))`, clamped. A GPU pulls
a shard every ~1–2 s → even thousands of shards is a handful of pipe messages/s (µs each) —
local latency is a non-issue. Remote hops are amortized by the two-level split below.

## Two-level for the hive

Single supervisor → per-machine **sub-supervisor** → local GPU workers. The top hands each
machine a **coarse** super-shard (amortizes the SSH round-trip); the machine subdivides it
into fine shards across its own GPUs with ~µs local latency. Local-only runs are just the
one-machine case. Machine-level chunks are themselves re-queueable if a whole box vanishes.

## Implementation stages

### Stage 1 — local work-queue (this change)
1. **`setup()` / `sweep(start,count,&res)` refactor.** Split each crack mode into context+
   upload done ONCE and a re-entrant range sweep. Keep byte-exact vs the current gates at
   every step (this is the risky part — gate after each mode). Start with `mode_crack_addr`
   (the main path), then `mode_missing`, `mode_crack_pass`, `mode_crack`.
2. **`--worker`** — build once, read SHARD lines from stdin, run `sweep`, emit PROG/DONE/FOUND.
3. **Supervisor rewrite** — own the queue + `--order` policy, spawn N local workers over
   pipes, pump the protocol, aggregate PROG into one global line, re-queue on worker EOF,
   STOP siblings on FOUND, print the winner. New flags: `--shards N`, `--order first|ends|center|random[:seed]`.
4. **Gate** — extend `e2e_multigpu.js`: planted winner found under each ordering policy;
   index==J==librxe rank; kill a worker mid-shard and assert the shard is re-queued and the
   run still finds it (recovery test).

### Stage 2 — SSH transport + hive — **DONE (flat model)**
Shipped as the **flat hive**: every GPU across every machine is a peer worker in the one
shard queue, reached over `ssh`. It reuses the Stage-1 supervisor loop verbatim — only the
worker *spawn* changed (local `posix_spawn` vs `ssh dest …`); the poll/shard/PROG/DONE/FOUND/
re-queue path is transport-agnostic.
- `--hosts SPEC` — comma list of `[user@]host[:port][/ngpu]` (ngpu default 1; host `local`
  runs in-process). One worker per host×GPU. `--remote-bin PATH` (the binary on the remotes;
  an absolute path also sets the remote cwd so relative `cuda/…`, `--bloom ../data/…`, `-D dict/`
  resolve as locally). `--ssh "CMD"` (default `ssh`; space-split, so `-J`/`-p`/`-i` work).
  Args are shell-quoted for the remote; the remote command is `cd <dir> && exec <bin> <args>`.
- Assumes **passwordless key auth** and that the **tool + `.blf`/dict files already exist** on
  each box (per Kiko).
- READY handshake carries `g_ptx_key` (kernels hash) already; version-match check in place.
- **Partition/crash recovery works over SSH**: a dead `ssh` (machine dropping off) EOFs its
  pipe → the same re-queue branch hands its in-flight shard to a survivor. Verified on loopback
  (kill one worker mid-run → throughput halves, still FOUND at the right rank).
- Gate `gate/e2e_hive.js` (`make hive-e2e`): loopback `--hosts localhost/2` and
  `localhost/1,localhost/1`; SKIPs if passwordless `ssh localhost` is unavailable.

**Deferred (not needed for the flat model, revisit for very large hives):**
- Two-level machine chunking + a per-machine sub-supervisor (the flat model uses N ssh
  connections per host — one per GPU; fine for a handful of GPUs/box, chatty for dozens).
- Bounded reconnect (today a dropped machine's work is re-queued to survivors, not retried on
  that host). Prereq auto-checks (binary/blf presence) beyond the natural first-shard failure.

## Risks / notes
- **Byte-exact refactor** is the crux of Stage 1 — gate after every mode split.
- **Secrets on remote boxes**: the target/mnemonic ships to every worker (fine for the
  user's own recovery; note it — don't send to boxes you don't trust).
- **Keep the canonical global index everywhere** — never let a worker report a slice-local
  index; the whole hive relies on J == librxe rank.
- The current contiguous supervisor stays as the fallback if `--shards`/workers are off, or
  is replaced outright once the work-queue is gated — decide at Stage 1 close.
