# Passphrase patterns (regime A)

`--mnemonic "…" --passphrase PATTERN` cracks a **fixed mnemonic with an unknown
BIP39 passphrase** whose *shape* is known. It works with any target — a single
`--address` or a `--addresses`/`--bloom` SET — and with `--exhaustive`,
`--purpose`, `--account`, and multi-GPU fan-out, exactly like the other modes.

## Grammar (what `PATTERN` accepts today)

A **fixed-length** sequence of elements, each optionally repeated `{N}`:

| element        | meaning                                  | example        |
|----------------|------------------------------------------|----------------|
| literal char   | that exact byte                          | `bike`         |
| `\x`           | escaped literal (`\[`, `\{`, `\\`, …)     | `\[`           |
| `[set]`        | one char from the set; ranges `a-z` ok   | `[0-9]` `[a-f]`|
| `[set]{N}`     | N positions of that class                | `[a-z]{6}`     |
| `lit{N}`       | a literal repeated N times               | `x{3}`         |

Examples: `[0-9]{4}` (a PIN), `[a-z]{6}`, `[A-Za-z0-9]{8}`, `bike[0-9]{2}`,
`correct-horse-[0-9]{4}`.

## How it works (why it's fast and shardable)

The passphrase space is a **mixed-radix number over the per-position alphabets**
(last position least-significant), so candidate index `j` unranks to a string
*on the GPU*, per thread — no candidates cross the PCIe bus. `[0-9]{4}` produces
the identical enumeration to the old digit-only path (byte-for-byte backward
compatible). The host compiles `PATTERN` into a per-position charset table
(`pp_parse`), uploads it once, and `g_crack_pass{,_bloom}` build the PBKDF2 salt
from it via `pp_salt` (a mixed-radix decode). Because it's a plain index range,
the work-queue shards it like everything else, and PBKDF2 stays the one shared
cost, so widening the alphabet is essentially free.

## Not yet supported

- **`{m,n}` variable length** — rejected with a clear message; use a fixed `{N}`.
  (Would be a host-side length loop over fixed-length sub-spaces.)
- **Alternation / dictionaries** — `(spring|summer)` or `prefix [:dictionary:]`.
  These are the natural next step and are *not* a second engine: a position
  whose alphabet is a set of **strings** (instead of bytes) drops straight into
  the same mixed-radix framework and yields variable length via word choice.
  A char class is just the degenerate case (1-char strings). Planned.

## The generic fallback we deliberately don't have (yet)

The modes here each ship a **closed-form on-GPU unranker** for the shape they
handle (permutations for `--words`, mixed-radix for passphrases). We do **not**
have a general on-device port of `rxe` (the host regex enumerator: parsed AST +
GMP bignum unranking), because `rxe` is a host C+GMP library that doesn't map to
a register-frugal, branch-light per-thread device function.

For arbitrary patterns that none of the closed-form unrankers can express
(nested alternation, unbounded repetition, spaces exceeding 2⁶⁴), the intended
future fallback is a **generic streaming path**: enumerate candidates on the host
with `rxe` and stream them to the GPU in chunks. It is fully general, but it
trades away throughput (capped at host enumeration speed) and complicates
work-queue sharding (seek-per-shard, then sequential), so it's a *fallback* to
reach when no standard pattern fits — not the default. Not implemented yet.
