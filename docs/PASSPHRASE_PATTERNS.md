# Passphrase patterns (regime A)

`--mnemonic "…" --passphrase PATTERN` cracks a **fixed mnemonic with an unknown
BIP39 passphrase** whose *shape* is known. It works with any target — a single
`--address` or a `--addresses`/`--bloom` SET — and with `--exhaustive`,
`--purpose`, `--account`, and multi-GPU fan-out, exactly like the other modes.

## Grammar (what `PATTERN` accepts today)

A sequence of elements, each optionally repeated `{N}`:

| element         | meaning                                             | example              |
|-----------------|-----------------------------------------------------|----------------------|
| literal char    | that exact byte                                     | `bike`               |
| `\x`            | escaped literal (`\[`, `\(`, `\|`, `\\`, …)          | `\(`                 |
| `[set]`         | one char from the set; ranges `a-z` ok              | `[0-9]` `[a-f]`      |
| `[:name:]`      | a POSIX class **or** a dictionary (see below)        | `[:digit:]` `[:words:]` |
| `(a\|b\|c)`     | alternation of literal strings (variable length ok) | `(spring\|summer)`   |
| `…{N}`          | repeat the preceding element N times                | `[a-z]{6}`           |

POSIX classes (bodies match `rxe`): `alpha` `digit` `alnum` `upper` `lower`
`xdigit` `space` `blank` `punct` `graph` `print`.

**Dictionaries** — `[:name:]` that isn't a POSIX class loads `name.dict` (one
word per line) from the directories given by **`-D DIR`** (repeatable), then the
current dir — the same convention as `rxe`'s tools. Use this for alternations too
big to type by hand.

Examples: `[0-9]{4}` (PIN), `[A-Za-z0-9]{8}`, `bike[0-9]{2}`,
`(spring|summer|winter)[0-9]{2}`, `prefix[:words:][0-9]{1}`.

## How it works (why it's fast and shardable)

The passphrase space is a **mixed-radix number over a per-position STRING-set
table** (last position least-significant), so candidate index `j` unranks to a
string *on the GPU*, per thread — no candidates cross the PCIe bus. Each position
offers a set of *alternatives*: a char class is the degenerate case where every
alternative is one byte; alternation and dictionaries add multi-char,
variable-length alternatives (the salt length is computed per candidate).
`[0-9]{4}` produces the identical enumeration to the old digit-only path
(byte-for-byte backward compatible). The host compiles `PATTERN` into three flat
tables (`pos_stroff`, `str_off`, `str_bytes`) via `pp_parse`, uploads them once,
and `g_crack_pass{,_bloom}` build the PBKDF2 salt via `pp_salt` (mixed-radix
decode). Because it's a plain index range the work-queue shards it like
everything else, and PBKDF2 stays the one shared cost, so a bigger alphabet — or
a whole dictionary — is essentially free per candidate.

Limits: up to `PP_MAXPOS` (128) positions and `PP_MAXSALT` (256) passphrase bytes.

## Not yet supported

- **`{m,n}` variable length** — rejected with a clear message; use a fixed `{N}`.
  (Would be a host-side length loop over fixed-length sub-spaces.)
- **Nested / non-literal alternation** — alternatives in `(…)` are literal
  strings; no nested classes or groups inside them yet.

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
