#!/usr/bin/env node
/* gen_vectors.js -- emit byte-exact (input -> expected) test vectors for the
 * Phase-1 CUDA crypto gates, using the reseed39 BROWSER ORACLE as the sole
 * correctness authority (estimator/bip39crypto.js + estimator/bip39.js) plus
 * the published BIP39/BIP32 test vectors.
 *
 * THE LAW (cli/BRIEF.md): gate every kernel against the browser reference /
 * published vectors, never against a sibling GPU model. So the "expected"
 * column here is produced by the same JS the browser tool ships.
 *
 * Locate the oracle via RESEED39_DIR (default: the sibling reseed39 clone).
 * Output: line-oriented hex files under vectors/ that the C harness parses.
 *
 * Determinism: a fixed-seed xorshift PRNG, so re-running yields identical
 * vectors (the gate is reproducible).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const RESEED39 = process.env.RESEED39_DIR ||
  ['/root/bip39rxcrack', path.join(__dirname, '..', '..', 'reseed39'),
   path.join(__dirname, '..', '..', 'bip39rxcrack')].find(p => {
     try { return fs.existsSync(path.join(p, 'estimator', 'bip39crypto.js')); }
     catch { return false; }
   });
if (!RESEED39) { console.error('cannot locate reseed39 oracle; set RESEED39_DIR'); process.exit(2); }
const C = require(path.join(RESEED39, 'estimator', 'bip39crypto.js'));
const WORDS = fs.readFileSync(path.join(RESEED39, 'data', 'english.txt'), 'utf8')
  .split(/\r?\n/).map(s => s.trim()).filter(Boolean);
if (WORDS.length !== 2048) { console.error(`wordlist not 2048 (${WORDS.length})`); process.exit(2); }

const OUT = path.join(__dirname, '..', 'vectors');
fs.mkdirSync(OUT, { recursive: true });
const hex = b => Buffer.from(b).toString('hex');

// Anchor the oracle to published ground truth: the generator itself asserts the
// FIPS SHA-512 KAT and BIP32 test-vector-1 constants before emitting anything.
// (Belt-and-suspenders on top of the browser-gated oracle.)
(function anchor() {
  const must = (got, want, m) => { if (got !== want) { console.error(`ANCHOR FAIL ${m}\n  got  ${got}\n  want ${want}`); process.exit(3); } };
  must(hex(C.sha512(C.utf8('abc'))),
    'ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f', 'SHA-512("abc") FIPS KAT');
  must(hex(C.sha512(C.utf8(''))),
    'cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e', 'SHA-512("") FIPS KAT');
  const bm = C.seedToMaster(C.fromHex('000102030405060708090a0b0c0d0e0f'));
  must(hex(C.ser256(bm.k)), 'e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35', 'BIP32 vector-1 m priv');
  must(hex(bm.c),           '873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508', 'BIP32 vector-1 m chaincode');
  const bh = C.ckdHardened(bm, 0x80000000);
  must(hex(C.ser256(bh.k)), 'edb2e14f9ee77d26dd93b4ecede8d16ed408ce149b6cd80b0715a2d911a0afea', "BIP32 vector-1 m/0' priv");
  must(hex(bh.c),           '47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141', "BIP32 vector-1 m/0' chaincode");
})();

// ---- deterministic PRNG (xorshift128) --------------------------------------
let s0 = 0x9e3779b9, s1 = 0x243f6a88, s2 = 0xb7e15162, s3 = 0xdeadbeef;
function rnd() {
  let t = s3;
  const s = s0;
  s3 = s2; s2 = s1; s1 = s;
  t ^= t << 11; t ^= t >>> 8;
  s0 = (t ^ s ^ (s >>> 19)) >>> 0;
  return s0 / 4294967296;
}
function rbyte() { return Math.floor(rnd() * 256) & 0xff; }
function rbytes(n) { const a = Buffer.alloc(n); for (let i = 0; i < n; i++) a[i] = rbyte(); return a; }
function rint(n) { return Math.floor(rnd() * n); }

// ---- entropy -> valid BIP39 mnemonic (standard SHA-256 checksum) -----------
// English wordlist is pure ASCII so the mnemonic bytes == UTF-8 == NFKD(mnemonic).
function entropyToMnemonic(ent) {
  const ENT = ent.length * 8, CS = ENT / 32, total = ENT + CS;
  const dig = crypto.createHash('sha256').update(ent).digest();
  const bits = new Uint8Array(total);
  for (let i = 0; i < ENT; i++) bits[i] = (ent[i >> 3] >> (7 - (i & 7))) & 1;
  for (let i = 0; i < CS; i++) bits[ENT + i] = (dig[i >> 3] >> (7 - (i & 7))) & 1;
  const words = [];
  for (let w = 0; w < total / 11; w++) {
    let idx = 0;
    for (let b = 0; b < 11; b++) idx = (idx << 1) | bits[w * 11 + b];
    words.push(WORDS[idx]);
  }
  return words.join(' ');
}
const ENT_BYTES = { 12: 16, 15: 20, 18: 24, 21: 28, 24: 32 };

// ---------------------------------------------------------------------------
// Stage 1: SHA-512  (msg -> 64-byte digest)   format: <len> <in_hex|-> <out_hex>
// ---------------------------------------------------------------------------
{
  const lines = [];
  const push = (msg) => lines.push(`${msg.length} ${msg.length ? hex(msg) : '-'} ${hex(C.sha512(msg))}`);
  push(Buffer.from('abc'));            // KAT
  push(Buffer.alloc(0));               // KAT ""
  push(Buffer.from('a'.repeat(1000))); // long
  // boundary lengths around SHA-512 block (128) and padding edges
  for (const L of [1, 55, 111, 112, 113, 127, 128, 129, 200, 239, 240, 256]) push(rbytes(L));
  for (let i = 0; i < 64; i++) push(rbytes(rint(300)));
  fs.writeFileSync(path.join(OUT, 'vec_sha512.txt'), lines.join('\n') + '\n');
  console.log(`sha512:   ${lines.length} vectors`);
}

// ---------------------------------------------------------------------------
// Stage 2: PBKDF2-HMAC-SHA512(2048)==mnemonicToSeed
//   password = UTF-8(NFKD(mnemonic)), salt = UTF-8("mnemonic"+NFKD(passphrase))
//   format: <pwlen> <pw_hex> <saltlen> <salt_hex> <seed_hex(128)>
// ---------------------------------------------------------------------------
{
  const lines = [];
  const push = (mnemonic, passphrase) => {
    const pw = C.utf8(C.nfkd(mnemonic));
    const salt = C.utf8('mnemonic' + C.nfkd(passphrase));
    const seed = C.mnemonicToSeed(mnemonic, passphrase);       // == pbkdf2(pw,salt,2048,64)
    lines.push(`${pw.length} ${hex(pw)} ${salt.length} ${hex(salt)} ${hex(seed)}`);
  };
  // Official BIP39 Trezor vector
  push('abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about', 'TREZOR');
  // Random mnemonics across all word counts x assorted passphrases (incl. the
  // 24-word >128-byte password pre-hash boundary, and empty/long passphrases).
  const phrases = ['', 'TREZOR', 'swordfish7', 'a', 'x'.repeat(80), 'pässwördé́', '   spaces   '];
  for (const W of [12, 15, 18, 21, 24]) {
    for (let i = 0; i < 8; i++) {
      const mn = entropyToMnemonic(rbytes(ENT_BYTES[W]));
      push(mn, phrases[rint(phrases.length)]);
    }
  }
  // Force several 24-word cases (password length > 128 -> HMAC key pre-hash).
  for (let i = 0; i < 6; i++) {
    const mn = entropyToMnemonic(rbytes(32));
    if (C.utf8(C.nfkd(mn)).length <= 128) { i--; continue; }
    push(mn, i % 2 ? 'longsalt'.repeat(20) : '');
  }
  fs.writeFileSync(path.join(OUT, 'vec_pbkdf2.txt'), lines.join('\n') + '\n');
  const over128 = lines.filter(l => parseInt(l.split(' ')[0], 10) > 128).length;
  console.log(`pbkdf2:   ${lines.length} vectors (${over128} with >128-byte password / pre-hash)`);
}

// ---------------------------------------------------------------------------
// Stage 3: BIP39 checksum == sha256_1blk_h0 (top cs bits of SHA-256(entropy))
//   The kernel computes SHA-256(entropy) and we compare the top csBits of its
//   first byte. Emit the full first byte so the harness masks csBits itself.
//   format: <entlen> <ent_hex> <csbits> <sha256_firstbyte_hex(2)>
// ---------------------------------------------------------------------------
{
  const lines = [];
  for (const W of [12, 15, 18, 21, 24]) {
    const ENT = ENT_BYTES[W], CS = W / 3;
    for (let i = 0; i < 16; i++) {
      const ent = rbytes(ENT);
      const fb = crypto.createHash('sha256').update(ent).digest()[0];  // standard SHA-256
      lines.push(`${ENT} ${hex(ent)} ${CS} ${fb.toString(16).padStart(2, '0')}`);
    }
  }
  fs.writeFileSync(path.join(OUT, 'vec_checksum.txt'), lines.join('\n') + '\n');
  console.log(`checksum: ${lines.length} vectors`);
}

// ---------------------------------------------------------------------------
// Stage 4: BIP32 master + hardened EC-FREE CKDpriv -> node {chaincode, priv}
//   kernel: node = seedToMaster(seed); apply `levels` ckdHardened(idx_j) steps.
//   idx_j are FULL hardened indices (0x80000000|i). accountNode(seed,purpose)
//   == deriveHardenedPath(seed,[purpose,0,0]) all hardened.
//   format: <seedlen> <seed_hex> <levels> <i0> <i1> <i2> <chaincode_hex(64)> <priv_hex(64)>
// ---------------------------------------------------------------------------
{
  const HARD = 0x80000000;
  const lines = [];
  const nodeAt = (seed, idxs) => {
    let n = C.seedToMaster(seed);
    for (const i of idxs) n = C.ckdHardened(n, i >>> 0);
    return n;
  };
  const emit = (seed, idxs) => {
    const n = nodeAt(seed, idxs);
    const i = [idxs[0] || 0, idxs[1] || 0, idxs[2] || 0];
    lines.push(`${seed.length} ${hex(seed)} ${idxs.length} ${i[0]>>>0} ${i[1]>>>0} ${i[2]>>>0} ${hex(n.c)} ${hex(C.ser256(n.k))}`);
  };
  // BIP32 test vector 1 (seed 000102..0f): master and m/0'
  const bseed = C.fromHex('000102030405060708090a0b0c0d0e0f');
  emit(bseed, []);                  // master  -> c=873dff.., k=e8f32e..
  emit(bseed, [0x80000000]);        // m/0'    -> c=47fdac.., k=edb2e1..
  // Random seeds (16..64 bytes, and real 64-byte BIP39 seeds) x accountNode paths.
  const purposes = [44, 49, 84, 86];
  for (let i = 0; i < 12; i++) {
    const seed = rbytes(16 + rint(49));          // 16..64
    emit(seed, []);                               // master vs oracle
    emit(seed, [(rint(4)) + HARD]);               // one hardened level
    const p = purposes[rint(purposes.length)];
    emit(seed, [p + HARD, 0 + HARD, 0 + HARD]);   // accountNode(seed,p)
  }
  // Real BIP39 seeds too (the actual xpub-path input): mnemonic -> seed -> account.
  for (let i = 0; i < 8; i++) {
    const seed = C.mnemonicToSeed(entropyToMnemonic(rbytes(ENT_BYTES[[12,24][i&1]])), i & 1 ? 'pw' : '');
    for (const p of purposes) emit(seed, [p + HARD, 0 + HARD, 0 + HARD]);
  }
  fs.writeFileSync(path.join(OUT, 'vec_account.txt'), lines.join('\n') + '\n');
  console.log(`account:  ${lines.length} vectors`);
}

console.log('vectors written to', OUT);
