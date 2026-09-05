#!/usr/bin/env node
/* e2e.js -- Phase-2 end-to-end proof + sieve cross-check, gated by the oracle.
 *
 * 1. Plant a real (checksum-valid) 12-word mnemonic with DISTINCT words.
 * 2. target = accountNode(seed, purpose).c  (what an account xpub carries).
 * 3. expected index = librxe rank of the planted arrangement (via the CLI --rank).
 * 4. Run the GPU self-enumerate crack; assert found {index, mnemonic, path} match.
 * 5. Sieve cross-check: GPU checksum-valid flag == oracle validator over samples.
 *
 * All "expected" values come from the reseed39 oracle + librxe. Usage:
 *   node gate/e2e.js            # random planted seed (fixed PRNG seed)
 */
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const RESEED39 = process.env.RESEED39_DIR || '/root/bip39rxcrack';
const C = require(path.join(RESEED39, 'estimator', 'bip39crypto.js'));
const B = require(path.join(RESEED39, 'estimator', 'bip39.js'));
const WORDS = fs.readFileSync(path.join(RESEED39, 'data', 'english.txt'), 'utf8')
  .split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const validator = B.makeValidator(WORDS);
const CLI = path.join(__dirname, '..', 'bip39rxcrack');
const hex = b => Buffer.from(b).toString('hex');

// deterministic PRNG so the proof is reproducible
let st = 0x12345678 >>> 0;
function rnd() { st ^= st << 13; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; }
function rbytes(n) { const a = Buffer.alloc(n); for (let i = 0; i < n; i++) a[i] = Math.floor(rnd() * 256) & 255; return a; }
function entropyToMnemonic(ent) {
  const ENT = ent.length * 8, CS = ENT / 32, total = ENT + CS;
  const dig = crypto.createHash('sha256').update(ent).digest();
  const bits = new Uint8Array(total);
  for (let i = 0; i < ENT; i++) bits[i] = (ent[i >> 3] >> (7 - (i & 7))) & 1;
  for (let i = 0; i < CS; i++) bits[ENT + i] = (dig[i >> 3] >> (7 - (i & 7))) & 1;
  const w = [];
  for (let k = 0; k < total / 11; k++) { let idx = 0; for (let b = 0; b < 11; b++) idx = (idx << 1) | bits[k * 11 + b]; w.push(WORDS[idx]); }
  return w.join(' ');
}

// 1) plant a valid 12-word mnemonic whose 12 words are all distinct
let planted;
for (let tries = 0; ; tries++) {
  const mn = entropyToMnemonic(rbytes(16));
  const toks = mn.split(' ');
  if (new Set(toks).size === 12) { planted = mn; break; }
  if (tries > 10000) { console.error('could not find distinct-word mnemonic'); process.exit(2); }
}
const PURPOSE = 84;
const ptoks = planted.split(' ');
// base/word-set order handed to the CLI: sorted unique (canonical, order-independent proof)
const wordsArg = [...ptoks].sort().join(' ');
if (!validator.isValid(planted)) { console.error('planted mnemonic not valid?!'); process.exit(2); }
const targetCC = hex(C.accountNode(C.mnemonicToSeed(planted, ''), PURPOSE).c);

console.log('== Phase-2 end-to-end proof ==');
console.log('planted  :', planted);
console.log('wordset  :', wordsArg, '(base order handed to CLI)');
console.log('purpose  : m/' + PURPOSE + "'/0'/0'");
console.log('targetCC :', targetCC);

// 3) expected index from librxe (canonical enumerator), via the CLI --rank
const expectedIndex = execFileSync(CLI, ['--words', wordsArg, '--rank', planted], { encoding: 'utf8' }).trim();
console.log('expected index (librxe rank):', expectedIndex);

// 4) run the GPU self-enumerate crack
console.log('\n-- running GPU self-enumerate crack --');
const out = execFileSync(CLI, ['--words', wordsArg, '--target-chaincode', targetCC, '--purpose', String(PURPOSE)],
  { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
process.stdout.write(out);
const gotIndex = (out.match(/index\s*:\s*(\d+)/) || [])[1];
const gotMn = (out.match(/mnemonic:\s*(.+)/) || [])[1]?.trim();
const gotPurpose = (out.match(/purpose (\d+)/) || [])[1];

let fail = 0;
const check = (c, m) => { console.log((c ? '  ok   ' : '  FAIL ') + m); if (!c) fail++; };
console.log('\n-- assertions --');
check(gotIndex === expectedIndex, `found index ${gotIndex} == librxe rank ${expectedIndex}`);
check(gotMn === planted, `found mnemonic == planted`);
check(gotPurpose === String(PURPOSE), `found purpose ${gotPurpose} == ${PURPOSE}`);

// 5) sieve cross-check: GPU checksum-valid flag vs oracle validator over samples
console.log('\n-- checksum-sieve cross-check (GPU vs oracle) --');
const NS = 1000;
const dv = execFileSync(CLI, ['--words', wordsArg, '--dump-valid', String(NS)], { encoding: 'utf8' });
let sbad = 0, svalid = 0, n = 0;
for (const line of dv.split('\n')) {
  if (!line.trim()) continue;
  const tab = line.indexOf('\t'); if (tab < 0) continue;
  const gv = line.slice(0, tab) === '1', mn = line.slice(tab + 1);
  const ov = validator.isValid(mn);
  if (gv !== ov) { if (sbad < 3) console.log(`    MISMATCH: gpu=${gv} oracle=${ov} :: ${mn}`); sbad++; }
  if (ov) svalid++;
  n++;
}
check(sbad === 0, `GPU checksum-valid == oracle over ${n} samples (${svalid} valid, ${sbad} mismatches)`);

console.log(`\n==== e2e: ${fail ? 'FAILED (' + fail + ')' : 'PASSED'} ====`);
process.exit(fail ? 1 : 0);
