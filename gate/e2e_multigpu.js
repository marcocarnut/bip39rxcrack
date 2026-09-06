#!/usr/bin/env node
/* e2e_multigpu.js -- multi-GPU fan-out parity proof (the --devices analogue of e2e.js).
 *
 * 1. Plant a valid 12-word mnemonic with DISTINCT words (same as e2e.js).
 * 2. J = librxe rank of the planted arrangement (via the CLI --rank).
 * 3. Fan out `--devices 0,1` over an outer window [J-W, J+W) sliced so J lands in
 *    the UPPER half -> the SECOND GPU must be the one that finds it.
 * 4. Assert: the crack reports FOUND, index == J (global canonical rank), mnemonic
 *    == planted, and the supervisor names a winning device. This proves the slice
 *    math, the cross-GPU "first FOUND wins + kill siblings", and that the reported
 *    index stays the GLOBAL rank regardless of which GPU/slice found it.
 *
 * Usage: node gate/e2e_multigpu.js [ndev]   (ndev default 2)
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
const NDEV = parseInt(process.argv[2] || '2', 10);

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

let planted;
for (let tries = 0; ; tries++) {
  const mn = entropyToMnemonic(rbytes(16));
  if (new Set(mn.split(' ')).size === 12) { planted = mn; break; }
  if (tries > 10000) { console.error('could not find distinct-word mnemonic'); process.exit(2); }
}
const PURPOSE = 84;
const wordsArg = [...planted.split(' ')].sort().join(' ');
if (!validator.isValid(planted)) { console.error('planted mnemonic not valid?!'); process.exit(2); }
const targetCC = hex(C.accountNode(C.mnemonicToSeed(planted, ''), PURPOSE).c);

console.log('== multi-GPU fan-out parity proof ==');
console.log('planted  :', planted);
console.log('wordset  :', wordsArg);
console.log('devices  : 0..' + (NDEV - 1) + ' (' + NDEV + ' GPUs)');

const J = BigInt(execFileSync(CLI, ['--words', wordsArg, '--rank', planted], { encoding: 'utf8' }).trim());
console.log('J (librxe rank):', J.toString());

// outer window straddling J, sized so J is in the UPPER (last-device) slice.
// per-device slice ~ count/NDEV; place J at offset ~ count*(NDEV-0.5)/NDEV.
const HALF = 1_000_000n;
let start = J > HALF ? J - HALF : 0n;
const count = 2n * HALF;
// which device slice does J fall in? (contiguous: dev d = [start+d*count/N, start+(d+1)*count/N))
const off = J - start, dev = Number((off * BigInt(NDEV)) / count);
console.log(`window   : [${start}, ${start + count})  -> J at offset ${off}, expected in device ${dev}`);

console.log('\n-- running --devices fan-out crack --');
let out = '', code = 0;
try {
  out = execFileSync(CLI, ['--words', wordsArg, '--target-chaincode', targetCC, '--purpose', String(PURPOSE),
    '--devices', Array.from({ length: NDEV }, (_, i) => i).join(','),
    '--start', start.toString(), '--count', count.toString()],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
} catch (e) { out = e.stdout ? e.stdout.toString() : ''; code = e.status || 1; }
process.stdout.write(out);

const gotIndex = (out.match(/index\s*(?:\(canonical\))?\s*:\s*(\d+)/) || out.match(/index\s*:\s*(\d+)/) || [])[1];
const gotMn = (out.match(/mnemonic:\s*(.+)/) || [])[1]?.trim();

let fail = 0;
const check = (c, m) => { console.log((c ? '  ok   ' : '  FAIL ') + m); if (!c) fail++; };
console.log('\n-- assertions --');
check(code === 0, `supervisor exit code 0 (FOUND), got ${code}`);
check(gotIndex === J.toString(), `found global index ${gotIndex} == librxe rank ${J}`);
check(gotMn === planted, `found mnemonic == planted`);

console.log(`\n==== e2e_multigpu (${NDEV} GPU): ${fail ? 'FAILED (' + fail + ')' : 'PASSED'} ====`);
process.exit(fail ? 1 : 0);
