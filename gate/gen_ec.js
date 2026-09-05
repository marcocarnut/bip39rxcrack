#!/usr/bin/env node
/* gen_ec.js -- secp256k1/address gate vectors from the reseed39 oracle.
 * For each 32-byte scalar: compressed pub, hash160, p2pkh(BIP44) program,
 * p2sh-p2wpkh(BIP49) program -- all from the oracle (privToPub/hash160/pubToTarget).
 * Includes edge scalars (1, 2, n-1, n-2) + many random, per the gate-rigor ask.
 * Format per line: <sk_hex(64)> <pub_hex(66)> <h160_hex(40)> <p2pkh_hex(40)> <p2sh_hex(40)>
 */
'use strict';
const fs = require('fs'); const path = require('path'); const crypto = require('crypto');
const RESEED39 = process.env.RESEED39_DIR || '/root/bip39rxcrack';
const C = require(path.join(RESEED39, 'estimator', 'bip39crypto.js'));
const OUT = path.join(__dirname, '..', 'vectors', 'vec_ec.txt');
const hex = b => Buffer.from(b).toString('hex');
const N = C.SECP_N;

// deterministic PRNG
let s=0xC0FFEE ^ 0;
function rb(){ s^=s<<13; s^=s>>>17; s^=s<<5; s>>>=0; return s&0xff; }
function rand32big(){ let x=0n; for(let i=0;i<32;i++) x=(x<<8n)|BigInt(rb()); return (x % (N-1n))+1n; } // [1,n-1]

function line(k){
  const sk = C.ser256(k);
  const pub = C.privToPub(k);                         // 33-byte compressed
  const h = C.hash160(pub);                           // 20
  const p44 = C.pubToTarget(pub, 44).program;         // p2pkh, 20
  const p49 = C.pubToTarget(pub, 49).program;         // p2sh-p2wpkh, 20
  return `${hex(sk)} ${hex(pub)} ${hex(h)} ${hex(p44)} ${hex(p49)}`;
}

const lines = [];
for (const k of [1n, 2n, 3n, N-1n, N-2n, 0x1234567890abcdefn]) lines.push(line(k));
const NR = parseInt(process.argv[2] || '4000', 10);
for (let i=0;i<NR;i++) lines.push(line(rand32big()));
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, lines.join('\n') + '\n');
console.log(`ec: ${lines.length} vectors -> ${OUT}`);
