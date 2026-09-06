#!/usr/bin/env node
/* e2e_addr.js -- address-target crack proof (covers mode_crack_addr, both the
 * compacted and fused paths). Complements e2e.js (which covers the chaincode mode).
 *
 * 1. Plant a valid 12-word distinct-word mnemonic.
 * 2. addr = its m/84'/0'/0'/0/0 p2wpkh address (via the oracle).
 * 3. J = librxe rank of the planted arrangement (CLI --rank).
 * 4. Crack `--words <sorted> --address addr` on one GPU; assert FOUND, index==J,
 *    mnemonic==planted. Run under both --compact (default) and --no-compact.
 *
 * Usage: node gate/e2e_addr.js
 */
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const { execFileSync } = require('child_process');
const RESEED39 = process.env.RESEED39_DIR || '/root/bip39rxcrack';
const C = require(path.join(RESEED39, 'estimator', 'bip39crypto.js'));
const B = require(path.join(RESEED39, 'estimator', 'bip39.js'));
const WORDS = fs.readFileSync(path.join(RESEED39, 'data', 'english.txt'), 'utf8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
const validator = B.makeValidator(WORDS);
const CLI = path.join(__dirname, '..', 'bip39rxcrack');

let st = 0x2468ace0 >>> 0;
function rnd(){ st ^= st<<13; st ^= st>>>17; st ^= st<<5; st>>>=0; return st/4294967296; }
function rbytes(n){ const a=Buffer.alloc(n); for(let i=0;i<n;i++) a[i]=Math.floor(rnd()*256)&255; return a; }
function entropyToMnemonic(ent){
  const ENT=ent.length*8, CS=ENT/32, total=ENT+CS;
  const dig=crypto.createHash('sha256').update(ent).digest();
  const bits=new Uint8Array(total);
  for(let i=0;i<ENT;i++) bits[i]=(ent[i>>3]>>(7-(i&7)))&1;
  for(let i=0;i<CS;i++) bits[ENT+i]=(dig[i>>3]>>(7-(i&7)))&1;
  const w=[]; for(let k=0;k<total/11;k++){ let idx=0; for(let b=0;b<11;b++) idx=(idx<<1)|bits[k*11+b]; w.push(WORDS[idx]); }
  return w.join(' ');
}

let planted;
for(let t=0;;t++){ const mn=entropyToMnemonic(rbytes(16)); if(new Set(mn.split(' ')).size===12){ planted=mn; break; } if(t>10000){console.error('no distinct mnemonic');process.exit(2);} }
const wordsArg=[...planted.split(' ')].sort().join(' ');
if(!validator.isValid(planted)){ console.error('planted invalid'); process.exit(2); }
const seed=C.mnemonicToSeed(planted,'');
const addr=C.encodeAddress(C.addressTarget(seed,84,0,0,0,0),'bc');
const J=execFileSync(CLI,['--words',wordsArg,'--rank',planted],{encoding:'utf8'}).trim();

console.log('== address-target crack proof ==');
console.log('planted :',planted);
console.log('address :',addr,'(m/84\'/0\'/0\'/0/0)');
console.log('J (rank):',J);

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
for(const mode of [['--compact','compacted'],['--no-compact','fused']]){
  console.log(`\n-- crack (${mode[1]}) --`);
  let out='',code=0;
  try{ out=execFileSync(CLI,['--words',wordsArg,'--address',addr,'--device','0',mode[0]],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }
  const gotIndex=(out.match(/index\s*(?:\(canonical\))?\s*:\s*(\d+)/)||[])[1];
  const gotMn=(out.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
  check(code===0,`${mode[1]}: exit 0 (FOUND)`);
  check(gotIndex===J,`${mode[1]}: index ${gotIndex} == rank ${J}`);
  check(gotMn===planted,`${mode[1]}: mnemonic == planted`);
}
console.log(`\n==== e2e_addr: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
