#!/usr/bin/env node
/* e2e_pass_bloom.js -- regime A (fixed mnemonic + passphrase [0-9]{N}) with a
 * bloom target SET and --exhaustive. Models the PIN/passphrase-confusion case:
 * one mnemonic, several PINs, each deriving a different funded address. Plant a
 * mnemonic; pick two PINs; put BOTH their p2wpkh addresses (+ decoys) in an
 * --addresses set; assert --exhaustive finds BOTH PINs (not just the first), each
 * with the right path. Then a single-target regression + a no-match negative.
 *
 * Usage: node gate/e2e_pass_bloom.js
 */
'use strict';
const path=require('path'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const CLI=path.join(__dirname,'..','bip39rxcrack');

const M='legal winner thank year wave sausage worth useful legal winner thank yellow';
const P1='1701', P2='1705';                                   // the two real PINs
const addr=(pin)=>C.encodeAddress(C.addressTarget(C.mnemonicToSeed(M,pin),84,0,0,0,0),'bc');
const a1=addr(P1), a2=addr(P2);
const decoys=[]; for(let i=0;i<6;i++) decoys.push(C.encodeAddress({type:'p2wpkh',program:crypto.randomBytes(20)},'bc'));
const set=[a1,...decoys,a2].join(',');

console.log('== regime A + bloom SET + --exhaustive (PIN/passphrase confusion) ==');
console.log('mnemonic:',M);
console.log(`PIN ${P1} -> ${a1}`); console.log(`PIN ${P2} -> ${a2}`);
let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };

// 1) exhaustive: BOTH PINs must be reported
let out='';
try{ out=execFileSync(CLI,['--mnemonic',M,'--passphrase','[0-9]{4}','--addresses',set,'--device','0','--exhaustive'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ out=e.stdout?e.stdout.toString():''; }
const found=new Set([...out.matchAll(/passphrase:\s*(\d{4})/g)].map(m=>m[1]));
check(/FOUND\s+2\s+match/.test(out), 'reports FOUND 2 matches');
check(found.has(P1), `found PIN ${P1}`);
check(found.has(P2), `found PIN ${P2}`);
check(/m\/84'\/0'\/0'\/0\/0/.test(out), "path m/84'/0'/0'/0/0 present");

// 2) single-target regression (no set): finds the one PIN, classic output
let s1='';
try{ s1=execFileSync(CLI,['--mnemonic',M,'--passphrase','[0-9]{4}','--address',a1,'--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ s1=e.stdout?e.stdout.toString():''; }
check(/FOUND/.test(s1) && new RegExp('passphrase\\s*:\\s*'+P1).test(s1), `single-target finds ${P1}`);

// 3) negative: a set of only decoys -> NOT FOUND
let neg='',ncode=0;
try{ neg=execFileSync(CLI,['--mnemonic',M,'--passphrase','[0-9]{4}','--addresses',decoys.join(','),'--device','0','--exhaustive'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ neg=e.stdout?e.stdout.toString():''; ncode=e.status||1; }
check(ncode===1 && /NOT FOUND/.test(neg), 'decoys-only -> NOT FOUND');

console.log(`\n==== e2e_pass_bloom: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
