#!/usr/bin/env node
/* e2e_missing.js -- missing-word ([:Nth:]) + address crack proof (mode_missing and
 * its work-queue worker). Plant a valid 12-word mnemonic, blank a middle word AND
 * the last word (so [:Nth:] checksum construction applies), derive the p2wpkh
 * address, then crack `--template --address` and assert the recovered mnemonic ==
 * planted -- single-GPU (mode_missing) and via the work-queue under two orderings.
 *
 * Usage: node gate/e2e_missing.js
 */
'use strict';
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const B=require(path.join(R,'estimator','bip39.js'));
const WL=fs.readFileSync(path.join(R,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const V=B.makeValidator(WL);
const CLI=path.join(__dirname,'..','bip39rxcrack');

function e2m(ent){const ENT=ent.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(ent).digest();
  const bits=new Uint8Array(T);for(let i=0;i<ENT;i++)bits[i]=(ent[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)bits[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let b=0;b<11;b++)x=(x<<1)|bits[k*11+b];w.push(WL[x]);}return w.join(' ');}
const mn=e2m(Buffer.from('0c1e24e5459335f0e5eee0b0e0e0aa11','hex'));
if(!V.isValid(mn)){ console.error('bad plant'); process.exit(2); }
const toks=mn.split(' ');
const addr=C.encodeAddress(C.addressTarget(C.mnemonicToSeed(mn,''),84,0,0,0,0),'bc');
const tmpl=toks.map((w,i)=>(i===1||i===11)?'[:bip39:]':w).join(' ');

console.log('== missing-word [:Nth:] + address crack proof ==');
console.log('planted :',mn);
console.log('template:',tmpl);
console.log('address :',addr,'(m/84\'/0\'/0\'/0/0)');

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
const runs=[['single','--device','0'],['wq first','--order','first','--shards','64'],['wq center','--order','center','--shards','64']];
for(const [label,...flags] of runs){
  let out='',code=0;
  try{ out=execFileSync(CLI,['--template',tmpl,'--address',addr,...flags],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }
  const gotMn=(out.match(/mnemonic\s*:\s*(.+)/)||[])[1]?.trim();
  check(code===0 && gotMn===mn, `${label}: FOUND, mnemonic == planted`);
}
console.log(`\n==== e2e_missing: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
