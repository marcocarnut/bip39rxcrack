#!/usr/bin/env node
/* e2e_bloom.js -- bloom target-SET crack proof (mode_crack_addr bloom path).
 * Plant a winner, hide its address among K-1 random decoys, crack
 * `--words --addresses <winner,decoys...>`, and assert it recovers the winner at
 * its librxe rank -- and that a run with ONLY decoys (winner absent) finds nothing.
 * Windowed around the rank for speed (the bloom+cull logic is what's under test).
 *
 * Usage: node gate/e2e_bloom.js
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

let st=0x51ede15 >>> 0;
function rnd(){ st^=st<<13; st^=st>>>17; st^=st<<5; st>>>=0; return st/4294967296; }
function rbytes(n){ const a=Buffer.alloc(n); for(let i=0;i<n;i++) a[i]=Math.floor(rnd()*256)&255; return a; }
function e2m(ent){ const ENT=ent.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(ent).digest();
  const bits=new Uint8Array(T); for(let i=0;i<ENT;i++)bits[i]=(ent[i>>3]>>(7-(i&7)))&1; for(let i=0;i<CS;i++)bits[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[]; for(let k=0;k<T/11;k++){ let x=0; for(let b=0;b<11;b++) x=(x<<1)|bits[k*11+b]; w.push(WL[x]); } return w.join(' '); }

let planted; for(let t=0;;t++){ const mn=e2m(rbytes(16)); if(new Set(mn.split(' ')).size===12){planted=mn;break;} if(t>10000){console.error('no mnemonic');process.exit(2);} }
const wordsArg=[...planted.split(' ')].sort().join(' ');
if(!V.isValid(planted)){ console.error('planted invalid'); process.exit(2); }
const addr=C.encodeAddress(C.addressTarget(C.mnemonicToSeed(planted,''),84,0,0,0,0),'bc');
const decoys=[]; for(let i=0;i<5;i++) decoys.push(C.encodeAddress({type:'p2wpkh',program:crypto.randomBytes(20)},'bc'));
const J=BigInt(execFileSync(CLI,['--words',wordsArg,'--rank',planted],{encoding:'utf8'}).trim());
const start=(J>2000n?J-2000n:0n).toString(), count='4000';

console.log('== bloom target-set crack proof ==');
console.log('planted :',planted);
console.log('winner  :',addr);
console.log('decoys  :',decoys.length,'random p2wpkh, J =',J.toString());

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
function run(list){ let out='',code=0;
  try{ out=execFileSync(CLI,['--words',wordsArg,'--addresses',list.join(','),'--device','0','--start',start,'--count',count],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; } return {out,code}; }

// 1) winner present among decoys -> FOUND at J
{ const shuffled=[decoys[0],decoys[1],addr,decoys[2],decoys[3],decoys[4]];
  const {out,code}=run(shuffled);
  const gi=(out.match(/index\s*(?:\(canonical\))?\s*:\s*(\d+)/)||[])[1];
  const gm=(out.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
  check(code===0 && gi===J.toString() && gm===planted, `winner among ${decoys.length} decoys: FOUND at J=${J} (index ${gi}, mnemonic ok)`); }

// 2) decoys ONLY (winner absent) -> NOT FOUND (bloom hits, if any, cull away)
{ const {out,code}=run(decoys);
  check(code===1 && /NOT FOUND/.test(out), `decoys only (no winner): NOT FOUND (code ${code})`); }

console.log(`\n==== e2e_bloom: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
