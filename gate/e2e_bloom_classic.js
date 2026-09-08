#!/usr/bin/env node
/* e2e_bloom_classic.js -- the CLASSIC (non-blocked, modulo arbitrary-size) filter1 +
 * --bloom-append, end to end on the GPU:
 *   1. plant a mnemonic; build a --classic1 filter (NON-pow2 size, exercises modulo)
 *      from DECOYS ONLY (winner absent) -> crack the window -> NOT FOUND (classic probe
 *      correctly rejects a non-member).
 *   2. --bloom-append the winner's address in place -> crack the same window -> FOUND at
 *      the librxe rank with the planted mnemonic (append is additive; the device classic
 *      probe matches the host build).
 * Proves: classic build, classic GPU probe (negative + positive), and incremental append.
 * Usage: node gate/e2e_bloom_classic.js
 */
'use strict';
const fs=require('fs'), path=require('path'), os=require('os'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const B=require(path.join(R,'estimator','bip39.js'));
const WL=fs.readFileSync(path.join(R,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const V=B.makeValidator(WL);
const CLI=path.join(__dirname,'..','bip39rxcrack');

let st=0x0ddba11>>>0; function rnd(){st^=st<<13;st^=st>>>17;st^=st<<5;st>>>=0;return st/4294967296;}
function rb(n){const a=Buffer.alloc(n);for(let i=0;i<n;i++)a[i]=Math.floor(rnd()*256)&255;return a;}
function e2m(e){const ENT=e.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(e).digest();
  const b=new Uint8Array(T);for(let i=0;i<ENT;i++)b[i]=(e[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)b[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let j=0;j<11;j++)x=(x<<1)|b[k*11+j];w.push(WL[x]);}return w.join(' ');}
let M;for(;;){M=e2m(rb(16));if(new Set(M.split(' ')).size===12)break;}
const seed=C.mnemonicToSeed(M,''); const wordsArg=[...M.split(' ')].sort().join(' ');
if(!V.isValid(M)){ console.error('plant invalid'); process.exit(2); }
const winner=C.encodeAddress(C.addressTarget(seed,84,0,0,0,0),'bc');   // p2wpkh
const decoys=[]; for(let i=0;i<200;i++) decoys.push(C.encodeAddress({type:'p2wpkh',program:crypto.randomBytes(20)},'bc'));
const blf=path.join(os.tmpdir(),`cle_${process.pid}.blf`);
const J=BigInt(execFileSync(CLI,['--words',wordsArg,'--rank',M],{encoding:'utf8'}).trim());
const start=(J>1000n?J-1000n:0n).toString();

console.log('== classic filter1 (modulo) + --bloom-append proof ==');
console.log('planted:',M,'\nwinner :',winner,' J =',J.toString());
let fail=0; const ck=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };

// 1) build a NON-pow2 classic filter from DECOYS ONLY (winner absent)
execFileSync(CLI,['--bloom-build','-',blf,'--bloom-gib','0.1,0.05','--classic1','--bloom-n','1000'],
  {input:decoys.join('\n')+'\n',stdio:['pipe','ignore','inherit']});
const crack=()=>{ let out='',code=0;
  try{ out=execFileSync(CLI,['--words',wordsArg,'--bloom',blf,'--start',start,'--count','3000','--device','0'],
    {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }
  const gi=(out.match(/index\s*(?:\(canonical\))?\s*:\s*(\d+)/)||[])[1];
  return {code,gi,out}; };

let r=crack();
ck(r.code===1 && /NOT FOUND/.test(r.out), 'winner ABSENT from classic filter -> NOT FOUND (classic probe rejects non-member)');

// 2) append the winner, then it must be FOUND at J
execFileSync(CLI,['--bloom-append','-',blf],{input:winner+'\n',stdio:['pipe','ignore','inherit']});
r=crack();
ck(r.code===0, 'after --bloom-append: crack exit 0 (FOUND)');
ck(r.gi===J.toString(), `index ${r.gi} == rank ${J} (device classic probe == host append)`);

try{ fs.unlinkSync(blf); }catch(e){}
console.log(`\n==== e2e_bloom_classic: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
