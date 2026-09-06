#!/usr/bin/env node
/* e2e_bloom_file.js -- prebuilt bloom FILE round-trip (--bloom-build + --bloom).
 * Plant a mnemonic; write an address list = its p2wpkh address + mixed-type decoys;
 * `--bloom-build` it to a .blf; crack `--words --bloom file.blf` and assert FOUND at
 * the librxe rank with the planted mnemonic and the matched purpose's path. Windowed.
 *
 * Usage: node gate/e2e_bloom_file.js
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
const seed=C.mnemonicToSeed(M,'');
const wordsArg=[...M.split(' ')].sort().join(' ');
if(!V.isValid(M)){ console.error('plant invalid'); process.exit(2); }
const winner=C.encodeAddress(C.addressTarget(seed,84,0,0,0,0),'bc');   // p2wpkh
const L=[winner];
for(let i=0;i<40;i++) L.push(C.encodeAddress({type:'p2pkh', program:crypto.randomBytes(20)},'bc'));
for(let i=0;i<40;i++) L.push(C.encodeAddress({type:'p2wpkh',program:crypto.randomBytes(20)},'bc'));
for(let i=0;i<40;i++) L.push(C.encodeAddress({type:'p2tr',  program:crypto.randomBytes(32)},'bc'));
const listf=path.join(os.tmpdir(),`bf_addrs_${process.pid}.txt`);
const blf=path.join(os.tmpdir(),`bf_${process.pid}.blf`);
fs.writeFileSync(listf, L.join('\n')+'\n');
const J=BigInt(execFileSync(CLI,['--words',wordsArg,'--rank',M],{encoding:'utf8'}).trim());
const start=(J>2000n?J-2000n:0n).toString();

console.log('== prebuilt bloom file (--bloom-build + --bloom) proof ==');
console.log('planted :',M);
console.log('winner  :',winner,'among',L.length-1,'mixed decoys; J =',J.toString());

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
try{ execFileSync(CLI,['--bloom-build',listf,blf],{stdio:['ignore','ignore','inherit']}); }
catch(e){ check(false,'--bloom-build failed'); }
check(fs.existsSync(blf), 'built .blf exists');

let out='',code=0;
try{ out=execFileSync(CLI,['--words',wordsArg,'--bloom',blf,'--start',start,'--count','4000','--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }
const gi=(out.match(/index\s*(?:\(canonical\))?\s*:\s*(\d+)/)||[])[1];
const gm=(out.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
const gp=(out.match(/path\s*:\s*(m\/\d+'[^\s]*)/)||[])[1];
check(code===0, 'crack exit 0 (FOUND)');
check(gi===J.toString(), `index ${gi} == rank ${J}`);
check(gm===M, 'mnemonic == planted');
check(!!gp && gp.startsWith("m/84'"), `path is m/84'/... (matched p2wpkh, got ${gp})`);

// negative: window without the winner -> NOT FOUND
let nout='',ncode=0;
try{ nout=execFileSync(CLI,['--words',wordsArg,'--bloom',blf,'--start','100000000','--count','4000','--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ nout=e.stdout?e.stdout.toString():''; ncode=e.status||1; }
check(ncode===1 && /NOT FOUND/.test(nout), 'no-winner window -> NOT FOUND');

try{ fs.unlinkSync(listf); fs.unlinkSync(blf); }catch(e){}
console.log(`\n==== e2e_bloom_file: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
