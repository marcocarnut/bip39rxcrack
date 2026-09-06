#!/usr/bin/env node
/* e2e_xpubs.js -- multi-xpub via a chaincode bloom (EC-free). Plant a mnemonic;
 * use its m/44'/0'/0' account xpub as the winner among a decoy xpub; crack
 * `--words --xpubs`. Assert FOUND at the librxe rank, mnemonic == planted, the
 * matched xpub is the winner, and the reported path is m/44'/... (the purpose the
 * account chaincode matched under -- derived from the default {44,49,84,86} set).
 * Only the chaincode matters to the cracker, so test xpubs carry a real chaincode
 * and a filler pubkey. Windowed for speed.
 *
 * Usage: node gate/e2e_xpubs.js
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

let st=0x1234abcd>>>0; function rnd(){st^=st<<13;st^=st>>>17;st^=st<<5;st>>>=0;return st/4294967296;}
function rb(n){const a=Buffer.alloc(n);for(let i=0;i<n;i++)a[i]=Math.floor(rnd()*256)&255;return a;}
function e2m(e){const ENT=e.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(e).digest();
  const b=new Uint8Array(T);for(let i=0;i<ENT;i++)b[i]=(e[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)b[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let j=0;j<11;j++)x=(x<<1)|b[k*11+j];w.push(WL[x]);}return w.join(' ');}
function mkxpub(cc){ const ver=Uint8Array.of(0x04,0x88,0xB2,0x1E),dep=Uint8Array.of(3),fp=Uint8Array.of(0,0,0,0),
    cn=Uint8Array.of(0x80,0,0,0),key=Buffer.concat([Uint8Array.of(2),crypto.randomBytes(32)]);
  return C.b58checkEncode(Buffer.concat([ver,dep,fp,cn,Buffer.from(cc),key])); }

let M;for(;;){M=e2m(rb(16));if(new Set(M.split(' ')).size===12)break;}
const seed=C.mnemonicToSeed(M,'');
const wordsArg=[...M.split(' ')].sort().join(' ');
if(!V.isValid(M)){ console.error('plant invalid'); process.exit(2); }
const winner=mkxpub(C.accountNode(seed,44).c);      // m/44'/0'/0' account
const decoy=mkxpub(crypto.randomBytes(32));
const J=BigInt(execFileSync(CLI,['--words',wordsArg,'--rank',M],{encoding:'utf8'}).trim());
const start=(J>2000n?J-2000n:0n).toString();

console.log('== multi-xpub (chaincode bloom, EC-free) proof ==');
console.log('planted :',M);
console.log('winner  :',winner.slice(0,20)+'… (m/44 account) among 1 decoy');
console.log('J       :',J.toString());

let out='',code=0;
try{ out=execFileSync(CLI,['--words',wordsArg,'--xpubs',[decoy,winner].join(','),'--start',start,'--count','4000','--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }

const gi=(out.match(/index\s*:\s*(\d+)/)||[])[1];
const gm=(out.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
const gp=(out.match(/path\s*:\s*(m\/\d+'[^\s]*)/)||[])[1];
const gx=(out.match(/xpub\s*:\s*(\S+)/)||[])[1];
let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
check(code===0, `exit 0 (FOUND)`);
check(gi===J.toString(), `index ${gi} == rank ${J}`);
check(gm===M, `mnemonic == planted`);
check(!!gp && gp.startsWith("m/44'"), `path is m/44'/... (matched purpose, got ${gp})`);
check(gx===winner, `matched xpub == the winner (not the decoy)`);
console.log(`\n==== e2e_xpubs: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
