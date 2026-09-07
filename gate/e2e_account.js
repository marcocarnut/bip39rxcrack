#!/usr/bin/env node
/* e2e_account.js -- --account (scan m/purpose'/0'/account'/change/index).
 * Plant a distinct-word mnemonic; derive its address at ACCOUNT 1 (not 0); put
 * that address (+ decoys) in an --addresses set. Assert:
 *   - default (account 0 only)  -> NOT FOUND (the funds are in account 1)
 *   - --account 3 (scan 0,1,2)  -> FOUND at m/49'/0'/1'/0/0 with the planted mnemonic
 * Windowed around the canonical rank for speed.
 *
 * Usage: node gate/e2e_account.js
 */
'use strict';
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const WL=fs.readFileSync(path.join(R,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const CLI=path.join(__dirname,'..','bip39rxcrack');

let st=0x0acc0117>>>0; function rb(n){const a=Buffer.alloc(n);for(let i=0;i<n;i++){st^=st<<13;st^=st>>>17;st^=st<<5;st>>>=0;a[i]=st&255;}return a;}
function e2m(e){const ENT=e.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(e).digest();
  const b=new Uint8Array(T);for(let i=0;i<ENT;i++)b[i]=(e[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)b[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let j=0;j<11;j++)x=(x<<1)|b[k*11+j];w.push(WL[x]);}return w.join(' ');}
let M; for(;;){M=e2m(rb(16)); if(new Set(M.split(' ')).size===12) break;}
const seed=C.mnemonicToSeed(M,'');
const a1=C.encodeAddress(C.addressTarget(seed,49,1,0,0,0),'bc');            // ACCOUNT 1, p2sh
const dec=[]; for(let i=0;i<4;i++) dec.push(C.encodeAddress({type:'p2sh',program:crypto.randomBytes(20)},'bc'));
const set=[dec[0],a1,dec[1],dec[2]].join(',');
const sorted=M.split(' ').slice().sort().join(' ');
const J=BigInt(execFileSync(CLI,['--words',sorted,'--rank',M],{encoding:'utf8'}).trim());
const start=(J>2000n?J-2000n:0n).toString();

console.log('== --account scan (m/49\'/0\'/account\'/..) ==');
console.log('mnemonic:',M);
console.log('account-1 addr:',a1,' among',dec.length-1,'decoys; J =',J.toString());
let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };

// [1] default: account 0 only -> the account-1 address is not derivable -> NOT FOUND
let n='',ncode=0;
try{ n=execFileSync(CLI,['--words',sorted,'--addresses',set,'--purpose','49','--start',start,'--count','4000','--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ n=e.stdout?e.stdout.toString():''; ncode=e.status||1; }
check(ncode===1 && /NOT FOUND/.test(n), 'default (account 0) -> NOT FOUND');

// [2] --account 3 -> FOUND at account 1
let o='',code=0;
try{ o=execFileSync(CLI,['--words',sorted,'--addresses',set,'--purpose','49','--account','3','--start',start,'--count','4000','--device','0'],
  {encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
catch(e){ o=e.stdout?e.stdout.toString():''; code=e.status||1; }
const gm=(o.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
const gp=(o.match(/path\s*:\s*(m\/\d+'[^\s]*)/)||[])[1];
check(code===0, '--account 3 -> FOUND (exit 0)');
check(gm===M, 'mnemonic == planted');
check(gp==="m/49'/0'/1'/0/0", `path is m/49'/0'/1'/0/0 (got ${gp})`);

console.log(`\n==== e2e_account: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
