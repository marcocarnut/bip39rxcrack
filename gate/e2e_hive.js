#!/usr/bin/env node
/* e2e_hive.js -- SSH hive over loopback. The same wq1 work-queue, but workers are
 * reached as `ssh <host> <remote-bin> ... --worker`. Plants a winner, windows it,
 * and asserts it's FOUND at the global canonical rank through the SSH transport --
 * once with --hosts localhost/2 (two GPUs on one machine) and once with
 * localhost/1,localhost/1 (two "machines", exercising the machine-level split).
 *
 * SKIPS (exit 0) if passwordless `ssh localhost` isn't available -- this gate needs
 * an sshd + a loopback key, which not every box has.
 *
 * Usage: node gate/e2e_hive.js
 */
'use strict';
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const { execFileSync, execSync } = require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const WL=fs.readFileSync(path.join(R,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const CLI=path.resolve(path.join(__dirname,'..','bip39rxcrack'));

// --- skip unless passwordless ssh localhost works ---
try{ execSync('ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=4 localhost true',{stdio:'ignore'}); }
catch(e){ console.log('== SSH hive ==\n  SKIP: passwordless `ssh localhost` not available on this box'); console.log('\n==== e2e_hive: SKIPPED ===='); process.exit(0); }

let st=0x51a7e>>>0; function rb(n){const a=Buffer.alloc(n);for(let i=0;i<n;i++){st^=st<<13;st^=st>>>17;st^=st<<5;st>>>=0;a[i]=st&255;}return a;}
function e2m(e){const ENT=e.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(e).digest();
  const b=new Uint8Array(T);for(let i=0;i<ENT;i++)b[i]=(e[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)b[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let j=0;j<11;j++)x=(x<<1)|b[k*11+j];w.push(WL[x]);}return w.join(' ');}
let M; for(;;){M=e2m(rb(16)); if(new Set(M.split(' ')).size===12) break;}
const seed=C.mnemonicToSeed(M,'');
const addr=C.encodeAddress(C.addressTarget(seed,84,0,0,0,0),'bc');
const sorted=M.split(' ').slice().sort().join(' ');
const J=BigInt(execFileSync(CLI,['--words',sorted,'--rank',M],{encoding:'utf8'}).trim());
const start=(J>2000000n?J-2000000n:0n).toString(); const count='4000000';   // ~4M window around the rank

console.log('== SSH hive (loopback) ==');
console.log('planted:',M);
console.log('winner addr:',addr,' J =',J.toString());
let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
function hive(hostspec){ let out='',code=0;
  try{ out=execFileSync(CLI,['--words',sorted,'--address',addr,'--hosts',hostspec,'--remote-bin',CLI,
      '--start',start,'--count',count,'--shards','8'],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ out=e.stdout?e.stdout.toString():''; code=e.status||1; }
  const idx=(out.match(/index\s*\(canonical\)\s*:\s*(\d+)/)||[])[1];
  const mn=(out.match(/mnemonic:\s*(.+)/)||[])[1]?.trim();
  return {code,idx,mn}; }

let r=hive('localhost/2');
check(r.code===0 && r.idx===J.toString() && r.mn===M, `localhost/2 (2 workers over SSH): FOUND at ${r.idx} (==J, mnemonic ok)`);
r=hive('localhost/1,localhost/1');
check(r.code===0 && r.idx===J.toString() && r.mn===M, `localhost/1,localhost/1 (2 "machines"): FOUND at ${r.idx} (==J, mnemonic ok)`);

console.log(`\n==== e2e_hive: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
