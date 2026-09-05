#!/usr/bin/env node
/* gen_addr.js -- seed->address gate vectors from the oracle: the FULL chain
 * seed -> m/purpose'/0'/0'/change/index -> pubkey -> program, for p2pkh(44),
 * p2sh-p2wpkh(49), p2wpkh(84). Exercises the non-hardened ckdNormal (EC) path.
 * Format: <seed_hex(128)> <purpose> <change> <index> <program_hex(40)>
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const RESEED39=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(RESEED39,'estimator','bip39crypto.js'));
const WORDS=fs.readFileSync(path.join(RESEED39,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const OUT=path.join(__dirname,'..','vectors','vec_addr.txt');
const hex=b=>Buffer.from(b).toString('hex');
let s=0xBEEF^0; function rb(){s^=s<<13;s^=s>>>17;s^=s<<5;s>>>=0;return s&0xff;}
function rbytes(n){const a=Buffer.alloc(n);for(let i=0;i<n;i++)a[i]=rb();return a;}
function entToMn(ent){const ENT=ent.length*8,CS=ENT/32,T=ENT+CS,d=crypto.createHash('sha256').update(ent).digest();
  const bits=new Uint8Array(T);for(let i=0;i<ENT;i++)bits[i]=(ent[i>>3]>>(7-(i&7)))&1;for(let i=0;i<CS;i++)bits[ENT+i]=(d[i>>3]>>(7-(i&7)))&1;
  const w=[];for(let k=0;k<T/11;k++){let x=0;for(let b=0;b<11;b++)x=(x<<1)|bits[k*11+b];w.push(WORDS[x]);}return w.join(' ');}
const lines=[];
const NR=parseInt(process.argv[2]||'300',10);
for(let i=0;i<NR;i++){
  const seed=C.mnemonicToSeed(entToMn(rbytes(i&1?32:16)), i%3?'':'pass'+i);
  const purpose=[44,49,84,86][i%4];
  const change=i%2, index=i%5;
  const prog=C.pubToTarget(C.privToPub(C.addressNode(seed,purpose,0,0,change,index).k),purpose).program;
  lines.push(`${hex(seed)} ${purpose} ${change} ${index} ${hex(prog)}`);
}
fs.mkdirSync(path.dirname(OUT),{recursive:true});
fs.writeFileSync(OUT,lines.join('\n')+'\n');
console.log(`addr: ${lines.length} vectors -> ${OUT}`);
