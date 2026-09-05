#!/usr/bin/env node
/* nth_gate.js -- gate the [:Nth:] last-word CONSTRUCTION against BRUTE+SIEVE.
 * For a template with the last position unknown (the checksum word) and one
 * other unknown middle position, over a sample of middle values:
 *   brute:      sweep all 2048 last-words, keep the checksum-VALID ones (oracle)
 *   construct:  enumerate the 2^(11-CS) free entropy values of the last word,
 *               SHA-256 the entropy, append the CS checksum bits -> valid last word
 * Assert the two SETS of last-word indices are byte-identical (same members).
 * This is the correctness proof rxe-b5 requires before trusting any A/B number.
 */
'use strict';
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const RESEED39=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const B=require(path.join(RESEED39,'estimator','bip39.js'));
const WORDS=fs.readFileSync(path.join(RESEED39,'data','english.txt'),'utf8').split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
const IDX=new Map(WORDS.map((w,i)=>[w,i]));
const validator=B.makeValidator(WORDS);

// CASE templates (known words + [:bip39-en:] unknowns): last position is unknown.
const CASES={
 12:{tpl:"audit vapor excuse note pledge X bundle start regular burden reveal L", mid:5},
 24:{tpl:"audit vapor excuse note pledge rough bundle start regular burden reveal X vehicle bird van hero harvest service toss enter equip truly march L", mid:11},
};
// pack the first ENT bits of a W-word index vector into entropy bytes
function packEntropy(g,W,ENT){ const ent=Buffer.alloc(ENT/8);
  for(let i=0;i<W;i++){ const idx=g[i]; for(let b=0;b<11;b++){ const pos=i*11+b; if(pos<ENT && ((idx>>(10-b))&1)) ent[pos>>3]|=(0x80>>(pos&7)); } } return ent; }

let bad=0, total=0;
for(const W of [12,24]){
  const CS=W/3, freebits=11-CS, ENT=W*11-CS;
  const c=CASES[W]; const toks=c.tpl.split(' ');
  // sample of middle values
  const mids=[0,1,42,1000,2047, 500, c.mid*137%2048];
  for(const m of mids){
    total++;
    // build the known index vector with middle=m, last=placeholder
    const g=toks.map((t,i)=> t==='X'?m : (t==='L'?0 : IDX.get(t)));
    const Wn=g.length;
    // BRUTE+SIEVE: all 2048 last words, keep valid
    const brute=new Set();
    for(let last=0;last<2048;last++){ g[Wn-1]=last;
      const mn=g.map(i=>WORDS[i]).join(' ');
      if(validator.isValid(mn)) brute.add(last);
    }
    // CONSTRUCT: free bits -> entropy -> sha256 -> checksum -> last word
    const construct=new Set();
    for(let f=0; f<(1<<freebits); f++){
      g[Wn-1]=(f<<CS);
      const ent=packEntropy(g,Wn,ENT);
      const cs=crypto.createHash('sha256').update(ent).digest()[0] >> (8-CS);
      construct.add((f<<CS)|cs);
    }
    const a=[...brute].sort((x,y)=>x-y), b=[...construct].sort((x,y)=>x-y);
    const eq = a.length===b.length && a.every((v,i)=>v===b[i]);
    if(!eq){ bad++; if(bad<=3) console.log(`  FAIL W=${W} mid=${m}: brute ${a.length} vs construct ${b.length}`); }
    else console.log(`  ok   W=${W} mid=${m}: ${a.length} valid last-words, construct==brute (2^${freebits}=${1<<freebits})`);
  }
}
console.log(`\n  ==== [:Nth:] construction gate ${bad?'FAILED':'PASSED'} (${total-bad}/${total} middle samples; construct==brute+sieve) ====`);
process.exit(bad?1:0);
