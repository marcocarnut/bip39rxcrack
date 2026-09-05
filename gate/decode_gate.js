#!/usr/bin/env node
/* decode_gate.js -- gate the CLI's address target decode vs the oracle
 * decodeAddress {type->purpose, program} over base58 + bech32/bech32m targets. */
'use strict';
const path=require('path'); const { execFileSync }=require('child_process');
const C=require(path.join(process.env.RESEED39_DIR||'/root/bip39rxcrack','estimator','bip39crypto.js'));
const CLI=path.join(__dirname,'..','bip39rxcrack');
const TYPE2PUR={p2pkh:44,p2sh:49,p2wpkh:84,p2tr:86};
const addrs=[
  '36TaauZymS8sLXYh9HqYeqMvShJcMajK7j', '144uEF4Yc4TuWFSBttRYDXDFD44BCDi7Hc',
  'bc1qxsgxntswmwg0q7pju5dxrreh6x6py2fpwcra3z',
  'bc1plnvatrnarr60d3fxr5647turr09yr8agrf3d3k8fc4xvpzktj6xscs3zya',
];
let bad=0;
for(const a of addrs){
  const d=C.decodeAddress(a); const wantPur=TYPE2PUR[d.type], wantProg=Buffer.from(d.program).toString('hex');
  const out=execFileSync(CLI,['--decode',a],{encoding:'utf8'}).trim().split(/\s+/);
  const gotPur=parseInt(out[0],10), gotProg=out[1];
  const ok=gotPur===wantPur && gotProg===wantProg;
  console.log(`  ${ok?'ok  ':'FAIL'} ${d.type.padEnd(6)} ${a}`); if(!ok){ console.log(`       got  ${gotPur} ${gotProg}\n       want ${wantPur} ${wantProg}`); bad++; }
}
console.log(`  ==== decode gate ${bad?'FAILED':'PASSED'} (${addrs.length} targets: base58 p2pkh/p2sh + bech32 p2wpkh/p2tr) ====`);
process.exit(bad?1:0);
