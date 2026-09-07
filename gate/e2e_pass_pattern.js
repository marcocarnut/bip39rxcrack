#!/usr/bin/env node
/* e2e_pass_pattern.js -- generic passphrase PATTERNS (charsets/literals/mixed-radix).
 * Fixed mnemonic; plant an address for a known passphrase; crack it with the
 * matching pattern and assert the passphrase is recovered. Covers a lowercase
 * class, a literal+digit mix, digit backward-compat, the bloom SET path, and a
 * wrong-charset negative.
 *
 * Usage: node gate/e2e_pass_pattern.js
 */
'use strict';
const path=require('path'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const CLI=path.join(__dirname,'..','bip39rxcrack');
const M='legal winner thank year wave sausage worth useful legal winner thank yellow';
const addr=(pp)=>C.encodeAddress(C.addressTarget(C.mnemonicToSeed(M,pp),49,0,0,0,0),'bc'); // p2sh

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
function crack(pattern,args){ let o='',code=0;
  try{ o=execFileSync(CLI,['--mnemonic',M,'--passphrase',pattern,'--device','0',...args],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ o=e.stdout?e.stdout.toString():''; code=e.status||1; }
  return {out:o,code}; }
function pp(o){ return (o.match(/passphrase\s*:\s*(\S+)/)||[])[1]; }

console.log('== generic passphrase patterns ==');
// [a] lowercase class
let r=crack('[a-z]{4}',['--address',addr('cafe'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='cafe', "[a-z]{4} -> cafe");
// [b] literal + digit class
r=crack('pass[0-9]{2}',['--address',addr('pass07'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='pass07', "pass[0-9]{2} -> pass07");
// [c] digit backward compat
r=crack('[0-9]{4}',['--address',addr('1234'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='1234', "[0-9]{4} -> 1234 (backward compat)");
// [d] mixed alnum
r=crack('[a-z0-9]{3}',['--address',addr('a1z'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='a1z', "[a-z0-9]{3} -> a1z");
// [e] bloom SET path (via --addresses) with a pattern
r=crack('[a-z]{4}',['--addresses',[addr('cafe'),C.encodeAddress({type:'p2sh',program:crypto.randomBytes(20)},'bc')].join(','),'--purpose','49']);
check(r.code===0 && pp(r.out)==='cafe', "[a-z]{4} + --addresses SET -> cafe");
// [f] negative: right length, wrong charset (digits only) -> NOT FOUND
r=crack('[0-9]{4}',['--address',addr('cafe'),'--purpose','49']);
check(r.code===1 && /NOT FOUND/.test(r.out), "[0-9]{4} cannot match 'cafe' -> NOT FOUND");
// [g] {m,n} rejected cleanly (exit 2 = parse error, vs 0 found / 1 not-found; msg on stderr)
r=crack('[0-9]{2,4}',['--address',addr('1234'),'--purpose','49']);
check(r.code===2, "{m,n} -> rejected with exit 2 (not yet supported)");

console.log(`\n==== e2e_pass_pattern: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
