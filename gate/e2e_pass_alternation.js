#!/usr/bin/env node
/* e2e_pass_alternation.js -- passphrase patterns with POSIX classes, inline
 * alternation (a|b|c), and external [:dict:] files (via -D), on top of the
 * mixed-radix string-set model. Fixed mnemonic; plant an address for a known
 * passphrase; crack it with the matching pattern and assert recovery. Also a
 * negative and the unknown-dict error.
 *
 * Usage: node gate/e2e_pass_alternation.js
 */
'use strict';
const fs=require('fs'), os=require('os'), path=require('path'), crypto=require('crypto');
const { execFileSync }=require('child_process');
const R=process.env.RESEED39_DIR||'/root/bip39rxcrack';
const C=require(path.join(R,'estimator','bip39crypto.js'));
const CLI=path.join(__dirname,'..','bip39rxcrack');
const M='legal winner thank year wave sausage worth useful legal winner thank yellow';
const addr=(pp)=>C.encodeAddress(C.addressTarget(C.mnemonicToSeed(M,pp),49,0,0,0,0),'bc');

const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ppdict_'));
fs.writeFileSync(path.join(dir,'words.dict'),'apple\nbanana\ncorrecthorse\nzebra\n');

let fail=0; const check=(c,m)=>{ console.log((c?'  ok   ':'  FAIL ')+m); if(!c)fail++; };
function crack(pattern,extra){ let o='',code=0;
  try{ o=execFileSync(CLI,['--mnemonic',M,'--passphrase',pattern,'--device','0',...extra],{encoding:'utf8',stdio:['ignore','pipe','inherit']}); }
  catch(e){ o=e.stdout?e.stdout.toString():''; code=e.status||1; }
  return {out:o,code}; }
const pp=(o)=>(o.match(/passphrase\s*:\s*(\S+)/)||[])[1];

console.log('== passphrase alternation / POSIX / dictionaries ==');
let r=crack('[:digit:]{4}',['--address',addr('1234'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='1234', '[:digit:]{4} -> 1234 (POSIX class)');
r=crack('[:alpha:]{4}',['--address',addr('cafe'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='cafe', '[:alpha:]{4} -> cafe (POSIX class)');
r=crack('(spring|summer|winter)2023',['--address',addr('summer2023'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='summer2023', '(spring|summer|winter)2023 -> summer2023 (alternation)');
r=crack('[:words:]',['-D',dir,'--address',addr('correcthorse'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='correcthorse', '[:words:] + -D -> correcthorse (dictionary)');
r=crack('prefix[:words:][0-9]{1}',['-D',dir,'--address',addr('prefixbanana7'),'--purpose','49']);
check(r.code===0 && pp(r.out)==='prefixbanana7', 'prefix[:words:][0-9]{1} -> prefixbanana7 (mixed)');
// negative: alternation lacking the right word
r=crack('(spring|winter)2023',['--address',addr('summer2023'),'--purpose','49']);
check(r.code===1 && /NOT FOUND/.test(r.out), 'alternation without "summer" -> NOT FOUND');
// unknown dict -> parse error (exit 2)
r=crack('[:nope:]',['--address',addr('cafe'),'--purpose','49']);
check(r.code===2, 'unknown [:dict:] -> exit 2 (needs -D dir with name.dict)');

try{ fs.rmSync(dir,{recursive:true,force:true}); }catch(e){}
console.log(`\n==== e2e_pass_alternation: ${fail?'FAILED ('+fail+')':'PASSED'} ====`);
process.exit(fail?1:0);
