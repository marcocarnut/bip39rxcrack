/* bip39rxcrack -- native CUDA BIP39 seed cracker (Phase 2: xpub, self-enumerate).
 *
 * v1 pattern: "known words, unknown order" -- a permutation of N known distinct
 * BIP39 words (reseed39's buildWordSetPattern -> "(  w0| w1|...){{N!}}"). The GPU
 * self-enumerates each permutation index on-die (decode_perm, byte-identical to
 * librxe), checksum-sieves, PBKDF2s survivors, derives the account node EC-free,
 * and compares its 32-byte chain code to the target (an account xpub's chaincode).
 *
 * librxe (RXE_DIR=../rxe) is the canonical enumerator: we parse the same wordset
 * pattern with it to get the count and to GATE the GPU unrank byte-exact.
 *
 * Modes:
 *   (crack)          --words "w1..wN" --xpub XPUB | --target-chaincode HEX [--purpose ..]
 *   --recon-gate [N] gate GPU unrank+reconstruction vs librxe over N sample indices
 *   --rank "w1..wN"  print the librxe index of one arrangement (canonical order)
 *   --dump-valid N   print "index mnemonic gpuvalid" for N samples (node cross-check)
 *
 * Correctness authority: the reseed39 browser oracle + librxe. Gate first.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctype.h>
#include <gmp.h>
#include <cuda.h>
#include <nvrtc.h>
#include "rxe.h"

#define CU(x)  do{ CUresult r=(x); if(r!=CUDA_SUCCESS){ const char*s=0; cuGetErrorString(r,&s); \
                   fprintf(stderr,"CUDA error %d (%s) at %s:%d\n",r,s?s:"?",__FILE__,__LINE__); exit(2);} }while(0)
#define NVR(x) do{ nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){ \
                   fprintf(stderr,"NVRTC error %d (%s) at %s:%d\n",r,nvrtcGetErrorString(r),__FILE__,__LINE__); exit(2);} }while(0)
#define MN_STRIDE 256

/* ------------------------------- utils --------------------------------- */
static char *slurp(const char *p){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*b=malloc(n+1);
  if(fread(b,1,n,f)!=(size_t)n){fprintf(stderr,"read %s\n",p);exit(2);} b[n]=0; fclose(f); return b; }
static int hexnib(int c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1;}
static int hex2bin(const char*s,uint8_t*o,int max){ int n=0; while(s[0]&&s[1]&&n<max){int hi=hexnib(s[0]),lo=hexnib(s[1]); if(hi<0||lo<0)break; o[n++]=(hi<<4)|lo; s+=2;} return n; }
static void tohex_(const uint8_t*b,int n,char*o){ static const char*hx="0123456789abcdef"; for(int i=0;i<n;i++){o[i*2]=hx[b[i]>>4];o[i*2+1]=hx[b[i]&15];} o[n*2]=0; }

/* ----------------------------- --resume: parse a saved CSV log ---------- */
typedef struct { char input_kind[16],target_kind[16],purpose[64],wp[1024],pp[256],target[256],found[256];
  int change,gap,checksum,nth,compact,loginterval_ms;
  unsigned long long start,total,swept,hashed; int have_data,outcome_found; } Resume;
/* copy the value of a "# key: value" header line (first match) into out */
static void rz_val(const char*content,const char*key,char*out,int outmax){
  out[0]=0; char pfx[64]; snprintf(pfx,sizeof pfx,"# %s: ",key); size_t pl=strlen(pfx);
  for(const char*p=content;p;){ const char*nl=strchr(p,'\n'); size_t len=nl?(size_t)(nl-p):strlen(p);
    if(len>=pl && !strncmp(p,pfx,pl)){ size_t v=len-pl; if(v>=(size_t)outmax) v=outmax-1;
      memcpy(out,p+pl,v); out[v]=0; return; }
    p=nl?nl+1:0; }
}
static int resume_load(const char*path,Resume*R){
  FILE*f=fopen(path,"rb"); if(!f){ fprintf(stderr,"--resume: cannot open %s\n",path); return -1; }
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*b=malloc(n+1);
  if(fread(b,1,n,f)!=(size_t)n){ fclose(f); free(b); fprintf(stderr,"--resume: read %s\n",path); return -1; }
  b[n]=0; fclose(f);
  memset(R,0,sizeof *R); R->change=R->gap=1; R->checksum=R->compact=1; R->nth=-1;
  char t[1024];
  rz_val(b,"input_kind",R->input_kind,sizeof R->input_kind);
  rz_val(b,"wp",R->wp,sizeof R->wp);  rz_val(b,"pp",R->pp,sizeof R->pp);
  rz_val(b,"target_kind",R->target_kind,sizeof R->target_kind);
  rz_val(b,"target",R->target,sizeof R->target);
  rz_val(b,"purpose",R->purpose,sizeof R->purpose);
  rz_val(b,"change",t,sizeof t); if(t[0])R->change=atoi(t);
  rz_val(b,"gap",t,sizeof t);    if(t[0])R->gap=atoi(t);
  rz_val(b,"checksum",t,sizeof t);if(t[0])R->checksum=atoi(t);
  rz_val(b,"nth",t,sizeof t);    if(t[0])R->nth=atoi(t);
  rz_val(b,"compact",t,sizeof t);if(t[0])R->compact=atoi(t);
  rz_val(b,"start",t,sizeof t);  R->start=t[0]?strtoull(t,0,10):0;
  rz_val(b,"total",t,sizeof t);  R->total=t[0]?strtoull(t,0,10):0;
  rz_val(b,"loginterval_ms",t,sizeof t); if(t[0])R->loginterval_ms=atoi(t);
  rz_val(b,"outcome",t,sizeof t); if(!strcmp(t,"FOUND")){ R->outcome_found=1; rz_val(b,"found",R->found,sizeof R->found); }
  /* last DATA row (leading digit): columns t_ms,swept,swept_total,pct,rate,hashed,eta */
  { char*p=b,*last=0; while(p&&*p){ char*nl=strchr(p,'\n'); if(*p>='0'&&*p<='9') last=p; p=nl?nl+1:0; }
    if(last){ R->have_data=1; unsigned long long fld[7]={0}; int k=0; char*q=last;
      while(k<7&&q){ fld[k++]=strtoull(q,0,10); q=strchr(q,','); if(q)q++; }
      R->swept=fld[1]; R->hashed=fld[5]; } }
  free(b); return 0;
}

/* ------------------------ NVRTC build (with include inliner) ------------ */
static CUcontext g_ctx; static CUmodule g_mod; static CUdevice g_dev;
static int g_pflag=0, g_loginterval_ms=0, g_csv_own=0; static double g_p_secs=0; static FILE *g_csv=0;
static const char *g_target_str=0, *g_pattern_str=0;
/* resume/header params (recorded in the CSV header, reconstructed by --resume) */
static const char *g_input_kind=0,*g_wp=0,*g_pp=0,*g_target_kind=0,*g_purpose_str=0;
static int g_change_v=1,g_gap_v=1,g_checksum_v=1,g_nth_v=-1,g_compact_v=1;
static unsigned long long g_orig_start=0;
/* --resume overrides: report global progress over the ORIGINAL job window */
static unsigned long long g_prog_total=0, g_swept_base=0, g_hashed_base=0; static int g_resuming=0;
static char *inline_includes(char *src,const char *cu){
  const char*tag="#include \""; char*p=strstr(src,tag); if(!p) return src;
  char dir[512]; snprintf(dir,sizeof dir,"%s",cu); char*sl=strrchr(dir,'/'); if(sl)*sl=0; else strcpy(dir,".");
  char*q=p+strlen(tag),*e=strchr(q,'"'); if(!e) return src; char h[512]; int hn=(int)(e-q); if(hn>500)hn=500; memcpy(h,q,hn); h[hn]=0;
  char full[1100]; snprintf(full,sizeof full,"%s/%s",dir,h); char*inc=slurp(full);
  char*le=strchr(e,'\n'); if(!le) le=e+1; else le++; size_t pre=p-src,post=strlen(le),il=strlen(inc);
  char*out=malloc(pre+il+post+2); memcpy(out,src,pre); memcpy(out+pre,inc,il); out[pre+il]='\n'; memcpy(out+pre+il+1,le,post+1);
  free(inc); free(src); return inline_includes(out,cu);
}
/* FNV-1a of a string (PTX cache key). */
static unsigned long long fnv1a(const char*s){ unsigned long long h=1469598103934665603ULL; for(;*s;s++){ h^=(unsigned char)*s; h*=1099511628211ULL; } return h; }
static void build_module(const char *cu_path){
  const char *arch="--gpu-architecture=compute_120";
  const char*mr=getenv("CRACK_MAXREG");
  const char*def=getenv("CRACK_DEF");   /* e.g. -DSHA512_UNROLL16 for A/B experiments */
  char *src=inline_includes(slurp(cu_path),cu_path);
  /* PTX cache: NVRTC compile of the full EC+taproot module is slow (~2-3 min);
     cache the PTX keyed by source+arch hash so unchanged source loads instantly. */
  char key[128]; snprintf(key,sizeof key,"%llx",fnv1a(src)^fnv1a(arch)^(def?fnv1a(def):0));
  char cpath[256]; snprintf(cpath,sizeof cpath,"/tmp/bip39rxcrack_ptx_%s.ptx",key);
  char *ptx=0; FILE*cf=fopen(cpath,"rb");
  if(cf && !getenv("CRACK_NOCACHE")){
    fseek(cf,0,SEEK_END); long pn=ftell(cf); fseek(cf,0,SEEK_SET); ptx=malloc(pn+1);
    if(fread(ptx,1,pn,cf)==(size_t)pn){ ptx[pn]=0; } else { free(ptx); ptx=0; } fclose(cf);
  } else if(cf) fclose(cf);
  if(!ptx){
    const char *opts[]={ arch, def?def:"" }; int nopt = def?2:1;
    nvrtcProgram prog; NVR(nvrtcCreateProgram(&prog,src,"crack_kernels.cu",0,0,0));
    nvrtcResult cr=nvrtcCompileProgram(prog,nopt,opts);
    size_t logn=0; nvrtcGetProgramLogSize(prog,&logn);
    if(logn>1){ char*log=malloc(logn); nvrtcGetProgramLog(prog,log);
      if(cr!=NVRTC_SUCCESS||getenv("CRACK_VERBOSE")) fprintf(stderr,"NVRTC log:\n%s\n",log);
      free(log);
    }
    if(cr!=NVRTC_SUCCESS){ fprintf(stderr,"kernel compile failed\n"); exit(2); }
    size_t ptxn=0; NVR(nvrtcGetPTXSize(prog,&ptxn)); ptx=malloc(ptxn); NVR(nvrtcGetPTX(prog,ptx)); nvrtcDestroyProgram(&prog);
    FILE*wf=fopen(cpath,"wb"); if(wf){ fwrite(ptx,1,strlen(ptx),wf); fclose(wf); }
  }
  CUdevice dev; CU(cuInit(0)); CU(cuDeviceGet(&dev,0)); g_dev=dev;
  char name[128]; int M=0,m=0; cuDeviceGetName(name,sizeof name,dev);
  cuDeviceGetAttribute(&M,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
  cuDeviceGetAttribute(&m,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
  CU(cuCtxCreate(&g_ctx,0,dev));
  { CUjit_option jopt[1]; void*jval[1]; int njit=0;
    if(mr){ jopt[njit]=CU_JIT_MAX_REGISTERS; jval[njit]=(void*)(size_t)atoi(mr); njit++;
      fprintf(stderr,"(JIT max registers = %s)\n",mr); }
    CU(cuModuleLoadDataEx(&g_mod,ptx,njit,jopt,jval)); }
  /* Build the fixed-base comb table once, then point the device global d_comb
     at it so k*G uses the comb (64 adds) instead of double-and-add (~384 ops). */
  { CUdeviceptr tbl; CU(cuMemAlloc(&tbl,(size_t)64*16*8*sizeof(unsigned long long)));
    CUfunction gi; if(cuModuleGetFunction(&gi,g_mod,"g_comb_init")==CUDA_SUCCESS){
      void*a[]={&tbl}; CU(cuLaunchKernel(gi,1,1,1,1,1,1,0,0,a,0)); CU(cuCtxSynchronize());
      CUdeviceptr sym; size_t sz; if(cuModuleGetGlobal(&sym,&sz,g_mod,"d_comb")==CUDA_SUCCESS) CU(cuMemcpyHtoD(sym,&tbl,sizeof tbl));
    } }
  fprintf(stderr,"device: %s (sm_%d%d), NVRTC13->compute_120 PTX->sm_120\n",name,M,m);
  if(getenv("KERN_INFO")){
    const char*ks[]={"g_crack_pass","g_crack_addr","g_crack"};
    for(int i=0;i<3;i++){ CUfunction f; if(cuModuleGetFunction(&f,g_mod,ks[i])!=CUDA_SUCCESS) continue;
      int regs=0,lmem=0,smem=0,maxb=0; cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f);
      cuFuncGetAttribute(&lmem,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f); cuFuncGetAttribute(&smem,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);
      cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,f,128,0);
      fprintf(stderr,"  [kern] %-13s regs=%d local=%dB shared=%dB maxblocks@128=%d\n",ks[i],regs,lmem,smem,maxb); }
  }
  free(ptx);
}
static CUfunction kern(const char*n){ CUfunction f; CU(cuModuleGetFunction(&f,g_mod,n)); return f; }
static CUdeviceptr up(const void*h,size_t n){ CUdeviceptr d; CU(cuMemAlloc(&d,n?n:1)); if(n&&h) CU(cuMemcpyHtoD(d,h,n)); return d; }

/* ------------------------- progress reporting -------------------------- */
/* -p [SECS] : human line to stderr every SECS (default 1.0); ETA from a
 *      TRAILING-window rate (settles faster than cumulative). --loginterval
 *      MS[:FILE] : CSV every MS ms (reseed39 browser log shape: # header, rows,
 *      # footer). The two cadences are INDEPENDENT. hashed = cumulative
 *      survivors that reached PBKDF2 (the real work). */
typedef struct { unsigned long long total,swept,hashed,swept_base,hashed_base; double t0;
  int p; double p_interval,p_last,p_win_t; unsigned long long p_win_swept;   /* -p cadence */
  FILE*csv; int csv_own; double csv_interval,csv_last; } Prog;               /* CSV cadence */
static double now_s(void){ struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec+tv.tv_usec/1e6; }
static void prog_init(Prog*P, unsigned long long total, int p, double p_secs, int loginterval_ms, FILE*csv, int csv_own){
  memset(P,0,sizeof *P); P->total = g_prog_total?g_prog_total:total; P->p=p; P->csv=csv; P->csv_own=csv_own;
  P->swept_base=g_swept_base; P->hashed_base=g_hashed_base;
  P->p_interval = p_secs>0?p_secs:1.0;
  P->csv_interval = loginterval_ms>0?loginterval_ms/1000.0:0.0;
  P->t0=P->p_last=P->csv_last=P->p_win_t=now_s();
}
/* finest active cadence, for chunk sizing (so both cadences get fresh data). */
static double prog_intv(const Prog*P){
  double m=1e9;
  if(P->p && P->p_interval<m) m=P->p_interval;
  if(P->csv && P->csv_interval>0 && P->csv_interval<m) m=P->csv_interval;
  return m>=1e9?1.0:m;
}
static void prog_hdr(Prog*P,const char*mode,const char*target,const char*pattern,const char*path,
                     unsigned long long chunk,unsigned long long budget_mb){
  if(!P->csv) return;
  if(g_resuming){   /* appending to an existing log: just a marker, header already present */
    fprintf(P->csv,"# --- resumed: from swept=%llu (global index %llu) ts_ms=%.0f ---\n",
      g_swept_base, g_orig_start+g_swept_base, now_s()*1000.0); fflush(P->csv); return; }
  /* explicit, one-per-line "# key: value" so --resume can reconstruct the run */
  fprintf(P->csv,"# bip39rxcrack progress log\n# engine: cuda NVRTC13->compute_120->sm_120  device: RTX 5090\n");
  fprintf(P->csv,"# mode: %s\n",mode);
  fprintf(P->csv,"# input_kind: %s\n# wp: %s\n# pp: %s\n",g_input_kind?g_input_kind:"",g_wp?g_wp:"",g_pp?g_pp:"");
  fprintf(P->csv,"# target_kind: %s\n# target: %s\n# path: %s\n",g_target_kind?g_target_kind:"",target?target:"",path?path:"");
  fprintf(P->csv,"# purpose: %s\n# change: %d\n# gap: %d\n# checksum: %d\n# nth: %d\n# compact: %d\n",
    g_purpose_str?g_purpose_str:"auto",g_change_v,g_gap_v,g_checksum_v,g_nth_v,g_compact_v);
  fprintf(P->csv,"# start: %llu\n# total: %llu\n# chunk: %llu\n# budget_mb: %llu\n# loginterval_ms: %d\n# start_ts_ms: %.0f\n",
    g_orig_start,P->total,chunk,budget_mb,g_loginterval_ms,now_s()*1000.0);
  fprintf(P->csv,"t_ms,swept,swept_total,pct,rate,hashed,eta_s\n"); fflush(P->csv);
  (void)pattern;
}
/* -p human line: ETA from a TRAILING-window rate (settles faster than cumulative). */
static void prog_emit_p(Prog*P){
  double now=now_s(), el=now-P->t0, cum=el>0?P->swept/el/1e6:0.0;
  unsigned long long gsw=P->swept_base+P->swept, ghash=P->hashed_base+P->hashed;
  double dwt=now-P->p_win_t; unsigned long long dsw=P->swept-P->p_win_swept;
  double inst=dwt>0.01?dsw/dwt/1e6:cum, pct=P->total?100.0*gsw/P->total:0.0;
  double eta=(inst>0&&P->total>gsw)?(P->total-gsw)/(inst*1e6):0.0;
  fprintf(stderr,"\r[%7.1fs] %.1f/%.1fM (%.1f%%) %.2f Mc/s  hashed %.2fM  ETA %.0fs    ",
      el,gsw/1e6,P->total/1e6,pct,inst,ghash/1e6,eta);
  P->p_last=now; P->p_win_t=now; P->p_win_swept=P->swept;
}
/* CSV row: cumulative rate (t_ms+swept let you derive instantaneous by deltas). */
static void prog_emit_csv(Prog*P){
  double now=now_s(), el=now-P->t0, cum=el>0?P->swept/el/1e6:0.0;   /* rate is THIS run's throughput */
  unsigned long long gsw=P->swept_base+P->swept, ghash=P->hashed_base+P->hashed;
  double pct=P->total?100.0*gsw/P->total:0.0;
  double eta=(cum>0&&P->total>gsw)?(P->total-gsw)/(cum*1e6):0.0;
  fprintf(P->csv,"%.0f,%llu,%llu,%.3f,%.3f,%llu,%.1f\n",el*1000.0,gsw,P->total,pct,cum,ghash,eta);
  fflush(P->csv); P->csv_last=now;
}
static void prog_tick(Prog*P, unsigned long long swept, unsigned long long hashed){
  P->swept=swept; P->hashed=hashed; double now=now_s();
  if(P->p && now-P->p_last>=P->p_interval) prog_emit_p(P);
  if(P->csv && P->csv_interval>0 && now-P->csv_last>=P->csv_interval) prog_emit_csv(P);
}
static void prog_finish(Prog*P, const char*outcome, const char*found, const char*path){
  if(P->p){ prog_emit_p(P); fprintf(stderr,"\n"); }
  if(P->csv){ prog_emit_csv(P);
    fprintf(P->csv,"# outcome: %s\n# found: %s  path: %s\n# elapsed_ms: %.0f\n",
      outcome,found?found:"",path?path:"",(now_s()-P->t0)*1000.0); fflush(P->csv); if(P->csv_own) fclose(P->csv); }
}
/* choose a chunk so reports land ~every interval; smaller when reporting is on. */
static unsigned long long report_chunk(int reporting){ return reporting ? (1ULL<<18) : (64ULL<<20); }
/* size the NEXT chunk so it takes ~interval seconds (report lands ~every tick). */
static unsigned long long next_chunk(unsigned long long ccount,double csecs,double interval,int reporting){
  if(!reporting) return 64ULL<<20;
  if(csecs<=0||interval<=0) return ccount;
  double c=(double)ccount/csecs*interval*0.5;           /* ~half-interval chunks -> tighter cadence */
  unsigned long long r=(unsigned long long)c;
  if(r<(1ULL<<18)) r=1ULL<<18;                          /* min 256K: launch overhead floor */
  if(r>(64ULL<<20)) r=64ULL<<20;                        /* max 64M: bounded buffer */
  return r;
}


/* ------------------------- base words / wordlist ------------------------ */
typedef struct { int n; char word[32][16]; uint32_t bip39[32]; } Words;
static void load_wordlist(char names[2048][16]){
  const char*rd=getenv("RESEED39_DIR"); if(!rd) rd="/root/bip39rxcrack";
  char p[600]; snprintf(p,sizeof p,"%s/data/english.txt",rd); char*t=slurp(p);
  int i=0; char*s=strtok(t,"\r\n"); while(s&&i<2048){ snprintf(names[i],16,"%s",s); i++; s=strtok(0,"\r\n"); }
  if(i!=2048){ fprintf(stderr,"wordlist not 2048 (%d) from %s\n",i,p); exit(2);} free(t);
}
static int bip39_index(char names[2048][16],const char*w){ for(int i=0;i<2048;i++) if(!strcmp(names[i],w)) return i; return -1; }
static void parse_words(Words*W,const char*csv){
  static char names[2048][16]; load_wordlist(names);
  W->n=0; char buf[2048]; snprintf(buf,sizeof buf,"%s",csv);
  char*s=strtok(buf," ,\t\r\n");
  while(s){ if(W->n>=32){fprintf(stderr,"max 32 words (v1)\n");exit(2);} snprintf(W->word[W->n],16,"%s",s);
    int bi=bip39_index(names,s); if(bi<0){fprintf(stderr,"'%s' is not a BIP39 English word\n",s);exit(2);}
    W->bip39[W->n]=(uint32_t)bi; W->n++; s=strtok(0," ,\t\r\n"); }
  /* distinctness (v1 permutation assumes distinct alternatives) */
  for(int i=0;i<W->n;i++) for(int j=i+1;j<W->n;j++) if(!strcmp(W->word[i],W->word[j])){
    fprintf(stderr,"duplicate word '%s' (v1 needs distinct words)\n",W->word[i]); exit(2); }
}
/* build the reseed39 wordset pattern: "(  w0| w1|...){{N!}}" (space-prefixed) */
static void wordset_pattern(const Words*W,char*out,int max){
  int L=snprintf(out,max,"(");
  for(int i=0;i<W->n;i++) L+=snprintf(out+L,max-L,"%s %s",i?"|":"",W->word[i]);
  snprintf(out+L,max-L,"){{%d!}}",W->n);
}

/* -------------------------------- base58 xpub --------------------------- */
static const char*B58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int b58decode(const char*s,uint8_t*out,int outmax){
  uint8_t tmp[128]; int tn=0; memset(tmp,0,sizeof tmp);
  for(const char*c=s;*c;c++){ const char*p=strchr(B58,*c); if(!p) return -1; int carry=(int)(p-B58);
    for(int i=0;i<tn;i++){ carry+=tmp[i]*58; tmp[i]=carry&0xff; carry>>=8; }
    while(carry){ if(tn>=(int)sizeof tmp) return -1; tmp[tn++]=carry&0xff; carry>>=8; } }
  int zeros=0; for(const char*c=s;*c==B58[0];c++) zeros++;
  int total=zeros+tn; if(total>outmax) return -1;
  for(int i=0;i<zeros;i++) out[i]=0;
  for(int i=0;i<tn;i++) out[zeros+i]=tmp[tn-1-i];
  return total;
}
/* xpub/xprv (base58check, 82 bytes = 78 payload + 4 checksum): chaincode @ [13..44] */
static int xpub_chaincode(const char*xp,uint8_t cc[32]){
  uint8_t raw[128]; int n=b58decode(xp,raw,sizeof raw);
  if(n!=82){ fprintf(stderr,"xpub base58 decode length %d (want 82)\n",n); return -1; }
  memcpy(cc,raw+13,32); return 0;
}
/* ------------------------------ bech32 --------------------------------- */
static const char*BECH="qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static uint32_t bech_polymod(const uint8_t*v,int n){
  static const uint32_t G[5]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
  uint32_t chk=1;
  for(int i=0;i<n;i++){ uint32_t b=chk>>25; chk=((chk&0x1ffffff)<<5)^v[i]; for(int k=0;k<5;k++) if((b>>k)&1) chk^=G[k]; }
  return chk;
}
/* decode a bech32/bech32m segwit address -> witver, program bytes, spec. */
static int bech32_decode(const char*addr,int*witver,uint8_t*prog,int*proglen,int*is_m){
  char s[130]; int L=strlen(addr); if(L<8||L>120) return -1;
  for(int i=0;i<L;i++){ char c=addr[i]; if(c>='A'&&c<='Z') c=c-'A'+'a'; s[i]=c; } s[L]=0;
  int pos=-1; for(int i=L-1;i>=0;i--) if(s[i]=='1'){ pos=i; break; }
  if(pos<1||pos+7>L) return -1;
  int hlen=pos; uint8_t values[130]; int vn=0;
  for(int i=0;i<hlen;i++) values[vn++]=s[i]>>5;      /* hrp expand high */
  values[vn++]=0;
  for(int i=0;i<hlen;i++) values[vn++]=s[i]&31;       /* hrp expand low */
  int dstart=vn;
  for(int i=pos+1;i<L;i++){ const char*p=strchr(BECH,s[i]); if(!p) return -1; values[vn++]=(uint8_t)(p-BECH); }
  int dlen=vn-dstart;                                  /* data incl. 6-char checksum */
  if(dlen<7) return -1;
  uint32_t pm=bech_polymod(values,vn);
  if(pm==1) *is_m=0; else if(pm==0x2bc830a3) *is_m=1; else return -1;
  const uint8_t*data=values+dstart;
  *witver=data[0];
  /* convertBits 5->8 over data[1 .. dlen-6) */
  int n5=dlen-6-1; const uint8_t*d5=data+1;
  uint32_t acc=0; int bits=0; *proglen=0;
  for(int i=0;i<n5;i++){ acc=(acc<<5)|d5[i]; bits+=5; while(bits>=8){ bits-=8; prog[(*proglen)++]=(uint8_t)((acc>>bits)&0xff); } }
  if(bits>=5 || ((acc<<(8-bits))&0xff)) return -1;     /* leftover / padding must be zero */
  return 0;
}
/* Any supported address target -> program (up to 32B) + proglen + purpose:
 *   base58 p2pkh 0x00 -> 44 (20B) ; p2sh 0x05 -> 49 (20B)
 *   bech32 v0 20B     -> 84 (p2wpkh) ; bech32m v1 32B -> 86 (p2tr) */
static int decode_address(const char*addr,uint8_t prog[32],int*proglen,int*purpose){
  if(!strncmp(addr,"bc1",3)||!strncmp(addr,"tb1",3)||!strncmp(addr,"bcrt1",5)){
    int wv,pl,ism; if(bech32_decode(addr,&wv,prog,&pl,&ism)){ fprintf(stderr,"bad bech32 address\n"); return -1; }
    if(wv==0&&pl==20&&ism==0){ *purpose=84; *proglen=20; return 0; }
    if(wv==1&&pl==32&&ism==1){ *purpose=86; *proglen=32; return 0; }
    fprintf(stderr,"unsupported witness v%d len %d (v1: p2wpkh bc1q / p2tr bc1p)\n",wv,pl); return -1;
  }
  uint8_t raw[64]; int n=b58decode(addr,raw,sizeof raw);
  if(n!=25){ fprintf(stderr,"address base58 decode length %d (want 25)\n",n); return -1; }
  int ver=raw[0]; memcpy(prog,raw+1,20); *proglen=20;
  if(ver==0x00) *purpose=44; else if(ver==0x05) *purpose=49;
  else { fprintf(stderr,"unsupported address version 0x%02x\n",ver); return -1; }
  return 0;
}

/* ------------------------------ GPU launch ------------------------------ */
static void gpu_upload_words(const Words*W, CUdeviceptr*d_data,CUdeviceptr*d_off,CUdeviceptr*d_len,CUdeviceptr*d_idx){
  uint8_t data[512]; int off[32],len[32]; int p=0;
  for(int i=0;i<W->n;i++){ off[i]=p; int l=strlen(W->word[i]); len[i]=l; memcpy(data+p,W->word[i],l); p+=l; }
  *d_data=up(data,p); *d_off=up(off,W->n*sizeof(int)); *d_len=up(len,W->n*sizeof(int)); *d_idx=up(W->bip39,W->n*sizeof(uint32_t));
}

/* --------------------------------- modes -------------------------------- */
static int mode_rank(const Words*W,const char*arrangement){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse: %s\n",r?rxe_error_message(r):"null");return 2;}
  /* build the canonical member string: wordset members render space-prefixed
     (" w0 w1 .. wN"); rxe_rank wants that exact whole-member form. */
  char member[1024]; int L=0; char tmp[1024]; snprintf(tmp,sizeof tmp,"%s",arrangement);
  char*s=strtok(tmp," ,\t\r\n"); while(s){ L+=snprintf(member+L,sizeof member-L," %s",s); s=strtok(0," ,\t\r\n"); }
  mpz_t idx; mpz_init(idx); int rc=rxe_rank(r,member,idx);
  if(rc){ fprintf(stderr,"rxe_rank: not a member (%s)\n",rxe_rank_reason()); return 2; }
  gmp_printf("%Zd\n",idx); mpz_clear(idx); rxe_free(r); return 0;
}

static int mode_recon_gate(const Words*W,int nsamp,const char*cu){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse: %s\n",r?rxe_error_message(r):"null");return 2;}
  mpz_t total; mpz_init(total); mpz_set(total,r->nitems);
  build_module(cu);
  CUdeviceptr dd,dof,dln,dix; gpu_upload_words(W,&dd,&dof,&dln,&dix);
  /* choose sample indices spread across [0,total) */
  unsigned long long *lo=malloc(nsamp*8),*hi=malloc(nsamp*8);
  mpz_t j,step,acc; mpz_init(j); mpz_init(step); mpz_init(acc);
  mpz_fdiv_q_ui(step,total,nsamp>0?nsamp:1); if(mpz_sgn(step)==0) mpz_set_ui(step,1);
  char (*libstr)[MN_STRIDE]=malloc((size_t)nsamp*MN_STRIDE);
  mpz_set_ui(acc,0);
  for(int i=0;i<nsamp;i++){
    if(mpz_cmp(acc,total)>=0) mpz_mod(acc,acc,total);
    mpz_set(j,acc);
    /* librxe reference render */
    rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
    char*t=buf; while(*t==' ')t++; snprintf(libstr[i],MN_STRIDE,"%s",t);
    /* split j into lo/hi 64-bit */
    mpz_t hiz,loz,tmp; mpz_init(hiz); mpz_init(loz); mpz_init(tmp);
    mpz_fdiv_q_2exp(hiz,j,64); mpz_fdiv_r_2exp(loz,j,64);
    lo[i]=mpz_get_ui(loz); hi[i]=mpz_get_ui(hiz);
    mpz_clear(hiz); mpz_clear(loz); mpz_clear(tmp);
    mpz_add(acc,acc,step);
  }
  CUdeviceptr dlo=up(lo,nsamp*8),dhi=up(hi,nsamp*8),dmn,dlen,dval;
  CU(cuMemAlloc(&dmn,(size_t)nsamp*MN_STRIDE)); CU(cuMemAlloc(&dlen,nsamp*sizeof(int))); CU(cuMemAlloc(&dval,nsamp));
  int n=W->n,size=W->n;
  void*args[]={&dd,&dof,&dln,&dix,&n,&size,&dlo,&dhi,&nsamp,&dmn,&dlen,&dval};
  int tpb=64,grid=(nsamp+tpb-1)/tpb; CU(cuLaunchKernel(kern("g_recon"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  char *gmn=malloc((size_t)nsamp*MN_STRIDE); int*glen=malloc(nsamp*sizeof(int));
  CU(cuMemcpyDtoH(gmn,dmn,(size_t)nsamp*MN_STRIDE)); CU(cuMemcpyDtoH(glen,dlen,nsamp*sizeof(int)));
  int bad=0;
  for(int i=0;i<nsamp;i++){ char g[MN_STRIDE]; int L=glen[i]; if(L>=MN_STRIDE)L=MN_STRIDE-1; memcpy(g,gmn+(size_t)i*MN_STRIDE,L); g[L]=0;
    if(getenv("DBG")&&i<6) fprintf(stderr,"  DBG #%d lo=%llu hi=%llu glen=%d\n    gpu:    %s\n    librxe: %s\n",i,lo[i],hi[i],glen[i],g,libstr[i]);
    if(strcmp(g,libstr[i])){ if(bad<3) fprintf(stderr,"  MISMATCH recon #%d\n    gpu:    %s\n    librxe: %s\n",i,g,libstr[i]); bad++; } }
  printf("  [%s] recon vs librxe   : %d/%d indices byte-identical (unrank+reconstruction)\n", bad?"FAIL":"PASS", nsamp-bad, nsamp);
  gmp_printf("  (pattern nitems = %Zd)\n",total);
  return bad?1:0;
}

/* forward decls (defined below) */
static int parse_template(const char*tpl,uint32_t tmpl[32],int upos[32],int*W,int*U,int*last_unknown);
static const char* WLNAME(int idx);
/* ---- missing-word ORDER gate: kernel unrank index == librxe canonical rank --- */
static int mnemonic_to_idx(const char*mn,uint32_t*out,int max){
  static char names[2048][16]; static int ld=0; if(!ld){load_wordlist(names);ld=1;}
  char buf[512]; snprintf(buf,sizeof buf,"%s",mn); int n=0; char*s=strtok(buf," \t\r\n");
  while(s&&n<max){ int bi=bip39_index(names,s); if(bi<0) return -1; out[n++]=(uint32_t)bi; s=strtok(0," \t\r\n"); }
  return n;
}
static struct rxe* rxe_from_template(const uint32_t*tmpl,const int*upos,int W,int U){
  static int reg=0; static char names[2048][16]; static const char*wp[2048];
  if(!reg){ load_wordlist(names); for(int i=0;i<2048;i++) wp[i]=names[i]; rxe_register_dict("bip39gate",wp,2048); reg=1; }
  char pat[4096]; int L=0,ui=0;
  for(int p=0;p<W;p++){ int unk=(ui<U && upos[ui]==p);
    L+=snprintf(pat+L,sizeof pat-L,"%s%s", p?" ":"", unk?"[:bip39gate:]":WLNAME((int)tmpl[p]));
    if(unk) ui++; }
  struct rxe*r=rxe_parse(pat,0);
  if(!r||rxe_error(r)){ fprintf(stderr,"rxe parse (gate): %s\n", r?rxe_error_message(r):"null"); return 0; }
  return r;
}
static int miss_gate_run(const char*label,int nth,const uint32_t*tmpl,const int*upos,int U,int W,
                         struct rxe*r,int nsamp,const char*cu){ (void)cu;
  int CS=W/3, freebits=11-CS;
  mpz_t total; mpz_init(total);
  if(nth){ mpz_ui_pow_ui(total,2048,U-1); mpz_mul_2exp(total,total,freebits); }
  else     mpz_ui_pow_ui(total,2048,U);
  unsigned long long *idx=malloc((size_t)nsamp*8);
  mpz_t acc,step,jz; mpz_init(acc); mpz_init(step); mpz_init(jz);
  mpz_fdiv_q_ui(step,total,nsamp>0?nsamp:1); if(mpz_sgn(step)==0) mpz_set_ui(step,1);
  mpz_set_ui(acc,0);
  for(int i=0;i<nsamp;i++){ idx[i]=mpz_get_ui(acc); mpz_add(acc,acc,step); if(mpz_cmp(acc,total)>=0) mpz_mod(acc,acc,total); }
  CUdeviceptr dtmpl=up(tmpl,W*sizeof(uint32_t)), dupos=up(upos,(U?U:1)*sizeof(int)), didx=up(idx,(size_t)nsamp*8);
  CUdeviceptr dout; CU(cuMemAlloc(&dout,(size_t)nsamp*W*sizeof(uint32_t)));
  int passU=U; void*args[]={&dtmpl,&W,&dupos,&passU,&didx,&nsamp,&dout};
  int tpb=64,grid=(nsamp+tpb-1)/tpb;
  CU(cuLaunchKernel(kern(nth?"g_unrank_nth":"g_unrank_missing"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  uint32_t *out=malloc((size_t)nsamp*W*sizeof(uint32_t));
  CU(cuMemcpyDtoH(out,dout,(size_t)nsamp*W*sizeof(uint32_t)));
  int bad=0;
  for(int i=0;i<nsamp;i++){ uint32_t*g=out+(size_t)i*W;
    if(nth){ unsigned long long j=idx[i], f=j&((1ULL<<freebits)-1), mid=j>>freebits; uint32_t last=g[W-1];
      if((last>>CS)!=f){ if(bad<3) fprintf(stderr,"  nth f-bit mismatch #%d: last>>CS=%u f=%llu\n",i,last>>CS,(unsigned long long)f); bad++; continue; }
      mpz_set_ui(jz,mid); mpz_mul_ui(jz,jz,2048); mpz_add_ui(jz,jz,last); }
    else mpz_set_ui(jz,idx[i]);
    rxe_seek(r,jz); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r); char*t=buf; while(*t==' ')t++;
    uint32_t lib[32]; int ln=mnemonic_to_idx(t,lib,32);
    int eq=(ln==W); for(int p=0;p<W&&eq;p++){ if(lib[p]!=g[p]) eq=0; }
    if(!eq){ if(bad<3){ fprintf(stderr,"  MISMATCH %s #%d idx=%llu\n    gpu:   ",label,i,idx[i]);
        for(int p=0;p<W;p++){ fprintf(stderr," %s",WLNAME((int)g[p])); } fprintf(stderr,"\n    librxe: %s\n",t);} bad++; } }
  gmp_printf("  [%s] %-9s vs librxe : %d/%d indices byte-identical (space = %Zd)\n", bad?"FAIL":"PASS", label, nsamp-bad, nsamp, total);
  free(idx); free(out); mpz_clear(total); mpz_clear(acc); mpz_clear(step); mpz_clear(jz);
  return bad?1:0;
}
static int mode_miss_gate(const char*tpl,int nsamp,const char*cu){
  uint32_t tmpl[32]; int upos[32],W,U,last_unknown;
  if(parse_template(tpl,tmpl,upos,&W,&U,&last_unknown)) return 2;
  build_module(cu);
  struct rxe*r=rxe_from_template(tmpl,upos,W,U); if(!r) return 2;
  printf("missing-word ORDER gate  (kernel unrank index == librxe canonical rank):\n");
  printf("  template: %s\n", tpl);
  int bad=0;
  bad |= miss_gate_run("baseline",0,tmpl,upos,U,W,r,nsamp,cu);
  if(last_unknown) bad |= miss_gate_run("[:Nth:]",1,tmpl,upos,U,W,r,nsamp,cu);
  else printf("  ([:Nth:] gate skipped: last position is a known word)\n");
  rxe_free(r);
  printf("  ==== missing-word order gate %s (%d samples/mode) ====\n", bad?"FAILED":"PASSED", nsamp);
  return bad?1:0;
}

static int mode_dump_valid(const Words*W,int nsamp,const char*cu){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse err\n");return 2;}
  mpz_t total; mpz_init(total); mpz_set(total,r->nitems);
  build_module(cu);
  CUdeviceptr dd,dof,dln,dix; gpu_upload_words(W,&dd,&dof,&dln,&dix);
  unsigned long long *lo=malloc(nsamp*8),*hi=malloc(nsamp*8);
  mpz_t acc,step; mpz_init(acc); mpz_init(step); mpz_fdiv_q_ui(step,total,nsamp>0?nsamp:1); if(mpz_sgn(step)==0)mpz_set_ui(step,1);
  mpz_set_ui(acc,0);
  for(int i=0;i<nsamp;i++){ mpz_t hiz,loz; mpz_init(hiz);mpz_init(loz);
    mpz_fdiv_q_2exp(hiz,acc,64); mpz_fdiv_r_2exp(loz,acc,64); lo[i]=mpz_get_ui(loz); hi[i]=mpz_get_ui(hiz);
    mpz_clear(hiz);mpz_clear(loz); mpz_add(acc,acc,step); if(mpz_cmp(acc,total)>=0)mpz_mod(acc,acc,total); }
  CUdeviceptr dlo=up(lo,nsamp*8),dhi=up(hi,nsamp*8),dmn,dlen,dval;
  CU(cuMemAlloc(&dmn,(size_t)nsamp*MN_STRIDE)); CU(cuMemAlloc(&dlen,nsamp*sizeof(int))); CU(cuMemAlloc(&dval,nsamp));
  int n=W->n,size=W->n; void*args[]={&dd,&dof,&dln,&dix,&n,&size,&dlo,&dhi,&nsamp,&dmn,&dlen,&dval};
  int tpb=64,grid=(nsamp+tpb-1)/tpb; CU(cuLaunchKernel(kern("g_recon"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  char*gmn=malloc((size_t)nsamp*MN_STRIDE); int*glen=malloc(nsamp*4); uint8_t*gval=malloc(nsamp);
  CU(cuMemcpyDtoH(gmn,dmn,(size_t)nsamp*MN_STRIDE)); CU(cuMemcpyDtoH(glen,dlen,nsamp*4)); CU(cuMemcpyDtoH(gval,dval,nsamp));
  for(int i=0;i<nsamp;i++){ char g[MN_STRIDE]; int L=glen[i]; if(L>=MN_STRIDE)L=MN_STRIDE-1; memcpy(g,gmn+(size_t)i*MN_STRIDE,L); g[L]=0;
    printf("%d\t%s\n",gval[i],g); }
  return 0;
}

static int mode_crack(const Words*W,const uint8_t target_cc[32],uint32_t*purposes,int npurp,int require_ck,
                      unsigned long long user_start,unsigned long long user_count,const char*cu){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse err\n");return 2;}
  mpz_t total; mpz_init(total); mpz_set(total,r->nitems);
  if(W->n>20){ fprintf(stderr,"v1 self-enumerate hit-index is u64: max 20 words (20! < 2^64). Got %d.\n",W->n); return 2; }
  build_module(cu);
  CUdeviceptr dd,dof,dln,dix; gpu_upload_words(W,&dd,&dof,&dln,&dix);
  CUdeviceptr dtc=up(target_cc,32), dpu=up(purposes,npurp*sizeof(uint32_t));
  unsigned long long init_idx=~0ULL; int zero=0;
  CUdeviceptr dhi=up(&init_idx,8), dfound=up(&zero,4), dpur=up(&zero,4);
  unsigned long long total_u=mpz_get_ui(total); /* <=20! fits u64 */
  unsigned long long start_lo=user_start, start_hi=0;
  if(start_lo>total_u) start_lo=total_u;
  unsigned long long count = user_count ? user_count : (total_u-start_lo);
  if(start_lo+count>total_u) count=total_u-start_lo;
  int n=W->n,size=W->n;
  int tpb=128, grid=1024;   /* grid-stride covers the whole shard */
  unsigned long long z0=0; CUdeviceptr dhashed=up(&z0,8);
  unsigned long long cstart=start_lo,ccount=0;
  void*args[]={&dd,&dof,&dln,&dix,&n,&size,&cstart,&start_hi,&ccount,&dpu,&npurp,&dtc,&require_ck,&dhi,&dfound,&dpur,&dhashed};
  int reporting=(g_pflag||g_loginterval_ms);
  char path[64]; snprintf(path,sizeof path,"xpub account chaincode (%d purpose%s)",npurp,npurp>1?"s":"");
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  prog_hdr(&P,"xpub (words, EC-free)",g_target_str,g_pattern_str,path,0,0);
  double intv=prog_intv(&P);
  fprintf(stderr,"enumerating %llu candidates from %llu (checksum-%s, %d purpose%s)...\n",
          count,start_lo,require_ck?"ON":"OFF",npurp,npurp>1?"s":"");
  int found=0; unsigned long long swept=0,hashed=0, chunk=reporting?report_chunk(1):(64ULL<<20); double tt0=now_s();
  for(cstart=start_lo; cstart<start_lo+count; ){
    ccount=(start_lo+count-cstart<chunk)?(start_lo+count-cstart):chunk; double c0=now_s();
    CU(cuLaunchKernel(kern("g_crack"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    CU(cuMemcpyDtoH(&hashed,dhashed,8)); prog_tick(&P,swept,require_ck?hashed:swept);
    CU(cuMemcpyDtoH(&found,dfound,4)); if(found) break;
    chunk=next_chunk(ccount,csecs,intv,reporting);
  }
  double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (enumerate+sieve%s%s)\n",
          swept,secs,swept/secs/1e6, require_ck?"->PBKDF2 on survivors":"+PBKDF2 all", swept<count?", early-exit":"");
  int fpur=0; unsigned long long hidx=0;
  CU(cuMemcpyDtoH(&hidx,dhi,8)); CU(cuMemcpyDtoH(&fpur,dpur,4));
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND (no candidate derived the target chain code)\n"); return 1; }
  /* render the found mnemonic via librxe (canonical) */
  mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
  char*t=buf; while(*t==' ')t++;
  prog_finish(&P,"FOUND",t,path);
  printf("FOUND\n");
  printf("  index   : %llu\n",hidx);
  printf("  mnemonic: %s\n",t);
  printf("  path    : m/%d'/0'/0'  (purpose %d, account 0)\n",fpur,fpur);
  mpz_clear(j); rxe_free(r); return 0;
}

/* ------------------ Regime B: words permutation, ADDRESS target --------- */
static int mode_crack_addr(const Words*W,const uint8_t tprog[32],int purpose,
                           uint32_t changes,uint32_t gap,int require_ck,int compact,
                           unsigned long long ustart,unsigned long long ucount,const char*cu){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse err\n");return 2;}
  mpz_t total; mpz_init(total); mpz_set(total,r->nitems);
  if(W->n>20){ fprintf(stderr,"v1 self-enumerate hit-index is u64: max 20 words. Got %d.\n",W->n); return 2; }
  build_module(cu);
  CUdeviceptr dd,dof,dln,dix; gpu_upload_words(W,&dd,&dof,&dln,&dix);
  CUdeviceptr dtp=up(tprog,32);
  unsigned long long init=~0ULL; int zero=0;
  CUdeviceptr dhi=up(&init,8),dfound=up(&zero,4);
  unsigned long long total_u=mpz_get_ui(total);
  unsigned long long start=ustart>total_u?total_u:ustart;
  unsigned long long count=ucount?ucount:(total_u-start); if(start+count>total_u) count=total_u-start;
  int n=W->n,size=W->n; uint32_t pu=(uint32_t)purpose;
  int tpb=128,grid=1024; struct timeval t0,t1; double secs=0; (void)t0;(void)t1;
  int reporting=(g_pflag||g_loginterval_ms);
  char path[64]; snprintf(path,sizeof path,"m/%d'/0'/0'/[0,%u)/[0,%u)",purpose,changes,gap);
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  double intv=prog_intv(&P);
  unsigned long long z0=0; CUdeviceptr dhashed=up(&z0,8);
  uint32_t hci0[2]={0,0}; CUdeviceptr dhit_ci=up(hci0,8);
  int used_compact=0;
  if(compact && require_ck){
    /* CHUNKED compaction: size the survivor buffer for a chunk's FULL size
     * (worst case: every candidate survives) -> overflow is STRUCTURALLY
     * impossible. Chunk size C = budget/8 (default ~1 GiB; COMPACT_BUDGET_MB env
     * override). One global hit_index (atomicMin across chunks) -> lowest-index
     * wins across chunk boundaries; early-exit once a chunk finds a hit (later
     * chunks only hold higher indices). Fall back to fused if no chunk fits. */
    unsigned long long budget = 1024ULL*1024*1024;
    const char*mb=getenv("COMPACT_BUDGET_MB"); if(mb){ long v=atol(mb); if(v>16) budget=(unsigned long long)v*1024*1024; }
    unsigned long long C = budget/8;
    CUdeviceptr dsurv=0;
    while(C>=(1ULL<<20)){ if(cuMemAlloc(&dsurv,C*8)==CUDA_SUCCESS) break; C/=2; dsurv=0; }
    if(dsurv){
      used_compact=1;
      unsigned long long z64=0; CUdeviceptr dctr=up(&z64,8);
      prog_hdr(&P,"regime B (words, compacted)",g_target_str,g_pattern_str,path,C,C*8/1024/1024);
      fprintf(stderr,"regime B (COMPACTED, chunked): %llu candidates, buffer %llu MiB -> %s ...\n",count,C*8/1024/1024,path);
      double tt0=now_s(); int found=0; unsigned long long tot_surv=0, swept=0;
      unsigned long long cchunk = reporting?(1ULL<<20):C;
      for(unsigned long long cs=start; cs<start+count; ){
        unsigned long long cc = (start+count-cs < cchunk) ? (start+count-cs) : cchunk;
        double c0=now_s();
        CU(cuMemcpyHtoD(dctr,&z64,8));
        void*sa[]={&dd,&dof,&dln,&dix,&n,&size,&cs,&cc,&dsurv,&C,&dctr};
        CU(cuLaunchKernel(kern("g_sieve_perm"),grid,1,1,tpb,1,1,0,0,sa,0)); CU(cuCtxSynchronize());
        unsigned long long nsurv=0; CU(cuMemcpyDtoH(&nsurv,dctr,8)); tot_surv+=nsurv;
        if(nsurv>C){ fprintf(stderr,"ERROR: chunk survivors %llu > chunk size %llu -- impossible (bug)\n",nsurv,C); return 2; }
        if(nsurv){ void*pa[]={&dd,&dof,&dln,&dix,&n,&size,&dsurv,&nsurv,&pu,&changes,&gap,&dtp,&dhi,&dfound,&dhit_ci};
          CU(cuLaunchKernel(kern("g_pbkdf2_perm"),grid,1,1,tpb,1,1,0,0,pa,0)); CU(cuCtxSynchronize()); }
        double csecs=now_s()-c0; swept+=cc; cs+=cc; prog_tick(&P,swept,tot_surv);
        CU(cuMemcpyDtoH(&found,dfound,4)); if(found) break;
        cchunk=next_chunk(cc,csecs,intv,reporting); if(cchunk>C) cchunk=C;
      }
      secs=now_s()-tt0; cuMemFree(dsurv);
      fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s  (%llu survivors -> dense PBKDF2%s)\n",swept,secs,swept/secs/1e6,tot_surv,swept<count?", early-exit":"");
    } else {
      fprintf(stderr,"regime B: compaction buffer won't allocate on this GPU -- falling back to fused.\n");
    }
  }
  int found=0;
  if(!used_compact){
    unsigned long long cstart=start,ccount=0;
    void*args[]={&dd,&dof,&dln,&dix,&n,&size,&cstart,&ccount,&pu,&changes,&gap,&dtp,&require_ck,&dhi,&dfound,&dhashed,&dhit_ci};
    prog_hdr(&P,require_ck?"regime B (words, fused)":"regime B (words, no-sieve)",g_target_str,g_pattern_str,path,0,0);
    fprintf(stderr,"regime B: %llu permutations (checksum-%s) -> %s ...\n",count,require_ck?"ON":"OFF",path);
    double tt0=now_s(); unsigned long long swept=0,hashed=0, chunk=reporting?report_chunk(1):(64ULL<<20);
    for(cstart=start; cstart<start+count; ){
      ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk;
      double c0=now_s();
      CU(cuLaunchKernel(kern("g_crack_addr"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
      double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
      CU(cuMemcpyDtoH(&hashed,dhashed,8)); prog_tick(&P,swept,require_ck?hashed:swept);
      CU(cuMemcpyDtoH(&found,dfound,4)); if(found) break;
      chunk=next_chunk(ccount,csecs,intv,reporting);
    }
    secs=now_s()-tt0;
    fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (sieve->PBKDF2+EC on survivors%s)\n",swept,secs,swept/secs/1e6,swept<count?", early-exit":"");
  }
  (void)secs;
  unsigned long long hidx=0; CU(cuMemcpyDtoH(&found,dfound,4)); CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND\n"); return 1; }
  mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
  char*t=buf; while(*t==' ')t++;
  uint32_t hci[2]; CU(cuMemcpyDtoH(hci,dhit_ci,8));
  char fpath[80]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'/%u/%u",purpose,hci[0],hci[1]);
  prog_finish(&P,"FOUND",t,fpath);
  printf("FOUND\n  index (canonical): %llu\n  mnemonic: %s\n  path    : %s\n",hidx,t,fpath);
  mpz_clear(j); rxe_free(r); return 0;
}

/* ---------------------- Missing-word ([:bip39-en:]) --------------------- */
/* pack + upload the full 2048-word list (strings + off + len). */
static void gpu_upload_wordlist(CUdeviceptr *d_data,CUdeviceptr *d_off,CUdeviceptr *d_len){
  static char names[2048][16]; load_wordlist(names);
  uint8_t data[20000]; int off[2048],len[2048]; int p=0;
  for(int i=0;i<2048;i++){ off[i]=p; int l=(int)strlen(names[i]); len[i]=l; memcpy(data+p,names[i],l); p+=l; }
  *d_data=up(data,p); *d_off=up(off,2048*sizeof(int)); *d_len=up(len,2048*sizeof(int));
}
/* [:bip39:] and its aliases (all the same 2048-word English list, per reseed39
 * crackworker: bip39-en / bip39en / bip39 / en / english / electrum-en /
 * electrumen / electrum). Returns 1 if the token is a BIP39-English wildcard. */
static int is_bip39_dict(const char*t){
  static const char*al[]={"[:bip39-en:]","[:bip39en:]","[:bip39:]","[:en:]","[:english:]",
                          "[:electrum-en:]","[:electrumen:]","[:electrum:]",
                          "[:12th:]","[:15th:]","[:18th:]","[:21st:]","[:24th:]",0};
  for(int i=0;al[i];i++) if(!strcmp(t,al[i])) return 1;
  return 0;
}
/* Extract the alternatives from a permutation wordset pattern
 * "((w0|w1|..) ){{N!..}}" or "( w0| w1|..){{N!}}" into a space-joined word list.
 * Everything before "{{" is stripped of ()/spaces and split on '|'. */
static void pattern_to_words(const char*pat,char*out,int max){
  char buf[2048]; snprintf(buf,sizeof buf,"%s",pat);
  char*bb=strstr(buf,"{{"); if(bb) *bb=0;              /* drop the {{..}} */
  int L=0; char*p=buf;
  while(*p){
    if(*p=='('||*p==')'||*p==' '||*p=='\t'){ p++; continue; }
    /* read a word token up to | ( ) space */
    char w[32]; int n=0;
    while(*p && *p!='|' && *p!='(' && *p!=')' && *p!=' ' && *p!='\t' && n<31) w[n++]=*p++;
    w[n]=0; if(n) L+=snprintf(out+L,max-L,"%s%s",L?" ":"",w);
    if(*p=='|') p++;
  }
}
/* Parse "w0 w1 [:bip39-en:] .. [:bip39-en:]" -> tmpl[W] word indices (unknown=0),
 * upos[U] unknown positions, W, U, last_unknown. */
static int parse_template(const char*tpl,uint32_t tmpl[32],int upos[32],int*W,int*U,int*last_unknown){
  static char names[2048][16]; load_wordlist(names);
  char buf[2048]; snprintf(buf,sizeof buf,"%s",tpl); *W=0; *U=0;
  char*tok=strtok(buf," \t\r\n");
  while(tok){ if(*W>=32){fprintf(stderr,"max 32 positions\n");return -1;}
    if(is_bip39_dict(tok)){ tmpl[*W]=0; upos[(*U)++]=*W; }
    else { int bi=bip39_index(names,tok); if(bi<0){fprintf(stderr,"'%s' is not a BIP39 word\n",tok);return -1;} tmpl[*W]=(uint32_t)bi; }
    (*W)++; tok=strtok(0," \t\r\n"); }
  int vc[6]={12,15,18,21,24,0}; int ok=0; for(int i=0;vc[i];i++) if(*W==vc[i]) ok=1;
  if(!ok){ fprintf(stderr,"word count %d not in {12,15,18,21,24}\n",*W); return -1; }
  if(*U<1){ fprintf(stderr,"template has no [:bip39-en:] unknown\n"); return -1; }
  *last_unknown = (upos[*U-1]==*W-1);
  return 0;
}
static const char* WLNAME(int idx){ static char names[2048][16]; static int loaded=0; if(!loaded){load_wordlist(names);loaded=1;} return names[idx]; }
static int mode_missing(const char*tpl,const uint8_t tprog[32],int purpose,uint32_t changes,uint32_t gap,
                        int nthmode,unsigned long long ustart,unsigned long long ucount,const char*cu){
  uint32_t tmpl[32]; int upos[32],W,U,last_unknown;
  if(parse_template(tpl,tmpl,upos,&W,&U,&last_unknown)) return 2;
  int CS=W/3, freebits=11-CS;
  int nth;   /* -1 auto (nth iff last unknown), 1 force, 0 off */
  if(nthmode==1){ if(!last_unknown){ fprintf(stderr,"--nth needs the LAST position to be [:bip39-en:]\n"); return 2; } nth=1; }
  else if(nthmode==0) nth=0;
  else nth = last_unknown;
  build_module(cu);
  CUdeviceptr dwl,dwoff,dwlen; gpu_upload_wordlist(&dwl,&dwoff,&dwlen);
  CUdeviceptr dtmpl=up(tmpl,W*sizeof(uint32_t)), dtp=up(tprog,32);
  /* unknown positions passed to the kernel: baseline=all U; nth=others (exclude last W-1) */
  int kpos[32],kU; unsigned long long total=1;
  if(nth){ kU=0; for(int i=0;i<U;i++) if(upos[i]!=W-1) kpos[kU++]=upos[i];
    for(int i=0;i<kU;i++) total*=2048ULL;
    total <<= freebits; }
  else { kU=U; for(int i=0;i<U;i++) kpos[i]=upos[i]; for(int i=0;i<U;i++) total*=2048ULL; }
  CUdeviceptr dupos=up(kpos, (kU?kU:1)*sizeof(int));
  unsigned long long start=ustart>total?total:ustart;
  unsigned long long count=ucount?ucount:(total-start); if(start+count>total) count=total-start;
  unsigned long long init=~0ULL; int zero=0;
  CUdeviceptr dhi=up(&init,8),dfound=up(&zero,4),dhg=up(0,W*sizeof(uint32_t));
  uint32_t pu=(uint32_t)purpose; int passU=U;
  unsigned long long z0=0; CUdeviceptr dhashed=up(&z0,8);
  uint32_t hci0[2]={0,0}; CUdeviceptr dhit_ci=up(hci0,8);
  unsigned long long cstart=start,ccount=0;
  void*args[]={&dwl,&dwoff,&dwlen,&dtmpl,&W,&dupos,&passU,&cstart,&ccount,&pu,&changes,&gap,&dtp,&dhi,&dfound,&dhg,&dhashed,&dhit_ci};
  int tpb=128,grid=1024, reporting=(g_pflag||g_loginterval_ms);
  char path[64]; snprintf(path,sizeof path,"m/%d'/0'/0'/[0,%u)/[0,%u)",purpose,changes,gap);
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  prog_hdr(&P, nth?"missing-word [:Nth:]":"missing-word baseline", g_target_str,g_pattern_str,path,0,0);
  double intv=prog_intv(&P);
  fprintf(stderr,"missing-word %s: W=%d, %d unknown(s)%s -> %llu candidates (%s)...\n",
          nth?"[:Nth:] CONSTRUCTION":"baseline sieve", W, U, nth?" (last=checksum-constructed)":"", count, path);
  int found=0; unsigned long long swept=0,hashed=0, chunk=reporting?report_chunk(1):(64ULL<<20); double tt0=now_s();
  for(cstart=start; cstart<start+count; ){
    ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
    CU(cuLaunchKernel(kern(nth?"g_crack_nth":"g_crack_missing"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    if(nth) hashed=swept; else CU(cuMemcpyDtoH(&hashed,dhashed,8));
    prog_tick(&P,swept,hashed);
    CU(cuMemcpyDtoH(&found,dfound,4)); if(found) break;
    chunk=next_chunk(ccount,csecs,intv,reporting);
  }
  double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.3fs = %.3f Mcand/s (%s%s)\n",swept,secs,swept/secs/1e6,
          nth?"all valid, no sieve":"sieve->PBKDF2+EC on survivors", swept<count?", early-exit":"");
  unsigned long long hidx=0; CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND\n"); return 1; }
  uint32_t hg[32]; CU(cuMemcpyDtoH(hg,dhg,W*sizeof(uint32_t)));
  /* nth uses a constructed subspace; also report the librxe full-space rank so it
     cross-checks against --rank / (re)seed39 (baseline hidx already equals it). */
  unsigned long long librxe_rank = nth ? ((hidx>>freebits)*2048ULL + hg[W-1]) : hidx;
  printf("FOUND\n  index    : %llu%s\n",hidx, nth?" (nth-space)":" (== librxe rank)");
  if(nth) printf("  librxe rank: %llu\n",librxe_rank);
  printf("  found words:");
  for(int i=0;i<U;i++) printf(" [pos %d]=%s",upos[i],WLNAME((int)hg[upos[i]]));
  printf("\n  mnemonic :");
  for(int p=0;p<W;p++) printf(" %s",WLNAME((int)hg[p]));
  uint32_t hci[2]; CU(cuMemcpyDtoH(hci,dhit_ci,8));
  char fpath[80]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'/%u/%u",purpose,hci[0],hci[1]);
  printf("\n  path     : %s\n",fpath);
  { char fw[256]; int L=0; for(int i=0;i<U;i++) L+=snprintf(fw+L,sizeof fw-L,"%s%s",i?" ":"",WLNAME((int)hg[upos[i]])); prog_finish(&P,"FOUND",fw,fpath); }
  (void)CS; return 0;
}

/* --------------------------- EC gate (vs oracle) ------------------------ */
static int mode_ec_gate(const char*vecfile,const char*cu){
  build_module(cu);
  FILE*fp=fopen(vecfile,"rb"); if(!fp){fprintf(stderr,"open %s (run: node gate/gen_ec.js)\n",vecfile);return 2;}
  int cap=0,n=0; uint8_t *sk=0,*epub=0,*eh=0,*e44=0,*e49=0;
  char*line=0; size_t lc=0; ssize_t rd; char a[200],b[200],c[200],d[200],e[200];
  while((rd=getline(&line,&lc,fp))>0){
    if(sscanf(line,"%199s %199s %199s %199s %199s",a,b,c,d,e)!=5) continue;
    if(n==cap){cap=cap?cap*2:64; sk=realloc(sk,cap*32);epub=realloc(epub,cap*33);eh=realloc(eh,cap*20);e44=realloc(e44,cap*20);e49=realloc(e49,cap*20);}
    hex2bin(a,sk+(size_t)n*32,32); hex2bin(b,epub+(size_t)n*33,33); hex2bin(c,eh+(size_t)n*20,20);
    hex2bin(d,e44+(size_t)n*20,20); hex2bin(e,e49+(size_t)n*20,20); n++;
  }
  free(line); fclose(fp);
  CUdeviceptr dsk=up(sk,(size_t)n*32),dpub,dh,d44,d49;
  CU(cuMemAlloc(&dpub,(size_t)n*33)); CU(cuMemAlloc(&dh,(size_t)n*20)); CU(cuMemAlloc(&d44,(size_t)n*20)); CU(cuMemAlloc(&d49,(size_t)n*20));
  void*args[]={&dsk,&n,&dpub,&dh,&d44,&d49};
  int tpb=64,grid=(n+tpb-1)/tpb; CU(cuLaunchKernel(kern("g_ec"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  uint8_t*gpub=malloc((size_t)n*33),*gh=malloc((size_t)n*20),*g44=malloc((size_t)n*20),*g49=malloc((size_t)n*20);
  CU(cuMemcpyDtoH(gpub,dpub,(size_t)n*33)); CU(cuMemcpyDtoH(gh,dh,(size_t)n*20));
  CU(cuMemcpyDtoH(g44,d44,(size_t)n*20)); CU(cuMemcpyDtoH(g49,d49,(size_t)n*20));
  int bpub=0,bh=0,b44=0,b49=0;
  for(int i=0;i<n;i++){
    if(memcmp(gpub+(size_t)i*33,epub+(size_t)i*33,33)){ if(bpub<3){char x[70],y[70];tohex_(gpub+(size_t)i*33,33,x);tohex_(epub+(size_t)i*33,33,y);fprintf(stderr,"  pub #%d\n    gpu %s\n    ora %s\n",i,x,y);} bpub++; }
    if(memcmp(gh+(size_t)i*20,eh+(size_t)i*20,20)) bh++;
    if(memcmp(g44+(size_t)i*20,e44+(size_t)i*20,20)) b44++;
    if(memcmp(g49+(size_t)i*20,e49+(size_t)i*20,20)) b49++;
  }
  printf("  [%s] secp256k1 privToPub : %d/%d\n", bpub?"FAIL":"PASS", n-bpub,n);
  printf("  [%s] hash160(pub33)      : %d/%d\n", bh?"FAIL":"PASS", n-bh,n);
  printf("  [%s] p2pkh program (44)  : %d/%d\n", b44?"FAIL":"PASS", n-b44,n);
  printf("  [%s] p2sh-p2wpkh (49)    : %d/%d\n", b49?"FAIL":"PASS", n-b49,n);
  int bad=bpub+bh+b44+b49;
  printf("  ==== EC gate %s (%d vectors) ====\n", bad?"FAILED":"PASSED", n);
  return bad?1:0;
}

/* ----------------------- seed->address gate (vs oracle) ----------------- */
static int mode_addr_gate(const char*vecfile,const char*cu){
  build_module(cu);
  FILE*fp=fopen(vecfile,"rb"); if(!fp){fprintf(stderr,"open %s (run: node gate/gen_addr.js)\n",vecfile);return 2;}
  int cap=0,n=0; uint8_t *seed=0,*eprog=0; uint32_t *pur=0,*chg=0,*idx=0;
  char*line=0; size_t lc=0; ssize_t rd; char sh[300],ph[80]; unsigned int P,CH,IX;
  while((rd=getline(&line,&lc,fp))>0){
    if(sscanf(line,"%299s %u %u %u %79s",sh,&P,&CH,&IX,ph)!=5) continue;
    if(n==cap){cap=cap?cap*2:64; seed=realloc(seed,(size_t)cap*64);eprog=realloc(eprog,(size_t)cap*32);pur=realloc(pur,cap*4);chg=realloc(chg,cap*4);idx=realloc(idx,cap*4);}
    hex2bin(sh,seed+(size_t)n*64,64); pur[n]=P; chg[n]=CH; idx[n]=IX; hex2bin(ph,eprog+(size_t)n*32,32); n++;
  }
  free(line); fclose(fp);
  CUdeviceptr dseed=up(seed,(size_t)n*64),dpur=up(pur,n*4),dchg=up(chg,n*4),didx=up(idx,n*4),dprog;
  CU(cuMemAlloc(&dprog,(size_t)n*32));
  void*args[]={&dseed,&dpur,&dchg,&didx,&n,&dprog};
  int tpb=64,grid=(n+tpb-1)/tpb; CU(cuLaunchKernel(kern("g_addr"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  uint8_t*gp=malloc((size_t)n*32); CU(cuMemcpyDtoH(gp,dprog,(size_t)n*32));
  int bad=0,b[4]={0},c[4]={0}; const int PI[4]={44,49,84,86};
  for(int i=0;i<n;i++){ int pl=(pur[i]==86)?32:20; int mm=memcmp(gp+(size_t)i*32,eprog+(size_t)i*32,pl)!=0;
    int k=pur[i]==44?0:pur[i]==49?1:pur[i]==84?2:3; c[k]++; if(mm)b[k]++;
    if(mm){ if(bad<3){char x[66],y[66];tohex_(gp+(size_t)i*32,pl,x);tohex_(eprog+(size_t)i*32,pl,y);fprintf(stderr,"  addr #%d p%u\n    gpu %s\n    ora %s\n",i,pur[i],x,y);} bad++; }
  }
  printf("  [%s] seed->p2pkh(44)      : %d/%d\n",b[0]?"FAIL":"PASS",c[0]-b[0],c[0]);
  printf("  [%s] seed->p2sh-p2wpkh(49): %d/%d\n",b[1]?"FAIL":"PASS",c[1]-b[1],c[1]);
  printf("  [%s] seed->p2wpkh(84)     : %d/%d\n",b[2]?"FAIL":"PASS",c[2]-b[2],c[2]);
  printf("  [%s] seed->p2tr(86)       : %d/%d\n",b[3]?"FAIL":"PASS",c[3]-b[3],c[3]);
  printf("  ==== seed->address gate %s (%d vectors, incl. non-hardened ckd + BIP86 TapTweak) ====\n",bad?"FAILED":"PASSED",n);
  (void)PI; return bad?1:0;
}

/* ----------------------- Regime A: passphrase crack --------------------- */
/* parse "[0-9]{N}" -> N (decimal width). Returns -1 on unsupported grammar. */
static int passphrase_width(const char*pat){
  int w=0; if(sscanf(pat,"[0-9]{%d}",&w)==1 && w>0 && w<=18) return w; return -1;
}
static int mode_crack_pass(const char*mnemonic,int pwidth,const uint8_t tprog[32],
                           int purpose,uint32_t changes,uint32_t gap,
                           unsigned long long ustart,unsigned long long ucount,const char*cu){
  build_module(cu);
  int mnlen=(int)strlen(mnemonic);
  CUdeviceptr dmn=up(mnemonic,mnlen), dtp=up(tprog,32);
  /* precompute the fixed-mnemonic HMAC key context once (regime A) */
  CUdeviceptr dhctx; CU(cuMemAlloc(&dhctx,16*sizeof(unsigned long long)));
  { void*ia[]={&dmn,&mnlen,&dhctx}; CU(cuLaunchKernel(kern("g_hctx_init"),1,1,1,1,1,1,0,0,ia,0)); CU(cuCtxSynchronize()); }
  unsigned long long total=1; for(int i=0;i<pwidth;i++) total*=10ULL;
  unsigned long long start=ustart>total?total:ustart;
  unsigned long long count=ucount?ucount:(total-start); if(start+count>total) count=total-start;
  unsigned long long init=~0ULL; int zero=0;
  CUdeviceptr dhi=up(&init,8),dfound=up(&zero,4);
  uint32_t pu=(uint32_t)purpose;
  unsigned long long cstart=start,ccount=0;
  uint32_t hci0[2]={0,0}; CUdeviceptr dhit_ci=up(hci0,8);
  void*args[]={&dhctx,&pwidth,&cstart,&ccount,&pu,&changes,&gap,&dtp,&dhi,&dfound,&dhit_ci};
  int tpb=128,grid=1024, reporting=(g_pflag||g_loginterval_ms);
  unsigned long long chunk=reporting?report_chunk(1):(64ULL<<20); if(chunk==0)chunk=1;
  char path[64]; snprintf(path,sizeof path,"m/%d'/0'/0'/[0,%u)/[0,%u)",purpose,changes,gap);
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  prog_hdr(&P,"regime A (passphrase)",g_target_str,g_pattern_str,path,chunk,0);
  fprintf(stderr,"regime A: fixed mnemonic, passphrase [0-9]{%d} = %llu candidates from %llu (purpose %d)...\n",pwidth,count,start,purpose);
  int found=0; unsigned long long swept=0; double tt0=now_s();
  double intv=prog_intv(&P);
  for(cstart=start; cstart<start+count; ){
    ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk;
    double c0=now_s();
    CU(cuLaunchKernel(kern("g_crack_pass"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0;
    swept+=ccount; cstart+=ccount; prog_tick(&P,swept,swept);   /* no sieve: hashed==swept */
    CU(cuMemcpyDtoH(&found,dfound,4)); if(found) break;
    chunk=next_chunk(ccount,csecs,intv,reporting);
  }
  double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (PBKDF2+EC every candidate%s)\n",swept,secs,swept/secs/1e6,swept<count?", early-exit":"");
  unsigned long long hidx=0; CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND\n"); return 1; }
  char pass[24]; unsigned long long q=hidx; for(int p=pwidth-1;p>=0;p--){ pass[p]=(char)('0'+(int)(q%10)); q/=10; } pass[pwidth]=0;
  uint32_t hci[2]; CU(cuMemcpyDtoH(hci,dhit_ci,8));
  char fpath[80]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'/%u/%u",purpose,hci[0],hci[1]);
  prog_finish(&P,"FOUND",pass,fpath);
  printf("FOUND\n  index      : %llu\n  passphrase : %s\n  path       : %s\n",hidx,pass,fpath);
  return 0;
}

/* --------------------------------- CLI ---------------------------------- */
/* ---- --profile: static occupancy/register/spill audit of the hot kernels ---- */
static int mode_profile(const char*cu){
  build_module(cu);
  int sm=0,warp=0,maxtpm=0,maxwpsm=0,regsm=0,shsm=0,clk=0,mem=0;
  cuDeviceGetAttribute(&sm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,g_dev);
  cuDeviceGetAttribute(&warp,CU_DEVICE_ATTRIBUTE_WARP_SIZE,g_dev);
  cuDeviceGetAttribute(&maxtpm,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,g_dev);
  cuDeviceGetAttribute(&regsm,CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,g_dev);
  cuDeviceGetAttribute(&shsm,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,g_dev);
  cuDeviceGetAttribute(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,g_dev);
  cuDeviceGetAttribute(&mem,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,g_dev);
  maxwpsm=maxtpm/warp;
  printf("device: SMs=%d  warp=%d  maxThreads/SM=%d (=%d warps)  regs/SM=%d  smem/SM=%dB  clock=%.0fMHz\n",
    sm,warp,maxtpm,maxwpsm,regsm,shsm,clk/1000.0);
  const char*ks[]={"g_pbkdf2","g_pbkdf2_perm","g_crack_pass","g_crack_nth","g_crack_missing","g_addr",0};
  printf("\n%-16s %5s %6s %7s %8s   occupancy @ block size (active warps/SM, %% of max)\n",
    "kernel","regs","smem","local","maxtpb");
  printf("%-16s %5s %6s %7s %8s   %-11s %-11s %-11s %-11s\n","","","","(spill)","", "64","128","256","512");
  for(int i=0;ks[i];i++){ CUfunction fn;
    if(cuModuleGetFunction(&fn,g_mod,ks[i])!=CUDA_SUCCESS){ printf("%-16s (not present)\n",ks[i]); continue; }
    int regs=0,sh=0,loc=0,maxtpb=0;
    cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,fn);
    cuFuncGetAttribute(&sh,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn);
    cuFuncGetAttribute(&loc,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn);
    cuFuncGetAttribute(&maxtpb,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,fn);
    printf("%-16s %5d %6d %7d %8d  ",ks[i],regs,sh,loc,maxtpb);
    int bs[4]={64,128,256,512};
    for(int b=0;b<4;b++){ int nb=0; cuOccupancyMaxActiveBlocksPerMultiprocessor(&nb,fn,bs[b],0);
      int aw=nb*bs[b]/warp; double pct=maxwpsm?100.0*aw/maxwpsm:0;
      printf(" %2dw %3.0f%%   ",aw,pct); }
    printf("\n");
  }
  printf("\n(local>0 = register spill to local memory -> a real optimization target.\n"
         " low occupancy on a latency-bound serial hash = headroom via fewer regs / more ILP.)\n");
  return 0;
}
static void usage(void){
  fprintf(stderr,
   "bip39rxcrack -- CUDA BIP39 seed cracker (GPU self-enumerate)\n"
   "usage: bip39rxcrack <input> <target> [scan] [engine] [progress]\n"
   "\n"
   "INPUT (choose one):\n"
   "  --pattern PAT           raw librxe mnemonic pattern (== reseed39 wp):\n"
   "                            \"((w0|w1|..) ){{N!}}\"  -> known words, unknown order\n"
   "                            \"w0 w1 [:bip39:] .. [:bip39:]\"  -> missing word(s)\n"
   "  --words \"w1 .. wN\"      N known distinct words, unknown order ({{N!}})\n"
   "  --template \"w .. [:bip39:] ..\"  known words + [:bip39:]/[:en:]/[:24th:] wildcards\n"
   "  --mnemonic \"w .. w\" --passphrase [0-9]{N}   fixed mnemonic, unknown PIN\n"
   "  ([:Nth:] last-word checksum construction auto-applies; --nth/--no-nth to force)\n"
   "\n"
   "TARGET (choose one):\n"
   "  --address ADDR          base58 (1../3..) or bech32 (bc1q p2wpkh / bc1p p2tr)\n"
   "  --xpub XPUB             account extended pubkey (EC-free chaincode compare)\n"
   "  --target-chaincode HEX  32-byte account chain code directly\n"
   "  --purpose 44,49,84,86   BIP purpose(s) to try (default 84; auto from address)\n"
   "\n"
   "SCAN (address target):\n"
   "  --gap N                 receive indices 0..N-1  (default 1 = first address)\n"
   "  --change N              change chains  0..N-1    (default 1 = external only)\n"
   "  --no-checksum           disable the BIP39 checksum sieve\n"
   "\n"
   "ENGINE:\n"
   "  --compact / --no-compact   dense-survivor path (default on; words+address+sieve)\n"
   "  --start IDX --count N    sweep a fixed index slice (windowed benchmark)\n"
   "  --limit N               alias of --count\n"
   "  --kernels PATH          crack_kernels.cu (default cuda/crack_kernels.cu)\n"
   "  env: COMPACT_BUDGET_MB (survivor buffer, default 1024), CRACK_NOCACHE=1 (recompile)\n"
   "\n"
   "PROGRESS:\n"
   "  -p [SECS]               live progress to stderr every SECS (default 1.0)\n"
   "  --loginterval MS[:FILE] CSV log every MS ms (to stderr, or FILE); columns\n"
   "                            t_ms,swept,swept_total,pct,rate,hashed,eta_s\n"
   "                          (-p and --loginterval cadences are independent)\n"
   "  --resume LOG            reconstruct a killed run from its CSV LOG (header\n"
   "                          carries every param) and continue from the last\n"
   "                          swept row, appending to the same LOG\n"
   "\n"
   "GATES/UTIL:\n"
   "  --ec-gate [F] --addr-gate [F] --recon-gate [N] --miss-gate [N] --dump-valid N --rank \"..\" --decode ADDR\n"
   "  (--miss-gate: prove missing-word/[:Nth:] unrank index == librxe canonical rank)\n");
}
int main(int argc,char**argv){
  /* NVRTC 13 dlopens libnvrtc-builtins.so.13.x via LD_LIBRARY_PATH; ensure it's
     set (re-exec once so the loader picks it up at startup). */
  { const char *nl="/usr/local/cuda-13.2/lib64"; const char *cur=getenv("LD_LIBRARY_PATH");
    if(!cur || !strstr(cur,nl)){ char buf[4096]; snprintf(buf,sizeof buf,"%s%s%s",nl,cur?":":"",cur?cur:"");
      setenv("LD_LIBRARY_PATH",buf,1); execv("/proc/self/exe",argv); /* falls through on failure */ } }
  const char *words=0,*xpub=0,*tcc_hex=0,*rankarg=0,*cu="cuda/crack_kernels.cu";
  int recon=0,recon_n=512,dumpv=0,dumpv_n=0,require_ck=1,compact=1; const char*ecgate=0,*addrgate=0;
  int missgate=0,missgate_n=64,profile=0;
  unsigned long long cstart=0,ccount=0;
  uint32_t purposes[8]={84}; int npurp=1,purpose_set=0;
  const char *mnemonic=0,*passphrase=0,*address=0,*decodearg=0,*templ=0,*patt=0; uint32_t a_changes=1,a_gap=1; int nthmode=-1; /* -1 auto, 1 force, 0 off */
  const char *resumearg=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--words")&&i+1<argc) words=argv[++i];
    else if(!strcmp(argv[i],"--xpub")&&i+1<argc) xpub=argv[++i];
    else if(!strcmp(argv[i],"--target-chaincode")&&i+1<argc) tcc_hex=argv[++i];
    else if(!strcmp(argv[i],"--purpose")&&i+1<argc){ npurp=0; purpose_set=1; char*s=strtok(argv[++i],","); while(s&&npurp<8){purposes[npurp++]=(uint32_t)atoi(s);s=strtok(0,",");} }
    else if(!strcmp(argv[i],"--no-checksum")) require_ck=0;
    else if(!strcmp(argv[i],"--compact")) compact=1;
    else if(!strcmp(argv[i],"--no-compact")) compact=0;
    else if(!strcmp(argv[i],"-p")){ g_pflag=1; if(i+1<argc && (isdigit((unsigned char)argv[i+1][0])||argv[i+1][0]=='.')) g_p_secs=atof(argv[++i]); }
    else if(!strcmp(argv[i],"--loginterval")&&i+1<argc){ char*a=argv[++i]; char*colon=strchr(a,':');
      if(colon){ *colon=0; g_loginterval_ms=atoi(a); g_csv=fopen(colon+1,"w"); g_csv_own=1; if(!g_csv){fprintf(stderr,"cannot open %s\n",colon+1);return 2;} }
      else { g_loginterval_ms=atoi(a); g_csv=stderr; } }
    else if(!strcmp(argv[i],"--recon-gate")){ recon=1; if(i+1<argc&&argv[i+1][0]!='-') recon_n=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--miss-gate")){ missgate=1; if(i+1<argc&&isdigit((unsigned char)argv[i+1][0])) missgate_n=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--profile")) profile=1;
    else if(!strcmp(argv[i],"--rank")&&i+1<argc) rankarg=argv[++i];
    else if(!strcmp(argv[i],"--dump-valid")&&i+1<argc){ dumpv=1; dumpv_n=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--start")&&i+1<argc) cstart=strtoull(argv[++i],0,10);
    else if((!strcmp(argv[i],"--count")||!strcmp(argv[i],"--limit"))&&i+1<argc) ccount=strtoull(argv[++i],0,10);
    else if(!strcmp(argv[i],"--ec-gate")){ ecgate="vectors/vec_ec.txt"; if(i+1<argc&&strncmp(argv[i+1],"-",1)!=0) ecgate=argv[++i]; }
    else if(!strcmp(argv[i],"--addr-gate")){ addrgate="vectors/vec_addr.txt"; if(i+1<argc&&strncmp(argv[i+1],"-",1)!=0) addrgate=argv[++i]; }
    else if(!strcmp(argv[i],"--mnemonic")&&i+1<argc) mnemonic=argv[++i];
    else if(!strcmp(argv[i],"--passphrase")&&i+1<argc) passphrase=argv[++i];
    else if(!strcmp(argv[i],"--address")&&i+1<argc) address=argv[++i];
    else if(!strcmp(argv[i],"--template")&&i+1<argc) templ=argv[++i];
    else if(!strcmp(argv[i],"--pattern")&&i+1<argc) patt=argv[++i];
    else if(!strcmp(argv[i],"--nth")) nthmode=1;
    else if(!strcmp(argv[i],"--no-nth")) nthmode=0;
    else if(!strcmp(argv[i],"--decode")&&i+1<argc) decodearg=argv[++i];
    else if(!strcmp(argv[i],"--change")&&i+1<argc) a_changes=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--gap")&&i+1<argc) a_gap=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--kernels")&&i+1<argc) cu=argv[++i];
    else if(!strcmp(argv[i],"--resume")&&i+1<argc) resumearg=argv[++i];
    else { fprintf(stderr,"unknown arg: %s\n",argv[i]); usage(); return 2; }
  }
  /* --resume LOG: reconstruct the run from a saved CSV log and continue from
     the last swept position. The log header carries every parameter; the last
     data row says how far it got (columns are GLOBAL, so re-resume also works). */
  if(resumearg){
    Resume R; if(resume_load(resumearg,&R)) return 2;
    if(R.outcome_found){ printf("--resume: log already reports FOUND: %s\n",R.found); return 0; }
    if(!R.have_data){ fprintf(stderr,"--resume: no progress rows in %s (nothing to resume)\n",resumearg); return 2; }
    if(R.total && R.swept>=R.total){ printf("--resume: window already complete (%llu/%llu swept, not found)\n",R.swept,R.total); return 1; }
    /* input axis */
    if(!strcmp(R.input_kind,"pattern"))      patt=R.wp;
    else if(!strcmp(R.input_kind,"template"))templ=R.wp;
    else if(!strcmp(R.input_kind,"passphrase")){ mnemonic=R.wp; passphrase=R.pp; }
    else if(!strcmp(R.input_kind,"words"))   words=R.wp;
    else { fprintf(stderr,"--resume: unknown input_kind '%s'\n",R.input_kind); return 2; }
    /* target axis */
    if(!strcmp(R.target_kind,"address"))       address=R.target;
    else if(!strcmp(R.target_kind,"xpub"))     xpub=R.target;
    else if(!strcmp(R.target_kind,"chaincode"))tcc_hex=R.target;
    /* purpose: "auto" => re-derive from the address; else the recorded list */
    if(R.purpose[0] && strcmp(R.purpose,"auto")){ npurp=0; purpose_set=1;
      char pb[64]; snprintf(pb,sizeof pb,"%s",R.purpose); char*sp=strtok(pb,","); while(sp&&npurp<8){purposes[npurp++]=(uint32_t)atoi(sp);sp=strtok(0,",");} }
    a_changes=(uint32_t)R.change; a_gap=(uint32_t)R.gap; require_ck=R.checksum; nthmode=R.nth; compact=R.compact;
    cstart = R.start + R.swept;                 /* resume from where it left off */
    ccount = (R.total>R.swept)?(R.total-R.swept):0;
    /* report GLOBAL progress over the ORIGINAL window; continue the same log */
    g_orig_start=R.start; g_swept_base=R.swept; g_hashed_base=R.hashed; g_prog_total=R.total; g_resuming=1;
    if(!g_csv){ g_csv=fopen(resumearg,"a"); g_csv_own=1; if(!g_csv){fprintf(stderr,"--resume: cannot append %s\n",resumearg);return 2;}
      if(R.loginterval_ms>0) g_loginterval_ms=R.loginterval_ms; }
    fprintf(stderr,"--resume: %s | resuming at global index %llu (%llu/%llu, %.1f%%), %llu remaining\n",
      resumearg,cstart,R.swept,R.total,R.total?100.0*R.swept/R.total:0.0,ccount);
  }
  /* --pattern: a raw librxe mnemonic pattern (== reseed39 wp). {{..}} => words
     permutation; else a missing-word template with [:dict:] wildcards. */
  static char _wbuf[2048];
  if(patt){ if(strstr(patt,"{{")){ pattern_to_words(patt,_wbuf,sizeof _wbuf); words=_wbuf; }
            else templ=patt; }
  g_target_str = address?address:(xpub?xpub:tcc_hex);
  if(patt) g_pattern_str = patt;
  g_pattern_str = words?words:(templ?templ:passphrase);
  /* record the run's parameters for the CSV header / --resume (fresh runs) */
  if(!g_resuming){
    g_orig_start=cstart;
    g_change_v=(int)a_changes; g_gap_v=(int)a_gap; g_checksum_v=require_ck; g_nth_v=nthmode; g_compact_v=compact;
    g_input_kind = patt?"pattern":(templ?"template":((mnemonic&&passphrase)?"passphrase":(words?"words":"")));
    g_wp = patt?patt:(templ?templ:((mnemonic&&passphrase)?mnemonic:(words?words:"")));
    g_pp = (mnemonic&&passphrase)?passphrase:"";
    g_target_kind = address?"address":(xpub?"xpub":(tcc_hex?"chaincode":""));
    static char purpbuf[64];
    if(purpose_set){ int L=0; for(int i=0;i<npurp;i++) L+=snprintf(purpbuf+L,sizeof purpbuf-L,"%s%u",i?",":"",purposes[i]); g_purpose_str=purpbuf; }
    else g_purpose_str="auto";
  }
  /* gate/util modes need no pattern */
  if(decodearg){ uint8_t pr[32]; int pl,pu; if(decode_address(decodearg,pr,&pl,&pu)) return 2;
    char h[66]; tohex_(pr,pl,h); printf("%d %s\n",pu,h); return 0; }
  if(profile) return mode_profile(cu);
  if(ecgate) return mode_ec_gate(ecgate,cu);
  if(addrgate) return mode_addr_gate(addrgate,cu);
  if(missgate){ const char*t=templ?templ:"trial [:bip39:] gloom dragon try dirt rapid crawl soon fatal tool chronic rapid ladder salmon palace expect enrich helmet truth receive [:bip39:] [:bip39:] [:bip39:]"; return mode_miss_gate(t,missgate_n,cu); }
  /* Missing-word ([:bip39-en:]) template + address target */
  if(templ){
    if(!address){ fprintf(stderr,"--template needs --address\n"); return 2; }
    uint8_t prog[32]; int apl,apu; if(decode_address(address,prog,&apl,&apu)) return 2;
    int purpose = purpose_set?(int)purposes[0]:apu;
    /* auto: construction if the last position is [:bip39-en:]; --nth/--no-nth override */
    return mode_missing(templ,prog,purpose,a_changes,a_gap,nthmode,cstart,ccount,cu);
  }
  /* Regime A: fixed mnemonic + passphrase [0-9]{N} + address target */
  if(mnemonic && passphrase){
    int w=passphrase_width(passphrase);
    if(w<0){ fprintf(stderr,"v1 passphrase supports [0-9]{N} only, got '%s'\n",passphrase); return 2; }
    if(!address){ fprintf(stderr,"regime A needs --address (p2pkh/p2sh target)\n"); return 2; }
    uint8_t prog[32]; int aproglen,apurpose; if(decode_address(address,prog,&aproglen,&apurpose)) return 2;
    int purpose = purpose_set ? (int)purposes[0] : apurpose;
    return mode_crack_pass(mnemonic,w,prog,purpose,a_changes,a_gap,cstart,ccount,cu);
  }
  if(!words){ usage(); return 2; }
  Words W; parse_words(&W,words);
  if(rankarg) return mode_rank(&W,rankarg);
  if(recon)   return mode_recon_gate(&W,recon_n,cu);
  if(dumpv)   return mode_dump_valid(&W,dumpv_n,cu);
  if(address){ uint8_t prog[32]; int aproglen,apurpose; if(decode_address(address,prog,&aproglen,&apurpose))return 2;
    int purpose = purpose_set?(int)purposes[0]:apurpose;
    return mode_crack_addr(&W,prog,purpose,a_changes,a_gap,require_ck,compact,cstart,ccount,cu); }
  uint8_t tcc[32];
  if(xpub){ if(xpub_chaincode(xpub,tcc)) return 2; }
  else if(tcc_hex){ if(hex2bin(tcc_hex,tcc,32)!=32){ fprintf(stderr,"target-chaincode must be 32 bytes hex\n"); return 2; } }
  else { fprintf(stderr,"need --xpub or --target-chaincode\n"); usage(); return 2; }
  return mode_crack(&W,tcc,purposes,npurp,require_ck,cstart,ccount,cu);
}
