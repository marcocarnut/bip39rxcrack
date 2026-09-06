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
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <spawn.h>
#include <poll.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <gmp.h>
#include <cuda.h>
#include <nvrtc.h>
#include "rxe.h"
#include "../cuda/bloom_common.h"

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
static int g_device=0;        /* CUDA ordinal to run on (--device) */
static int g_print_total=0;   /* --print-total: print the job size and exit before any GPU work */
static int g_warm=0;          /* --warm: build/JIT the module (populate PTX cache) and exit */
static int g_devtag=0;        /* prefix -p progress with the device index (multi-GPU children) */
static char g_li_raw[320];    /* raw --loginterval value (survives in-place arg mutation) */
static const char *g_csv_path=0;  /* CSV file to open (deferred: supervisors don't log) */
static int g_worker=0;        /* --worker: persistent shard-servicing worker over stdin/stdout */
static char g_ptx_key[128];   /* kernels/PTX hash, reported in the worker READY handshake */
#define WQ_PROTO "wq1"        /* work-queue line-protocol version */
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
  if(g_mod) return;   /* idempotent: a persistent worker builds its context once */
  const char *arch="--gpu-architecture=compute_120";
  const char*mr=getenv("CRACK_MAXREG");
  const char*def=getenv("CRACK_DEF");   /* e.g. -DSHA512_UNROLL16 for A/B experiments */
  char *src=inline_includes(slurp(cu_path),cu_path);
  /* PTX cache: NVRTC compile of the full EC+taproot module is slow (~2-3 min);
     cache the PTX keyed by source+arch hash so unchanged source loads instantly. */
  char key[128]; snprintf(key,sizeof key,"%llx",fnv1a(src)^fnv1a(arch)^(def?fnv1a(def):0));
  snprintf(g_ptx_key,sizeof g_ptx_key,"%s",key);   /* kernels-hash for the worker handshake */
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
    /* Atomic write (tmp+rename): concurrent multi-GPU children never read a
       half-written PTX -- each sees the old-complete or the new-complete file. */
    char tpath[300]; snprintf(tpath,sizeof tpath,"%s.tmp.%d",cpath,(int)getpid());
    FILE*wf=fopen(tpath,"wb"); if(wf){ fwrite(ptx,1,strlen(ptx),wf); fclose(wf); if(rename(tpath,cpath)) unlink(tpath); }
  }
  CUdevice dev; CU(cuInit(0)); CU(cuDeviceGet(&dev,g_device)); g_dev=dev;
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
  fprintf(stderr,"device %d: %s (sm_%d%d), NVRTC13->compute_120 PTX->sm_120\n",g_device,name,M,m);
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
  if(g_devtag) fprintf(stderr,"[dev%d] %.1f/%.1fM (%.1f%%) %.2f Mc/s  hashed %.2fM  ETA %.0fs\n",
      g_device,gsw/1e6,P->total/1e6,pct,inst,ghash/1e6,eta);
  else fprintf(stderr,"\r[%7.1fs] %.1f/%.1fM (%.1f%%) %.2f Mc/s  hashed %.2fM  ETA %.0fs    ",
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

/* --------------- address ENCODE (program+purpose -> string) --------------
 * Inverse of decode_address, for reporting a found address from a prebuilt bloom
 * (which carries no strings). Needs host SHA-256 for base58check. */
#define ROR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_host(const uint8_t*msg,size_t len,uint8_t out[32]){
  static const uint32_t K[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  size_t padlen=((len+8)/64+1)*64; uint8_t*m=calloc(padlen,1); memcpy(m,msg,len); m[len]=0x80;
  uint64_t bits=(uint64_t)len*8; for(int i=0;i<8;i++) m[padlen-1-i]=(uint8_t)(bits>>(8*i));
  for(size_t off=0;off<padlen;off+=64){
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=((uint32_t)m[off+i*4]<<24)|((uint32_t)m[off+i*4+1]<<16)|((uint32_t)m[off+i*4+2]<<8)|m[off+i*4+3];
    for(int i=16;i<64;i++){ uint32_t s0=ROR32(w[i-15],7)^ROR32(w[i-15],18)^(w[i-15]>>3); uint32_t s1=ROR32(w[i-2],17)^ROR32(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(int i=0;i<64;i++){ uint32_t S1=ROR32(e,6)^ROR32(e,11)^ROR32(e,25); uint32_t ch=(e&f)^((~e)&g); uint32_t t1=hh+S1+ch+K[i]+w[i];
      uint32_t S0=ROR32(a,2)^ROR32(a,13)^ROR32(a,22); uint32_t maj=(a&b)^(a&c)^(b&c); uint32_t t2=S0+maj;
      hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
  }
  free(m); for(int i=0;i<8;i++){ out[i*4]=h[i]>>24; out[i*4+1]=h[i]>>16; out[i*4+2]=h[i]>>8; out[i*4+3]=h[i]; }
}
static void b58encode(const uint8_t*data,int len,char*out){
  int zeros=0; while(zeros<len && data[zeros]==0) zeros++;
  uint8_t tmp[256]; int tn=0; memset(tmp,0,sizeof tmp);
  for(int i=zeros;i<len;i++){ int carry=data[i];
    for(int j=0;j<tn;j++){ carry+=(int)tmp[j]<<8; tmp[j]=carry%58; carry/=58; }
    while(carry){ tmp[tn++]=carry%58; carry/=58; } }
  int oi=0; for(int i=0;i<zeros;i++) out[oi++]='1';
  for(int i=tn-1;i>=0;i--) out[oi++]=B58[tmp[i]];
  out[oi]=0;
}
static void b58check(const uint8_t*payload,int plen,char*out){
  uint8_t buf[64]; memcpy(buf,payload,plen);
  uint8_t d1[32],d2[32]; sha256_host(payload,plen,d1); sha256_host(d1,32,d2);
  memcpy(buf+plen,d2,4); b58encode(buf,plen+4,out);
}
static void bech32_encode(const char*hrp,const uint8_t*data5,int n5,int is_m,char*out){
  uint8_t values[130]; int vn=0,hlen=(int)strlen(hrp);
  for(int i=0;i<hlen;i++) values[vn++]=hrp[i]>>5;
  values[vn++]=0;
  for(int i=0;i<hlen;i++) values[vn++]=hrp[i]&31;
  for(int i=0;i<n5;i++) values[vn++]=data5[i];
  for(int i=0;i<6;i++) values[vn+i]=0;
  uint32_t pm=bech_polymod(values,vn+6)^(is_m?0x2bc830a3u:1u);
  int oi=0; for(int i=0;i<hlen;i++) out[oi++]=hrp[i]; out[oi++]='1';
  for(int i=0;i<n5;i++) out[oi++]=BECH[data5[i]];
  for(int i=0;i<6;i++) out[oi++]=BECH[(pm>>(5*(5-i)))&31];
  out[oi]=0;
}
static void segwit_encode(const char*hrp,int witver,const uint8_t*prog,int plen,char*out){
  uint8_t d5[90]; int n5=0; d5[n5++]=(uint8_t)witver; uint32_t acc=0; int bits=0;
  for(int i=0;i<plen;i++){ acc=(acc<<8)|prog[i]; bits+=8; while(bits>=5){ bits-=5; d5[n5++]=(acc>>bits)&31; } }
  if(bits) d5[n5++]=(acc<<(5-bits))&31;
  bech32_encode(hrp,d5,n5,witver>=1?1:0,out);
}
/* program (20 or 32 B) + BIP purpose -> address string. Returns 0 on success. */
static int encode_address(const uint8_t*prog,int purpose,char*out){
  if(purpose==44){ uint8_t p[21]; p[0]=0x00; memcpy(p+1,prog,20); b58check(p,21,out); return 0; }
  if(purpose==49){ uint8_t p[21]; p[0]=0x05; memcpy(p+1,prog,20); b58check(p,21,out); return 0; }
  if(purpose==84){ segwit_encode("bc",0,prog,20,out); return 0; }
  if(purpose==86){ segwit_encode("bc",1,prog,32,out); return 0; }
  return -1;
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
  if(g_print_total){ gmp_printf("%Zd\n",total); return 0; }
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
/* ---- setup-once / sweep(start,count) split (foundation for the work-queue) ----
 * A persistent worker builds the CUDA context + uploads once (crack_addr_setup),
 * then sweeps arbitrary [start,count) shards repeatedly (crack_addr_sweep) on the
 * SAME context. mode_crack_addr is just setup + one whole-window sweep + render,
 * so single-GPU behaviour stays byte-identical (gated). */
typedef void (*prog_cb)(void*ud, unsigned long long swept, unsigned long long hashed);
/* A set of target programs (bloom mode), SORTED for the exact cull. Each entry
 * keeps its address string + purpose (script type). Programs are zero-padded to
 * 32 bytes and the cull compares all 32, so a set may MIX script types (20B h160
 * + 32B taproot). `purposes` is the distinct set of script types to derive under
 * (multi-purpose). NULL/n==0 => single-target (the existing exact-compare path). */
typedef struct { uint8_t prog[32]; char *str; int purpose; } AEnt;
/* Two culling backends: `ent` (in-RAM, keeps address strings; --addresses) OR a
 * second HOST bloom `host_filter` over sha256(program) -- independent of the GPU
 * filter, so their false positives multiply (dual-bloom; prebuilt --bloom). It's
 * checked only on the rare GPU-filter hits. `prefilter` (when set) is filter 1,
 * uploaded to the GPU as-is instead of being built. */
typedef struct { AEnt *ent; long n; uint32_t purposes[8]; int npurp;
                 const uint32_t *host_filter; uint32_t host_filter_nblocks;
                 const uint32_t *prefilter; uint32_t prefilter_nblocks; } AddrSet;
static int aent_cmp(const void*a,const void*b){ return memcmp(((const AEnt*)a)->prog,((const AEnt*)b)->prog,32); }
static void aset_sort(AddrSet*A){ qsort(A->ent,(size_t)A->n,sizeof(AEnt),aent_cmp); }
static const AEnt* aset_lookup(const AddrSet*A,const uint8_t*prog){   /* only ent-backed carries strings */
  if(A->host_filter||!A->ent) return 0;
  AEnt key; memcpy(key.prog,prog,32); return (const AEnt*)bsearch(&key,A->ent,(size_t)A->n,sizeof(AEnt),aent_cmp); }
static int aset_member(const AddrSet*A,const uint8_t*prog){
  if(A->host_filter){ uint8_t h[32]; sha256_host(prog,32,h); return bloom_probe(A->host_filter,h,A->host_filter_nblocks-1); }
  return aset_lookup(A,prog)!=0; }
static void aset_free(AddrSet*A){ if(!A->ent) return; for(long i=0;i<A->n;i++) free(A->ent[i].str); free(A->ent); A->ent=0; }

typedef struct {
  struct rxe*r; unsigned long long total;
  CUdeviceptr dd,dof,dln,dix, dtp, dhi,dfound,dhashed,dhit_ci, dsurv,dctr;
  unsigned long long C;   /* survivor-buffer capacity (compact); 0 = fused */
  int n,size, require_ck, purpose; uint32_t pu,changes,gap; int grid,tpb;
  CUdeviceptr d_bloom, d_hits, d_hitcnt, d_purposes; uint32_t bloom_mask, hitcap; int bnpurp;  /* bloom mode */
  const AddrSet *aset;    /* exact cull set; 0 => single-target */
} CrackCtx;

static int crack_addr_setup(CrackCtx*X,const Words*W,const uint8_t tprog[32],int purpose,
                            uint32_t changes,uint32_t gap,int require_ck,int compact,const char*cu,
                            const AddrSet*aset){
  memset(X,0,sizeof *X);
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  X->r=rxe_parse(pat,0); if(!X->r||rxe_error(X->r)){fprintf(stderr,"rxe parse err\n");return 2;}
  if(W->n>20){ fprintf(stderr,"v1 self-enumerate hit-index is u64: max 20 words. Got %d.\n",W->n); return 2; }
  { mpz_t total; mpz_init(total); mpz_set(total,X->r->nitems); X->total=mpz_get_ui(total); mpz_clear(total); }
  build_module(cu);
  gpu_upload_words(W,&X->dd,&X->dof,&X->dln,&X->dix);
  X->dtp=up(tprog,32);
  unsigned long long init=~0ULL,z0=0; int zero=0; uint32_t hci0[2]={0,0};
  X->dhi=up(&init,8); X->dfound=up(&zero,4); X->dhashed=up(&z0,8); X->dhit_ci=up(hci0,8);
  X->n=W->n; X->size=W->n; X->pu=(uint32_t)purpose; X->changes=changes; X->gap=gap;
  X->require_ck=require_ck; X->purpose=purpose; X->grid=1024; X->tpb=128;
  /* survivor buffer (chunked compaction): sized for a chunk's FULL worst case so
     overflow is structurally impossible. Allocated ONCE here, reused per sweep.
     Used by both single-target and bloom (g_pbkdf2_perm_bloom on the survivors). */
  if(compact && require_ck){
    unsigned long long budget=1024ULL*1024*1024;
    const char*mb=getenv("COMPACT_BUDGET_MB"); if(mb){ long v=atol(mb); if(v>16) budget=(unsigned long long)v*1024*1024; }
    unsigned long long C=budget/8; CUdeviceptr dsurv=0;
    while(C>=(1ULL<<20)){ if(cuMemAlloc(&dsurv,C*8)==CUDA_SUCCESS) break; C/=2; dsurv=0; }
    if(dsurv){ X->dsurv=dsurv; X->C=C; X->dctr=up(&z0,8); }
    else fprintf(stderr,"regime B: compaction buffer won't allocate on this GPU -- falling back to fused.\n");
  }
  /* bloom target set: build the blocked filter on the host, upload it, and alloc
     the GPU hit buffer. The fused *_bloom kernel probes + appends; the host culls. */
  if(aset && (aset->n>0 || aset->prefilter)){
    X->aset=aset; long ntgt=aset->n; uint32_t nb; size_t fbytes;
    if(aset->prefilter){ nb=aset->prefilter_nblocks; fbytes=(size_t)nb*8u*4u; X->d_bloom=up(aset->prefilter,fbytes); }
    else { double bpk=24.0; const char*e=getenv("BLOOM_BPK"); if(e){ double v=atof(e); if(v>=4) bpk=v; }
      nb=bloom_nblocks((uint64_t)ntgt,bpk); fbytes=(size_t)nb*8u*4u;
      uint32_t *hf=calloc(fbytes,1); for(long i=0;i<aset->n;i++) bloom_insert(hf,aset->ent[i].prog,nb-1);
      X->d_bloom=up(hf,fbytes); free(hf); }
    X->bloom_mask=nb-1;
    X->hitcap=1u<<18;                                /* 262144 hits/chunk */
    CU(cuMemAlloc(&X->d_hits,(size_t)X->hitcap*sizeof(BloomHit)));
    X->d_hitcnt=up(&z0,4);
    X->bnpurp=aset->npurp; X->d_purposes=up(aset->purposes,(size_t)aset->npurp*sizeof(uint32_t));
    if(aset->prefilter)
      fprintf(stderr,"bloom: prebuilt GPU filter %.1f MiB (%u blocks), host cull filter %u blocks, %d purpose(s)\n",
              (double)fbytes/1048576.0, nb, aset->host_filter_nblocks, aset->npurp);
    else
      fprintf(stderr,"bloom: %ld target(s), filter %.1f MiB (%u blocks, %.1f bits/key), %d purpose(s), cull on host\n",
              ntgt, (double)fbytes/1048576.0, nb, ntgt?(double)nb*256.0/(double)ntgt:0.0, aset->npurp);
  }
  return 0;
}

/* Bloom sweep [start,count): fused *_bloom kernel probes each program, appends
   hits; the host culls per chunk. The FIRST chunk with a culled-true hit holds the
   lowest global index (chunks run in increasing-index order), so we take the min
   over that chunk's true hits and stop. */
static int crack_addr_sweep_bloom(CrackCtx*X,unsigned long long start,unsigned long long count,
                                  double intv,int reporting,prog_cb cb,void*ud,
                                  unsigned long long*out_idx,uint32_t out_ci[2],char*out_addr,int addrsz,int*out_purpose){
  int rck=X->require_ck, n=X->n, size=X->size, compact=(X->dsurv!=0);
  unsigned long long z0=0; CU(cuMemcpyHtoD(X->dhashed,&z0,8));
  unsigned long long cstart=start,ccount=0,swept=0,hashed=0;
  unsigned long long chunk = compact ? (reporting?(1ULL<<20):X->C) : (reporting?report_chunk(1):(64ULL<<20));
  int found=0; unsigned long long best=~0ULL; uint32_t best_ci[2]={0,0}; unsigned char best_prog[32]={0}; int best_purpose=0;
  BloomHit *hb=malloc((size_t)X->hitcap*sizeof(BloomHit));
  for(cstart=start; cstart<start+count; ){
    ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
    unsigned int zc=0; CU(cuMemcpyHtoD(X->d_hitcnt,&zc,4));
    if(compact){   /* sieve -> dense PBKDF2 on survivors -> probe (the ~20 Mc/s path) */
      unsigned long long z64=0; CU(cuMemcpyHtoD(X->dctr,&z64,8));
      void*sa[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&cstart,&ccount,&X->dsurv,&X->C,&X->dctr};
      CU(cuLaunchKernel(kern("g_sieve_perm"),X->grid,1,1,X->tpb,1,1,0,0,sa,0)); CU(cuCtxSynchronize());
      unsigned long long nsurv=0; CU(cuMemcpyDtoH(&nsurv,X->dctr,8)); hashed+=nsurv;
      if(nsurv>X->C){ fprintf(stderr,"ERROR: chunk survivors %llu > %llu (bug)\n",nsurv,X->C); free(hb); return -1; }
      if(nsurv){ void*pa[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&X->dsurv,&nsurv,&X->d_purposes,&X->bnpurp,&X->changes,&X->gap,
                            &X->d_bloom,&X->bloom_mask,&X->d_hits,&X->d_hitcnt,&X->hitcap};
        CU(cuLaunchKernel(kern("g_pbkdf2_perm_bloom"),X->grid,1,1,X->tpb,1,1,0,0,pa,0)); CU(cuCtxSynchronize()); }
    } else {       /* fused sieve+PBKDF2+EC -> probe */
      void*args[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&cstart,&ccount,&X->d_purposes,&X->bnpurp,&X->changes,&X->gap,
                   &X->d_bloom,&X->bloom_mask,&rck,&X->d_hits,&X->d_hitcnt,&X->hitcap,&X->dhashed};
      CU(cuLaunchKernel(kern("g_crack_addr_bloom"),X->grid,1,1,X->tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    }
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    unsigned int hc=0; CU(cuMemcpyDtoH(&hc,X->d_hitcnt,4));
    if(hc>0){
      unsigned int nc=hc>X->hitcap?X->hitcap:hc;
      CU(cuMemcpyDtoH(hb,X->d_hits,(size_t)nc*sizeof(BloomHit)));
      for(unsigned int i=0;i<nc;i++) if(aset_member(X->aset,hb[i].prog)){
        found=1; if(hb[i].gidx<best){ best=hb[i].gidx; best_ci[0]=hb[i].change; best_ci[1]=hb[i].index; best_purpose=(int)hb[i].purpose; memcpy(best_prog,hb[i].prog,32); } }
      if(hc>X->hitcap) fprintf(stderr,"  (bloom hit-buffer overflow %u>%u; raise it if this recurs)\n",hc,X->hitcap);
    }
    if(!compact) CU(cuMemcpyDtoH(&hashed,X->dhashed,8));
    if(cb) cb(ud,swept,rck?hashed:swept);
    if(found) break;
    chunk=next_chunk(ccount,csecs,intv,reporting); if(compact && chunk>X->C) chunk=X->C;
  }
  free(hb);
  if(found){ *out_idx=best; out_ci[0]=best_ci[0]; out_ci[1]=best_ci[1]; if(out_purpose) *out_purpose=best_purpose;
    const AEnt*me=aset_lookup(X->aset,best_prog);
    if(out_addr&&addrsz){ if(me) snprintf(out_addr,(size_t)addrsz,"%s",me->str);
      else if(encode_address(best_prog,best_purpose,out_addr)){ char hx[66]; tohex_(best_prog,best_purpose==86?32:20,hx); snprintf(out_addr,(size_t)addrsz,"program:%s",hx); } } }
  return found;
}

/* Sweep [start,count). Resets per-sweep hit state, returns 1 if found (sets
   *out_idx + out_ci[2]), 0 if not. Streams progress via cb (nullable). */
static int crack_addr_sweep(CrackCtx*X,unsigned long long start,unsigned long long count,
                            double intv,int reporting,prog_cb cb,void*ud,
                            unsigned long long*out_idx,uint32_t out_ci[2],char*out_addr,int addrsz,int*out_purpose){
  if(out_addr&&addrsz) out_addr[0]=0;
  if(X->d_bloom) return crack_addr_sweep_bloom(X,start,count,intv,reporting,cb,ud,out_idx,out_ci,out_addr,addrsz,out_purpose);
  unsigned long long init=~0ULL,z0=0; int zero=0; uint32_t hci0[2]={0,0};
  CU(cuMemcpyHtoD(X->dhi,&init,8)); CU(cuMemcpyHtoD(X->dfound,&zero,4));
  CU(cuMemcpyHtoD(X->dhashed,&z0,8)); CU(cuMemcpyHtoD(X->dhit_ci,hci0,8));
  int found=0; int rck=X->require_ck; int n=X->n,size=X->size;
  if(X->dsurv){   /* compacted: sieve -> dense PBKDF2 on survivors */
    unsigned long long z64=0, tot_surv=0, swept=0, cchunk=reporting?(1ULL<<20):X->C;
    for(unsigned long long cs=start; cs<start+count; ){
      unsigned long long cc=(start+count-cs<cchunk)?(start+count-cs):cchunk; double c0=now_s();
      CU(cuMemcpyHtoD(X->dctr,&z64,8));
      void*sa[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&cs,&cc,&X->dsurv,&X->C,&X->dctr};
      CU(cuLaunchKernel(kern("g_sieve_perm"),X->grid,1,1,X->tpb,1,1,0,0,sa,0)); CU(cuCtxSynchronize());
      unsigned long long nsurv=0; CU(cuMemcpyDtoH(&nsurv,X->dctr,8)); tot_surv+=nsurv;
      if(nsurv>X->C){ fprintf(stderr,"ERROR: chunk survivors %llu > chunk size %llu (bug)\n",nsurv,X->C); return -1; }
      if(nsurv){ void*pa[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&X->dsurv,&nsurv,&X->pu,&X->changes,&X->gap,&X->dtp,&X->dhi,&X->dfound,&X->dhit_ci};
        CU(cuLaunchKernel(kern("g_pbkdf2_perm"),X->grid,1,1,X->tpb,1,1,0,0,pa,0)); CU(cuCtxSynchronize()); }
      double csecs=now_s()-c0; swept+=cc; cs+=cc; if(cb) cb(ud,swept,tot_surv);
      CU(cuMemcpyDtoH(&found,X->dfound,4)); if(found) break;
      cchunk=next_chunk(cc,csecs,intv,reporting); if(cchunk>X->C) cchunk=X->C;
    }
  } else {        /* fused: sieve+PBKDF2+EC in one kernel */
    unsigned long long cstart=start,ccount=0,swept=0,hashed=0, chunk=reporting?report_chunk(1):(64ULL<<20);
    void*args[]={&X->dd,&X->dof,&X->dln,&X->dix,&n,&size,&cstart,&ccount,&X->pu,&X->changes,&X->gap,&X->dtp,&rck,&X->dhi,&X->dfound,&X->dhashed,&X->dhit_ci};
    for(cstart=start; cstart<start+count; ){
      ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
      CU(cuLaunchKernel(kern("g_crack_addr"),X->grid,1,1,X->tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
      double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
      CU(cuMemcpyDtoH(&hashed,X->dhashed,8)); if(cb) cb(ud,swept,rck?hashed:swept);
      CU(cuMemcpyDtoH(&found,X->dfound,4)); if(found) break;
      chunk=next_chunk(ccount,csecs,intv,reporting);
    }
  }
  CU(cuMemcpyDtoH(&found,X->dfound,4));
  if(found){ CU(cuMemcpyDtoH(out_idx,X->dhi,8)); CU(cuMemcpyDtoH(out_ci,X->dhit_ci,8)); }
  return found;
}

/* progress callback for the single-GPU mode path: drives the CSV/-p machinery */
static void mode_prog_cb(void*ud,unsigned long long swept,unsigned long long hashed){ prog_tick((Prog*)ud,swept,hashed); }

static int mode_crack_addr(const Words*W,const uint8_t tprog[32],int purpose,
                           uint32_t changes,uint32_t gap,int require_ck,int compact,
                           unsigned long long ustart,unsigned long long ucount,const char*cu,
                           const AddrSet*aset){
  if(g_print_total){ char pat[2048]; wordset_pattern(W,pat,sizeof pat);
    struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse err\n");return 2;}
    if(W->n>20){ fprintf(stderr,"v1 self-enumerate hit-index is u64: max 20 words. Got %d.\n",W->n); return 2; }
    gmp_printf("%Zd\n",r->nitems); rxe_free(r); return 0; }
  CrackCtx X; int rc=crack_addr_setup(&X,W,tprog,purpose,changes,gap,require_ck,compact,cu,aset); if(rc) return rc;
  unsigned long long start=ustart>X.total?X.total:ustart;
  unsigned long long count=ucount?ucount:(X.total-start); if(start+count>X.total) count=X.total-start;
  int reporting=(g_pflag||g_loginterval_ms);
  char path[64]; snprintf(path,sizeof path,"m/%d'/0'/0'/[0,%u)/[0,%u)",purpose,changes,gap);
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  double intv=prog_intv(&P);
  if(X.dsurv){ prog_hdr(&P,"regime B (words, compacted)",g_target_str,g_pattern_str,path,X.C,X.C*8/1024/1024);
    fprintf(stderr,"regime B (COMPACTED, chunked): %llu candidates, buffer %llu MiB -> %s ...\n",count,X.C*8/1024/1024,path); }
  else { prog_hdr(&P,require_ck?"regime B (words, fused)":"regime B (words, no-sieve)",g_target_str,g_pattern_str,path,0,0);
    fprintf(stderr,"regime B: %llu permutations (checksum-%s) -> %s ...\n",count,require_ck?"ON":"OFF",path); }
  double tt0=now_s(); unsigned long long hidx=0; uint32_t hci[2]={0,0}; char maddr[100]=""; int mpurpose=purpose;
  int found=crack_addr_sweep(&X,start,count,intv,reporting,mode_prog_cb,&P,&hidx,hci,maddr,sizeof maddr,&mpurpose);
  if(found<0) return 2;
  double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (%s%s)\n",
    P.swept,secs,secs>0?P.swept/secs/1e6:0.0,
    X.dsurv?"compacted->dense PBKDF2":"fused sieve->PBKDF2+EC", (found&&P.swept<count)?", early-exit":"");
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND\n"); return 1; }
  mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(X.r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,X.r);
  char*t=buf; while(*t==' ')t++;
  char fpath[80]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'/%u/%u",mpurpose,hci[0],hci[1]);
  prog_finish(&P,"FOUND",t,fpath);
  printf("FOUND\n  index (canonical): %llu\n  mnemonic: %s\n  path    : %s\n",hidx,t,fpath);
  if(maddr[0]) printf("  address : %s\n",maddr);
  mpz_clear(j); return 0;
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
/* setup-once / sweep split for the missing-word ([:Nth:]) + address path, mirroring
   the crack_addr_* pair. The [:Nth:] subspace uses nth-space indices; the librxe
   full-space rank is derived at report time. */
typedef struct {
  CUdeviceptr dwl,dwoff,dwlen,dtmpl,dupos,dtp,dhi,dfound,dhg,dhashed,dhit_ci;
  int W,U,passU,nth,freebits, upos[32]; uint32_t pu,changes,gap; int grid,tpb;
  unsigned long long total;
  CUdeviceptr d_bloom, d_hits, d_hitcnt, d_purposes; uint32_t bloom_mask, hitcap; int bnpurp;  /* bloom mode */
  const AddrSet *aset;
} MissCtx;
static int crack_missing_setup(MissCtx*M,const char*tpl,const uint8_t tprog[32],int purpose,
                               uint32_t changes,uint32_t gap,int nthmode,const char*cu,
                               const AddrSet*aset){
  memset(M,0,sizeof *M);
  uint32_t tmpl[32]; int upos[32],W,U,last_unknown;
  if(parse_template(tpl,tmpl,upos,&W,&U,&last_unknown)) return 2;
  int CS=W/3, freebits=11-CS; (void)CS;
  int nth;
  if(nthmode==1){ if(!last_unknown){ fprintf(stderr,"--nth needs the LAST position to be [:bip39-en:]\n"); return 2; } nth=1; }
  else if(nthmode==0) nth=0; else nth=last_unknown;
  int kpos[32],kU; unsigned long long total=1;
  if(nth){ kU=0; for(int i=0;i<U;i++) if(upos[i]!=W-1) kpos[kU++]=upos[i]; for(int i=0;i<kU;i++) total*=2048ULL; total<<=freebits; }
  else { kU=U; for(int i=0;i<U;i++) kpos[i]=upos[i]; for(int i=0;i<U;i++) total*=2048ULL; }
  M->total=total; M->W=W; M->U=U; M->passU=U; M->nth=nth; M->freebits=freebits;
  for(int i=0;i<U;i++) M->upos[i]=upos[i];
  M->pu=(uint32_t)purpose; M->changes=changes; M->gap=gap; M->grid=1024; M->tpb=128;
  build_module(cu);
  gpu_upload_wordlist(&M->dwl,&M->dwoff,&M->dwlen);
  M->dtmpl=up(tmpl,W*sizeof(uint32_t)); M->dtp=up(tprog,32); M->dupos=up(kpos,(kU?kU:1)*sizeof(int));
  unsigned long long init=~0ULL,z0=0; int zero=0; uint32_t hci0[2]={0,0};
  M->dhi=up(&init,8); M->dfound=up(&zero,4); M->dhg=up(0,W*sizeof(uint32_t)); M->dhashed=up(&z0,8); M->dhit_ci=up(hci0,8);
  if(aset && (aset->n>0 || aset->prefilter)){    /* bloom target set (same as crack_addr_setup) */
    M->aset=aset; long ntgt=aset->n; uint32_t nb; size_t fbytes;
    if(aset->prefilter){ nb=aset->prefilter_nblocks; fbytes=(size_t)nb*8u*4u; M->d_bloom=up(aset->prefilter,fbytes); }
    else { double bpk=24.0; const char*e=getenv("BLOOM_BPK"); if(e){ double v=atof(e); if(v>=4) bpk=v; }
      nb=bloom_nblocks((uint64_t)ntgt,bpk); fbytes=(size_t)nb*8u*4u;
      uint32_t *hf=calloc(fbytes,1); for(long i=0;i<aset->n;i++) bloom_insert(hf,aset->ent[i].prog,nb-1);
      M->d_bloom=up(hf,fbytes); free(hf); }
    M->bloom_mask=nb-1;
    M->hitcap=1u<<18; CU(cuMemAlloc(&M->d_hits,(size_t)M->hitcap*sizeof(BloomHit))); M->d_hitcnt=up(&z0,4);
    M->bnpurp=aset->npurp; M->d_purposes=up(aset->purposes,(size_t)aset->npurp*sizeof(uint32_t));
    if(aset->prefilter)
      fprintf(stderr,"bloom: prebuilt GPU filter %.1f MiB (%u blocks), host cull filter %u blocks, %d purpose(s)\n",
              (double)fbytes/1048576.0,nb,aset->host_filter_nblocks,aset->npurp);
    else
      fprintf(stderr,"bloom: %ld target(s), filter %.1f MiB (%u blocks, %.1f bits/key), %d purpose(s), cull on host\n",
              ntgt,(double)fbytes/1048576.0,nb,ntgt?(double)nb*256.0/(double)ntgt:0.0,aset->npurp);
  }
  return 0;
}
/* Sweep [start,count) in nth-space. On hit: *out_hidx (nth-space), out_ci, out_hg[W]. */
static int crack_missing_sweep_single(MissCtx*M,unsigned long long start,unsigned long long count,double intv,int rep,
                               prog_cb cb,void*ud,unsigned long long*out_hidx,uint32_t out_ci[2],uint32_t out_hg[32]){
  unsigned long long init=~0ULL,z0=0; int zero=0; uint32_t hci0[2]={0,0};
  CU(cuMemcpyHtoD(M->dhi,&init,8)); CU(cuMemcpyHtoD(M->dfound,&zero,4));
  CU(cuMemcpyHtoD(M->dhashed,&z0,8)); CU(cuMemcpyHtoD(M->dhit_ci,hci0,8));
  int found=0; unsigned long long cstart=start,ccount=0,swept=0,hashed=0, chunk=rep?report_chunk(1):(64ULL<<20);
  void*args[]={&M->dwl,&M->dwoff,&M->dwlen,&M->dtmpl,&M->W,&M->dupos,&M->passU,&cstart,&ccount,&M->pu,&M->changes,&M->gap,&M->dtp,&M->dhi,&M->dfound,&M->dhg,&M->dhashed,&M->dhit_ci};
  for(cstart=start; cstart<start+count; ){
    ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
    CU(cuLaunchKernel(kern(M->nth?"g_crack_nth":"g_crack_missing"),M->grid,1,1,M->tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    if(M->nth) hashed=swept; else CU(cuMemcpyDtoH(&hashed,M->dhashed,8));
    if(cb) cb(ud,swept,hashed);
    CU(cuMemcpyDtoH(&found,M->dfound,4)); if(found) break;
    chunk=next_chunk(ccount,csecs,intv,rep);
  }
  CU(cuMemcpyDtoH(&found,M->dfound,4));
  if(found){ CU(cuMemcpyDtoH(out_hidx,M->dhi,8)); CU(cuMemcpyDtoH(out_ci,M->dhit_ci,8)); CU(cuMemcpyDtoH(out_hg,M->dhg,M->W*sizeof(uint32_t))); }
  return found;
}
/* Bloom sweep for missing-word: the *_bloom kernels probe + append; host culls per
   chunk. On a culled-true hit we re-derive that ONE index single-target (target =
   the hit's program) to recover the word array for rendering. */
static int crack_missing_sweep_bloom(MissCtx*M,unsigned long long start,unsigned long long count,double intv,int rep,
                               prog_cb cb,void*ud,unsigned long long*out_hidx,uint32_t out_ci[2],uint32_t out_hg[32],
                               char*out_addr,int addrsz,int*out_purpose){
  unsigned long long z0=0; CU(cuMemcpyHtoD(M->dhashed,&z0,8));
  unsigned long long cstart=start,ccount=0,swept=0,hashed=0, chunk=rep?report_chunk(1):(64ULL<<20);
  int found=0; unsigned long long best=~0ULL; uint32_t best_ci[2]={0,0}; unsigned char best_prog[32]={0}; int best_purpose=0;
  BloomHit *hb=malloc((size_t)M->hitcap*sizeof(BloomHit));
  CUfunction kf=kern(M->nth?"g_crack_nth_bloom":"g_crack_missing_bloom");
  for(cstart=start; cstart<start+count; ){
    ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
    unsigned int zc=0; CU(cuMemcpyHtoD(M->d_hitcnt,&zc,4));
    void*args[]={&M->dwl,&M->dwoff,&M->dwlen,&M->dtmpl,&M->W,&M->dupos,&M->passU,&cstart,&ccount,&M->d_purposes,&M->bnpurp,&M->changes,&M->gap,
                 &M->d_bloom,&M->bloom_mask,&M->d_hits,&M->d_hitcnt,&M->hitcap,&M->dhashed};
    CU(cuLaunchKernel(kf,M->grid,1,1,M->tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    unsigned int hc=0; CU(cuMemcpyDtoH(&hc,M->d_hitcnt,4));
    if(hc>0){ unsigned int nc=hc>M->hitcap?M->hitcap:hc;
      CU(cuMemcpyDtoH(hb,M->d_hits,(size_t)nc*sizeof(BloomHit)));
      for(unsigned int i=0;i<nc;i++) if(aset_member(M->aset,hb[i].prog)){
        found=1; if(hb[i].gidx<best){ best=hb[i].gidx; best_ci[0]=hb[i].change; best_ci[1]=hb[i].index; best_purpose=(int)hb[i].purpose; memcpy(best_prog,hb[i].prog,32); } }
      if(hc>M->hitcap) fprintf(stderr,"  (bloom hit-buffer overflow %u>%u)\n",hc,M->hitcap);
    }
    if(M->nth) hashed=swept; else CU(cuMemcpyDtoH(&hashed,M->dhashed,8));
    if(cb) cb(ud,swept,hashed);
    if(found) break;
    chunk=next_chunk(ccount,csecs,intv,rep);
  }
  free(hb);
  if(found){ CU(cuMemcpyHtoD(M->dtp,best_prog,32));   /* re-derive the winner to get its words */
    uint32_t ci2[2]; crack_missing_sweep_single(M,best,1,0,0,0,0,out_hidx,ci2,out_hg);
    *out_hidx=best; out_ci[0]=best_ci[0]; out_ci[1]=best_ci[1]; if(out_purpose) *out_purpose=best_purpose;
    const AEnt*me=aset_lookup(M->aset,best_prog);
    if(out_addr&&addrsz){ if(me) snprintf(out_addr,(size_t)addrsz,"%s",me->str);
      else if(encode_address(best_prog,best_purpose,out_addr)){ char hx[66]; tohex_(best_prog,best_purpose==86?32:20,hx); snprintf(out_addr,(size_t)addrsz,"program:%s",hx); } } }
  return found;
}
static int crack_missing_sweep(MissCtx*M,unsigned long long start,unsigned long long count,double intv,int rep,
                               prog_cb cb,void*ud,unsigned long long*out_hidx,uint32_t out_ci[2],uint32_t out_hg[32],
                               char*out_addr,int addrsz,int*out_purpose){
  if(out_addr&&addrsz) out_addr[0]=0;
  if(M->d_bloom) return crack_missing_sweep_bloom(M,start,count,intv,rep,cb,ud,out_hidx,out_ci,out_hg,out_addr,addrsz,out_purpose);
  return crack_missing_sweep_single(M,start,count,intv,rep,cb,ud,out_hidx,out_ci,out_hg);
}
static int mode_missing(const char*tpl,const uint8_t tprog[32],int purpose,uint32_t changes,uint32_t gap,
                        int nthmode,unsigned long long ustart,unsigned long long ucount,const char*cu,
                        const AddrSet*aset){
  if(g_print_total){   /* host-only job size, no GPU */
    uint32_t tmpl[32]; int upos[32],W,U,last_unknown;
    if(parse_template(tpl,tmpl,upos,&W,&U,&last_unknown)) return 2;
    int fb=11-W/3; int nth=(nthmode==1)?1:(nthmode==0)?0:last_unknown;
    if(nthmode==1 && !last_unknown){ fprintf(stderr,"--nth needs the LAST position to be [:bip39-en:]\n"); return 2; }
    unsigned long long total=1; int kU=0; for(int i=0;i<U;i++) if(!(nth&&upos[i]==W-1)) kU++;
    for(int i=0;i<kU;i++) total*=2048ULL;
    if(nth) total<<=fb;
    printf("%llu\n",total); return 0;
  }
  MissCtx M; if(crack_missing_setup(&M,tpl,tprog,purpose,changes,gap,nthmode,cu,aset)) return 2;
  unsigned long long start=ustart>M.total?M.total:ustart;
  unsigned long long count=ucount?ucount:(M.total-start); if(start+count>M.total) count=M.total-start;
  int reporting=(g_pflag||g_loginterval_ms);
  char path[64]; snprintf(path,sizeof path,"m/%d'/0'/0'/[0,%u)/[0,%u)",purpose,changes,gap);
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  prog_hdr(&P, M.nth?"missing-word [:Nth:]":"missing-word baseline", g_target_str,g_pattern_str,path,0,0);
  double intv=prog_intv(&P);
  fprintf(stderr,"missing-word %s: W=%d, %d unknown(s)%s -> %llu candidates (%s)...\n",
          M.nth?"[:Nth:] CONSTRUCTION":"baseline sieve", M.W, M.U, M.nth?" (last=checksum-constructed)":"", count, path);
  double tt0=now_s(); unsigned long long hidx=0; uint32_t hci[2]={0,0},hg[32]; char maddr[100]=""; int mpurpose=purpose;
  int found=crack_missing_sweep(&M,start,count,intv,reporting,mode_prog_cb,&P,&hidx,hci,hg,maddr,sizeof maddr,&mpurpose);
  if(found<0) return 2;
  double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.3fs = %.3f Mcand/s (%s%s)\n",P.swept,secs,secs>0?P.swept/secs/1e6:0.0,
          M.nth?"all valid, no sieve":"sieve->PBKDF2+EC on survivors", (found&&P.swept<count)?", early-exit":"");
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND\n"); return 1; }
  unsigned long long librxe_rank = M.nth ? ((hidx>>M.freebits)*2048ULL + hg[M.W-1]) : hidx;
  printf("FOUND\n  index    : %llu%s\n",hidx, M.nth?" (nth-space)":" (== librxe rank)");
  if(M.nth) printf("  librxe rank: %llu\n",librxe_rank);
  printf("  found words:");
  for(int i=0;i<M.U;i++) printf(" [pos %d]=%s",M.upos[i],WLNAME((int)hg[M.upos[i]]));
  printf("\n  mnemonic :");
  for(int p=0;p<M.W;p++) printf(" %s",WLNAME((int)hg[p]));
  char fpath[80]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'/%u/%u",mpurpose,hci[0],hci[1]);
  printf("\n  path     : %s\n",fpath);
  if(maddr[0]) printf("  address  : %s\n",maddr);
  { char fw[256]; int L=0; for(int i=0;i<M.U;i++) L+=snprintf(fw+L,sizeof fw-L,"%s%s",i?" ":"",WLNAME((int)hg[M.upos[i]])); prog_finish(&P,"FOUND",fw,fpath); }
  return 0;
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
  unsigned long long total=1; for(int i=0;i<pwidth;i++) total*=10ULL;
  if(g_print_total){ printf("%llu\n",total); return 0; }
  build_module(cu);
  int mnlen=(int)strlen(mnemonic);
  CUdeviceptr dmn=up(mnemonic,mnlen), dtp=up(tprog,32);
  /* precompute the fixed-mnemonic HMAC key context once (regime A) */
  CUdeviceptr dhctx; CU(cuMemAlloc(&dhctx,16*sizeof(unsigned long long)));
  { void*ia[]={&dmn,&mnlen,&dhctx}; CU(cuLaunchKernel(kern("g_hctx_init"),1,1,1,1,1,1,0,0,ia,0)); CU(cuCtxSynchronize()); }
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
/* ------------------------------- worker -----------------------------------
 * --worker: a persistent process that builds its CUDA context ONCE, announces
 * READY, then services shard requests from stdin over the wq1 line protocol
 * until STOP/EOF. The transport is a pipe today (local fork+exec) and `ssh host
 * ... --worker` later -- the protocol is identical. All indices are GLOBAL
 * canonical ranks. v1 handles the address-words path (mode_crack_addr). */
typedef struct { unsigned long long id, base_swept, base_hashed, cur_swept, cur_hashed; double last; } WorkerProg;
static void worker_prog_cb(void*ud,unsigned long long swept,unsigned long long hashed){
  WorkerProg*w=ud; w->cur_swept=swept; w->cur_hashed=hashed; double now=now_s();
  if(now-w->last<0.3) return;   /* rate-limit PROG to ~3/s */
  w->last=now; printf("PROG %llu %llu %llu\n",w->id,w->base_swept+swept,w->base_hashed+hashed); fflush(stdout);
}
/* A Sweeper adapts a mode's setup/sweep to the worker: sweep a shard and, on a
   hit, hand back the GLOBAL canonical index, change/index, and the mnemonic. */
typedef int (*sweep_fn)(void*ctx,unsigned long long start,unsigned long long count,double intv,int rep,
                        prog_cb cb,void*ud,unsigned long long*out_idx,uint32_t out_ci[2],char*out_mn,int mnsz,
                        char*out_addr,int addrsz,int*out_purpose);
typedef struct { sweep_fn sweep; void*ctx; unsigned long long total; int purpose; } Sweeper;
static int wq_sweep_addr(void*c,unsigned long long s,unsigned long long n,double intv,int rep,prog_cb cb,void*ud,
                         unsigned long long*oi,uint32_t oc[2],char*mn,int mnsz,char*addr,int addrsz,int*opur){
  CrackCtx*X=c; uint32_t ci[2]; unsigned long long hidx=0;
  int f=crack_addr_sweep(X,s,n,intv,rep,cb,ud,&hidx,ci,addr,addrsz,opur);
  if(f>0){ *oi=hidx; oc[0]=ci[0]; oc[1]=ci[1];
    mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(X->r,j); char b[MN_STRIDE]; rxe_current(b,sizeof b,X->r);
    char*t=b; while(*t==' ')t++; snprintf(mn,mnsz,"%s",t); mpz_clear(j); }
  return f;
}
static int wq_sweep_missing(void*c,unsigned long long s,unsigned long long n,double intv,int rep,prog_cb cb,void*ud,
                            unsigned long long*oi,uint32_t oc[2],char*mn,int mnsz,char*addr,int addrsz,int*opur){
  MissCtx*M=c; uint32_t ci[2],hg[32]; unsigned long long hidx=0;
  int f=crack_missing_sweep(M,s,n,intv,rep,cb,ud,&hidx,ci,hg,addr,addrsz,opur);
  if(f>0){ *oi = M->nth ? ((hidx>>M->freebits)*2048ULL + hg[M->W-1]) : hidx; oc[0]=ci[0]; oc[1]=ci[1];
    int L=0; for(int p=0;p<M->W;p++) L+=snprintf(mn+L,mnsz-L,"%s%s",p?" ":"",WLNAME((int)hg[p])); }
  return f;
}
static int run_worker(Sweeper*S){
  printf("READY %s dev%d kern=%s total=%llu\n",WQ_PROTO,g_device,g_ptx_key,S->total); fflush(stdout);
  char line[512];
  WorkerProg wp={0};   /* cumulative across ALL this worker's shards (base) + in-flight (cur) */
  while(fgets(line,sizeof line,stdin)){
    if(!strncmp(line,"STOP",4)) break;
    unsigned long long id,s,c;
    if(sscanf(line,"SHARD %llu %llu %llu",&id,&s,&c)!=3) continue;
    if(s>S->total) s=S->total;
    if(s+c>S->total) c=S->total-s;
    wp.id=id; wp.cur_swept=0; wp.cur_hashed=0;
    unsigned long long oi=0; uint32_t oc[2]={0,0}; char mn[512]=""; char addr[100]=""; int mpur=S->purpose;
    int f=S->sweep(S->ctx,s,c,0.3,1,worker_prog_cb,&wp,&oi,oc,mn,sizeof mn,addr,sizeof addr,&mpur);
    if(f<0){ printf("ERROR %llu\n",id); fflush(stdout); continue; }
    if(f){ printf("FOUND %llu %llu %d %u %u %s %s\n",id,oi,mpur,oc[0],oc[1],addr[0]?addr:"-",mn); fflush(stdout); }
    else { /* whole shard swept: bank its full candidate count, emit a final PROG, then DONE */
      wp.base_swept+=c; wp.base_hashed+=wp.cur_hashed;
      printf("PROG %llu %llu %llu\nDONE %llu\n",id,wp.base_swept,wp.base_hashed,id); fflush(stdout); }
  }
  return 0;
}
static int run_worker_addr(const Words*W,const uint8_t tprog[32],int purpose,uint32_t changes,uint32_t gap,
                           int require_ck,int compact,const char*cu,const AddrSet*aset){
  CrackCtx X; if(crack_addr_setup(&X,W,tprog,purpose,changes,gap,require_ck,compact,cu,aset)) return 2;
  Sweeper S={wq_sweep_addr,&X,X.total,purpose}; return run_worker(&S);
}
static int run_worker_missing(const char*tpl,const uint8_t tprog[32],int purpose,uint32_t changes,uint32_t gap,
                              int nthmode,const char*cu,const AddrSet*aset){
  MissCtx M; if(crack_missing_setup(&M,tpl,tprog,purpose,changes,gap,nthmode,cu,aset)) return 2;
  Sweeper S={wq_sweep_missing,&M,M.total,purpose}; return run_worker(&S);
}

/* ---------------------- multi-GPU fan-out supervisor ----------------------
 * --devices/--gpus turns this process into a supervisor: it computes the job
 * size once (a --print-total child, no GPU), warms the PTX cache once, then
 * forks one crack child per GPU over a contiguous slice of the canonical index
 * space (reusing --device/--start/--count). First child to exit 0 (FOUND) wins;
 * the supervisor SIGTERMs the siblings' process group. Ctrl-C kills them too. */
extern char **environ;
static volatile sig_atomic_t g_sup_pgid=0;
static void sup_sigint(int sig){ (void)sig; if(g_sup_pgid) killpg(g_sup_pgid,SIGTERM); _exit(130); }

/* Copy argv, dropping the fan-out/slice flags (and their values), then append
   `extra` verbatim. Returns a NULL-terminated malloc'd argv (argv[0] kept). */
static char**child_argv(int argc,char**argv,char*const*extra,int nextra){
  char**out=calloc((size_t)argc+nextra+2,sizeof(char*)); int o=0;
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],"--devices")||!strcmp(argv[i],"--gpus")||!strcmp(argv[i],"--device")||
       !strcmp(argv[i],"--start")||!strcmp(argv[i],"--count")||!strcmp(argv[i],"--limit")||
       !strcmp(argv[i],"--loginterval")||!strcmp(argv[i],"--shards")||!strcmp(argv[i],"--order")){ i++; continue; }   /* drop flag + its value */
    if(!strcmp(argv[i],"--print-total")||!strcmp(argv[i],"--warm")||!strcmp(argv[i],"--worker")||!strcmp(argv[i],"--devtag")) continue;
    out[o++]=argv[i];
  }
  for(int i=0;i<nextra;i++) out[o++]=extra[i];
  out[o]=0; return out;
}
/* Spawn cargv, capture its stdout into out[]. Returns 0 iff it exited 0. */
static int run_capture(char**cargv,char*out,size_t outn){
  int pfd[2]; if(pipe(pfd)) return -1;
  posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa,pfd[1],STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&fa,pfd[0]);
  posix_spawn_file_actions_addclose(&fa,pfd[1]);
  pid_t pid; int rc=posix_spawn(&pid,"/proc/self/exe",&fa,0,cargv,environ);
  posix_spawn_file_actions_destroy(&fa); close(pfd[1]);
  if(rc){ close(pfd[0]); return -1; }
  size_t k=0; ssize_t r; while(k+1<outn && (r=read(pfd[0],out+k,outn-1-k))>0) k+=(size_t)r;
  out[k]=0; close(pfd[0]); int st; waitpid(pid,&st,0);
  return (WIFEXITED(st)&&WEXITSTATUS(st)==0)?0:-1;
}
static int run_supervisor(int argc,char**argv,int*devs,int ndev,
                          unsigned long long ostart,unsigned long long ocount){
  /* 1) job size via a no-GPU --print-total child */
  char*pt[]={"--print-total"}; char**ptv=child_argv(argc,argv,pt,1);
  char nbuf[64]; if(run_capture(ptv,nbuf,sizeof nbuf)){ fprintf(stderr,"supervisor: could not compute job total\n"); free(ptv); return 2; }
  free(ptv);
  unsigned long long total=strtoull(nbuf,0,10);
  if(!total){ fprintf(stderr,"supervisor: job total is 0 (nothing to sweep)\n"); return 2; }
  if(ostart>total) ostart=total;
  unsigned long long owin = ocount?ocount:(total-ostart);
  if(ostart+owin>total) owin=total-ostart;
  fprintf(stderr,"supervisor: total=%llu, window=[%llu,%llu), fan-out to %d GPU(s)\n",total,ostart,ostart+owin,ndev);

  /* 2) warm the PTX cache once so the N children don't all compile in parallel */
  char dv0[16]; snprintf(dv0,sizeof dv0,"%d",devs[0]);
  char*we[]={"--warm","--device",dv0}; char**wv=child_argv(argc,argv,we,3);
  { pid_t wp; if(!posix_spawn(&wp,"/proc/self/exe",0,0,wv,environ)){ int st; waitpid(wp,&st,0);
      if(!(WIFEXITED(st)&&WEXITSTATUS(st)==0)){ fprintf(stderr,"supervisor: warm build failed\n"); free(wv); return 2; } } }
  free(wv);

  /* original --loginterval (if any): give each child its own .devN file.
     Use the captured globals -- argv was mutated in place during parse. */
  int li_ms = g_loginterval_ms>0 ? g_loginterval_ms : -1;
  const char *li_file = g_csv_path;

  /* 3) fan out one child per device over a contiguous slice */
  pid_t pids[16]={0}; int alive=0; pid_t pgid=0;
  signal(SIGINT,sup_sigint); signal(SIGTERM,sup_sigint);  /* armed before the first child exists */
  for(int d=0; d<ndev; d++){
    unsigned long long s = ostart + (unsigned long long)((__int128)owin*d/ndev);
    unsigned long long e = ostart + (unsigned long long)((__int128)owin*(d+1)/ndev);
    if(e<=s) continue;   /* empty slice (fewer indices than GPUs): --count 0 would mean
                            "sweep to end" to the child, so skip spawning it entirely */
    char dv[16],sd[24],cd[24],liarg[320];
    snprintf(dv,sizeof dv,"%d",devs[d]); snprintf(sd,sizeof sd,"%llu",s); snprintf(cd,sizeof cd,"%llu",e-s);
    char*extra[12]; int ne=0;
    extra[ne++]="--device"; extra[ne++]=dv;
    extra[ne++]="--start";  extra[ne++]=sd;
    extra[ne++]="--count";  extra[ne++]=cd;
    extra[ne++]="--devtag";
    if(li_ms>=0){
      if(li_file){ const char*dot=strrchr(li_file,'.');
        if(dot) snprintf(liarg,sizeof liarg,"%d:%.*s.dev%d%s",li_ms,(int)(dot-li_file),li_file,devs[d],dot);
        else    snprintf(liarg,sizeof liarg,"%d:%s.dev%d",li_ms,li_file,devs[d]); }
      else snprintf(liarg,sizeof liarg,"%d",li_ms);
      extra[ne++]="--loginterval"; extra[ne++]=liarg;
    }
    char**cv=child_argv(argc,argv,extra,ne);
    posix_spawnattr_t at; posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at,POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&at,pgid);   /* 0 => new group (this child leads it) */
    pid_t pid; int rc=posix_spawn(&pid,"/proc/self/exe",0,&at,cv,environ);
    posix_spawnattr_destroy(&at); free(cv);
    if(rc){ fprintf(stderr,"supervisor: spawn dev %d failed: %s\n",devs[d],strerror(rc)); continue; }
    if(!pgid){ pgid=pid; g_sup_pgid=pgid; }  /* first child's pid is the group id; arm the handler */
    pids[d]=pid; alive++;
  }
  if(alive==0){ fprintf(stderr,"supervisor: no children spawned\n"); return 2; }

  /* 4) supervise: first child to exit 0 (FOUND) wins; kill the siblings */
  int winner=-1, err=0;
  while(alive>0){
    int st; pid_t pid=waitpid(-1,&st,0);
    if(pid<0){ if(errno==EINTR) continue; break; }
    int which=-1; for(int d=0;d<ndev;d++) if(pids[d]==pid) which=d;
    alive--;
    int code = WIFEXITED(st)?WEXITSTATUS(st):-1;
    if(code==0){ winner=which; killpg(pgid,SIGTERM);
      while(alive>0){ if(waitpid(-1,0,0)>0) alive--; else if(errno!=EINTR) break; }
      break;
    } else if(code!=1) err=1;   /* 1 = this slice NOT FOUND; anything else = error */
  }
  if(winner>=0){ fprintf(stderr,"supervisor: device %d found it.\n",devs[winner]); return 0; }
  if(err){ fprintf(stderr,"supervisor: a child errored; result inconclusive.\n"); return 2; }
  fprintf(stderr,"supervisor: NOT FOUND across all %d slice(s).\n",ndev); return 1;
}

/* ------------------- work-queue supervisor (fine shards) -------------------
 * Owns a queue of fine shards handed out in a policy order; persistent --worker
 * processes pull shards, stream PROG, report DONE/FOUND. First FOUND wins ->
 * STOP all. A worker that dies (EOF) has its in-flight shard re-queued. The
 * supervisor prints ONE consolidated live line summing all workers. */
static const char*ORDER_NAMES[]={"first","ends","center","random"};
static void build_order(int policy,long n,long*out,unsigned long long seed){
  if(policy==1){ long lo=0,hi=n-1,k=0; while(lo<=hi){ out[k++]=lo++; if(lo<=hi) out[k++]=hi--; } }
  else if(policy==2){ long c=n/2,k=0; out[k++]=c; for(long off=1;k<n;off++){ if(c-off>=0)out[k++]=c-off; if(k<n&&c+off<n)out[k++]=c+off; } }
  else if(policy==3){ for(long i=0;i<n;i++) out[i]=i;
    unsigned long long r=seed?seed:88172645463325252ULL;
    for(long i=n-1;i>0;i--){ r^=r<<13;r^=r>>7;r^=r<<17; long j=(long)(r%(unsigned long long)(i+1)); long t=out[i];out[i]=out[j];out[j]=t; } }
  else { for(long i=0;i<n;i++) out[i]=i; }   /* 0 = first-to-last */
}
typedef struct { pid_t pid; int wfd,rfd,alive,ready; long shard;
  unsigned long long swept,hashed; char buf[8192]; int blen; } Wrk;
static int run_workqueue(int argc,char**argv,int*devs,int ndev,
                         unsigned long long ostart,unsigned long long ocount,
                         int order_policy,unsigned long long order_seed,long nshards_arg){
  /* 1) total (no-GPU child) + 2) warm the PTX cache once */
  char*pt[]={"--print-total"}; char**ptv=child_argv(argc,argv,pt,1);
  char nbuf[64]; if(run_capture(ptv,nbuf,sizeof nbuf)){ fprintf(stderr,"workqueue: could not compute total\n"); free(ptv); return 2; }
  free(ptv); unsigned long long total=strtoull(nbuf,0,10);
  if(!total){ fprintf(stderr,"workqueue: total is 0\n"); return 2; }
  if(ostart>total) ostart=total;
  unsigned long long owin=ocount?ocount:(total-ostart);
  if(ostart+owin>total) owin=total-ostart;
  if(owin==0){ fprintf(stderr,"workqueue: empty window\n"); return 1; }
  char dv0[16]; snprintf(dv0,sizeof dv0,"%d",devs[0]);
  char*we[]={"--warm","--device",dv0}; char**wv=child_argv(argc,argv,we,3);
  { pid_t wp; if(!posix_spawn(&wp,"/proc/self/exe",0,0,wv,environ)){ int st; waitpid(wp,&st,0);
      if(!(WIFEXITED(st)&&WEXITSTATUS(st)==0)){ fprintf(stderr,"workqueue: warm build failed\n"); free(wv); return 2; } } }
  free(wv);

  /* 3) shard plan: ~8M candidates/shard by default (or --shards N) */
  unsigned long long SHARD_CAND=8ULL<<20;
  long nshards = nshards_arg>0 ? nshards_arg : (long)((owin+SHARD_CAND-1)/SHARD_CAND);
  if(nshards<ndev) nshards=ndev;
  if(nshards<1) nshards=1;
  if(nshards>4000000) nshards=4000000;
  unsigned long long ss=(owin+nshards-1)/nshards; nshards=(long)((owin+ss-1)/ss);
  long*order=malloc((size_t)nshards*sizeof(long)); build_order(order_policy,nshards,order,order_seed);
  fprintf(stderr,"workqueue: total=%llu window=[%llu,%llu) shards=%ld (~%llu each) order=%s across %d GPU(s)\n",
    total,ostart,ostart+owin,nshards,ss,ORDER_NAMES[order_policy],ndev);

  /* 4) spawn workers, each on a stdin/stdout pipe pair */
  Wrk W[16]; int nw=0; pid_t pgid=0;
  for(int d=0; d<ndev; d++){
    int tw[2],fw[2]; if(pipe(tw)||pipe(fw)){ fprintf(stderr,"workqueue: pipe failed\n"); return 2; }
    char dv[16]; snprintf(dv,sizeof dv,"%d",devs[d]);
    char*extra[3]={"--worker","--device",dv}; char**cv=child_argv(argc,argv,extra,3);
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa,tw[0],0); posix_spawn_file_actions_adddup2(&fa,fw[1],1);
    posix_spawn_file_actions_addclose(&fa,tw[1]); posix_spawn_file_actions_addclose(&fa,fw[0]);
    posix_spawn_file_actions_addclose(&fa,tw[0]); posix_spawn_file_actions_addclose(&fa,fw[1]);
    posix_spawnattr_t at; posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at,POSIX_SPAWN_SETPGROUP); posix_spawnattr_setpgroup(&at,pgid);
    pid_t pid; int rc=posix_spawn(&pid,"/proc/self/exe",&fa,&at,cv,environ);
    posix_spawn_file_actions_destroy(&fa); posix_spawnattr_destroy(&at); free(cv);
    close(tw[0]); close(fw[1]);
    if(rc){ fprintf(stderr,"workqueue: spawn dev %d failed: %s\n",devs[d],strerror(rc)); close(tw[1]); close(fw[0]); continue; }
    if(!pgid){ pgid=pid; g_sup_pgid=pgid; }
    memset(&W[nw],0,sizeof W[nw]); W[nw].pid=pid; W[nw].wfd=tw[1]; W[nw].rfd=fw[0]; W[nw].alive=1; W[nw].shard=-1; nw++;
  }
  if(nw==0){ fprintf(stderr,"workqueue: no workers\n"); free(order); return 2; }
  signal(SIGINT,sup_sigint); signal(SIGTERM,sup_sigint);
  signal(SIGPIPE,SIG_IGN);   /* a dead worker's pipe write must not kill the supervisor */

  /* queue state: next unclaimed order[] slot + a re-queue stack for dead workers */
  long next=0; long*requeue=malloc((size_t)nshards*sizeof(long)); long rq=0;
  #define NEXT_SHARD() (rq>0 ? requeue[--rq] : (next<nshards ? order[next++] : -1))
  int found=0; unsigned long long win_idx=0; uint32_t win_ci[2]={0,0}; char win_mn[512]=""; int win_purpose=0; char win_addr[100]="";
  int winner_dev=-1;
  double t0=now_s(), last_print=0, last_rate_t=t0; unsigned long long last_rate_sum=0;

  /* assign each ready worker its first shard as soon as READY arrives (below) */
  for(;;){
    /* termination: found, or no shards left and every worker idle */
    int outstanding=0; for(int i=0;i<nw;i++) if(W[i].alive&&W[i].shard>=0) outstanding++;
    int any_alive=0; for(int i=0;i<nw;i++) if(W[i].alive) any_alive++;
    if(found) break;
    if(!any_alive) break;
    if(rq==0 && next>=nshards && outstanding==0) break;   /* NOT FOUND: queue drained */

    struct pollfd pfd[16]; int map[16],np=0;
    for(int i=0;i<nw;i++) if(W[i].alive){ pfd[np].fd=W[i].rfd; pfd[np].events=POLLIN; map[np]=i; np++; }
    int pr=poll(pfd,np,250);
    if(pr<0){ if(errno==EINTR) continue; break; }
    for(int p=0;p<np && !found;p++){
      if(!(pfd[p].revents&(POLLIN|POLLHUP|POLLERR))) continue;
      Wrk*w=&W[map[p]];
      int n=read(w->rfd,w->buf+w->blen,sizeof w->buf-1-w->blen);
      if(n<=0){ /* worker died: re-queue its in-flight shard */
        w->alive=0; close(w->rfd); close(w->wfd);
        if(w->shard>=0){ requeue[rq++]=w->shard; w->shard=-1;
          fprintf(stderr,"workqueue: worker dev%d died -> re-queued shard\n",map[p]); }
        continue;
      }
      w->blen+=n; w->buf[w->blen]=0;
      char*ln=w->buf, *nl;
      while((nl=strchr(ln,'\n'))){
        *nl=0;
        if(!strncmp(ln,"READY",5)){ long s=NEXT_SHARD();
          if(s<0){ dprintf(w->wfd,"STOP\n"); }
          else { w->shard=s; unsigned long long st=ostart+(unsigned long long)s*ss, cc=(st+ss>ostart+owin)?(ostart+owin-st):ss;
            dprintf(w->wfd,"SHARD %ld %llu %llu\n",s,st,cc); } }
        else { unsigned long long id,a,b; char mn[480];
          if(sscanf(ln,"PROG %llu %llu %llu",&id,&a,&b)==3){ w->swept=a; w->hashed=b; }
          else if(sscanf(ln,"DONE %llu",&id)==1){ long s=NEXT_SHARD();
            if(s<0){ w->shard=-1; dprintf(w->wfd,"STOP\n"); }
            else { w->shard=s; unsigned long long st=ostart+(unsigned long long)s*ss, cc=(st+ss>ostart+owin)?(ostart+owin-st):ss;
              dprintf(w->wfd,"SHARD %ld %llu %llu\n",s,st,cc); } }
          else if(sscanf(ln,"FOUND %llu %llu %d %u %u %99s %479[^\n]",&id,&win_idx,&win_purpose,&win_ci[0],&win_ci[1],win_addr,mn)>=7){
            found=1; winner_dev=map[p]; snprintf(win_mn,sizeof win_mn,"%s",mn); break; }
          else if(sscanf(ln,"ERROR %llu",&id)==1){ long s=NEXT_SHARD();   /* shard failed: re-hand a new one */
            if(s<0){ w->shard=-1; } else { w->shard=s; unsigned long long st=ostart+(unsigned long long)s*ss, cc=(st+ss>ostart+owin)?(ostart+owin-st):ss; dprintf(w->wfd,"SHARD %ld %llu %llu\n",s,st,cc); } }
        }
        ln=nl+1;
      }
      w->blen-=(int)(ln-w->buf); memmove(w->buf,ln,w->blen); w->buf[w->blen]=0;
    }
    /* consolidated live line (supervisor owns it) */
    if(g_pflag){ double now=now_s();
      if(now-last_print>=(g_p_secs>0?g_p_secs:1.0)){
        unsigned long long sw=0,ha=0; for(int i=0;i<nw;i++){ sw+=W[i].swept; ha+=W[i].hashed; }
        double dt=now-last_rate_t; double rate=dt>0?(double)(sw-last_rate_sum)/dt/1e6:0;
        last_rate_t=now; last_rate_sum=sw; last_print=now;
        double pct=owin?100.0*sw/owin:0, eta=(rate>0&&owin>sw)?(owin-sw)/(rate*1e6):0;
        int live=0; for(int i=0;i<nw;i++) if(W[i].alive) live++;
        fprintf(stderr,"\r[%6.1fs] %.1f/%.1fM (%.1f%%) %.2f Mc/s  hashed %.1fM  ETA %.0fs  %dgpu   ",
          now-t0, sw/1e6, owin/1e6, pct, rate, ha/1e6, eta, live); fflush(stderr); }
    }
  }
  if(g_pflag) fprintf(stderr,"\n");
  for(int i=0;i<nw;i++) if(W[i].alive) dprintf(W[i].wfd,"STOP\n");
  if(pgid) killpg(pgid,SIGTERM);
  while(waitpid(-1,0,0)>0 || errno==EINTR){ if(errno==EINTR) continue; }
  free(order); free(requeue);
  if(found){
    char fpath[80]; snprintf(fpath,sizeof fpath,"m/?'/0'/0'/%u/%u",win_ci[0],win_ci[1]);
    fprintf(stderr,"workqueue: device %d found it.\n",winner_dev);
    printf("FOUND\n  index (canonical): %llu\n  mnemonic: %s\n  path    : m/%d'/0'/0'/%u/%u\n",win_idx,win_mn,win_purpose,win_ci[0],win_ci[1]);
    if(win_addr[0] && strcmp(win_addr,"-")) printf("  address : %s\n",win_addr);
    return 0;
  }
  printf("NOT FOUND\n"); return 1;
}
/* Decode --addresses (comma list) and/or --addresses-file (one per line) into a
   sorted AddrSet for the bloom + cull. v1: all targets must share one script type
   (one derivation purpose). Returns the shared purpose in *purpose_out. */
static int build_addrset(const char*csv,const char*file,AddrSet*A,int*purpose_out){
  char **addrs=0; long na=0,cap=0;
  #define ADDR_PUSH(s) do{ if(na==cap){ cap=cap?cap*2:64; addrs=realloc(addrs,(size_t)cap*sizeof(char*)); } addrs[na++]=strdup(s); }while(0)
  if(csv){ char*buf=strdup(csv); char*t=strtok(buf,","); while(t){ while(*t==' ')t++; if(*t) ADDR_PUSH(t); t=strtok(0,","); } free(buf); }
  if(file){ FILE*f=fopen(file,"r"); if(!f){ fprintf(stderr,"cannot open %s\n",file); return 2; }
    char line[256]; while(fgets(line,sizeof line,f)){ char*s=line; while(*s==' '||*s=='\t')s++;
      char*e=s+strlen(s); while(e>s&&(e[-1]=='\n'||e[-1]=='\r'||e[-1]==' '||e[-1]=='\t')) *--e=0; if(*s) ADDR_PUSH(s); } fclose(f); }
  if(na==0){ fprintf(stderr,"--addresses: no addresses given\n"); return 2; }
  AEnt *ent=malloc((size_t)na*sizeof(AEnt)); A->npurp=0;
  for(long i=0;i<na;i++){ uint8_t pr[32]; int pl,pu; if(decode_address(addrs[i],pr,&pl,&pu)){ fprintf(stderr,"--addresses: bad address '%s'\n",addrs[i]); return 2; }
    memset(ent[i].prog,0,32); memcpy(ent[i].prog,pr,(size_t)pl); ent[i].str=addrs[i]; ent[i].purpose=pu;
    int seen=0; for(int k=0;k<A->npurp;k++) if(A->purposes[k]==(uint32_t)pu) seen=1;   /* distinct script types */
    if(!seen && A->npurp<8) A->purposes[A->npurp++]=(uint32_t)pu; }
  free(addrs);
  A->ent=ent; A->n=na; aset_sort(A); *purpose_out=(int)A->purposes[0];
  return 0;
}
/* --xpubs / --xpubs-file -> a sorted set of 32-byte account CHAINCODES (bloom +
   cull). An xpub doesn't encode its BIP purpose, so the derive tries a default
   set {44,49,84,86} (--purpose overrides it). */
static int build_xpubset(const char*csv,const char*file,AddrSet*A){
  char **xs=0; long nx=0,cap=0;
  #define XPUB_PUSH(s) do{ if(nx==cap){ cap=cap?cap*2:32; xs=realloc(xs,(size_t)cap*sizeof(char*)); } xs[nx++]=strdup(s); }while(0)
  if(csv){ char*buf=strdup(csv); char*t=strtok(buf,","); while(t){ while(*t==' ')t++; if(*t) XPUB_PUSH(t); t=strtok(0,","); } free(buf); }
  if(file){ FILE*f=fopen(file,"r"); if(!f){ fprintf(stderr,"cannot open %s\n",file); return 2; }
    char line[256]; while(fgets(line,sizeof line,f)){ char*s=line; while(*s==' '||*s=='\t')s++;
      char*e=s+strlen(s); while(e>s&&(e[-1]=='\n'||e[-1]=='\r'||e[-1]==' '||e[-1]=='\t')) *--e=0; if(*s) XPUB_PUSH(s); } fclose(f); }
  #undef XPUB_PUSH
  if(nx==0){ fprintf(stderr,"--xpubs: no xpubs given\n"); return 2; }
  AEnt *ent=malloc((size_t)nx*sizeof(AEnt));
  for(long i=0;i<nx;i++){ uint8_t cc[32]; if(xpub_chaincode(xs[i],cc)){ fprintf(stderr,"--xpubs: bad xpub '%s'\n",xs[i]); return 2; }
    memset(ent[i].prog,0,32); memcpy(ent[i].prog,cc,32); ent[i].str=xs[i]; ent[i].purpose=0; }
  free(xs);
  A->ent=ent; A->n=nx; aset_sort(A);
  A->npurp=4; A->purposes[0]=44; A->purposes[1]=49; A->purposes[2]=84; A->purposes[3]=86;
  return 0;
}

/* ---------------- prebuilt bloom file (.blf): build + load ----------------
 * Dual-bloom layout: [BlfHeader][filter1: nblocks1*32 B][filter2: nblocks2*32 B].
 * Filter 1 (over the raw program) is the GPU prefilter; filter 2 (over sha256 of
 * the program, independent) is the host cull probed on filter 1's rare hits. Two
 * independent blooms multiply FPRs (behave like k=32), so no exact address list is
 * stored and there is no sort -- ~12 GiB gets FPR ~1e-13 at 1.5e9. The main program
 * mmaps it: filter1 -> GPU, filter2 stays paged on the host. */
#define BLF_MAGIC 0x32464C42u   /* 'B','L','F','2' (dual bloom) */
typedef struct { uint32_t magic,version,nblocks1,nblocks2,k,npurp; uint32_t purposes[8]; uint64_t n_addrs; } BlfHeader;

/* FPR of one k=16 blocked bloom of `nblocks` for `n` keys. */
static double blf_fpr(unsigned long long nblocks,unsigned long long n){
  if(!nblocks) return 1.0;
  double lambda=(double)n/(double)nblocks, fill=1.0-pow(255.0/256.0,16.0*lambda);
  return pow(fill,16.0);
}
/* pick two power-of-two filter sizes (nb1>=nb2) of minimal total blocks with
   combined FPR <= target -- two independent blooms behave like k=32, so this beats
   one filter (capped at k=16 by the 20-byte program). */
static void blf_size(unsigned long long n,double target,uint32_t*nb1,uint32_t*nb2){
  int bb1=28,bb2=28; unsigned long long best=~0ULL;
  for(int b1=4;b1<=31;b1++) for(int b2=4;b2<=b1;b2++){
    if(blf_fpr(1ULL<<b1,n)*blf_fpr(1ULL<<b2,n)<=target){
      unsigned long long tot=(1ULL<<b1)+(1ULL<<b2); if(tot<best){ best=tot; bb1=b1; bb2=b2; } } }
  *nb1=1u<<bb1; *nb2=1u<<bb2;
}
/* Read addresses (one per line; IN='-' = stdin, so pipe curl|zcat) -> a dual-bloom
   .blf: filter 1 over the raw program, filter 2 over sha256(program) (independent).
   Streamed -- no sort, no stored programs; only the two filters live in RAM. */
static int build_bloom_file(const char*infile,const char*outfile,unsigned long long n_hint,double fpr){
  if(!n_hint){ fprintf(stderr,"--bloom-build needs --bloom-n N (approx address count, to size the filters)\n"); return 2; }
  uint32_t nb1,nb2; blf_size(n_hint,fpr,&nb1,&nb2);
  size_t f1b=(size_t)nb1*32u, f2b=(size_t)nb2*32u;
  uint32_t *f1=calloc(f1b,1), *f2=calloc(f2b,1);
  if(!f1||!f2){ fprintf(stderr,"oom (filters %.2f GiB)\n",(double)(f1b+f2b)/1073741824.0); return 2; }
  fprintf(stderr,"bloom-build: filters %u+%u blocks = %.2f GiB (target FPR %.0e), streaming...\n",nb1,nb2,(double)(f1b+f2b)/1073741824.0,fpr);
  FILE*f=(!strcmp(infile,"-"))?stdin:fopen(infile,"r"); if(!f){ fprintf(stderr,"cannot open %s\n",infile); return 2; }
  long n=0,bad=0; uint32_t purposes[8]; int npurp=0; char line[256];
  while(fgets(line,sizeof line,f)){ char*s=line; while(*s==' '||*s=='\t')s++;
    char*e=s+strlen(s); while(e>s&&(e[-1]=='\n'||e[-1]=='\r'||e[-1]==' '||e[-1]=='\t')) *--e=0; if(!*s) continue;
    uint8_t pr[32]; memset(pr,0,32); int pl,pu; if(decode_address(s,pr,&pl,&pu)){ if(bad<5) fprintf(stderr,"  skip bad address: %s\n",s); bad++; continue; }
    (void)pl;
    bloom_insert(f1,pr,nb1-1);
    uint8_t hh[32]; sha256_host(pr,32,hh); bloom_insert(f2,hh,nb2-1);   /* filter 2 = independent */
    int seen=0; for(int k=0;k<npurp;k++) if(purposes[k]==(uint32_t)pu) seen=1;
    if(!seen && npurp<8) purposes[npurp++]=(uint32_t)pu;
    n++; if((n&0x3FFFFFF)==0) fprintf(stderr,"  ... %ld addresses\r",n); }
  if(f!=stdin) fclose(f);
  if(!n){ fprintf(stderr,"bloom-build: no valid addresses\n"); return 2; }
  double efpr=blf_fpr(nb1,(uint64_t)n)*blf_fpr(nb2,(uint64_t)n);
  FILE*o=fopen(outfile,"wb"); if(!o){ fprintf(stderr,"cannot write %s\n",outfile); return 2; }
  BlfHeader h; memset(&h,0,sizeof h); h.magic=BLF_MAGIC; h.version=2; h.nblocks1=nb1; h.nblocks2=nb2; h.k=BLOOM_K; h.npurp=(uint32_t)npurp;
  for(int i=0;i<npurp;i++) h.purposes[i]=purposes[i];
  h.n_addrs=(uint64_t)n;
  fwrite(&h,sizeof h,1,o); fwrite(f1,f1b,1,o); fwrite(f2,f2b,1,o);
  if(fclose(o)){ fprintf(stderr,"write error %s\n",outfile); return 2; }
  free(f1); free(f2);
  fprintf(stderr,"bloom-build: %ld addresses (%ld skipped), dual bloom %u+%u blocks = %.2f GiB, %d purpose(s), combined FPR ~%.1e -> %s\n",
          n,bad,nb1,nb2,(double)(f1b+f2b)/1073741824.0,npurp,efpr,outfile);
  return 0;
}
/* mmap a .blf: filter 1 -> GPU (prefilter), filter 2 -> host (the cull). */
static int load_bloom_file(const char*path,AddrSet*A){
  int fd=open(path,O_RDONLY); if(fd<0){ fprintf(stderr,"cannot open %s\n",path); return 2; }
  struct stat st; if(fstat(fd,&st)){ close(fd); return 2; } size_t sz=(size_t)st.st_size;
  void*base=mmap(0,sz,PROT_READ,MAP_SHARED,fd,0); close(fd);
  if(base==MAP_FAILED){ fprintf(stderr,"mmap %s failed\n",path); return 2; }
  BlfHeader*h=(BlfHeader*)base;
  if(h->magic!=BLF_MAGIC){ fprintf(stderr,"%s: not a v2 .blf (rebuild with --bloom-build)\n",path); return 2; }
  if(h->k!=BLOOM_K){ fprintf(stderr,"%s: k=%u != build k=%d (rebuild)\n",path,h->k,BLOOM_K); return 2; }
  size_t f1b=(size_t)h->nblocks1*32u;
  memset(A,0,sizeof *A);
  A->prefilter=(const uint32_t*)((uint8_t*)base+sizeof(BlfHeader)); A->prefilter_nblocks=h->nblocks1;
  A->host_filter=(const uint32_t*)((uint8_t*)base+sizeof(BlfHeader)+f1b); A->host_filter_nblocks=h->nblocks2;
  A->npurp=(int)h->npurp; for(int i=0;i<A->npurp&&i<8;i++) A->purposes[i]=h->purposes[i];
  if(A->npurp==0){ A->npurp=4; A->purposes[0]=44; A->purposes[1]=49; A->purposes[2]=84; A->purposes[3]=86; }
  double efpr=blf_fpr(h->nblocks1,h->n_addrs)*blf_fpr(h->nblocks2,h->n_addrs);
  fprintf(stderr,"bloom: loaded %s -- %llu addresses, dual %u+%u blocks (%.2f GiB), combined FPR ~%.1e, %d purpose(s)\n",
          path,(unsigned long long)h->n_addrs,h->nblocks1,h->nblocks2,(double)(f1b+(size_t)h->nblocks2*32u)/1073741824.0,efpr,A->npurp);
  return 0;
}
/* xpub SET (words, EC-free): a blocked bloom of account chaincodes; multi-purpose
   derive; host cull; reports the matched xpub + its purpose. Fans out over GPUs via
   the contiguous supervisor (like --xpub). */
static int mode_crack_xpubs(const Words*W,const AddrSet*xset,int require_ck,
                            unsigned long long ustart,unsigned long long ucount,const char*cu){
  char pat[2048]; wordset_pattern(W,pat,sizeof pat);
  struct rxe*r=rxe_parse(pat,0); if(!r||rxe_error(r)){fprintf(stderr,"rxe parse err\n");return 2;}
  mpz_t total; mpz_init(total); mpz_set(total,r->nitems);
  if(W->n>20){ fprintf(stderr,"v1 self-enumerate hit-index is u64: max 20 words. Got %d.\n",W->n); return 2; }
  if(g_print_total){ gmp_printf("%Zd\n",total); return 0; }
  build_module(cu);
  CUdeviceptr dd,dof,dln,dix; gpu_upload_words(W,&dd,&dof,&dln,&dix);
  double bpk=24.0; const char*e=getenv("BLOOM_BPK"); if(e){ double v=atof(e); if(v>=4) bpk=v; }
  uint32_t nb=bloom_nblocks((uint64_t)xset->n,bpk), mask=nb-1; size_t fbytes=(size_t)nb*8u*4u;
  uint32_t *hf=calloc(fbytes,1); for(long i=0;i<xset->n;i++) bloom_insert(hf,xset->ent[i].prog,mask);
  CUdeviceptr d_bloom=up(hf,fbytes); free(hf);
  unsigned long long z0=0; uint32_t hitcap=1u<<18;
  CUdeviceptr d_hits; CU(cuMemAlloc(&d_hits,(size_t)hitcap*sizeof(BloomHit)));
  CUdeviceptr d_hitcnt=up(&z0,4);
  int npurp=xset->npurp; CUdeviceptr d_purposes=up(xset->purposes,(size_t)npurp*sizeof(uint32_t));
  unsigned long long total_u=mpz_get_ui(total);
  unsigned long long start=ustart>total_u?total_u:ustart;
  unsigned long long count=ucount?ucount:(total_u-start); if(start+count>total_u) count=total_u-start;
  int n=W->n,size=W->n,rck=require_ck,tpb=128,grid=1024,reporting=(g_pflag||g_loginterval_ms);
  CUdeviceptr dhashed=up(&z0,8);
  char path[80]; snprintf(path,sizeof path,"xpub chaincode set (%ld, %d purpose%s)",xset->n,npurp,npurp>1?"s":"");
  Prog P; prog_init(&P,count,g_pflag,g_p_secs,g_loginterval_ms,g_csv,g_csv_own);
  prog_hdr(&P,"xpubs (words, EC-free, bloom)",g_target_str,g_pattern_str,path,0,0);
  fprintf(stderr,"bloom: %ld xpub(s), filter %.1f MiB, %d purpose(s), cull on host\n",xset->n,(double)fbytes/1048576.0,npurp);
  double intv=prog_intv(&P);
  BloomHit *hb=malloc((size_t)hitcap*sizeof(BloomHit));
  int found=0; unsigned long long best=~0ULL; int best_purpose=0; unsigned char best_cc[32]={0};
  unsigned long long swept=0,hashed=0, chunk=reporting?report_chunk(1):(64ULL<<20); double tt0=now_s();
  for(unsigned long long cstart=start; cstart<start+count; ){
    unsigned long long ccount=(start+count-cstart<chunk)?(start+count-cstart):chunk; double c0=now_s();
    unsigned int zc=0; CU(cuMemcpyHtoD(d_hitcnt,&zc,4));
    void*args[]={&dd,&dof,&dln,&dix,&n,&size,&cstart,&ccount,&d_purposes,&npurp,&rck,&d_bloom,&mask,&d_hits,&d_hitcnt,&hitcap,&dhashed};
    CU(cuLaunchKernel(kern("g_crack_bloom"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    double csecs=now_s()-c0; swept+=ccount; cstart+=ccount;
    unsigned int hc=0; CU(cuMemcpyDtoH(&hc,d_hitcnt,4));
    if(hc>0){ unsigned int ncp=hc>hitcap?hitcap:hc; CU(cuMemcpyDtoH(hb,d_hits,(size_t)ncp*sizeof(BloomHit)));
      for(unsigned int i=0;i<ncp;i++) if(aset_member(xset,hb[i].prog)){ found=1; if(hb[i].gidx<best){ best=hb[i].gidx; best_purpose=(int)hb[i].purpose; memcpy(best_cc,hb[i].prog,32); } } }
    CU(cuMemcpyDtoH(&hashed,dhashed,8)); prog_tick(&P,swept,rck?hashed:swept);
    if(found) break;
    chunk=next_chunk(ccount,csecs,intv,reporting);
  }
  free(hb); double secs=now_s()-tt0;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (enumerate+sieve->PBKDF2, EC-free%s)\n",
          swept,secs,secs>0?swept/secs/1e6:0.0, (found&&swept<count)?", early-exit":"");
  if(!found){ prog_finish(&P,"NOT_FOUND",0,path); printf("NOT FOUND (no candidate derived a target chain code)\n"); return 1; }
  mpz_t j; mpz_init_set_ui(j,best); rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
  char*t=buf; while(*t==' ')t++;
  const AEnt*me=aset_lookup(xset,best_cc);
  char fpath[64]; snprintf(fpath,sizeof fpath,"m/%d'/0'/0'",best_purpose);
  prog_finish(&P,"FOUND",t,fpath);
  printf("FOUND\n  index   : %llu\n  mnemonic: %s\n  path    : %s  (account 0)\n",best,t,fpath);
  if(me) printf("  xpub    : %s\n",me->str);
  mpz_clear(j); rxe_free(r); return 0;
}
/* Host-only: validate the blocked-bloom construction empirically -- build a
 * filter of N random uniform "programs", confirm zero false NEGATIVES, and
 * measure the false-POSITIVE rate across a sweep of bits/key. No GPU. */
static uint64_t bst_splitmix(uint64_t *x){   /* high-quality uniform in every bit */
  uint64_t z=(*x+=0x9E3779B97F4A7C15ULL);
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);
}
static void bst_fill(uint64_t *x,uint8_t *p,int n){ for(int i=0;i<n;i+=8){ uint64_t r=bst_splitmix(x); for(int b=0;b<8&&i+b<n;b++) p[i+b]=(uint8_t)(r>>(8*b)); } }
static int mode_bloom_selftest(long N){
  if(N<=0) N=1000000;
  uint64_t s=0x243f6a8885a308d3ULL;
  uint8_t *mem=malloc((size_t)N*20);
  for(long i=0;i<N;i++) bst_fill(&s,mem+(size_t)i*20,20);
  const long M=10000000;   /* non-member probes for the FPR estimate */
  printf("bloom self-test: N=%ld members, 256-bit block, k=%d (sliced, no re-hash), %ld FP probes\n",N,BLOOM_K,M);
  printf("  bits/key   filter      members      FP/probes        measured FPR    ~1 in\n");
  double bpks[]={8,12,16,20,24,32}; int fail=0;
  double prev_fpr=1.0;
  for(unsigned bi=0; bi<sizeof bpks/sizeof*bpks; bi++){
    double bpk=bpks[bi];
    uint32_t nb=bloom_nblocks((uint64_t)N,bpk), mask=nb-1;
    uint32_t *filt=calloc((size_t)nb*8,4);
    for(long i=0;i<N;i++) bloom_insert(filt,mem+(size_t)i*20,mask);
    long mok=0; for(long i=0;i<N;i++) mok+=bloom_probe(filt,mem+(size_t)i*20,mask);
    long fp=0; for(long i=0;i<M;i++){ uint8_t q[20]; bst_fill(&s,q,20); fp+=bloom_probe(filt,q,mask); }
    double fpr=(double)fp/M; double mib=(double)nb*32.0/1048576.0;
    printf("  %6.0f   %8.2f MiB  %ld/%ld  %8ld/%ld   %.3e   %.0f\n",
           bpk, mib, mok, N, fp, M, fpr, fp?1.0/fpr:0.0);
    if(mok!=N){ printf("    FAIL: %ld/%ld members missed (false negative -- construction bug)\n",N-mok,N); fail=1; }
    (void)prev_fpr; prev_fpr=fpr;
    free(filt);
  }
  free(mem);
  printf("==== bloom self-test: %s ====\n", fail?"FAILED (false negatives)":"PASSED (no false negatives; FPR falls with bits/key)");
  return fail?1:0;
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
   "  --addresses A,B,..      a SET of target addresses (blocked bloom + host cull);\n"
   "  --addresses-file PATH   ... or one per line. MIXED script types ok (derives each\n"
   "                          candidate under every type present; --purpose overrides the\n"
   "                          set). Works with --words and --template; fans out over GPUs.\n"
   "  --bloom FILE.blf        load a PREBUILT address bloom (any funded address); like\n"
   "                          --addresses but from a file (reports the matched hash160)\n"
   "  --bloom-build IN OUT    build a .blf from an address list IN (one per line) -> OUT.\n"
   "                          IN='-' reads stdin, e.g.  curl -s LIST | zcat | \\\n"
   "                            ... --bloom-build - f.blf --bloom-n 1500000000 --fpr 1e-12\n"
   "                          --bloom-n N  (required) approx address count, sizes the filters\n"
   "                          --fpr P      target combined false-positive rate (default 1e-12)\n"
   "  --xpub XPUB             account extended pubkey (EC-free chaincode compare)\n"
   "  --xpubs X,.. / --xpubs-file PATH  a SET of account xpubs (chaincode bloom,\n"
   "                          EC-free; tries purposes 44/49/84/86, --purpose overrides)\n"
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
   "  (default: crack jobs fan out over ALL visible GPUs; pin one with --device)\n"
   "  --device D              run on a single CUDA ordinal D (no fan-out)\n"
   "  --devices D0,D1,..      fan out one child process per GPU over disjoint index\n"
   "  --gpus N                slices; first to FOUND wins, siblings are killed\n"
   "                          (--devices 0,1  ==  --gpus 2). Honours an outer --start/--count.\n"
   "  --order P[:seed]        (--words/--template + --address) work-queue sweep order:\n"
   "                          first|ends|center|random -- exploit a prior on where\n"
   "                          the key is; the supervisor shows one consolidated line\n"
   "  --shards N              split the space into N fine shards (default ~8M each)\n"
   "  --print-total           print the job size (index count) and exit (no GPU)\n"
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
  const char *addresses=0,*addresses_file=0;   /* bloom target set */
  const char *xpubs=0,*xpubs_file=0;            /* xpub (chaincode) bloom set */
  const char *bloom_file=0,*bbuild_in=0,*bbuild_out=0;  /* prebuilt address bloom */
  unsigned long long bloom_n=0; double bloom_fpr=1e-12;   /* --bloom-build sizing */
  const char *resumearg=0;
  int devs[16], ndev=0, device_set=0;   /* --devices/--gpus: fan out one child per GPU */
  int order_policy=0, order_given=0; unsigned long long order_seed=0; long nshards_arg=0;  /* work-queue */
  int bloom_selftest=0; long bloom_selftest_n=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--words")&&i+1<argc) words=argv[++i];
    else if(!strcmp(argv[i],"--xpub")&&i+1<argc) xpub=argv[++i];
    else if(!strcmp(argv[i],"--target-chaincode")&&i+1<argc) tcc_hex=argv[++i];
    else if(!strcmp(argv[i],"--purpose")&&i+1<argc){ npurp=0; purpose_set=1; char*s=strtok(argv[++i],","); while(s&&npurp<8){purposes[npurp++]=(uint32_t)atoi(s);s=strtok(0,",");} }
    else if(!strcmp(argv[i],"--no-checksum")) require_ck=0;
    else if(!strcmp(argv[i],"--compact")) compact=1;
    else if(!strcmp(argv[i],"--no-compact")) compact=0;
    else if(!strcmp(argv[i],"-p")){ g_pflag=1; if(i+1<argc && (isdigit((unsigned char)argv[i+1][0])||argv[i+1][0]=='.')) g_p_secs=atof(argv[++i]); }
    else if(!strcmp(argv[i],"--loginterval")&&i+1<argc){
      /* capture raw first (the fan-out supervisor re-reads it), and DEFER the fopen:
         a supervisor must not open the log -- each child opens its own per-device file. */
      snprintf(g_li_raw,sizeof g_li_raw,"%s",argv[++i]);
      char*colon=strchr(g_li_raw,':'); if(colon){ *colon=0; g_csv_path=colon+1; }
      g_loginterval_ms=atoi(g_li_raw); }
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
    else if(!strcmp(argv[i],"--addresses")&&i+1<argc) addresses=argv[++i];
    else if(!strcmp(argv[i],"--addresses-file")&&i+1<argc) addresses_file=argv[++i];
    else if(!strcmp(argv[i],"--xpubs")&&i+1<argc) xpubs=argv[++i];
    else if(!strcmp(argv[i],"--xpubs-file")&&i+1<argc) xpubs_file=argv[++i];
    else if(!strcmp(argv[i],"--bloom")&&i+1<argc) bloom_file=argv[++i];
    else if(!strcmp(argv[i],"--bloom-build")&&i+2<argc){ bbuild_in=argv[++i]; bbuild_out=argv[++i]; }
    else if(!strcmp(argv[i],"--bloom-n")&&i+1<argc) bloom_n=strtoull(argv[++i],0,10);
    else if(!strcmp(argv[i],"--fpr")&&i+1<argc){ double v=atof(argv[++i]); if(v>0&&v<1) bloom_fpr=v; }
    else if(!strcmp(argv[i],"--template")&&i+1<argc) templ=argv[++i];
    else if(!strcmp(argv[i],"--pattern")&&i+1<argc) patt=argv[++i];
    else if(!strcmp(argv[i],"--nth")) nthmode=1;
    else if(!strcmp(argv[i],"--no-nth")) nthmode=0;
    else if(!strcmp(argv[i],"--decode")&&i+1<argc) decodearg=argv[++i];
    else if(!strcmp(argv[i],"--change")&&i+1<argc) a_changes=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--gap")&&i+1<argc) a_gap=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--kernels")&&i+1<argc) cu=argv[++i];
    else if(!strcmp(argv[i],"--resume")&&i+1<argc) resumearg=argv[++i];
    else if(!strcmp(argv[i],"--device")&&i+1<argc){ g_device=atoi(argv[++i]); device_set=1; }
    else if(!strcmp(argv[i],"--devices")&&i+1<argc){ ndev=0; char*s=strtok(argv[++i],","); while(s&&ndev<16){ devs[ndev++]=atoi(s); s=strtok(0,","); } }
    else if(!strcmp(argv[i],"--gpus")&&i+1<argc){ int N=atoi(argv[++i]); if(N>16)N=16; ndev=N; for(int d=0;d<N;d++)devs[d]=d; }
    else if(!strcmp(argv[i],"--print-total")) g_print_total=1;
    else if(!strcmp(argv[i],"--warm")) g_warm=1;
    else if(!strcmp(argv[i],"--worker")) g_worker=1;
    else if(!strcmp(argv[i],"--shards")&&i+1<argc) nshards_arg=atol(argv[++i]);
    else if(!strcmp(argv[i],"--order")&&i+1<argc){ char*o=argv[++i]; char*colon=strchr(o,':'); if(colon){*colon=0; order_seed=strtoull(colon+1,0,10);}
      if(!strcmp(o,"first"))order_policy=0; else if(!strcmp(o,"ends"))order_policy=1; else if(!strcmp(o,"center"))order_policy=2; else if(!strcmp(o,"random"))order_policy=3;
      else { fprintf(stderr,"--order: first|ends|center|random[:seed], got '%s'\n",o); return 2; } order_given=1; }
    else if(!strcmp(argv[i],"--devtag")) g_devtag=1;
    else if(!strcmp(argv[i],"--bloom-selftest")){ bloom_selftest=1; if(i+1<argc&&isdigit((unsigned char)argv[i+1][0])) bloom_selftest_n=atol(argv[++i]); }
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
  /* --warm: build/JIT the module (populate the PTX cache) and exit */
  if(g_warm){ build_module(cu); return 0; }
  /* Fan-out applies only to real crack jobs, not gate/util modes (which run once
     on a single device). */
  int addr_set = (addresses||addresses_file||bloom_file);   /* a bloom target SET */
  int crackjob = !(profile||ecgate||addrgate||missgate||decodearg||rankarg||recon||dumpv||g_print_total||g_worker||bloom_selftest||bbuild_out);
  /* Default: use ALL local GPUs. If the user pinned neither --device nor --devices,
     enumerate the visible CUDA devices and fan out across them (honours
     CUDA_VISIBLE_DEVICES). A single-GPU box falls through to the in-process path. */
  if(crackjob && ndev==0 && !device_set){
    CU(cuInit(0)); int nd=0; cuDeviceGetCount(&nd);
    if(nd>1){ ndev = nd>16?16:nd; for(int d=0;d<ndev;d++) devs[d]=d;
      fprintf(stderr,"(auto: %d GPUs detected -> fanning out; use --device N to pin one)\n",ndev); }
  }
  /* Route crack jobs to a multi-GPU driver. The address-words path has a
     persistent-worker WORK-QUEUE (fine shards + ordering + consolidated stats);
     other modes still use the contiguous supervisor until they get setup/sweep. */
  if(crackjob && (ndev>0 || order_given || nshards_arg>0)){
    int tgt=(address||addr_set);           /* single addr OR a bloom set */
    if(tgt && ((words && !templ) || templ)){ int nd=ndev>0?ndev:1; if(ndev==0) devs[0]=g_device;
      return run_workqueue(argc,argv,devs,nd,cstart,ccount,order_policy,order_seed,nshards_arg); }
    if(ndev>0) return run_supervisor(argc,argv,devs,ndev,cstart,ccount);
  }
  /* open the deferred CSV now that we know we're an actual cracker, not a supervisor
     (--resume already opened its own append handle, so only open when unset) */
  if(g_loginterval_ms && !g_csv){
    if(g_csv_path){ g_csv=fopen(g_csv_path,"w"); g_csv_own=1; if(!g_csv){ fprintf(stderr,"cannot open %s\n",g_csv_path); return 2; } }
    else g_csv=stderr;
  }
  /* gate/util modes need no pattern */
  if(bloom_selftest) return mode_bloom_selftest(bloom_selftest_n);
  if(bbuild_out){ return build_bloom_file(bbuild_in,bbuild_out,bloom_n,bloom_fpr); }
  if(decodearg){ uint8_t pr[32]; int pl,pu; if(decode_address(decodearg,pr,&pl,&pu)) return 2;
    char h[66]; tohex_(pr,pl,h); printf("%d %s\n",pu,h); return 0; }
  if(profile) return mode_profile(cu);
  if(ecgate) return mode_ec_gate(ecgate,cu);
  if(addrgate) return mode_addr_gate(addrgate,cu);
  if(missgate){ const char*t=templ?templ:"trial [:bip39:] gloom dragon try dirt rapid crawl soon fatal tool chronic rapid ladder salmon palace expect enrich helmet truth receive [:bip39:] [:bip39:] [:bip39:]"; return mode_miss_gate(t,missgate_n,cu); }
  /* Missing-word ([:bip39-en:]) template + address (single) or --addresses (set) */
  if(templ){
    uint8_t prog[32]={0}; int purpose; AddrSet A; const AddrSet*aset=0;
    if(addresses||addresses_file||bloom_file){ int apu=84;
      if(bloom_file){ if(load_bloom_file(bloom_file,&A)) return 2; if(A.npurp) apu=(int)A.purposes[0]; }
      else { if(build_addrset(addresses,addresses_file,&A,&apu)) return 2; }
      if(purpose_set){ A.npurp=npurp>8?8:npurp; for(int k=0;k<A.npurp;k++) A.purposes[k]=purposes[k]; }
      aset=&A; purpose=purpose_set?(int)purposes[0]:apu; if(A.ent) memcpy(prog,A.ent[0].prog,32); }
    else if(address){ int apl,apu; if(decode_address(address,prog,&apl,&apu)) return 2; purpose=purpose_set?(int)purposes[0]:apu; }
    else { fprintf(stderr,"--template needs --address or --addresses\n"); return 2; }
    /* auto: construction if the last position is [:bip39-en:]; --nth/--no-nth override */
    int rc;
    if(g_worker) rc=run_worker_missing(templ,prog,purpose,a_changes,a_gap,nthmode,cu,aset);
    else rc=mode_missing(templ,prog,purpose,a_changes,a_gap,nthmode,cstart,ccount,cu,aset);
    if(aset) aset_free(&A);
    return rc;
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
  if(addresses||addresses_file||bloom_file){ AddrSet A; int apu=84; uint8_t dummy[32]={0};
    if(bloom_file){ if(load_bloom_file(bloom_file,&A)) return 2; if(A.npurp) apu=(int)A.purposes[0]; }
    else { if(build_addrset(addresses,addresses_file,&A,&apu)) return 2; }
    if(purpose_set){ A.npurp=npurp>8?8:npurp; for(int k=0;k<A.npurp;k++) A.purposes[k]=purposes[k]; }  /* --purpose overrides the derive set */
    uint8_t*tp = A.ent?A.ent[0].prog:dummy;
    int purpose = purpose_set?(int)purposes[0]:apu; int rc;
    if(g_worker) rc=run_worker_addr(&W,tp,purpose,a_changes,a_gap,require_ck,compact,cu,&A);
    else rc=mode_crack_addr(&W,tp,purpose,a_changes,a_gap,require_ck,compact,cstart,ccount,cu,&A);
    aset_free(&A); return rc; }
  if(address){ uint8_t prog[32]; int aproglen,apurpose; if(decode_address(address,prog,&aproglen,&apurpose))return 2;
    int purpose = purpose_set?(int)purposes[0]:apurpose;
    if(g_worker) return run_worker_addr(&W,prog,purpose,a_changes,a_gap,require_ck,compact,cu,0);
    return mode_crack_addr(&W,prog,purpose,a_changes,a_gap,require_ck,compact,cstart,ccount,cu,0); }
  if(g_worker){ fprintf(stderr,"--worker supports the --words/--address and --template/--address paths\n"); return 2; }
  if(xpubs||xpubs_file){ AddrSet Xs; if(build_xpubset(xpubs,xpubs_file,&Xs)) return 2;
    if(purpose_set){ Xs.npurp=npurp>8?8:npurp; for(int k=0;k<Xs.npurp;k++) Xs.purposes[k]=purposes[k]; }
    int rc=mode_crack_xpubs(&W,&Xs,require_ck,cstart,ccount,cu); aset_free(&Xs); return rc; }
  uint8_t tcc[32];
  if(xpub){ if(xpub_chaincode(xpub,tcc)) return 2; }
  else if(tcc_hex){ if(hex2bin(tcc_hex,tcc,32)!=32){ fprintf(stderr,"target-chaincode must be 32 bytes hex\n"); return 2; } }
  else { fprintf(stderr,"need --xpub or --target-chaincode\n"); usage(); return 2; }
  return mode_crack(&W,tcc,purposes,npurp,require_ck,cstart,ccount,cu);
}
