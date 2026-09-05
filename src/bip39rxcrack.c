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

/* ------------------------ NVRTC build (with include inliner) ------------ */
static CUcontext g_ctx; static CUmodule g_mod;
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
  char *src=inline_includes(slurp(cu_path),cu_path);
  /* PTX cache: NVRTC compile of the full EC+taproot module is slow (~2-3 min);
     cache the PTX keyed by source+arch hash so unchanged source loads instantly. */
  char key[128]; snprintf(key,sizeof key,"%llx",fnv1a(src)^fnv1a(arch));
  char cpath[256]; snprintf(cpath,sizeof cpath,"/tmp/bip39rxcrack_ptx_%s.ptx",key);
  char *ptx=0; FILE*cf=fopen(cpath,"rb");
  if(cf && !getenv("CRACK_NOCACHE")){
    fseek(cf,0,SEEK_END); long pn=ftell(cf); fseek(cf,0,SEEK_SET); ptx=malloc(pn+1);
    if(fread(ptx,1,pn,cf)==(size_t)pn){ ptx[pn]=0; } else { free(ptx); ptx=0; } fclose(cf);
  } else if(cf) fclose(cf);
  if(!ptx){
    const char *opts[]={ arch };
    nvrtcProgram prog; NVR(nvrtcCreateProgram(&prog,src,"crack_kernels.cu",0,0,0));
    nvrtcResult cr=nvrtcCompileProgram(prog,1,opts);
    size_t logn=0; nvrtcGetProgramLogSize(prog,&logn);
    if(logn>1){ char*log=malloc(logn); nvrtcGetProgramLog(prog,log);
      if(cr!=NVRTC_SUCCESS||getenv("CRACK_VERBOSE")) fprintf(stderr,"NVRTC log:\n%s\n",log);
      free(log);
    }
    if(cr!=NVRTC_SUCCESS){ fprintf(stderr,"kernel compile failed\n"); exit(2); }
    size_t ptxn=0; NVR(nvrtcGetPTXSize(prog,&ptxn)); ptx=malloc(ptxn); NVR(nvrtcGetPTX(prog,ptx)); nvrtcDestroyProgram(&prog);
    FILE*wf=fopen(cpath,"wb"); if(wf){ fwrite(ptx,1,strlen(ptx),wf); fclose(wf); }
  }
  CUdevice dev; CU(cuInit(0)); CU(cuDeviceGet(&dev,0));
  char name[128]; int M=0,m=0; cuDeviceGetName(name,sizeof name,dev);
  cuDeviceGetAttribute(&M,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
  cuDeviceGetAttribute(&m,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
  CU(cuCtxCreate(&g_ctx,0,dev)); CU(cuModuleLoadDataEx(&g_mod,ptx,0,0,0));
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
  W->n=0; char buf[1024]; snprintf(buf,sizeof buf,"%s",csv);
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
  void*args[]={&dd,&dof,&dln,&dix,&n,&size,&start_lo,&start_hi,&count,&dpu,&npurp,&dtc,&require_ck,&dhi,&dfound,&dpur};
  fprintf(stderr,"enumerating %llu candidates from index %llu (checksum-%s, %d purpose%s) on GPU...\n",
          count,start_lo,require_ck?"ON":"OFF",npurp,npurp>1?"s":"");
  struct timeval t0,t1; gettimeofday(&t0,0);
  CU(cuLaunchKernel(kern("g_crack"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  gettimeofday(&t1,0);
  double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (raw enumerate+sieve%s)\n",
          count,secs,count/secs/1e6, require_ck?"->PBKDF2 on survivors":"+PBKDF2 all");
  int found=0,fpur=0; unsigned long long hidx=0;
  CU(cuMemcpyDtoH(&found,dfound,4)); CU(cuMemcpyDtoH(&hidx,dhi,8)); CU(cuMemcpyDtoH(&fpur,dpur,4));
  if(!found){ printf("NOT FOUND (no candidate derived the target chain code)\n"); return 1; }
  /* render the found mnemonic via librxe (canonical) */
  mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
  char*t=buf; while(*t==' ')t++;
  printf("FOUND\n");
  printf("  index   : %llu\n",hidx);
  printf("  mnemonic: %s\n",t);
  printf("  path    : m/%d'/0'/0'  (purpose %d, account 0)\n",fpur,fpur);
  mpz_clear(j); rxe_free(r); return 0;
}

/* ------------------ Regime B: words permutation, ADDRESS target --------- */
static int mode_crack_addr(const Words*W,const uint8_t tprog[32],int purpose,
                           uint32_t change,uint32_t index,int require_ck,int compact,
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
  int tpb=128,grid=1024; struct timeval t0,t1; double secs=0;
  if(compact && require_ck){
    unsigned long long survcap=count/4+1000000ULL; /* 12w pass ~1/16; 4x margin */
    unsigned long long z64=0; CUdeviceptr dsurv,dctr; CU(cuMemAlloc(&dsurv,survcap*8)); dctr=up(&z64,8);
    fprintf(stderr,"regime B (COMPACTED): sieve %llu -> dense survivors -> full-warp PBKDF2 -> m/%d'/0'/0'/%u/%u ...\n",count,purpose,change,index);
    gettimeofday(&t0,0);
    void*sa[]={&dd,&dof,&dln,&dix,&n,&size,&start,&count,&dsurv,&survcap,&dctr};
    CU(cuLaunchKernel(kern("g_sieve_perm"),grid,1,1,tpb,1,1,0,0,sa,0)); CU(cuCtxSynchronize());
    unsigned long long nsurv=0; CU(cuMemcpyDtoH(&nsurv,dctr,8));
    if(nsurv>survcap){ fprintf(stderr,"  WARN: survivors %llu > cap %llu (dropped; perf-only)\n",nsurv,survcap); nsurv=survcap; }
    void*pa[]={&dd,&dof,&dln,&dix,&n,&size,&dsurv,&nsurv,&pu,&change,&index,&dtp,&dhi,&dfound};
    CU(cuLaunchKernel(kern("g_pbkdf2_perm"),grid,1,1,tpb,1,1,0,0,pa,0)); CU(cuCtxSynchronize());
    gettimeofday(&t1,0); secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
    fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s  (%llu survivors -> dense PBKDF2)\n",count,secs,count/secs/1e6,nsurv);
  } else {
    void*args[]={&dd,&dof,&dln,&dix,&n,&size,&start,&count,&pu,&change,&index,&dtp,&require_ck,&dhi,&dfound};
    fprintf(stderr,"regime B: %llu permutations from %llu (checksum-%s) -> m/%d'/0'/0'/%u/%u ...\n",
            count,start,require_ck?"ON":"OFF",purpose,change,index);
    gettimeofday(&t0,0);
    CU(cuLaunchKernel(kern("g_crack_addr"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
    gettimeofday(&t1,0); secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
    fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (sieve->PBKDF2+EC on survivors)\n",count,secs,count/secs/1e6);
  }
  (void)secs;
  int found=0; unsigned long long hidx=0; CU(cuMemcpyDtoH(&found,dfound,4)); CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ printf("NOT FOUND\n"); return 1; }
  mpz_t j; mpz_init_set_ui(j,hidx); rxe_seek(r,j); char buf[MN_STRIDE]; rxe_current(buf,sizeof buf,r);
  char*t=buf; while(*t==' ')t++;
  printf("FOUND\n  index (canonical): %llu\n  mnemonic: %s\n  path    : m/%d'/0'/0'/%u/%u\n",hidx,t,purpose,change,index);
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
/* Parse "w0 w1 [:bip39-en:] .. [:bip39-en:]" -> tmpl[W] word indices (unknown=0),
 * upos[U] unknown positions, W, U, last_unknown. */
static int parse_template(const char*tpl,uint32_t tmpl[32],int upos[32],int*W,int*U,int*last_unknown){
  static char names[2048][16]; load_wordlist(names);
  char buf[2048]; snprintf(buf,sizeof buf,"%s",tpl); *W=0; *U=0;
  char*tok=strtok(buf," \t\r\n");
  while(tok){ if(*W>=32){fprintf(stderr,"max 32 positions\n");return -1;}
    if(!strcmp(tok,"[:bip39-en:]")){ tmpl[*W]=0; upos[(*U)++]=*W; }
    else { int bi=bip39_index(names,tok); if(bi<0){fprintf(stderr,"'%s' is not a BIP39 word\n",tok);return -1;} tmpl[*W]=(uint32_t)bi; }
    (*W)++; tok=strtok(0," \t\r\n"); }
  int vc[6]={12,15,18,21,24,0}; int ok=0; for(int i=0;vc[i];i++) if(*W==vc[i]) ok=1;
  if(!ok){ fprintf(stderr,"word count %d not in {12,15,18,21,24}\n",*W); return -1; }
  if(*U<1){ fprintf(stderr,"template has no [:bip39-en:] unknown\n"); return -1; }
  *last_unknown = (upos[*U-1]==*W-1);
  return 0;
}
static const char* WLNAME(int idx){ static char names[2048][16]; static int loaded=0; if(!loaded){load_wordlist(names);loaded=1;} return names[idx]; }
static int mode_missing(const char*tpl,const uint8_t tprog[32],int purpose,uint32_t change,uint32_t index,
                        int use_nth,unsigned long long ustart,unsigned long long ucount,const char*cu){
  uint32_t tmpl[32]; int upos[32],W,U,last_unknown;
  if(parse_template(tpl,tmpl,upos,&W,&U,&last_unknown)) return 2;
  int CS=W/3, freebits=11-CS;
  int nth = use_nth && last_unknown;
  if(use_nth && !last_unknown){ fprintf(stderr,"[:Nth:] construction needs the LAST position to be [:bip39-en:]\n"); return 2; }
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
  uint32_t pu=(uint32_t)purpose; int Uarg = nth ? U : U;   /* kernel U: nth uses U-1 others but reads U-1; pass total for nth */
  int passU = nth ? U : U;
  void*args[]={&dwl,&dwoff,&dwlen,&dtmpl,&W,&dupos,&passU,&start,&count,&pu,&change,&index,&dtp,&dhi,&dfound,&dhg};
  int tpb=128,grid=1024;
  fprintf(stderr,"missing-word %s: W=%d, %d unknown(s)%s -> %llu candidates (m/%d'/0'/0'/%u/%u)...\n",
          nth?"[:Nth:] CONSTRUCTION":"baseline sieve", W, U, nth?" (last=checksum-constructed)":"", count, purpose, change, index);
  struct timeval t0,t1; gettimeofday(&t0,0);
  CU(cuLaunchKernel(kern(nth?"g_crack_nth":"g_crack_missing"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  gettimeofday(&t1,0); double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  fprintf(stderr,"  swept %llu candidates in %.3fs = %.3f Mcand/s (%s)\n",count,secs,count/secs/1e6,
          nth?"all valid, no sieve":"sieve->PBKDF2+EC on survivors");
  int found=0; unsigned long long hidx=0; CU(cuMemcpyDtoH(&found,dfound,4)); CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ printf("NOT FOUND\n"); return 1; }
  uint32_t hg[32]; CU(cuMemcpyDtoH(hg,dhg,W*sizeof(uint32_t)));
  printf("FOUND\n  index    : %llu\n  found words:",hidx);
  for(int i=0;i<U;i++) printf(" [pos %d]=%s",upos[i],WLNAME((int)hg[upos[i]]));
  printf("\n  mnemonic :");
  for(int p=0;p<W;p++) printf(" %s",WLNAME((int)hg[p]));
  printf("\n  path     : m/%d'/0'/0'/%u/%u\n",purpose,change,index);
  (void)CS;(void)Uarg; return 0;
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
                           int purpose,uint32_t change,uint32_t index,
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
  void*args[]={&dhctx,&pwidth,&start,&count,&pu,&change,&index,&dtp,&dhi,&dfound};
  int tpb=128,grid=1024;
  fprintf(stderr,"regime A: fixed mnemonic, passphrase [0-9]{%d} = %llu candidates from %llu (purpose %d)...\n",pwidth,count,start,purpose);
  struct timeval t0,t1; gettimeofday(&t0,0);
  CU(cuLaunchKernel(kern("g_crack_pass"),grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
  gettimeofday(&t1,0); double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_usec-t0.tv_usec)/1e6;
  fprintf(stderr,"  swept %llu candidates in %.2fs = %.3f Mcand/s (PBKDF2+EC every candidate)\n",count,secs,count/secs/1e6);
  int found=0; unsigned long long hidx=0; CU(cuMemcpyDtoH(&found,dfound,4)); CU(cuMemcpyDtoH(&hidx,dhi,8));
  if(!found){ printf("NOT FOUND\n"); return 1; }
  char pass[24]; unsigned long long q=hidx; for(int p=pwidth-1;p>=0;p--){ pass[p]=(char)('0'+(int)(q%10)); q/=10; } pass[pwidth]=0;
  printf("FOUND\n  index      : %llu\n  passphrase : %s\n  path       : m/%d'/0'/0'/%u/%u\n",hidx,pass,purpose,change,index);
  return 0;
}

/* --------------------------------- CLI ---------------------------------- */
static void usage(void){
  fprintf(stderr,
   "usage: bip39rxcrack --words \"w1 w2 .. wN\" [target] [opts]\n"
   "  target: --xpub XPUB | --target-chaincode HEX(32B)\n"
   "  --purpose 44,49,84,86   BIP purposes to try (default 84)\n"
   "  --no-checksum           disable the BIP39 checksum sieve\n"
   "  --recon-gate [N]        gate GPU unrank+reconstruction vs librxe (default 512)\n"
   "  --rank \"w1 .. wN\"       print the librxe index of one arrangement\n"
   "  --dump-valid N          print 'gpuvalid<TAB>mnemonic' for N samples\n"
   "  --start IDX --count N    shard the index space (multi-GPU / windowed proof)\n"
   "  --limit N               alias of --count (raw-throughput sweep)\n"
   "  --kernels PATH          crack_kernels.cu (default cuda/crack_kernels.cu)\n");
}
int main(int argc,char**argv){
  /* NVRTC 13 dlopens libnvrtc-builtins.so.13.x via LD_LIBRARY_PATH; ensure it's
     set (re-exec once so the loader picks it up at startup). */
  { const char *nl="/usr/local/cuda-13.2/lib64"; const char *cur=getenv("LD_LIBRARY_PATH");
    if(!cur || !strstr(cur,nl)){ char buf[4096]; snprintf(buf,sizeof buf,"%s%s%s",nl,cur?":":"",cur?cur:"");
      setenv("LD_LIBRARY_PATH",buf,1); execv("/proc/self/exe",argv); /* falls through on failure */ } }
  const char *words=0,*xpub=0,*tcc_hex=0,*rankarg=0,*cu="cuda/crack_kernels.cu";
  int recon=0,recon_n=512,dumpv=0,dumpv_n=0,require_ck=1,compact=0; const char*ecgate=0,*addrgate=0;
  unsigned long long cstart=0,ccount=0;
  uint32_t purposes[8]={84}; int npurp=1,purpose_set=0;
  const char *mnemonic=0,*passphrase=0,*address=0,*decodearg=0,*templ=0; uint32_t achange=0,aindex=0; int nthmode=-1; /* -1 auto, 1 force, 0 off */
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--words")&&i+1<argc) words=argv[++i];
    else if(!strcmp(argv[i],"--xpub")&&i+1<argc) xpub=argv[++i];
    else if(!strcmp(argv[i],"--target-chaincode")&&i+1<argc) tcc_hex=argv[++i];
    else if(!strcmp(argv[i],"--purpose")&&i+1<argc){ npurp=0; purpose_set=1; char*s=strtok(argv[++i],","); while(s&&npurp<8){purposes[npurp++]=(uint32_t)atoi(s);s=strtok(0,",");} }
    else if(!strcmp(argv[i],"--no-checksum")) require_ck=0;
    else if(!strcmp(argv[i],"--compact")) compact=1;
    else if(!strcmp(argv[i],"--recon-gate")){ recon=1; if(i+1<argc&&argv[i+1][0]!='-') recon_n=atoi(argv[++i]); }
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
    else if(!strcmp(argv[i],"--nth")) nthmode=1;
    else if(!strcmp(argv[i],"--no-nth")) nthmode=0;
    else if(!strcmp(argv[i],"--decode")&&i+1<argc) decodearg=argv[++i];
    else if(!strcmp(argv[i],"--change")&&i+1<argc) achange=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--index")&&i+1<argc) aindex=(uint32_t)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--kernels")&&i+1<argc) cu=argv[++i];
    else { fprintf(stderr,"unknown arg: %s\n",argv[i]); usage(); return 2; }
  }
  /* gate/util modes need no pattern */
  if(decodearg){ uint8_t pr[32]; int pl,pu; if(decode_address(decodearg,pr,&pl,&pu)) return 2;
    char h[66]; tohex_(pr,pl,h); printf("%d %s\n",pu,h); return 0; }
  if(ecgate) return mode_ec_gate(ecgate,cu);
  if(addrgate) return mode_addr_gate(addrgate,cu);
  /* Missing-word ([:bip39-en:]) template + address target */
  if(templ){
    if(!address){ fprintf(stderr,"--template needs --address\n"); return 2; }
    uint8_t prog[32]; int apl,apu; if(decode_address(address,prog,&apl,&apu)) return 2;
    int purpose = purpose_set?(int)purposes[0]:apu;
    /* auto: construction if the last position is [:bip39-en:]; --nth/--no-nth override */
    int use_nth = (nthmode==1) ? 1 : (nthmode==0 ? 0 : 1);
    return mode_missing(templ,prog,purpose,achange,aindex,use_nth,cstart,ccount,cu);
  }
  /* Regime A: fixed mnemonic + passphrase [0-9]{N} + address target */
  if(mnemonic && passphrase){
    int w=passphrase_width(passphrase);
    if(w<0){ fprintf(stderr,"v1 passphrase supports [0-9]{N} only, got '%s'\n",passphrase); return 2; }
    if(!address){ fprintf(stderr,"regime A needs --address (p2pkh/p2sh target)\n"); return 2; }
    uint8_t prog[32]; int aproglen,apurpose; if(decode_address(address,prog,&aproglen,&apurpose)) return 2;
    int purpose = purpose_set ? (int)purposes[0] : apurpose;
    return mode_crack_pass(mnemonic,w,prog,purpose,achange,aindex,cstart,ccount,cu);
  }
  if(!words){ usage(); return 2; }
  Words W; parse_words(&W,words);
  if(rankarg) return mode_rank(&W,rankarg);
  if(recon)   return mode_recon_gate(&W,recon_n,cu);
  if(dumpv)   return mode_dump_valid(&W,dumpv_n,cu);
  if(address){ uint8_t prog[32]; int aproglen,apurpose; if(decode_address(address,prog,&aproglen,&apurpose))return 2;
    int purpose = purpose_set?(int)purposes[0]:apurpose;
    return mode_crack_addr(&W,prog,purpose,achange,aindex,require_ck,compact,cstart,ccount,cu); }
  uint8_t tcc[32];
  if(xpub){ if(xpub_chaincode(xpub,tcc)) return 2; }
  else if(tcc_hex){ if(hex2bin(tcc_hex,tcc,32)!=32){ fprintf(stderr,"target-chaincode must be 32 bytes hex\n"); return 2; } }
  else { fprintf(stderr,"need --xpub or --target-chaincode\n"); usage(); return 2; }
  return mode_crack(&W,tcc,purposes,npurp,require_ck,cstart,ccount,cu);
}
