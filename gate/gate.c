/* gate.c -- Phase-1 crypto gate harness for bip39rxcrack.
 *
 * Compiles cuda/gate_kernels.cu at runtime via NVRTC (compute_90 PTX; the
 * sm_120 driver JIT-forwards it), runs each kernel over vectors generated from
 * the reseed39 browser oracle, and diffs byte-exact. Prints PASS/FAIL per
 * stage with counts. THE LAW: expected values come from the oracle / published
 * BIP vectors, never a sibling GPU model.
 *
 *   ./gate [vectors_dir] [kernels.cu]     (defaults: ./vectors ./cuda/gate_kernels.cu)
 *
 * Exit 0 iff all stages PASS with zero mismatches.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <nvrtc.h>

#define CU(x)  do{ CUresult r=(x); if(r!=CUDA_SUCCESS){ const char*s=0; cuGetErrorString(r,&s); \
                   fprintf(stderr,"CUDA error %d (%s) at %s:%d\n",r,s?s:"?",__FILE__,__LINE__); exit(2);} }while(0)
#define NVR(x) do{ nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){ \
                   fprintf(stderr,"NVRTC error %d (%s) at %s:%d\n",r,nvrtcGetErrorString(r),__FILE__,__LINE__); exit(2);} }while(0)

static CUcontext g_ctx;
static CUmodule  g_mod;

/* ------- read whole file ------- */
static char *slurp(const char *path){
  FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",path); exit(2);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){fprintf(stderr,"read %s\n",path);exit(2);} b[n]=0; fclose(f); return b;
}
static int hexv(int c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1;}
/* decode hex token ("-" => empty) into out; return byte count */
static int hex2bin(const char *s, unsigned char *out){
  if(s[0]=='-'&&s[1]==0) return 0;
  int n=0; while(s[0]&&s[1]){ int hi=hexv(s[0]),lo=hexv(s[1]); if(hi<0||lo<0) break; out[n++]=(hi<<4)|lo; s+=2; }
  return n;
}
static void tohex(const unsigned char *b,int n,char *out){ static const char*h="0123456789abcdef";
  for(int i=0;i<n;i++){ out[i*2]=h[b[i]>>4]; out[i*2+1]=h[b[i]&15]; } out[n*2]=0; }

/* ------- NVRTC compile + module load ------- */
/* Resolve a single local `#include "header"` by textual substitution, so NVRTC
 * never needs a filesystem include path (avoids quoted-include search quirks).
 * Header is looked up relative to the .cu's directory. */
static char *inline_includes(char *src, const char *cu_path){
  const char *tag="#include \"";
  char *p=strstr(src,tag);
  if(!p) return src;
  char dir[512]; snprintf(dir,sizeof dir,"%s",cu_path);
  char *sl=strrchr(dir,'/'); if(sl) *sl=0; else strcpy(dir,".");
  char *q=p+strlen(tag); char *e=strchr(q,'"'); if(!e) return src;
  char hdr[512]; int hn=(int)(e-q); if(hn>500) hn=500; memcpy(hdr,q,hn); hdr[hn]=0;
  char full[1100]; snprintf(full,sizeof full,"%s/%s",dir,hdr);
  char *inc=slurp(full);
  char *lineend=strchr(e,'\n'); if(!lineend) lineend=e+1; else lineend++;
  size_t pre=p-src, post=strlen(lineend), ilen=strlen(inc);
  char *out=malloc(pre+ilen+post+2);
  memcpy(out,src,pre); memcpy(out+pre,inc,ilen); out[pre+ilen]='\n';
  memcpy(out+pre+ilen+1,lineend,post+1);
  free(inc); free(src);
  return inline_includes(out,cu_path);   // resolve any further includes
}
static void build_module(const char *cu_path){
  char *src=inline_includes(slurp(cu_path),cu_path);
  const char *opts[]={ "--gpu-architecture=compute_120" };
  nvrtcProgram prog;
  NVR(nvrtcCreateProgram(&prog,src,"gate_kernels.cu",0,0,0));
  nvrtcResult cr=nvrtcCompileProgram(prog,1,opts);
  size_t logn=0; nvrtcGetProgramLogSize(prog,&logn);
  if(logn>1){
    char*log=malloc(logn); nvrtcGetProgramLog(prog,log);
    if(cr!=NVRTC_SUCCESS||getenv("GATE_VERBOSE")) fprintf(stderr,"NVRTC log:\n%s\n",log);
    free(log);
  }
  if(cr!=NVRTC_SUCCESS){ fprintf(stderr,"kernel compile failed\n"); exit(2); }
  size_t ptxn=0; NVR(nvrtcGetPTXSize(prog,&ptxn));
  char *ptx=malloc(ptxn); NVR(nvrtcGetPTX(prog,ptx)); nvrtcDestroyProgram(&prog);
  CUdevice dev; CU(cuInit(0)); CU(cuDeviceGet(&dev,0));
  char name[128]; int ccM=0,ccm=0;
  cuDeviceGetName(name,sizeof name,dev);
  cuDeviceGetAttribute(&ccM,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
  cuDeviceGetAttribute(&ccm,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
  CU(cuCtxCreate(&g_ctx,0,dev));
  CU(cuModuleLoadDataEx(&g_mod,ptx,0,0,0));
  printf("device: %s (sm_%d%d), NVRTC13->compute_120 PTX->sm_120\n\n",name,ccM,ccm);
  free(ptx); free(src);
}
static CUfunction kern(const char *n){ CUfunction f; CU(cuModuleGetFunction(&f,g_mod,n)); return f; }
static void launch(CUfunction f,int n,void**args){
  int tpb=64, grid=(n+tpb-1)/tpb;
  CU(cuLaunchKernel(f,grid,1,1,tpb,1,1,0,0,args,0)); CU(cuCtxSynchronize());
}
static CUdeviceptr dup2dev(const void*h,size_t n){ CUdeviceptr d; CU(cuMemAlloc(&d,n?n:1)); if(n) CU(cuMemcpyHtoD(d,h,n)); return d; }

/* dynamic byte buffer */
typedef struct { unsigned char *b; size_t n,cap; } Buf;
static void bput(Buf*B,const unsigned char*p,int n){ if(B->n+n>B->cap){ B->cap=(B->n+n)*2+64; B->b=realloc(B->b,B->cap);} memcpy(B->b+B->n,p,n); B->n+=n; }

/* getline wrapper iterating a file's lines */
#define MAXTOK 8192

/* ============================ Stage 1: SHA-512 ========================== */
static int gate_sha512(const char *dir){
  char path[512]; snprintf(path,sizeof path,"%s/vec_sha512.txt",dir);
  FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(2);}
  Buf data={0}; int *off=0,*len=0,cap=0,n=0; unsigned char (*exp)[64]=0;
  char *line=0; size_t lc=0; ssize_t rd;
  unsigned char tmp[MAXTOK];
  while((rd=getline(&line,&lc,f))>0){
    if(rd<3) continue;
    char inhex[MAXTOK], outhex[MAXTOK]; int L;
    if(sscanf(line,"%d %s %s",&L,inhex,outhex)!=3) continue;
    if(n==cap){cap=cap?cap*2:64; off=realloc(off,cap*sizeof(int)); len=realloc(len,cap*sizeof(int)); exp=realloc(exp,cap*64);}
    off[n]=(int)data.n; int bl=hex2bin(inhex,tmp); len[n]=bl; bput(&data,tmp,bl);
    hex2bin(outhex,exp[n]); n++;
  }
  free(line); fclose(f);
  CUdeviceptr dd=dup2dev(data.b,data.n), doff=dup2dev(off,n*sizeof(int)), dlen=dup2dev(len,n*sizeof(int)), dout;
  CU(cuMemAlloc(&dout,(size_t)n*64));
  void*args[]={&dd,&doff,&dlen,&n,&dout};
  launch(kern("g_sha512"),n,args);
  unsigned char *got=malloc((size_t)n*64); CU(cuMemcpyDtoH(got,dout,(size_t)n*64));
  int bad=0;
  for(int i=0;i<n;i++) if(memcmp(got+i*64,exp[i],64)){ if(bad<3){char a[130],b[130];tohex(got+i*64,64,a);tohex(exp[i],64,b);
      fprintf(stderr,"  MISMATCH sha512 #%d (len %d)\n    got  %s\n    want %s\n",i,len[i],a,b);} bad++; }
  cuMemFree(dd);cuMemFree(doff);cuMemFree(dlen);cuMemFree(dout);
  free(got);free(data.b);free(off);free(len);free(exp);
  printf("  [%s] SHA-512            : %d/%d match\n", bad?"FAIL":"PASS", n-bad, n);
  return bad;
}

/* ============================ Stage 2: PBKDF2 ========================== */
static int gate_pbkdf2(const char *dir){
  char path[512]; snprintf(path,sizeof path,"%s/vec_pbkdf2.txt",dir);
  FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(2);}
  Buf pw={0},salt={0}; int *poff=0,*plen=0,*soff=0,*slen=0,cap=0,n=0; unsigned char (*exp)[64]=0; int over128=0;
  char *line=0; size_t lc=0; ssize_t rd; unsigned char tmp[MAXTOK];
  while((rd=getline(&line,&lc,f))>0){
    int PL,SL; char ph[MAXTOK],sh[MAXTOK],oh[MAXTOK];
    if(sscanf(line,"%d %s %d %s %s",&PL,ph,&SL,sh,oh)!=5) continue;
    if(n==cap){cap=cap?cap*2:64; poff=realloc(poff,cap*4);plen=realloc(plen,cap*4);soff=realloc(soff,cap*4);slen=realloc(slen,cap*4);exp=realloc(exp,cap*64);}
    poff[n]=(int)pw.n; int b1=hex2bin(ph,tmp); plen[n]=b1; bput(&pw,tmp,b1); if(b1>128) over128++;
    soff[n]=(int)salt.n; int b2=hex2bin(sh,tmp); slen[n]=b2; bput(&salt,tmp,b2);
    hex2bin(oh,exp[n]); n++;
  }
  free(line); fclose(f);
  CUdeviceptr dpw=dup2dev(pw.b,pw.n),dpo=dup2dev(poff,n*4),dpl=dup2dev(plen,n*4),
              dsa=dup2dev(salt.b,salt.n),dso=dup2dev(soff,n*4),dsl=dup2dev(slen,n*4),dout;
  CU(cuMemAlloc(&dout,(size_t)n*64));
  void*args[]={&dpw,&dpo,&dpl,&dsa,&dso,&dsl,&n,&dout};
  launch(kern("g_pbkdf2"),n,args);
  unsigned char *got=malloc((size_t)n*64); CU(cuMemcpyDtoH(got,dout,(size_t)n*64));
  int bad=0;
  for(int i=0;i<n;i++) if(memcmp(got+i*64,exp[i],64)){ if(bad<3){char a[130],b[130];tohex(got+i*64,64,a);tohex(exp[i],64,b);
      fprintf(stderr,"  MISMATCH pbkdf2 #%d (pwlen %d)\n    got  %s\n    want %s\n",i,plen[i],a,b);} bad++; }
  cuMemFree(dpw);cuMemFree(dpo);cuMemFree(dpl);cuMemFree(dsa);cuMemFree(dso);cuMemFree(dsl);cuMemFree(dout);
  free(got);free(pw.b);free(salt.b);free(poff);free(plen);free(soff);free(slen);free(exp);
  printf("  [%s] PBKDF2-HMAC-SHA512 : %d/%d match  (2048 iters; %d vectors >128B key/pre-hash)\n",
         bad?"FAIL":"PASS", n-bad, n, over128);
  return bad;
}

/* ============================ Stage 3: checksum ========================= */
static int gate_checksum(const char *dir){
  char path[512]; snprintf(path,sizeof path,"%s/vec_checksum.txt",dir);
  FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(2);}
  Buf data={0}; int *off=0,*len=0,*csb=0,cap=0,n=0; unsigned char *expfb=0;
  char *line=0; size_t lc=0; ssize_t rd; unsigned char tmp[MAXTOK];
  while((rd=getline(&line,&lc,f))>0){
    int EL,CS; char eh[MAXTOK],fbh[8];
    if(sscanf(line,"%d %s %d %s",&EL,eh,&CS,fbh)!=4) continue;
    if(n==cap){cap=cap?cap*2:64; off=realloc(off,cap*4);len=realloc(len,cap*4);csb=realloc(csb,cap*4);expfb=realloc(expfb,cap);}
    off[n]=(int)data.n; int bl=hex2bin(eh,tmp); len[n]=bl; bput(&data,tmp,bl);
    csb[n]=CS; unsigned char fb; hex2bin(fbh,&fb); expfb[n]=fb; n++;
  }
  free(line); fclose(f);
  CUdeviceptr dd=dup2dev(data.b,data.n),doff=dup2dev(off,n*4),dlen=dup2dev(len,n*4),dout;
  CU(cuMemAlloc(&dout,n)); void*args[]={&dd,&doff,&dlen,&n,&dout};
  launch(kern("g_sha256"),n,args);
  unsigned char *got=malloc(n); CU(cuMemcpyDtoH(got,dout,n));
  int bad=0, per[9]={0}, tot[9]={0};
  for(int i=0;i<n;i++){
    int cs=csb[i]; unsigned char mask = cs>=8?0xff:(unsigned char)(0xff<<(8-cs));
    tot[cs]++;
    if((got[i]&mask)!=(expfb[i]&mask)){ if(bad<3) fprintf(stderr,"  MISMATCH checksum #%d cs=%d got %02x want %02x\n",i,cs,got[i],expfb[i]); bad++; per[cs]++; }
  }
  cuMemFree(dd);cuMemFree(doff);cuMemFree(dlen);cuMemFree(dout); free(got);free(data.b);free(off);free(len);free(csb);free(expfb);
  printf("  [%s] BIP39 checksum     : %d/%d match  (cs bits: 12w=4 15w=5 18w=6 21w=7 24w=8; %d/%d/%d/%d/%d each)\n",
         bad?"FAIL":"PASS", n-bad, n, tot[4],tot[5],tot[6],tot[7],tot[8]);
  return bad;
}

/* ============================ Stage 4: account ========================= */
static int gate_account(const char *dir){
  char path[512]; snprintf(path,sizeof path,"%s/vec_account.txt",dir);
  FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(2);}
  Buf seed={0}; int *off=0,*len=0,*lev=0,cap=0,n=0; unsigned int *idx=0;
  unsigned char (*expc)[32]=0,(*expk)[32]=0;
  char *line=0; size_t lc=0; ssize_t rd; unsigned char tmp[MAXTOK];
  while((rd=getline(&line,&lc,f))>0){
    int SL,L; unsigned int i0,i1,i2; char sh[MAXTOK],ch[80],kh[80];
    if(sscanf(line,"%d %s %d %u %u %u %s %s",&SL,sh,&L,&i0,&i1,&i2,ch,kh)!=8) continue;
    if(n==cap){cap=cap?cap*2:64; off=realloc(off,cap*4);len=realloc(len,cap*4);lev=realloc(lev,cap*4);
      idx=realloc(idx,cap*3*sizeof(unsigned int));expc=realloc(expc,cap*32);expk=realloc(expk,cap*32);}
    off[n]=(int)seed.n; int bl=hex2bin(sh,tmp); len[n]=bl; bput(&seed,tmp,bl);
    lev[n]=L; idx[n*3]=i0; idx[n*3+1]=i1; idx[n*3+2]=i2;
    hex2bin(ch,expc[n]); hex2bin(kh,expk[n]); n++;
  }
  free(line); fclose(f);
  CUdeviceptr ds=dup2dev(seed.b,seed.n),doff=dup2dev(off,n*4),dlen=dup2dev(len,n*4),
              dlev=dup2dev(lev,n*4),didx=dup2dev(idx,n*3*sizeof(unsigned int)),dc,dk;
  CU(cuMemAlloc(&dc,(size_t)n*32)); CU(cuMemAlloc(&dk,(size_t)n*32));
  void*args[]={&ds,&doff,&dlen,&dlev,&didx,&n,&dc,&dk};
  launch(kern("g_account"),n,args);
  unsigned char *gc=malloc((size_t)n*32),*gk=malloc((size_t)n*32);
  CU(cuMemcpyDtoH(gc,dc,(size_t)n*32)); CU(cuMemcpyDtoH(gk,dk,(size_t)n*32));
  int bad=0;
  for(int i=0;i<n;i++){
    int bc=memcmp(gc+i*32,expc[i],32), bk=memcmp(gk+i*32,expk[i],32);
    if(bc||bk){ if(bad<3){char a[70],b[70];
        tohex(gc+i*32,32,a);tohex(expc[i],32,b); fprintf(stderr,"  MISMATCH account #%d L=%d\n    chaincode got %s\n              want %s\n",i,lev[i],a,b);
        tohex(gk+i*32,32,a);tohex(expk[i],32,b); fprintf(stderr,"    priv      got %s\n              want %s\n",a,b);} bad++; }
  }
  cuMemFree(ds);cuMemFree(doff);cuMemFree(dlen);cuMemFree(dlev);cuMemFree(didx);cuMemFree(dc);cuMemFree(dk);
  free(gc);free(gk);free(seed.b);free(off);free(len);free(lev);free(idx);free(expc);free(expk);
  printf("  [%s] BIP32 EC-free deriv: %d/%d match  (master + hardened CKDpriv chaincode+priv; incl. BIP32 vector-1)\n",
         bad?"FAIL":"PASS", n-bad, n);
  return bad;
}

int main(int argc,char**argv){
  { const char *nl="/usr/local/cuda-13.2/lib64"; const char *cur=getenv("LD_LIBRARY_PATH");
    if(!cur || !strstr(cur,nl)){ char buf[4096]; snprintf(buf,sizeof buf,"%s%s%s",nl,cur?":":"",cur?cur:"");
      setenv("LD_LIBRARY_PATH",buf,1); execv("/proc/self/exe",argv); } }
  const char *dir = argc>1?argv[1]:"vectors";
  const char *cu  = argc>2?argv[2]:"cuda/gate_kernels.cu";
  printf("bip39rxcrack Phase-1 crypto gates (byte-exact vs reseed39 oracle / published BIP vectors)\n");
  build_module(cu);
  int bad=0;
  bad += gate_sha512(dir);
  bad += gate_pbkdf2(dir);
  bad += gate_checksum(dir);
  bad += gate_account(dir);
  printf("\n==== %s: %d total mismatches ====\n", bad?"GATE FAILED":"GATE PASSED", bad);
  return bad?1:0;
}
