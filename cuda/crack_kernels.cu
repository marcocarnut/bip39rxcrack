/* crack_kernels.cu -- Phase-2 self-enumerate xpub crack (single GPU).
 *
 * The GPU enumerates its OWN candidates from a global index (PLAN §3): each
 * thread unranks its permutation index on-die (decode_perm, byte-identical to
 * librxe), runs the pipeline, and reports hits. No candidate transfer over
 * PCIe. All device crypto is the Phase-1-proven bip39_device.cuh.
 *
 *   digits -> checksum sieve -> PBKDF2 -> hardened EC-free derive -> chaincode compare
 *
 * Base words (their strings + 11-bit BIP39 indices) are the only per-run data
 * the host uploads -- O(pattern), not O(candidates). The word's 11-bit index IS
 * the odometer digit; the password string is rebuilt from the chosen words.
 */
#include "bip39_device.cuh"
#include "secp256k1_device.cuh"

#define MN_STRIDE 256   /* max reconstructed mnemonic bytes (24w * ~9 + slack) */
#define HARD 0x80000000u

/* Build the BIP39 mnemonic bytes for permutation position digits dig[0..size-1]
 * from the base-word string table; single-space separated, no trailing space
 * (English wordlist is ASCII so NFKD is identity). Returns length. Also fills
 * g[p] = 11-bit BIP39 index of the word at position p. */
__device__ int build_mnemonic(const u8 *bw_data, const int *bw_off, const int *bw_len,
                              const u32 *bw_idx, const int *dig, int size,
                              u8 *out_mn, u32 *g){
  int L=0;
  for(int p=0; p<size; p++){
    int d=dig[p];
    if(p) out_mn[L++]=' ';
    const u8 *w=bw_data+bw_off[d]; int wl=bw_len[d];
    for(int c=0;c<wl;c++) out_mn[L++]=w[c];
    g[p]=bw_idx[d];
  }
  return L;
}

/* Reconstruction + checksum-validity gate: for each provided index, emit the
 * mnemonic bytes and the checksum-valid flag (to diff vs the oracle host-side). */
extern "C" __global__ void g_recon(const u8 *bw_data, const int *bw_off, const int *bw_len,
                                   const u32 *bw_idx, int n, int size,
                                   const unsigned long long *idx_lo, const unsigned long long *idx_hi,
                                   int nidx, u8 *out_mn, int *out_len, u8 *out_valid){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nidx) return;
  (void)idx_hi;                          /* <=20 words -> index fits u64 */
  unsigned long long j=idx_lo[i];
  int dig[32]; decode_perm(dig,n,size,j);
  u32 g[32];
  int L=build_mnemonic(bw_data,bw_off,bw_len,bw_idx,dig,size,out_mn+(size_t)i*MN_STRIDE,g);
  out_len[i]=L;
  out_valid[i]=(u8)bip39_checksum_ok(g,size);
}

/* The crack: grid-stride over [start, start+count). checksum-ON sieves before
 * PBKDF2. On a chaincode match, record the (lowest) global index + purpose.
 * hit_index is u64 (valid for <=20-word permutations, 20! < 2^64); the host
 * refuses >20-word self-enumerate until the u128 hit path lands. */
extern "C" __global__ void g_crack(const u8 *bw_data, const int *bw_off, const int *bw_len,
                                   const u32 *bw_idx, int n, int size,
                                   unsigned long long start_lo, unsigned long long start_hi,
                                   unsigned long long count,
                                   const u32 *purposes, int npurp,
                                   const u8 *target_cc, int require_ck,
                                   unsigned long long *hit_index, int *hit_found, int *hit_purpose){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  (void)start_hi;                        /* <=20 words -> start+count fit u64 */
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
      t<count; t+=stride){
    unsigned long long j=start_lo+t;
    int dig[32]; decode_perm(dig,n,size,j);
    u8 mn[MN_STRIDE]; u32 g[32];
    int L=build_mnemonic(bw_data,bw_off,bw_len,bw_idx,dig,size,mn,g);
    if(require_ck && !bip39_checksum_ok(g,size)) continue;
    u8 seed[64]; const u8 salt[8]={'m','n','e','m','o','n','i','c'};
    pbkdf2_seed(mn,(u32)L,salt,8,2048,seed);
    for(int pp=0; pp<npurp; pp++){
      u32 idx3[3]={ purposes[pp]|HARD, 0u|HARD, 0u|HARD };
      u8 cc[32],kk[32]; derive_hardened(seed,64,idx3,3,cc,kk);
      int eq=1; for(int b=0;b<32;b++) if(cc[b]!=target_cc[b]){ eq=0; break; }
      if(eq){
        unsigned long long ji=j;    // <=20w fits u64
        atomicMin(hit_index, ji);
        atomicExch(hit_found, 1);
        atomicExch(hit_purpose, (int)purposes[pp]);
      }
    }
  }
}

/* EC gate: for each 32-byte scalar, emit compressed pub(33), hash160(20),
 * p2pkh program(20), p2sh-p2wpkh program(20) -- diffed vs the oracle host-side. */
extern "C" __global__ void g_ec(const u8 *sk, int n, u8 *pub, u8 *h160, u8 *p2pkh, u8 *p2sh){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  u8 p[33]; scalar_mul_G(sk+(size_t)i*32, p);
  for(int b=0;b<33;b++) pub[(size_t)i*33+b]=p[b];
  u8 h[20]; hash160(p,33,h); for(int b=0;b<20;b++) h160[(size_t)i*20+b]=h[b];
  u8 pr[32]; int pl; pub_to_program(p,44,pr,&pl); for(int b=0;b<20;b++) p2pkh[(size_t)i*20+b]=pr[b];
  pub_to_program(p,49,pr,&pl); for(int b=0;b<20;b++) p2sh[(size_t)i*20+b]=pr[b];
}

/* seed->address gate: for each (seed, purpose, change, index) emit the 20-byte
 * program -- diffed vs the oracle pubToTarget(addressNode(...)) host-side. */
extern "C" __global__ void g_addr(const u8 *seed, const u32 *purpose, const u32 *change,
                                  const u32 *index, int n, u8 *prog){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  int pl; derive_address(seed+(size_t)i*64, purpose[i], change[i], index[i], prog+(size_t)i*32, &pl);
}

/* Regime A: FIXED mnemonic, passphrase varies over [0-9]{width} (decimal
 * odometer). No checksum sieve (fixed mnemonic already valid). Each thread:
 * passphrase digits from its index -> salt="mnemonic"+digits -> PBKDF2 ->
 * derive_address -> compare 20-byte program. */
/* one-time: build the fixed-mnemonic HMAC key context (regime A). */
extern "C" __global__ void g_hctx_init(const u8 *mn, int mnlen, HCTX *out){
  if(blockIdx.x*blockDim.x+threadIdx.x!=0) return;
  hmac512_ctx(mn,(u32)mnlen,out);
}
extern "C" __global__ void g_crack_pass(const HCTX *hkey, int pwidth,
                                        unsigned long long start, unsigned long long count,
                                        u32 purpose, u32 change, u32 index,
                                        const u8 *target_prog,
                                        unsigned long long *hit_index, int *hit_found){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; t<count; t+=stride){
    unsigned long long j=start+t;
    u8 salt[8+16]; salt[0]='m';salt[1]='n';salt[2]='e';salt[3]='m';salt[4]='o';salt[5]='n';salt[6]='i';salt[7]='c';
    unsigned long long q=j; for(int p=pwidth-1;p>=0;p--){ salt[8+p]=(u8)('0'+(int)(q%10)); q/=10; }
    u32 slen=8+pwidth;
    u8 seed[64]; pbkdf2_seed_ctx(hkey,salt,slen,2048,seed);
    u8 prog[32]; int pl; derive_address(seed,purpose,change,index,prog,&pl);
    int eq=1; for(int b=0;b<pl;b++) if(prog[b]!=target_prog[b]){ eq=0; break; }
    if(eq){ atomicMin(hit_index,j); atomicExch(hit_found,1); }
  }
}

/* Regime B (address target): words permutation self-enumerate -> checksum sieve
 * -> PBKDF2 -> derive_address(purpose,change,index) -> 20-byte program compare. */
extern "C" __global__ void g_crack_addr(const u8 *bw_data, const int *bw_off, const int *bw_len,
                                        const u32 *bw_idx, int n, int size,
                                        unsigned long long start_lo, unsigned long long count,
                                        u32 purpose, u32 change, u32 index,
                                        const u8 *target_prog, int require_ck,
                                        unsigned long long *hit_index, int *hit_found){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; t<count; t+=stride){
    unsigned long long j=start_lo+t;
    int dig[32]; decode_perm(dig,n,size,j);
    u8 mn[MN_STRIDE]; u32 g[32];
    int L=build_mnemonic(bw_data,bw_off,bw_len,bw_idx,dig,size,mn,g);
    if(require_ck && !bip39_checksum_ok(g,size)) continue;
    u8 seed[64]; const u8 salt[8]={'m','n','e','m','o','n','i','c'};
    pbkdf2_seed(mn,(u32)L,salt,8,2048,seed);
    u8 prog[32]; int pl; derive_address(seed,purpose,change,index,prog,&pl);
    int eq=1; for(int b=0;b<pl;b++) if(prog[b]!=target_prog[b]){ eq=0; break; }
    if(eq){ atomicMin(hit_index,j); atomicExch(hit_found,1); }
  }
}

/* Build the fixed-base comb table (one thread; one-time). d_comb must be set to
 * `table` by the host AFTER this returns. table = 64*16*8 u64 (x[4]||y[4]). */
extern "C" __global__ void g_comb_init(u64 *table){
  if(blockIdx.x*blockDim.x+threadIdx.x!=0) return;
  jpt base; fe_set(&base.X,(const fe*)FE_GX); fe_set(&base.Y,(const fe*)FE_GY); fe_set_u64(&base.Z,1); base.inf=0;
  for(int i=0;i<64;i++){
    jpt acc=base;                         // j=1 : 16^i * G
    for(int j=1;j<16;j++){
      fe x,y; j_affine(&acc,&x,&y);
      u64 *e=table+((size_t)(i*16+j))*8;
      e[0]=x.v[0];e[1]=x.v[1];e[2]=x.v[2];e[3]=x.v[3];
      e[4]=y.v[0];e[5]=y.v[1];e[6]=y.v[2];e[7]=y.v[3];
      if(j<15) j_add(&acc,&acc,&base);    // (j+1)*16^i*G
    }
    for(int d=0;d<4;d++) j_double(&base,&base);   // base *= 16
  }
}

/* ===================== Missing-word (odometer) crack ===================== */
/* Build a mnemonic from full word indices g[0..W-1] using the 2048-word list. */
__device__ int build_mnemonic_wl(const u8 *wl, const int *wloff, const int *wllen,
                                 const u32 *g, int W, u8 *out){
  int L=0;
  for(int p=0;p<W;p++){ if(p) out[L++]=' '; int wi=(int)g[p]; const u8 *w=wl+wloff[wi]; int wn=wllen[wi];
    for(int c=0;c<wn;c++) out[L++]=w[c]; }
  return L;
}
/* BASELINE: fixed known words + U unknown [:bip39-en:] positions (base-2048
 * odometer, 11 bits each -- no division since 2048=2^11) -> checksum sieve ->
 * PBKDF2 -> derive_address -> program compare. */
extern "C" __global__ void g_crack_missing(const u8 *wl, const int *wloff, const int *wllen,
                                           const u32 *tmpl, int W, const int *upos, int U,
                                           unsigned long long start, unsigned long long count,
                                           u32 purpose, u32 change, u32 index, const u8 *target_prog,
                                           unsigned long long *hit_index, int *hit_found, u32 *hit_g){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; t<count; t+=stride){
    unsigned long long j=start+t;
    u32 g[24]; for(int p=0;p<W;p++) g[p]=tmpl[p];
    for(int u=0;u<U;u++) g[upos[u]]=(u32)((j>>(11*u))&2047ULL);
    if(!bip39_checksum_ok(g,W)) continue;
    u8 mn[MN_STRIDE]; int L=build_mnemonic_wl(wl,wloff,wllen,g,W,mn);
    u8 seed[64]; const u8 salt[8]={'m','n','e','m','o','n','i','c'};
    pbkdf2_seed(mn,(u32)L,salt,8,2048,seed);
    u8 prog[32]; int pl; derive_address(seed,purpose,change,index,prog,&pl);
    int eq=1; for(int b=0;b<pl;b++) if(prog[b]!=target_prog[b]){ eq=0; break; }
    if(eq){ atomicMin(hit_index,j); atomicExch(hit_found,1); for(int p=0;p<W;p++) hit_g[p]=g[p]; }
  }
}
/* [:Nth:] CONSTRUCTION: the LAST position (W-1) is the unknown checksum word.
 * upos[0..U-2] are the OTHER unknowns (base-2048); the last word's free
 * (11-CS) entropy bits are enumerated and the CS checksum bits CONSTRUCTED from
 * SHA-256(entropy) -> every candidate is checksum-valid (no sieve, no divergence).
 * Index space = 2048^(U-1) * 2^(11-CS). */
extern "C" __global__ void g_crack_nth(const u8 *wl, const int *wloff, const int *wllen,
                                       const u32 *tmpl, int W, const int *upos, int U,
                                       unsigned long long start, unsigned long long count,
                                       u32 purpose, u32 change, u32 index, const u8 *target_prog,
                                       unsigned long long *hit_index, int *hit_found, u32 *hit_g){
  int total=W*11, ENT=total*32/33, CS=total-ENT;   // CS = W/3
  int freebits=11-CS;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; t<count; t+=stride){
    unsigned long long j=start+t;
    u32 g[24]; for(int p=0;p<W;p++) g[p]=tmpl[p];
    unsigned long long lastfree=j & ((1ULL<<freebits)-1);
    unsigned long long rest=j>>freebits;
    for(int u=0;u<U-1;u++) g[upos[u]]=(u32)((rest>>(11*u))&2047ULL);  // other unknowns
    g[W-1]=(u32)(lastfree<<CS);                       // top freebits set, checksum=0
    // entropy = top ENT bits of g[0..W-1]; SHA-256 -> top CS bits = checksum
    u8 ent[32]; for(int i=0;i<32;i++) ent[i]=0;
    for(int i=0;i<W;i++){ u32 idx=g[i]; for(int b=0;b<11;b++){ int pos=i*11+b; if(pos<ENT && ((idx>>(10-b))&1)) ent[pos>>3]|=(0x80>>(pos&7)); } }
    u8 dig[32]; sha256_1blk(ent,(u32)(ENT/8),dig);
    u32 cs=(dig[0]>>(8-CS))&((1u<<CS)-1);
    g[W-1]=(u32)((lastfree<<CS)|cs);                  // the constructed valid last word
    u8 mn[MN_STRIDE]; int L=build_mnemonic_wl(wl,wloff,wllen,g,W,mn);
    u8 seed[64]; const u8 salt[8]={'m','n','e','m','o','n','i','c'};
    pbkdf2_seed(mn,(u32)L,salt,8,2048,seed);
    u8 prog[32]; int pl; derive_address(seed,purpose,change,index,prog,&pl);
    int eq=1; for(int b=0;b<pl;b++) if(prog[b]!=target_prog[b]){ eq=0; break; }
    if(eq){ atomicMin(hit_index,j); atomicExch(hit_found,1); for(int p=0;p<W;p++) hit_g[p]=g[p]; }
  }
}

/* ===================== Stream compaction (permutation) =================== */
/* Phase 1: sieve -- decode_perm + checksum only (coherent, no PBKDF2). Write
 * the global index of each survivor into a dense array via an atomic counter. */
extern "C" __global__ void g_sieve_perm(const u8 *bw_data, const int *bw_off, const int *bw_len,
                                        const u32 *bw_idx, int n, int size,
                                        unsigned long long start_lo, unsigned long long count,
                                        unsigned long long *surv, unsigned long long survcap,
                                        unsigned long long *counter){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long t=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; t<count; t+=stride){
    unsigned long long j=start_lo+t;
    int dig[32]; decode_perm(dig,n,size,j);
    u32 g[32]; for(int p=0;p<size;p++) g[p]=bw_idx[dig[p]];
    if(bip39_checksum_ok(g,size)){ unsigned long long idx=atomicAdd(counter,1ULL); if(idx<survcap) surv[idx]=j; }
  }
}
/* Phase 2: dense PBKDF2 over the survivor array -- every lane is a survivor
 * (no warp divergence). */
extern "C" __global__ void g_pbkdf2_perm(const u8 *bw_data, const int *bw_off, const int *bw_len,
                                         const u32 *bw_idx, int n, int size,
                                         const unsigned long long *surv, unsigned long long nsurv,
                                         u32 purpose, u32 change, u32 index, const u8 *target_prog,
                                         unsigned long long *hit_index, int *hit_found){
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  for(unsigned long long s=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; s<nsurv; s+=stride){
    unsigned long long j=surv[s];
    int dig[32]; decode_perm(dig,n,size,j);
    u8 mn[MN_STRIDE]; u32 g[32];
    int L=build_mnemonic(bw_data,bw_off,bw_len,bw_idx,dig,size,mn,g);
    u8 seed[64]; const u8 salt[8]={'m','n','e','m','o','n','i','c'};
    pbkdf2_seed(mn,(u32)L,salt,8,2048,seed);
    u8 prog[32]; int pl; derive_address(seed,purpose,change,index,prog,&pl);
    int eq=1; for(int b=0;b<pl;b++) if(prog[b]!=target_prog[b]){ eq=0; break; }
    if(eq){ atomicMin(hit_index,j); atomicExch(hit_found,1); }
  }
}
