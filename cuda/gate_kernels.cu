/* gate_kernels.cu -- standalone CUDA kernels for the Phase-1 crypto gates.
 *
 * Compiled at runtime by NVRTC (targeting compute_90 PTX; the sm_120 driver
 * JIT-forwards it -- same route bip38rxcrack uses on this Blackwell box). One
 * thread per test vector; correctness-only (perf comes later).
 *
 * Kernels:
 *   g_sha512   : SHA-512(msg)                         -> 64 bytes
 *   g_sha256   : SHA-256(msg, <=55B one block) 1st B  -> 1 byte
 *   g_pbkdf2   : PBKDF2-HMAC-SHA512(pw, salt, 2048)   -> 64 bytes
 *   g_account  : seedToMaster + `levels` hardened     -> chaincode(32)+priv(32)
 *
 * All four are gated byte-exact against the browser oracle in gen_vectors.js.
 */
typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;

/* ============================ SHA-512 (streaming) ======================== */
__device__ __constant__ u64 K512[80] = {
 0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
 0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
 0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
 0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
 0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
 0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
 0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
 0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
 0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
 0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
 0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
 0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
 0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
 0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
 0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
 0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
 0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
 0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
 0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
 0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL };

__device__ __forceinline__ u64 ror64(u64 x, int n){ return (x >> n) | (x << (64 - n)); }

struct S512 { u64 H[8]; u8 buf[128]; u32 blen; u64 total; };

__device__ void s512_init(S512 *s){
  s->H[0]=0x6a09e667f3bcc908ULL; s->H[1]=0xbb67ae8584caa73bULL;
  s->H[2]=0x3c6ef372fe94f82bULL; s->H[3]=0xa54ff53a5f1d36f1ULL;
  s->H[4]=0x510e527fade682d1ULL; s->H[5]=0x9b05688c2b3e6c1fULL;
  s->H[6]=0x1f83d9abfb41bd6bULL; s->H[7]=0x5be0cd19137e2179ULL;
  s->blen=0; s->total=0;
}
__device__ void s512_block(S512 *s, const u8 *p){
  u64 w[80];
  #pragma unroll
  for(int i=0;i<16;i++){
    w[i]=((u64)p[i*8]<<56)|((u64)p[i*8+1]<<48)|((u64)p[i*8+2]<<40)|((u64)p[i*8+3]<<32)
        |((u64)p[i*8+4]<<24)|((u64)p[i*8+5]<<16)|((u64)p[i*8+6]<<8)|((u64)p[i*8+7]);
  }
  for(int i=16;i<80;i++){
    u64 s0=ror64(w[i-15],1)^ror64(w[i-15],8)^(w[i-15]>>7);
    u64 s1=ror64(w[i-2],19)^ror64(w[i-2],61)^(w[i-2]>>6);
    w[i]=w[i-16]+s0+w[i-7]+s1;
  }
  u64 a=s->H[0],b=s->H[1],c=s->H[2],d=s->H[3],e=s->H[4],f=s->H[5],g=s->H[6],h=s->H[7];
  for(int i=0;i<80;i++){
    u64 S1=ror64(e,14)^ror64(e,18)^ror64(e,41);
    u64 ch=(e&f)^((~e)&g);
    u64 t1=h+S1+ch+K512[i]+w[i];
    u64 S0=ror64(a,28)^ror64(a,34)^ror64(a,39);
    u64 maj=(a&b)^(a&c)^(b&c);
    u64 t2=S0+maj;
    h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
  }
  s->H[0]+=a;s->H[1]+=b;s->H[2]+=c;s->H[3]+=d;s->H[4]+=e;s->H[5]+=f;s->H[6]+=g;s->H[7]+=h;
}
__device__ void s512_update(S512 *s, const u8 *data, u32 len){
  s->total += len;
  while(len){
    if(s->blen==0 && len>=128){ s512_block(s,data); data+=128; len-=128; continue; }
    u32 take=128-s->blen; if(take>len) take=len;
    for(u32 i=0;i<take;i++) s->buf[s->blen+i]=data[i];
    s->blen+=take; data+=take; len-=take;
    if(s->blen==128){ s512_block(s,s->buf); s->blen=0; }
  }
}
__device__ void s512_final(S512 *s, u8 out[64]){
  u64 bits = s->total * 8ULL;
  u8 pad = 0x80;
  s512_update(s,&pad,1);
  u8 zero=0;
  while(s->blen!=112) s512_update(s,&zero,1);      // pad to 112 mod 128
  u8 lb[16];
  for(int i=0;i<8;i++) lb[i]=0;                     // high 64 bits of length = 0
  for(int i=0;i<8;i++) lb[15-i]=(u8)(bits>>(8*i));
  s512_update(s,lb,16);                            // now blen==0, block flushed
  for(int i=0;i<8;i++){
    out[i*8+0]=(u8)(s->H[i]>>56); out[i*8+1]=(u8)(s->H[i]>>48);
    out[i*8+2]=(u8)(s->H[i]>>40); out[i*8+3]=(u8)(s->H[i]>>32);
    out[i*8+4]=(u8)(s->H[i]>>24); out[i*8+5]=(u8)(s->H[i]>>16);
    out[i*8+6]=(u8)(s->H[i]>>8);  out[i*8+7]=(u8)(s->H[i]);
  }
}
__device__ void sha512(const u8 *msg, u32 len, u8 out[64]){
  S512 s; s512_init(&s); s512_update(&s,msg,len); s512_final(&s,out);
}

/* ============================ HMAC-SHA512 ================================ */
/* Streams ipad/opad blocks so no concat buffer is needed; key>128 pre-hashed. */
__device__ void hmac512_key(const u8 *key, u32 klen, u8 kb[128]){
  for(int i=0;i<128;i++) kb[i]=0;
  if(klen>128){ u8 hk[64]; sha512(key,klen,hk); for(int i=0;i<64;i++) kb[i]=hk[i]; }
  else { for(u32 i=0;i<klen;i++) kb[i]=key[i]; }
}
/* Precompute the two padded key blocks, then each HMAC over msg is 2 sha512s. */
struct HCTX { u8 ib[128], ob[128]; };
__device__ void hmac512_ctx(const u8 *key, u32 klen, HCTX *h){
  u8 kb[128]; hmac512_key(key,klen,kb);
  for(int i=0;i<128;i++){ h->ib[i]=kb[i]^0x36; h->ob[i]=kb[i]^0x5c; }
}
__device__ void hmac512_run(const HCTX *h, const u8 *msg, u32 mlen, u8 out[64]){
  S512 s; s512_init(&s); s512_update(&s,h->ib,128); s512_update(&s,msg,mlen);
  u8 inner[64]; s512_final(&s,inner);
  s512_init(&s); s512_update(&s,h->ob,128); s512_update(&s,inner,64); s512_final(&s,out);
}

/* ======================= SHA-256 (single block, <=55B) ================== */
__device__ __constant__ u32 K256[64]={
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ __forceinline__ u32 ror32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
__device__ void sha256_1blk(const u8 *msg, u32 len, u8 out[32]){  // len<=55
  u8 blk[64]; for(int i=0;i<64;i++) blk[i]=0;
  for(u32 i=0;i<len;i++) blk[i]=msg[i];
  blk[len]=0x80;
  u32 bits=len*8; blk[60]=(bits>>24)&255;blk[61]=(bits>>16)&255;blk[62]=(bits>>8)&255;blk[63]=bits&255;
  u32 w[64];
  for(int i=0;i<16;i++) w[i]=(blk[4*i]<<24)|(blk[4*i+1]<<16)|(blk[4*i+2]<<8)|blk[4*i+3];
  for(int i=16;i<64;i++){
    u32 s0=ror32(w[i-15],7)^ror32(w[i-15],18)^(w[i-15]>>3);
    u32 s1=ror32(w[i-2],17)^ror32(w[i-2],19)^(w[i-2]>>10);
    w[i]=w[i-16]+s0+w[i-7]+s1;
  }
  u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
  for(int i=0;i<64;i++){
    u32 S1=ror32(e,6)^ror32(e,11)^ror32(e,25);
    u32 ch=(e&f)^((~e)&g);
    u32 t1=h+S1+ch+K256[i]+w[i];
    u32 S0=ror32(a,2)^ror32(a,13)^ror32(a,22);
    u32 maj=(a&b)^(a&c)^(b&c);
    u32 t2=S0+maj;
    h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
  }
  u32 H[8]={0x6a09e667+a,0xbb67ae85+b,0x3c6ef372+c,0xa54ff53a+d,0x510e527f+e,0x9b05688c+f,0x1f83d9ab+g,0x5be0cd19+h};
  for(int i=0;i<8;i++){ out[i*4]=(H[i]>>24)&255;out[i*4+1]=(H[i]>>16)&255;out[i*4+2]=(H[i]>>8)&255;out[i*4+3]=H[i]&255; }
}

/* ===================== 256-bit int mod n (secp256k1) ==================== */
/* limbs little-endian: v[0]=LSW .. v[7]=MSW */
__device__ __constant__ u32 SECP_N[8]={0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,
                                       0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
__device__ void be_to_limbs(const u8 be[32], u32 v[8]){
  for(int j=0;j<8;j++){
    int base=28-4*j; // bytes base..base+3 (big-endian) -> limb j
    v[j]=((u32)be[base]<<24)|((u32)be[base+1]<<16)|((u32)be[base+2]<<8)|(u32)be[base+3];
  }
}
__device__ void limbs_to_be(const u32 v[8], u8 be[32]){
  for(int j=0;j<8;j++){
    int base=28-4*j;
    be[base]=(v[j]>>24)&255; be[base+1]=(v[j]>>16)&255; be[base+2]=(v[j]>>8)&255; be[base+3]=v[j]&255;
  }
}
// compare a vs n (both 8 limbs LE): return 1 if a>=n
__device__ int ge_n(const u32 a[8], u32 carry){
  if(carry) return 1;
  for(int j=7;j>=0;j--){ if(a[j]>SECP_N[j]) return 1; if(a[j]<SECP_N[j]) return 0; }
  return 1; // equal -> >=
}
__device__ u32 sub_n(u32 a[8], u32 carry){ // a -= n ; return new carry (borrow-adjusted top)
  u64 br=0;
  for(int j=0;j<8;j++){ u64 t=(u64)a[j]-SECP_N[j]-br; a[j]=(u32)t; br=(t>>63)&1; }
  return carry - (u32)br; // carry was 0 or 1; subtracting n reduces the 257th bit
}
// r = (a + b) mod n ; a,b big-endian 32 bytes -> r big-endian 32 bytes
__device__ void modn_add(const u8 a_be[32], const u8 b_be[32], u8 r_be[32]){
  u32 a[8],b[8]; be_to_limbs(a_be,a); be_to_limbs(b_be,b);
  u32 s[8]; u64 c=0;
  for(int j=0;j<8;j++){ u64 t=(u64)a[j]+b[j]+c; s[j]=(u32)t; c=t>>32; }
  u32 carry=(u32)c; // 0 or 1 (257th bit)
  // reduce: subtract n while (carry:s) >= n  (at most 3 times for a,b<2^256)
  for(int it=0; it<3 && ge_n(s,carry); it++) carry=sub_n(s,carry);
  limbs_to_be(s,r_be);
}

/* ============================== KERNELS ================================= */
extern "C" __global__ void g_sha512(const u8 *data, const int *off, const int *len, int n, u8 *out){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  sha512(data+off[i], (u32)len[i], out+i*64);
}
extern "C" __global__ void g_sha256(const u8 *data, const int *off, const int *len, int n, u8 *out){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  u8 d[32]; sha256_1blk(data+off[i],(u32)len[i],d); out[i]=d[0];
}
extern "C" __global__ void g_pbkdf2(const u8 *pw, const int *pwoff, const int *pwlen,
                                    const u8 *salt, const int *soff, const int *slen,
                                    int n, u8 *out){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  HCTX h; hmac512_ctx(pw+pwoff[i], (u32)pwlen[i], &h);
  // block index 1: msg = salt || 0x00000001
  const u8 *st = salt+soff[i]; u32 sl=(u32)slen[i];
  // build salt||INT(1) via a small local (salt bounded in gate data)
  // stream instead: inner hash over ib, salt, INT(1)
  u8 U[64], T[64];
  { S512 s; s512_init(&s); s512_update(&s,h.ib,128); s512_update(&s,st,sl);
    u8 be[4]={0,0,0,1}; s512_update(&s,be,4); u8 inner[64]; s512_final(&s,inner);
    s512_init(&s); s512_update(&s,h.ob,128); s512_update(&s,inner,64); s512_final(&s,U); }
  for(int j=0;j<64;j++) T[j]=U[j];
  for(int it=1; it<2048; it++){
    hmac512_run(&h,U,64,U);
    for(int j=0;j<64;j++) T[j]^=U[j];
  }
  for(int j=0;j<64;j++) out[i*64+j]=T[j];
}
extern "C" __global__ void g_account(const u8 *seed, const int *off, const int *len,
                                     const int *levels, const u32 *idx, int n,
                                     u8 *out_c, u8 *out_k){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  // master: I = HMAC-SHA512("Bitcoin seed", seed); k=I[0:32] (unreduced), c=I[32:64]
  const u8 bs[12]={'B','i','t','c','o','i','n',' ','s','e','e','d'};
  HCTX hm; hmac512_ctx(bs,12,&hm);
  u8 I[64]; hmac512_run(&hm,seed+off[i],(u32)len[i],I);
  u8 k[32], c[32];
  for(int j=0;j<32;j++){ k[j]=I[j]; c[j]=I[32+j]; }
  int L=levels[i];
  for(int s=0;s<L;s++){
    u32 index=idx[i*3+s];
    // data = 0x00 || ser256(k) || ser32(index)
    u8 data[37]; data[0]=0x00; for(int j=0;j<32;j++) data[1+j]=k[j];
    data[33]=(index>>24)&255; data[34]=(index>>16)&255; data[35]=(index>>8)&255; data[36]=index&255;
    HCTX hc; hmac512_ctx(c,32,&hc);
    u8 Ic[64]; hmac512_run(&hc,data,37,Ic);
    u8 IL[32]; for(int j=0;j<32;j++) IL[j]=Ic[j];
    u8 nk[32]; modn_add(IL,k,nk);            // k = (IL + k_par) mod n
    for(int j=0;j<32;j++){ k[j]=nk[j]; c[j]=Ic[32+j]; }
  }
  for(int j=0;j<32;j++){ out_c[i*32+j]=c[j]; out_k[i*32+j]=k[j]; }
}
