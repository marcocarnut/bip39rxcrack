/* secp256k1_device.cuh -- minimal secp256k1 + RIPEMD-160 + address programs for
 * the CUDA address-target crack. Correctness-only (double-and-add), gated
 * byte-exact against the reseed39 oracle (privToPub, ripemd160/hash160,
 * pubToTarget). Ported from estimator/bip39crypto.js. Depends on bip39_device.cuh
 * (u8/u32/u64, sha256_1blk, hmac512, modn_add).
 *
 * Field elements: u64 v[4] little-endian (v[0]=LSW), value in [0,p).
 * p = 2^256 - 2^32 - 977 ; the fold constant C = 2^32 + 977 = 0x1000003D1.
 */

typedef struct { u64 v[4]; } fe;

__device__ __constant__ u64 FE_P[4] = {
  0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
#define SECP_C 0x1000003D1ULL   /* 2^256 mod p */

/* generator G (affine), little-endian limbs */
__device__ __constant__ u64 FE_GX[4] = {
  0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
__device__ __constant__ u64 FE_GY[4] = {
  0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

__device__ __forceinline__ u64 mulhi(u64 a, u64 b){ return __umul64hi(a,b); }

__device__ __forceinline__ int fe_iszero(const fe *a){ return (a->v[0]|a->v[1]|a->v[2]|a->v[3])==0; }
__device__ __forceinline__ void fe_set(fe *r, const fe *a){ r->v[0]=a->v[0];r->v[1]=a->v[1];r->v[2]=a->v[2];r->v[3]=a->v[3]; }
__device__ __forceinline__ void fe_set_u64(fe *r, u64 x){ r->v[0]=x;r->v[1]=r->v[2]=r->v[3]=0; }
/* returns 1 if a>=p */
__device__ __forceinline__ int fe_ge_p(const u64 a[4]){
  for(int i=3;i>=0;i--){ if(a[i]>FE_P[i]) return 1; if(a[i]<FE_P[i]) return 0; } return 1;
}
/* r = a - p (assumes a>=p) */
__device__ __forceinline__ void fe_sub_p(u64 a[4]){
  u64 br=0; for(int i=0;i<4;i++){ u64 t=a[i]-FE_P[i]-br; br=(a[i]<FE_P[i]+br)||(FE_P[i]==~0ULL&&br)?1:( (a[i]<FE_P[i])||((a[i]==FE_P[i])&&br) ); a[i]=t; }
}
/* Robust subtract-with-borrow helpers (the fancy one above is error-prone). */
__device__ __forceinline__ u64 subb(u64 a, u64 b, u64 *borrow){
  u64 t=a-b; u64 br1=(a<b);
  u64 t2=t-*borrow; u64 br2=(t<*borrow);
  *borrow=br1|br2; return t2;
}
__device__ __forceinline__ u64 addc(u64 a, u64 b, u64 *carry){
  u64 t=a+b; u64 c1=(t<a);
  u64 t2=t+*carry; u64 c2=(t2<t);
  *carry=c1|c2; return t2;
}
__device__ void fe_cond_sub_p(u64 a[4]){ if(fe_ge_p(a)){ u64 br=0; for(int i=0;i<4;i++) a[i]=subb(a[i],FE_P[i],&br); } }

__device__ void fe_add(fe *r, const fe *a, const fe *b){
  u64 c=0,t[4]; for(int i=0;i<4;i++) t[i]=addc(a->v[i],b->v[i],&c);
  /* c is the 257th bit; reduce: if carry or >=p subtract p (add C on carry) */
  if(c){ u64 cc=0; t[0]=addc(t[0],SECP_C,&cc); for(int i=1;i<4&&cc;i++) t[i]=addc(t[i],0,&cc); }
  fe_cond_sub_p(t);
  r->v[0]=t[0];r->v[1]=t[1];r->v[2]=t[2];r->v[3]=t[3];
}
__device__ void fe_sub(fe *r, const fe *a, const fe *b){
  u64 br=0,t[4]; for(int i=0;i<4;i++) t[i]=subb(a->v[i],b->v[i],&br);
  if(br){ /* add p */ u64 c=0; for(int i=0;i<4;i++) t[i]=addc(t[i],FE_P[i],&c); }
  r->v[0]=t[0];r->v[1]=t[1];r->v[2]=t[2];r->v[3]=t[3];
}

/* full 256x256 -> 512 (t[8]) schoolbook via mulhi */
__device__ __noinline__ void mul_256(const u64 a[4], const u64 b[4], u64 t[8]){
  for(int i=0;i<8;i++) t[i]=0;
  for(int i=0;i<4;i++){
    u64 carry=0;
    for(int j=0;j<4;j++){
      u64 lo=a[i]*b[j], hi=mulhi(a[i],b[j]);
      u64 c1=0; u64 s=addc(t[i+j],lo,&c1);
      u64 c2=0; s=addc(s,carry,&c2);
      t[i+j]=s;
      carry=hi+c1+c2;   /* hi<2^64-1 so this can't overflow given c1,c2<=1 */
    }
    t[i+4]+=carry;
  }
}
/* multiply 4-limb by scalar c (u64) -> 5-limb out */
__device__ void mul_scalar(const u64 a[4], u64 c, u64 out[5]){
  u64 carry=0;
  for(int i=0;i<4;i++){ u64 lo=a[i]*c, hi=mulhi(a[i],c); u64 cc=0; out[i]=addc(lo,carry,&cc); carry=hi+cc; }
  out[4]=carry;
}
/* reduce 512-bit t[8] mod p into r (fe) */
__device__ __noinline__ void fe_reduce(fe *r, u64 t[8]){
  /* r0 = lo + hi*C  (hi = t[4..7]) */
  u64 hi[4]={t[4],t[5],t[6],t[7]};
  u64 m[5]; mul_scalar(hi,SECP_C,m);          /* hi*C : 5 limbs */
  u64 s[5]; u64 c=0;
  for(int i=0;i<4;i++) s[i]=addc(t[i],m[i],&c);
  s[4]=m[4]+c;                                 /* overflow above 2^256 */
  /* fold s[4] again: result = s[0..3] + s[4]*C (s[4] small) */
  u64 add0=s[4]*SECP_C; u64 add1=mulhi(s[4],SECP_C);
  u64 out[4]; u64 cc=0;
  out[0]=addc(s[0],add0,&cc);
  out[1]=addc(s[1],add1,&cc);
  out[2]=addc(s[2],0,&cc);
  out[3]=addc(s[3],0,&cc);
  if(cc){ u64 c2=0; out[0]=addc(out[0],SECP_C,&c2); for(int i=1;i<4&&c2;i++) out[i]=addc(out[i],0,&c2); }
  fe_cond_sub_p(out);
  r->v[0]=out[0];r->v[1]=out[1];r->v[2]=out[2];r->v[3]=out[3];
}
__device__ __noinline__ void fe_mul(fe *r, const fe *a, const fe *b){ u64 t[8]; mul_256(a->v,b->v,t); fe_reduce(r,t); }
__device__ void fe_sqr(fe *r, const fe *a){ fe_mul(r,a,a); }

/* inverse via Fermat: a^(p-2). p-2 = 0xFFFFFFFE...FC2D */
__device__ __noinline__ void fe_inv(fe *r, const fe *a){
  /* exponent p-2 limbs (little-endian) */
  const u64 e[4]={0xFFFFFFFEFFFFFC2DULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
  fe res; fe_set_u64(&res,1); fe base; fe_set(&base,a);
  for(int limb=0;limb<4;limb++){
    u64 x=e[limb];
    for(int b=0;b<64;b++){
      if((x>>b)&1) fe_mul(&res,&res,&base);
      fe_sqr(&base,&base);
    }
  }
  fe_set(r,&res);
}

/* Jacobian point */
typedef struct { fe X,Y,Z; int inf; } jpt;
__device__ void j_set_inf(jpt *p){ p->inf=1; fe_set_u64(&p->X,1); fe_set_u64(&p->Y,1); fe_set_u64(&p->Z,0); }
__device__ __noinline__ void j_double(jpt *r, const jpt *p){
  if(p->inf||fe_iszero(&p->Y)){ j_set_inf(r); return; }
  fe Y2,S,M,X3,Y3,Z3,t,t2;
  fe_sqr(&Y2,&p->Y);
  fe_mul(&S,&p->X,&Y2); fe c4; fe_set_u64(&c4,4); fe_mul(&S,&S,&c4);       // S=4*X*Y^2
  fe_sqr(&M,&p->X); fe c3; fe_set_u64(&c3,3); fe_mul(&M,&M,&c3);           // M=3*X^2
  fe_sqr(&X3,&M); fe_add(&t,&S,&S); fe_sub(&X3,&X3,&t);                    // X3=M^2-2S
  fe_sub(&t,&S,&X3); fe_mul(&Y3,&M,&t);                                    // M*(S-X3)
  fe_sqr(&t2,&Y2); fe c8; fe_set_u64(&c8,8); fe_mul(&t2,&t2,&c8); fe_sub(&Y3,&Y3,&t2); // -8*Y^4
  fe_mul(&Z3,&p->Y,&p->Z); fe_add(&Z3,&Z3,&Z3);                           // 2*Y*Z
  fe_set(&r->X,&X3); fe_set(&r->Y,&Y3); fe_set(&r->Z,&Z3); r->inf=0;
}
__device__ __noinline__ void j_add(jpt *r, const jpt *p, const jpt *q){
  if(p->inf){ *r=*q; return; } if(q->inf){ *r=*p; return; }
  fe Z1Z1,Z2Z2,U1,U2,S1,S2,H,R,HH,HHH,V,X3,Y3,Z3,t,t2;
  fe_sqr(&Z1Z1,&p->Z); fe_sqr(&Z2Z2,&q->Z);
  fe_mul(&U1,&p->X,&Z2Z2); fe_mul(&U2,&q->X,&Z1Z1);
  fe_mul(&S1,&p->Y,&q->Z); fe_mul(&S1,&S1,&Z2Z2);
  fe_mul(&S2,&q->Y,&p->Z); fe_mul(&S2,&S2,&Z1Z1);
  fe_sub(&H,&U2,&U1); fe_sub(&R,&S2,&S1);
  if(fe_iszero(&H)){ if(fe_iszero(&R)){ j_double(r,p); return; } j_set_inf(r); return; }
  fe_sqr(&HH,&H); fe_mul(&HHH,&H,&HH); fe_mul(&V,&U1,&HH);
  fe_sqr(&X3,&R); fe_sub(&X3,&X3,&HHH); fe_add(&t,&V,&V); fe_sub(&X3,&X3,&t);
  fe_sub(&t,&V,&X3); fe_mul(&Y3,&R,&t); fe_mul(&t2,&S1,&HHH); fe_sub(&Y3,&Y3,&t2);
  fe_mul(&Z3,&p->Z,&q->Z); fe_mul(&Z3,&Z3,&H);
  fe_set(&r->X,&X3); fe_set(&r->Y,&Y3); fe_set(&r->Z,&Z3); r->inf=0;
}

/* k*G (k = 32 big-endian bytes) -> Jacobian point (double-and-add). Used to
 * BUILD the comb table; the hot path uses scalar_mul_G_j (comb) below. */
__device__ __noinline__ void scalar_mul_G_dbl(const u8 k[32], jpt *R){
  j_set_inf(R);
  jpt G; fe_set(&G.X,(const fe*)FE_GX); fe_set(&G.Y,(const fe*)FE_GY); fe_set_u64(&G.Z,1); G.inf=0;
  for(int i=0;i<256;i++){
    j_double(R,R);
    int byte=k[i>>3]; int bit=(byte>>(7-(i&7)))&1;   // MSB-first
    if(bit) j_add(R,R,&G);
  }
}
/* ---- fixed-base comb (4-bit windows) --------------------------------------
 * Table d_comb holds, for each of 64 nibble positions i and value j in 1..15,
 * the AFFINE point (j << 4i)*G as 8 u64 (x[4] || y[4]). k*G = sum over nibbles
 * of table[i][nibble_i] -- <=64 adds, no doublings (~6x fewer point ops than
 * double-and-add). Built once on-device by g_comb_init. */
__device__ u64 *d_comb = 0;          // set by the host after g_comb_init
__device__ __noinline__ void scalar_mul_G_j(const u8 k[32], jpt *R){
  if(!d_comb){ scalar_mul_G_dbl(k,R); return; }
  j_set_inf(R);
  for(int i=0;i<64;i++){
    int byte=k[31-(i>>1)]; int nib=(byte>>((i&1)*4))&15;
    if(!nib) continue;
    const u64 *e=d_comb+((size_t)(i*16+nib))*8;
    jpt Q; Q.X.v[0]=e[0];Q.X.v[1]=e[1];Q.X.v[2]=e[2];Q.X.v[3]=e[3];
    Q.Y.v[0]=e[4];Q.Y.v[1]=e[5];Q.Y.v[2]=e[6];Q.Y.v[3]=e[7]; fe_set_u64(&Q.Z,1); Q.inf=0;
    j_add(R,R,&Q);
  }
}
/* Jacobian -> affine (x,y) with cond-normalized limbs. */
__device__ void j_affine(const jpt *R, fe *x, fe *y){
  fe zi,zi2,zi3; fe_inv(&zi,&R->Z); fe_sqr(&zi2,&zi); fe_mul(&zi3,&zi2,&zi);
  fe_mul(x,&R->X,&zi2); fe_mul(y,&R->Y,&zi3);
  fe_cond_sub_p(x->v); fe_cond_sub_p(y->v);
}
/* fe -> 32 big-endian bytes */
__device__ void fe_to_be(const fe *x, u8 out[32]){
  for(int i=0;i<32;i++){ int limb=(31-i)>>3, byteinlimb=(31-i)&7; out[i]=(u8)(x->v[limb]>>(8*byteinlimb)); }
}
/* fe <- 32 big-endian bytes */
__device__ void fe_from_be(fe *x, const u8 in[32]){
  x->v[0]=x->v[1]=x->v[2]=x->v[3]=0;
  for(int i=0;i<32;i++){ int limb=(31-i)>>3, byteinlimb=(31-i)&7; x->v[limb]|=((u64)in[i])<<(8*byteinlimb); }
}
/* k*G, k as 32 big-endian bytes; output compressed 33-byte pubkey */
__device__ void scalar_mul_G(const u8 k[32], u8 pub[33]){
  jpt R; scalar_mul_G_j(k,&R);
  if(R.inf){ for(int i=0;i<33;i++) pub[i]=0; return; }
  fe x,y; j_affine(&R,&x,&y);
  pub[0]=(y.v[0]&1)?0x03:0x02;
  fe_to_be(&x,pub+1);
}

/* ============================== RIPEMD-160 ============================== */
__device__ __forceinline__ u32 rol32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
__device__ void ripemd160(const u8 *msg, u32 len, u8 out[20]){
  u32 h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
  /* single-block only needed (our inputs <=55B): pad to 64 */
  u8 blk[64]; for(int i=0;i<64;i++) blk[i]=0;
  for(u32 i=0;i<len;i++) blk[i]=msg[i];
  blk[len]=0x80; u64 bits=(u64)len*8; for(int i=0;i<8;i++) blk[56+i]=(u8)(bits>>(8*i));
  u32 X[16]; for(int i=0;i<16;i++) X[i]=(u32)blk[4*i]|((u32)blk[4*i+1]<<8)|((u32)blk[4*i+2]<<16)|((u32)blk[4*i+3]<<24);
  const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
  const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
  const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
  const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
  const u32 KL[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
  const u32 KR[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
  u32 al=h0,bl=h1,cl=h2,dl=h3,el=h4, ar=h0,br=h1,cr=h2,dr=h3,er=h4;
  for(int j=0;j<80;j++){
    int rnd=j/16;
    u32 f,g;
    if(rnd==0) f=bl^cl^dl; else if(rnd==1) f=(bl&cl)|(~bl&dl); else if(rnd==2) f=(bl|~cl)^dl; else if(rnd==3) f=(bl&dl)|(cl&~dl); else f=bl^(cl|~dl);
    u32 t=rol32(al+f+X[rl[j]]+KL[rnd],sl[j])+el; al=el;el=dl;dl=rol32(cl,10);cl=bl;bl=t;
    int rndr=4-rnd;
    if(rndr==0) g=br^cr^dr; else if(rndr==1) g=(br&cr)|(~br&dr); else if(rndr==2) g=(br|~cr)^dr; else if(rndr==3) g=(br&dr)|(cr&~dr); else g=br^(cr|~dr);
    u32 tr=rol32(ar+g+X[rr[j]]+KR[rnd],sr[j])+er; ar=er;er=dr;dr=rol32(cr,10);cr=br;br=tr;
  }
  u32 T=h1+cl+dr; h1=h2+dl+er; h2=h3+el+ar; h3=h4+al+br; h4=h0+bl+cr; h0=T;
  u32 H[5]={h0,h1,h2,h3,h4};
  for(int i=0;i<5;i++){ out[i*4]=(u8)H[i]; out[i*4+1]=(u8)(H[i]>>8); out[i*4+2]=(u8)(H[i]>>16); out[i*4+3]=(u8)(H[i]>>24); }
}
__device__ void hash160(const u8 *msg, u32 len, u8 out[20]){ u8 sh[32]; sha256_1blk(msg,len,sh); ripemd160(sh,32,out); }

/* ============================ Taproot (BIP86) =========================== */
/* Multi-block SHA-256 (arbitrary length) -- tagged hash needs 96-byte input. */
__device__ __noinline__ void sha256_compress(u32 H[8], const u8 blk[64]){
  u32 w[64]; for(int i=0;i<16;i++) w[i]=(blk[4*i]<<24)|(blk[4*i+1]<<16)|(blk[4*i+2]<<8)|blk[4*i+3];
  for(int i=16;i<64;i++){ u32 s0=ror32(w[i-15],7)^ror32(w[i-15],18)^(w[i-15]>>3),s1=ror32(w[i-2],17)^ror32(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
  u32 a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
  for(int i=0;i<64;i++){ u32 S1=ror32(e,6)^ror32(e,11)^ror32(e,25),ch=(e&f)^((~e)&g),t1=h+S1+ch+K256[i]+w[i],S0=ror32(a,2)^ror32(a,13)^ror32(a,22),maj=(a&b)^(a&c)^(b&c),t2=S0+maj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
  H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
}
/* Multi-block SHA-256 (arbitrary length) -- tagged hash needs a 96-byte input. */
__device__ void sha256_full(const u8 *msg, u32 len, u8 out[32]){
  u32 H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  u32 off=0; u8 blk[64];
  while(len-off>=64){ for(int i=0;i<64;i++) blk[i]=msg[off+i]; sha256_compress(H,blk); off+=64; }
  u32 rem=len-off; for(u32 i=0;i<rem;i++) blk[i]=msg[off+i];
  blk[rem]=0x80; for(u32 i=rem+1;i<64;i++) blk[i]=0;
  if(rem>=56){ sha256_compress(H,blk); for(int i=0;i<64;i++) blk[i]=0; }
  u64 bits=(u64)len*8; for(int i=0;i<8;i++) blk[63-i]=(u8)(bits>>(8*i));
  sha256_compress(H,blk);
  for(int i=0;i<8;i++){ out[i*4]=(H[i]>>24)&255;out[i*4+1]=(H[i]>>16)&255;out[i*4+2]=(H[i]>>8)&255;out[i*4+3]=H[i]&255; }
}
/* modular exponent (r = a^e mod p), e little-endian u64[4] */
__device__ __noinline__ void fe_pow(fe *r, const fe *a, const u64 e[4]){
  fe res; fe_set_u64(&res,1); fe base; fe_set(&base,a);
  for(int limb=0;limb<4;limb++){ u64 x=e[limb];
    for(int b=0;b<64;b++){ if((x>>b)&1) fe_mul(&res,&res,&base); fe_sqr(&base,&base); } }
  fe_set(r,&res);
}
__device__ __forceinline__ int fe_eq(const fe *a,const fe *b){ return a->v[0]==b->v[0]&&a->v[1]==b->v[1]&&a->v[2]==b->v[2]&&a->v[3]==b->v[3]; }
/* lift_x: even-Y point on the curve with the given x (BIP340). Returns 0 if no
 * square root (x not on curve). */
__device__ __noinline__ int liftX(const u8 x_be[32], fe *Px, fe *Py){
  const u64 EXP[4]={0xFFFFFFFFBFFFFF0CULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0x3FFFFFFFFFFFFFFFULL}; // (p+1)/4
  fe x; fe_from_be(&x,x_be);
  fe x2,x3,c,seven; fe_sqr(&x2,&x); fe_mul(&x3,&x2,&x); fe_set_u64(&seven,7); fe_add(&c,&x3,&seven);
  fe y; fe_pow(&y,&c,EXP);
  fe y2; fe_sqr(&y2,&y); if(!fe_eq(&y2,&c)) return 0;
  if(y.v[0]&1){ fe zero; fe_set_u64(&zero,0); fe ny; fe_sub(&ny,&zero,&y); fe_set(&y,&ny); } // even Y
  fe_set(Px,&x); fe_set(Py,&y); return 1;
}
/* TapTweak tagged hash of x (32B) reduced mod n -> 32 big-endian bytes. */
__device__ void taptweak(const u8 x_be[32], u8 out_t[32]){
  const u8 tag[8]={'T','a','p','T','w','e','a','k'};
  u8 th[32]; sha256_full(tag,8,th);
  u8 buf[96]; for(int i=0;i<32;i++){ buf[i]=th[i]; buf[32+i]=th[i]; buf[64+i]=x_be[i]; }
  u8 ht[32]; sha256_full(buf,96,ht);
  u8 zero[32]; for(int i=0;i<32;i++) zero[i]=0;
  modn_add(ht,zero,out_t);   // ht mod n
}
/* BIP86 p2tr program: Q = liftX(x) + taptweak(x)*G ; program = Q.x (32B). */
__device__ void p2tr_program(const u8 pub[33], u8 prog[32]){
  const u8 *xb=pub+1;
  fe Px,Py; if(!liftX(xb,&Px,&Py)){ for(int i=0;i<32;i++) prog[i]=0; return; }
  u8 t[32]; taptweak(xb,t);
  jpt tG; scalar_mul_G_j(t,&tG);
  jpt P; fe_set(&P.X,&Px); fe_set(&P.Y,&Py); fe_set_u64(&P.Z,1); P.inf=0;
  jpt Q; j_add(&Q,&P,&tG);
  fe qx,qy; j_affine(&Q,&qx,&qy); fe_to_be(&qx,prog);
}

/* pubkey(33) + purpose -> address program; *proglen = 20 (p2pkh/p2sh/p2wpkh)
 * or 32 (p2tr). p2pkh/p2wpkh: hash160; p2sh-p2wpkh: hash160(0x0014||h160);
 * p2tr(86): x-only tweaked output key. */
__device__ void pub_to_program(const u8 pub[33], int purpose, u8 prog[32], int *proglen){
  if(purpose==86){ p2tr_program(pub,prog); *proglen=32; return; }
  u8 h[20]; hash160(pub,33,h);
  if(purpose==49){ u8 redeem[22]; redeem[0]=0x00; redeem[1]=0x14; for(int i=0;i<20;i++) redeem[2+i]=h[i]; hash160(redeem,22,prog); }
  else { for(int i=0;i<20;i++) prog[i]=h[i]; }   /* 44 p2pkh, 84 p2wpkh */
  *proglen=20;
}

/* Non-hardened CKDpriv (needs EC): k'=(IL+k) mod n, c'=IR, where
 * I=HMAC-SHA512(c, serP(k*G) || ser32(index)). k,c are 32 big-endian bytes,
 * updated in place. index < 2^31. */
__device__ void ckd_normal(u8 *k, u8 *c, u32 index){
  u8 pub[33]; scalar_mul_G(k,pub);
  u8 data[37]; for(int i=0;i<33;i++) data[i]=pub[i];
  data[33]=(index>>24)&255; data[34]=(index>>16)&255; data[35]=(index>>8)&255; data[36]=index&255;
  HCTX h; hmac512_ctx(c,32,&h); u8 I[64]; hmac512_run(&h,data,37,I);
  u8 IL[32]; for(int i=0;i<32;i++) IL[i]=I[i];
  u8 nk[32]; modn_add(IL,k,nk);
  for(int i=0;i<32;i++){ k[i]=nk[i]; c[i]=I[32+i]; }
}
/* seed -> m/purpose'/0'/0'/change/index pubkey -> 20-byte address program.
 * program type follows the purpose (44 p2pkh, 49 p2sh-p2wpkh, 84 p2wpkh). */
__device__ void derive_address(const u8 seed[64], u32 purpose, u32 change, u32 index, u8 prog[32], int *proglen){
  u32 hidx[3]={ purpose|0x80000000u, 0x80000000u, 0x80000000u };
  u8 c[32],k[32]; derive_hardened(seed,64,hidx,3,c,k);   // account node m/purpose'/0'/0'
  ckd_normal(k,c,change);
  ckd_normal(k,c,index);
  u8 pub[33]; scalar_mul_G(k,pub);
  pub_to_program(pub,(int)purpose,prog,proglen);
}
