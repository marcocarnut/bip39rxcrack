#ifndef BLOOM_COMMON_H
#define BLOOM_COMMON_H
/* Blocked bloom over ALREADY-UNIFORM 20/32-byte fingerprints (address programs /
 * account chaincodes). The key is already a hash, so we SLICE bit-fields out of
 * it instead of re-hashing.
 *
 *   block = 32 bytes = 256 bits = 8 x u32 = one GPU memory sector (one fetch).
 *   k bits per key, all inside the one block. block index from P[0..3];
 *   bit position j from byte P[4+j] (0..255 -> which of the block's 256 bits).
 *
 * BLOOM_K=16 uses P[0..19] (20 bytes = exactly a hash160) and lands near the
 * optimal k for our ~16-32 bits/key range. Identical on host (build) and device
 * (probe) -- coin-agnostic: it reads only the raw fingerprint bytes.
 *
 * Plain integer types only (no <stdint.h>) so this compiles under NVRTC too. */

#ifdef __CUDACC__
#define BLOOM_FN __host__ __device__ static inline
#else
#define BLOOM_FN static inline
#endif

#define BLOOM_K 16   /* bits set/tested per key; k position-bytes + 4 block-bytes <= 20 */

/* One bloom-probe hit, emitted by the GPU for the host to cull (bloom hits are
 * many -- true positive + false positives -- unlike the single-target atomicMin).
 * `purpose` is the BIP purpose the match was derived under (an xpub/chaincode
 * doesn't encode its type, so the derive is authoritative). prog holds the
 * fingerprint (20/32B address program, or a 32B account chaincode). */
typedef struct { unsigned long long gidx; unsigned int change, index, purpose; unsigned char prog[32]; } BloomHit;

/* 4 big-endian bytes -> u32 (same on host and device). */
BLOOM_FN unsigned int bloom_u32be(const unsigned char *p){
  return ((unsigned int)p[0]<<24)|((unsigned int)p[1]<<16)|((unsigned int)p[2]<<8)|(unsigned int)p[3];
}
/* which 32-byte block (nblocks is a power of two; mask = nblocks-1). */
BLOOM_FN unsigned int bloom_block(const unsigned char *P, unsigned int nblocks_mask){
  return bloom_u32be(P) & nblocks_mask;
}
/* set the k bits for fingerprint P into the filter (host build). */
BLOOM_FN void bloom_insert(unsigned int *filter, const unsigned char *P, unsigned int nblocks_mask){
  unsigned int *b = filter + (unsigned long long)bloom_block(P,nblocks_mask)*8ull;
  for(int j=0;j<BLOOM_K;j++){ unsigned int pos = P[4+j]; b[pos>>5] |= (1u<<(pos&31u)); }
}
/* probe: 1 if all k bits present (member or false positive), else 0. One block
 * load; the k tests are register ops. */
BLOOM_FN int bloom_probe(const unsigned int *filter, const unsigned char *P, unsigned int nblocks_mask){
  const unsigned int *b = filter + (unsigned long long)bloom_block(P,nblocks_mask)*8ull;
  for(int j=0;j<BLOOM_K;j++){ unsigned int pos = P[4+j]; if(!(b[pos>>5] & (1u<<(pos&31u)))) return 0; }
  return 1;
}
/* pick nblocks (power of two) for n keys at ~bits_per_key; filter = nblocks*32 B. */
BLOOM_FN unsigned int bloom_nblocks(unsigned long long n, double bits_per_key){
  double blocks = ((double)n * bits_per_key) / 256.0;
  unsigned int nb = 1;
  while((double)nb < blocks && nb < 0x40000000u) nb <<= 1;   /* cap 2^30 blocks = 32 GiB */
  return nb ? nb : 1u;
}
#endif
