/* gate_kernels.cu -- Phase-1 crypto gate kernels (one thread per vector).
 * Device crypto lives in the shared bip39_device.cuh (proven byte-exact);
 * this file is just the four gate entry points. NVRTC->compute_90->sm_120 JIT.
 */
#include "bip39_device.cuh"

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
  pbkdf2_seed(pw+pwoff[i],(u32)pwlen[i], salt+soff[i],(u32)slen[i], 2048, out+i*64);
}
extern "C" __global__ void g_account(const u8 *seed, const int *off, const int *len,
                                     const int *levels, const u32 *idx, int n,
                                     u8 *out_c, u8 *out_k){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  derive_hardened(seed+off[i],(u32)len[i], idx+i*3, levels[i], out_c+i*32, out_k+i*32);
}
