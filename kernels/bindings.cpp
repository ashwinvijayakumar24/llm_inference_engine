// nanobind module exposing the decode-attention kernels to Python.
//
// The bindings take raw CUDA device pointers as integers (from a torch
// tensor's .data_ptr()) rather than array objects, so nanobind never needs to
// understand torch tensors and no data crosses the host boundary.

#include <nanobind/nanobind.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace nb = nanobind;

void launch_attention_decode_v1(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale);

void launch_attention_decode_v2(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale);

void launch_attention_decode_v3(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale);

// Decode attention. Pointers are raw CUDA device addresses passed as integers
// (from torch tensor.data_ptr()). Data stays on the GPU — no host round-trip.
// Writes results into the tensor backing `out_ptr`.
void attention_decode_v1(
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr, uintptr_t out_ptr,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    launch_attention_decode_v1(
        reinterpret_cast<const __half*>(q_ptr),
        reinterpret_cast<const __half*>(k_ptr),
        reinterpret_cast<const __half*>(v_ptr),
        reinterpret_cast<__half*>(out_ptr),
        n_heads, n_kv_heads, kv_seq, head_dim, scale);
    cudaDeviceSynchronize();
}

void attention_decode_v2(
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr, uintptr_t out_ptr,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    launch_attention_decode_v2(
        reinterpret_cast<const __half*>(q_ptr),
        reinterpret_cast<const __half*>(k_ptr),
        reinterpret_cast<const __half*>(v_ptr),
        reinterpret_cast<__half*>(out_ptr),
        n_heads, n_kv_heads, kv_seq, head_dim, scale);
    cudaDeviceSynchronize();
}

void attention_decode_v3(
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr, uintptr_t out_ptr,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale)
{
    launch_attention_decode_v3(
        reinterpret_cast<const __half*>(q_ptr),
        reinterpret_cast<const __half*>(k_ptr),
        reinterpret_cast<const __half*>(v_ptr),
        reinterpret_cast<__half*>(out_ptr),
        n_heads, n_kv_heads, kv_seq, head_dim, scale);
    cudaDeviceSynchronize();
}

NB_MODULE(engine_kernels, m) {
    m.def("attention_decode_v1", &attention_decode_v1,
          "Decode attention v1 (one thread per head, streaming softmax)");
    m.def("attention_decode_v2", &attention_decode_v2,
          "Decode attention v2 (head_dim threads per head, shared-mem reduction)");
    m.def("attention_decode_v3", &attention_decode_v3,
          "Decode attention v3 (split-KV flash-decoding + warp-shuffle reduction)");
}
