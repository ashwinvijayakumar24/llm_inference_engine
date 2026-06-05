#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;

void launch_add_one(float* d_data, int n);

void launch_attention_decode_v1(
    const __half* Q, const __half* K, const __half* V, __half* out,
    int n_heads, int n_kv_heads, int kv_seq, int head_dim, float scale);

// CPU version
nb::ndarray<nb::numpy, float, nb::ndim<1>>
add_one(nb::ndarray<nb::numpy, float, nb::ndim<1>> arr) {
    size_t n = arr.shape(0);
    float* data = arr.data();
    for (size_t i = 0; i < n; i++) data[i] += 1.0f;
    return arr;
}

// CUDA version
nb::ndarray<nb::numpy, float, nb::ndim<1>>
add_one_cuda(nb::ndarray<nb::numpy, float, nb::ndim<1>> arr) {
    size_t n = arr.shape(0);
    float* h_data = arr.data();

    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    launch_add_one(d_data, (int)n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    return arr;
}

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

NB_MODULE(engine_kernels, m) {
    m.def("add_one", &add_one, "Add 1 to each element (CPU)");
    m.def("add_one_cuda", &add_one_cuda, "Add 1 to each element (CUDA)");
    m.def("attention_decode_v1", &attention_decode_v1,
          "Decode attention v1 (one thread per head, streaming softmax)");
}
