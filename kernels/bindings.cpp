#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace nb = nanobind;

void launch_add_one(float* d_data, int n);

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

NB_MODULE(engine_kernels, m) {
    m.def("add_one", &add_one, "Add 1 to each element (CPU)");
    m.def("add_one_cuda", &add_one_cuda, "Add 1 to each element (CUDA)");
}
