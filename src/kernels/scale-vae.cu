// by cjx
#include <cublas_v2.h>
#include <torch/extension.h>

// cuBLAS handle
cublasHandle_t cublas_handle;

// Initialize cuBLAS handle
void init_cublas() {
    cublasCreate(&cublas_handle);
}

// Destroy cuBLAS handle
void destroy_cublas() {
    cublasDestroy(cublas_handle);
}

// cuBLAS kernel for scaling the mean
torch::Tensor scale_mean_cublas(torch::Tensor mu, float scale_factor) {
  TORCH_CHECK(mu.is_cuda(), "Input tensor must be on the GPU");

  // retrieve pointer to the data
  float* mu_data = mu.data_ptr<float>();

  // get elements
  int n = mu.numel();

  // scales by cuBLAS
  cublasSscal(cublas_handle, n, &scale_factor, mu_data, 1);

  return mu; // new scaled tensor
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_cublas", &init_cublas, "Initialize cuBLAS handle");
    m.def("destroy_cublas", &destroy_cublas, "Destroy cuBLAS handle");
    m.def("scale_mean_cublas", &scale_mean_cublas, "Scale mean using cuBLAS");
}