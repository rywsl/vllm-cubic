#include <cstdint>
#include <stdexcept>

#include <torch/csrc/stable/tensor.h>

// FlashMLA's registration unit references the SM100 dense-prefill entry point
// even when only Hopper kernels are built. Keep the symbol resolvable on H200;
// the Kimi K3 path uses the SM90 sparse/dense kernels instead.
__attribute__((visibility("default"))) void FMHACutlassSM100FwdRun(
    torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
    torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
    torch::stable::Tensor, torch::stable::Tensor, int64_t, double, int64_t,
    int64_t, bool) {
  throw std::runtime_error(
      "FlashMLA SM100 dense prefill is unavailable on SM90");
}
