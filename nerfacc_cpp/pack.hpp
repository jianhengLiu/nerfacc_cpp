#pragma once

#include <torch/torch.h>
namespace nerfacc {
torch::Tensor pack_info(const torch::Tensor &ray_indices, int n_rays) {
  /* clang-format off
  """Pack `ray_indices` to `packed_info`. Useful for converting per sample data to per ray data.

  Note:
      this function is not differentiable to any inputs.

  Args:
      ray_indices: Ray indices of the samples. LongTensor with shape (n_sample).
      n_rays: Number of rays. If None, it is inferred from `ray_indices`. Default is None.

  Returns:
      A LongTensor of shape (n_rays, 2) that specifies the start and count
      of each chunk in the flattened input tensor, with in total n_rays chunks.

  Example:

  .. code-block:: python

      >>> ray_indices = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2], device="cuda")
      >>> packed_info = pack_info(ray_indices, n_rays=3)
      >>> packed_info
      tensor([[0, 2], [2, 3], [5, 4]], device='cuda:0')

  """
  clang-format on */
  TORCH_CHECK(ray_indices.dim() == 1,
              "ray_indices must be a 1D tensor with shape (n_samples).");

  if (ray_indices.is_cuda()) {
    torch::Device device = ray_indices.device();
    auto dtype = ray_indices.dtype();

    if (n_rays == -1) {
      n_rays = ray_indices.max().item<int>() + 1;
    }
    torch::Tensor chunk_cnts = torch::zeros({n_rays}, ray_indices.options());
    chunk_cnts.index_add_(0, ray_indices, torch::ones_like(ray_indices));

    torch::Tensor chunk_starts = chunk_cnts.cumsum(0) - chunk_cnts;

    torch::Tensor packed_info = torch::stack({chunk_starts, chunk_cnts}, -1);
    return packed_info;
  } else {
    throw std::runtime_error("Only support cuda inputs.");
  }
}
} // namespace nerfacc