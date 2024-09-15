#pragma once
#include "pack.hpp"
#include "scan.hpp"

namespace nerfacc {

torch::Tensor render_transmittance_from_alpha(
    const torch::Tensor &alphas, torch::Tensor packed_info = {},
    const torch::Tensor &ray_indices = {}, const int &n_rays = -1,
    const torch::Tensor &prefix_trans = {}) {
  if (ray_indices.defined() && packed_info.numel() == 0) {
    packed_info = pack_info(ray_indices, n_rays);
  }

  torch::Tensor trans = exclusive_prod(1 - alphas, packed_info);

  if (prefix_trans.defined()) {
    trans *= prefix_trans;
  }
  return trans;
}

std::vector<torch::Tensor> render_transmittance_from_density(
    const torch::Tensor &t_starts, const torch::Tensor &t_ends,
    const torch::Tensor &sigmas, torch::Tensor packed_info = {},
    const torch::Tensor &ray_indices = {}, const int &n_rays = -1,
    const torch::Tensor &prefix_trans = {}) {
  if (ray_indices.defined() && packed_info.numel() == 0) {
    packed_info = pack_info(ray_indices, n_rays);
  }

  torch::Tensor sigmas_dt = sigmas * (t_ends - t_starts);
  if ((sigmas_dt < 0).any().item<bool>()) {
    std::cout << "sigmas_dt < 0\n";
  }
  torch::Tensor alphas = 1.0 - torch::exp(-sigmas_dt);
  torch::Tensor trans = torch::exp(-exclusive_sum(sigmas_dt, packed_info));

  if (prefix_trans.defined()) {
    trans *= prefix_trans;
  }

  return {trans, alphas};
}

std::vector<torch::Tensor> render_transmittance_from_density_delta(
    const torch::Tensor &t_deltas, const torch::Tensor &sigmas,
    torch::Tensor packed_info = {}, const torch::Tensor &ray_indices = {},
    const int &n_rays = -1, const torch::Tensor &prefix_trans = {}) {
  if (ray_indices.defined() && packed_info.numel() == 0) {
    packed_info = pack_info(ray_indices, n_rays);
  }

  torch::Tensor sigmas_dt = sigmas * t_deltas;
  if ((sigmas_dt < 0).any().item<bool>()) {
    std::cout << "sigmas_dt < 0\n";
  }
  torch::Tensor alphas = 1.0 - torch::exp(-sigmas_dt);
  torch::Tensor trans = torch::exp(-exclusive_sum(sigmas_dt, packed_info));

  if (prefix_trans.defined()) {
    trans *= prefix_trans;
  }

  return {trans, alphas};
}

std::vector<torch::Tensor> render_weight_from_density(
    const torch::Tensor &t_starts, const torch::Tensor &t_ends,
    const torch::Tensor &sigmas, torch::Tensor packed_info = {},
    const torch::Tensor &ray_indices = {}, const int &n_rays = -1,
    const torch::Tensor &prefix_trans = {}) {
  /* clang-format off
  """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

  .. math::
      w_i = T_i(1 - exp(-\\sigma_i\delta_i)), \\quad\\textrm{where}\\quad T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

  This function supports both batched and flattened input tensor. For flattened input tensor, either
  (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

  Args:
      t_starts: The start time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      t_ends: The end time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
          of each chunk in the flattened samples, with in total n_rays chunks.
          Useful for flattened input.
      ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
      n_rays: Number of rays. Only useful when `ray_indices` is provided.
      prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

  Returns:
      The rendering weights, transmittance and opacities, both with the same shape as `sigmas`.

  Examples:

  .. code-block:: python

      >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
      >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
      >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
      >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
      >>> weights, transmittance, alphas = render_weight_from_density(
      >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
      weights: [0.33, 0.37, 0.03, 0.55, 0.04, 0.00, 0.59]
      transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
      alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

  """
  clang-format on */
  auto render_results = render_transmittance_from_density(
      t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans);
  auto trans = render_results[0];
  auto alphas = render_results[1];
  torch::Tensor weights = trans * alphas;
  return {weights, trans, alphas};
}

std::vector<torch::Tensor> render_weight_from_density_delta(
    const torch::Tensor &t_deltas, const torch::Tensor &sigmas,
    torch::Tensor packed_info = {}, const torch::Tensor &ray_indices = {},
    const int &n_rays = -1, const torch::Tensor &prefix_trans = {}) {
  /* clang-format off
  """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

  .. math::
      w_i = T_i(1 - exp(-\\sigma_i\delta_i)), \\quad\\textrm{where}\\quad T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

  This function supports both batched and flattened input tensor. For flattened input tensor, either
  (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

  Args:
      t_starts: The start time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      t_ends: The end time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
      packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
          of each chunk in the flattened samples, with in total n_rays chunks.
          Useful for flattened input.
      ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
      n_rays: Number of rays. Only useful when `ray_indices` is provided.
      prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

  Returns:
      The rendering weights, transmittance and opacities, both with the same shape as `sigmas`.

  Examples:

  .. code-block:: python

      >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
      >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
      >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
      >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
      >>> weights, transmittance, alphas = render_weight_from_density(
      >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
      weights: [0.33, 0.37, 0.03, 0.55, 0.04, 0.00, 0.59]
      transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
      alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

  """
  clang-format on */
  auto render_results = render_transmittance_from_density_delta(
      t_deltas, sigmas, packed_info, ray_indices, n_rays, prefix_trans);
  auto trans = render_results[0];
  auto alphas = render_results[1];
  torch::Tensor weights = trans * alphas;
  return {weights, trans, alphas};
}

std::vector<torch::Tensor>
render_weight_from_alpha(const torch::Tensor &alphas,
                         torch::Tensor packed_info = {},
                         const torch::Tensor &ray_indices = {}, int n_rays = -1,
                         const torch::Tensor &prefix_trans = {}) {
  torch::Tensor trans = render_transmittance_from_alpha(
      alphas, packed_info, ray_indices, n_rays, prefix_trans);
  torch::Tensor weights = trans * alphas;
  return {weights, trans};
}

torch::Tensor accumulate_along_rays(
    const torch::Tensor &weights, const torch::Tensor &values = torch::Tensor(),
    const torch::Tensor &ray_indices = torch::Tensor(), int n_rays = -1) {
  /* """Accumulate volumetric values along the ray.

  This function supports both batched inputs and flattened inputs with
  `ray_indices` and `n_rays` provided.

  Note:
      This function is differentiable to `weights` and `values`.

  Args:
      weights: Weights to be accumulated. If `ray_indices` not provided,
          `weights` must be batched with shape (n_rays, n_samples). Else it
          must be flattened with shape (all_samples,).
      values: Values to be accumulated. If `ray_indices` not provided,
          `values` must be batched with shape (n_rays, n_samples, D). Else it
          must be flattened with shape (all_samples, D). None means
          we accumulate weights along rays. Default: None.
      ray_indices: Ray indices of the samples with shape (all_samples,).
          If provided, `weights` must be a flattened tensor with shape
  (all_samples,) and values (if not None) must be a flattened tensor with shape
  (all_samples, D). Default: None. n_rays: Number of rays. Should be provided
  together with `ray_indices`. Default: None.

  Returns:
      Accumulated values with shape (n_rays, D). If `values` is not given we
  return the accumulated weights, in which case D == 1.

  Examples:

  .. code-block:: python

      # Rendering: accumulate rgbs, opacities, and depths along the rays.
      colors = accumulate_along_rays(weights, rgbs, ray_indices, n_rays)
      opacities = accumulate_along_rays(weights, None, ray_indices, n_rays)
      depths = accumulate_along_rays(
          weights,
          (t_starts + t_ends)[:, None] / 2.0,
          ray_indices,
          n_rays,
      )
      # (n_rays, 3), (n_rays, 1), (n_rays, 1)
      print(colors.shape, opacities.shape, depths.shape)

  """ */

  // If values are not provided, accumulate weights along rays
  torch::Tensor src;
  if (values.numel() == 0) {
    src = weights.unsqueeze(-1);
  } else {
    TORCH_CHECK(values.dim() == (weights.dim() + 1));
    src = weights.unsqueeze(-1) * values;
  }

  if (ray_indices.numel() != 0) {
    TORCH_CHECK(n_rays != -1);
    TORCH_CHECK(weights.dim() == 1);
    torch::Tensor outputs = torch::zeros({n_rays, src.size(-1)}, src.options());
    outputs.index_add_(0, ray_indices, src);
    return outputs;
  } else {
    return torch::sum(src, -2);
  }
}
} // namespace nerfacc
