#pragma once
#include "volrend.hpp"

namespace nerfacc {

enum ColorType {
  NONE,
  RANDOM,
  LAST_SAMPLE,
  WHITE,
  BLACK,
  RED,
  GREEN,
  BLUE,
};

torch::Tensor get_color(ColorType color_type) {
  switch (color_type) {
  case WHITE:
    return torch::tensor({1.0, 1.0, 1.0});
  case BLACK:
    return torch::tensor({0.0, 0.0, 0.0});
  case RED:
    return torch::tensor({1.0, 0.0, 0.0});
  case GREEN:
    return torch::tensor({0.0, 1.0, 0.0});
  case BLUE:
    return torch::tensor({0.0, 0.0, 1.0});
  default:
    throw std::runtime_error("Invalid color type");
  }
}

class RGBRenderer : public torch::nn::Module {
public:
  explicit RGBRenderer(ColorType background_color = RANDOM) {
    background_color_ = background_color;
  }

  static std::vector<torch::Tensor>
  combine_rgb(const torch::Tensor &rgb, const torch::Tensor &weights,
              const ColorType &background_color = RANDOM,
              const torch::Tensor &ray_indices = torch::Tensor(),
              int num_rays = -1) {
    /*
    """Composite samples along ray and render color image.
    If background color is random, no BG color is added - as if the background
    was black!

    Args:
        rgb: RGB for each sample
        weights: Weights for each sample
        background_color: Background color as RGB.
        ray_indices: Ray index for each sample, used when samples are packed.
        num_rays: Number of rays, used when samples are packed.

    Returns:
        Outputs rgb values.
    """
     */
    torch::Tensor comp_rgb, accumulated_weight;

    if (ray_indices.defined() && num_rays != -1) {
      // Necessary for packed samples from volumetric ray sampler
      if (background_color == LAST_SAMPLE) {
        throw std::runtime_error("Background color 'LAST_SAMPLE' not "
                                 "implemented for packed samples.");
      }
      comp_rgb = nerfacc::accumulate_along_rays(weights.select(-1, 0), rgb,
                                                ray_indices, num_rays);
      accumulated_weight = nerfacc::accumulate_along_rays(
          weights.select(-1, 0), torch::Tensor(), ray_indices, num_rays);
    } else {
      comp_rgb = torch::sum(weights * rgb, -2);
      accumulated_weight = torch::sum(weights, -2);
    }
    torch::Tensor bg_color;
    if (background_color == RANDOM) {
      // If background color is random, the predicted color is returned
      // without blending, as if the background color was black.
      return {comp_rgb, accumulated_weight};
    } else if (background_color == LAST_SAMPLE) {
      // Note, this is only supported for non-packed samples.
      bg_color = rgb.slice(-1, -1);
    } else {
      bg_color = get_background_color(background_color, comp_rgb.sizes(),
                                      comp_rgb.device());
    }
    comp_rgb = comp_rgb + bg_color * (1.0 - accumulated_weight);
    return {comp_rgb, accumulated_weight};
  }

  static torch::Tensor get_background_color(ColorType background_color,
                                            torch::IntArrayRef shape,
                                            torch::Device device) {
    TORCH_CHECK(shape.back() == 3);

    return get_color(background_color).expand(shape).to(device);
  }

  std::vector<torch::Tensor>
  forward(torch::Tensor rgb, torch::Tensor weights,
          const torch::Tensor &ray_indices = torch::Tensor(), int num_rays = -1,
          ColorType background_color = NONE) {

    if (background_color == NONE) {
      background_color = background_color_;
    }
    if (!is_training()) {
      rgb = torch::nan_to_num(rgb);
    }
    auto combine_results =
        combine_rgb(rgb, weights, background_color, ray_indices, num_rays);
    auto comp_rgb = combine_results[0];
    auto accumulated_weight = combine_results[1];

    if (!is_training()) {
      torch::clamp_(comp_rgb, 0.0, 1.0);
    }
    return {comp_rgb, accumulated_weight};
  }

private:
  ColorType background_color_;
};

class DepthRenderer : public torch::nn::Module {
public:
  explicit DepthRenderer(const std::string &_method = "median")
      : method_(_method) {}

  torch::Tensor forward(const torch::Tensor &weights,
                        const torch::Tensor &ray_samples_starts,
                        const torch::Tensor &ray_samples_ends,
                        torch::Tensor ray_indices = torch::Tensor(),
                        int64_t num_rays = -1) {
    if (method_ == "median") {
      torch::Tensor steps = (ray_samples_starts + ray_samples_ends) / 2;

      if (ray_indices.defined() && (num_rays != -1)) {
        throw std::runtime_error(
            "Median depth calculation is not implemented for packed samples.");
      }

      torch::Tensor cumulative_weights =
          torch::cumsum(weights.select(-1, 0), -1);
      torch::Tensor split = torch::ones_like(weights).select(-1, 0) *
                            0.5; // TODO: check dim is correct
      torch::Tensor median_index =
          torch::searchsorted(cumulative_weights, split, false, false, "left");
      median_index = torch::clamp(median_index, 0, steps.size(-2) - 1);
      torch::Tensor median_depth =
          torch::gather(steps.select(-1, 0), -1, median_index);
      return median_depth;
    } else if (method_ == "expected") {
      static float eps = 1e-6f;
      torch::Tensor steps = (ray_samples_starts + ray_samples_ends) / 2;

      if (ray_indices.defined() && (num_rays != -1)) {
        torch::Tensor depth = accumulate_along_rays(
            weights.select(-1, 0), steps, ray_indices, num_rays);
        torch::Tensor accumulation = accumulate_along_rays(
            weights.select(-1, 0), torch::Tensor(), ray_indices, num_rays);
        // depth = depth / (accumulation + eps);
        return depth;
      } else {
        torch::Tensor depth =
            torch::sum(weights * steps, -2) / (torch::sum(weights, -2) + eps);
        depth = torch::clip(depth, steps.min(), steps.max());
        return depth;
      }
    }

    throw std::runtime_error("Method " + method_ + " not implemented");
  }

private:
  std::string method_;
};
} // namespace nerfacc
