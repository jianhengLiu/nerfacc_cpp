#pragma once
#include <nerfacc/cuda/csrc/nerfacc.h>
namespace nerfacc {
class ExclusiveProdFunction
    : public torch::autograd::Function<ExclusiveProdFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor chunk_starts,
                               torch::Tensor chunk_cnts, torch::Tensor inputs) {
    chunk_starts = chunk_starts.contiguous();
    chunk_cnts = chunk_cnts.contiguous();
    inputs = inputs.contiguous();
    torch::Tensor outputs =
        exclusive_prod_forward(chunk_starts, chunk_cnts, inputs);
    // std::cout << ctx->needs_input_grad(0) << std::endl;
    // std::cout << ctx->needs_input_grad(1) << std::endl;
    // std::cout << ctx->needs_input_grad(2) << std::endl;
    // if (ctx->needs_input_grad(2))

    ctx->save_for_backward({chunk_starts, chunk_cnts, inputs, outputs});

    return outputs;
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs) {
    torch::Tensor grad_outputs_tensor = grad_outputs[0];
    grad_outputs_tensor = grad_outputs_tensor.contiguous();
    torch::Tensor grad_inputs;
    auto saved_variables = ctx->get_saved_variables();
    auto chunk_starts = saved_variables[0];
    auto chunk_cnts = saved_variables[1];
    auto inputs = saved_variables[2];
    auto outputs = saved_variables[3];

    grad_inputs = exclusive_prod_backward(chunk_starts, chunk_cnts, inputs,
                                          outputs, grad_outputs_tensor);
    return {torch::autograd::Variable(), torch::autograd::Variable(),
            grad_inputs};
  }
};

// https://pytorch.org/tutorials/advanced/dispatcher.html?highlight=autogradcontext
// TORCH_LIBRARY_IMPL(nerfacc, CUDA, m) {
//   m.impl("exclusive_prod_forward", TORCH_FN(ExclusiveProdFunction::forward));
//   m.impl("exclusive_prod_backward",
//   TORCH_FN(ExclusiveProdFunction::backward));
// }

// TORCH_LIBRARY(nerfacc, m) {
//   m.def("exclusive_prod_forward", ExclusiveProdFunction::forward);
//   m.def("exclusive_prod_backward", ExclusiveProdFunction::backward);
// }

torch::Tensor exclusive_prod(torch::Tensor inputs,
                             const torch::Tensor &packed_info = {}) {
  if (packed_info.numel() == 0) {
    torch::Tensor ones = torch::ones_like(inputs.slice(-1, 0, 1));
    torch::Tensor shifted_inputs =
        torch::cat({ones, inputs.slice(-1, 0, -1)}, -1);
    torch::Tensor outputs = torch::cumprod(shifted_inputs, -1);
    return outputs;
  } else {
    torch::Tensor chunk_starts = packed_info.select(-1, 0);
    torch::Tensor chunk_cnts = packed_info.select(-1, 1);
    torch::Tensor outputs = ExclusiveProdFunction::apply(
        chunk_starts.contiguous(), chunk_cnts.contiguous(),
        inputs.contiguous());
    return outputs;
  }
}

class ExclusiveSumFunction
    : public torch::autograd::Function<ExclusiveSumFunction> {
public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext *ctx, const torch::Tensor &chunk_starts,
      const torch::Tensor &chunk_cnts, torch::Tensor inputs,
      const torch::Tensor &normalize = torch::tensor({false}, torch::kBool)) {
    torch::Tensor outputs = exclusive_sum(chunk_starts, chunk_cnts, inputs,
                                          normalize.item<bool>(), false);

    ctx->save_for_backward({chunk_starts, chunk_cnts, normalize});

    return outputs;
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs) {
    torch::Tensor grad_outputs_tensor = grad_outputs[0];
    grad_outputs_tensor = grad_outputs_tensor.contiguous();
    torch::Tensor grad_inputs;
    auto saved_variables = ctx->get_saved_variables();
    auto chunk_starts = saved_variables[0];
    auto chunk_cnts = saved_variables[1];
    auto normalize = saved_variables[2];

    assert(normalize.item<bool>() == false);

    grad_inputs = exclusive_sum(chunk_starts, chunk_cnts, grad_outputs_tensor,
                                normalize.item<bool>(), true);
    return {torch::autograd::Variable(), torch::autograd::Variable(),
            grad_inputs, torch::autograd::Variable()};
  }
};

torch::Tensor exclusive_sum(const torch::Tensor &inputs,
                            const torch::Tensor &packed_info = {}) {
  if (packed_info.numel() == 0) {
    torch::Tensor outputs =
        torch::cumsum(torch::cat({torch::zeros_like(inputs.slice(-1, 0, 1)),
                                  inputs.slice(-1, 0, -1)},
                                 -1),
                      -1);
    return outputs;
  } else {
    assert(inputs.dim() == 1 && packed_info.dim() == 2 &&
           packed_info.size(-1) == 2);
    auto unbind_results = packed_info.unbind(-1);
    torch::Tensor chunk_starts = unbind_results[0];
    torch::Tensor chunk_cnts = unbind_results[1];
    torch::Tensor outputs = ExclusiveSumFunction::apply(
        chunk_starts.contiguous(), chunk_cnts.contiguous(), inputs.contiguous(),
        torch::tensor({false}, torch::kBool));
    return outputs;
  }
}

} // namespace nerfacc