# All Codes Here are copy from easydist

# Most functions in this file are copied from https://github.com/pytorch/pytorch/blob/main/torch/distributed/_spmd/api.py.
# We copy these functions because they are not ready in released pytorch.

from functools import partial

import torch
from torch._decomp.decompositions import mse_loss, mse_loss_backward
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value,
                                   _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach,
                                   _capturable_doc, _differentiable_doc, _foreach_doc, _fused_doc,
                                   _maximize_doc)
from torch.optim.adam import _single_tensor_adam
import contextlib

aten = torch.ops.aten  # pyre-ignore

# from torch/distributed/_spmd/api.py
def _fused_adam_decomp(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=1,
    beta1=1,
    beta2=1,
    weight_decay=1,
    eps=1,
    amsgrad=True,
    maximize=True,
    grad_scale=None,
    found_inf=None,
):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
    # print("here")
    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # skip gradient copying as we don't need to copy gradients back
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


# from torch/distributed/_spmd/api.py
def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_unaop_decomp(op, self):
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_binop_list_decomp(op, self, other):
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_binop_scalar_decomp(op, self, scalar=1):
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# from torch/distributed/_spmd/api.py
def _foreach_addcop_tensor_decomp(op, self, tensor1, tensor2, tensor):
    self_updated = op(self, tensor1, tensor2, tensor)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


# modified from torch/distributed/_spmd/api.py
EASYDIST_DECOMP_TABLE = {
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_add.Scalar),
    aten._foreach_addcdiv_.Scalar: partial(_foreach_addcop_scalar_decomp,
                                           aten._foreach_addcdiv.Scalar),
    aten._foreach_addcdiv_.Tensor: partial(_foreach_addcop_tensor_decomp,
                                           aten._foreach_addcdiv.Tensor),
    aten._foreach_addcmul_.Scalar: partial(_foreach_addcop_scalar_decomp,
                                           aten._foreach_addcmul.Scalar),
    aten._foreach_div_.List: partial(_foreach_binop_list_decomp, aten._foreach_div.List),
    aten._foreach_mul_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_mul.Scalar),
    aten._foreach_neg_.default: partial(_foreach_unaop_decomp, aten._foreach_neg.default),
    aten._foreach_reciprocal_.default: partial(_foreach_unaop_decomp,
                                               aten._foreach_reciprocal.default),
    aten._foreach_sub_.Scalar: partial(_foreach_binop_scalar_decomp, aten._foreach_sub.Scalar),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.mse_loss.default: mse_loss,
    aten.mse_loss_backward.default: mse_loss_backward,
}

def _traceable_single_tensor_adam(params,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                        grad_scale,
                        found_inf,
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):
    
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            # orig param.addcdiv_(exp_avg, denom)
            param.add_( exp_avg / denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            # orig param.addcdiv_(exp_avg, denom, value=-step_size)
            param.add_( exp_avg / denom *(-step_size))

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])

@contextlib.contextmanager
def adam_traceable_context():
    orig_adam_code = _single_tensor_adam.__code__
    torch._utils.is_compiling.__code__ = _traceable_single_tensor_adam.__code__
    try:
        yield
    finally:
        _single_tensor_adam.__code__ = orig_adam_code